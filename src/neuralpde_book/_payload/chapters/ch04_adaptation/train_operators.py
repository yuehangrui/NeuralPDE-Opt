"""Bounded non-PINN train/validation experiments for smooth control."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import minimize
from .operator_models import MIMO, RBDINO
from .operator_data import generate, validate, basis, kkt, MODES, EIGENVALUES, ALPHA
from .instance import quadrature
from chapters.ch04.training import write_json


def identity():
    folder = Path(__file__).parent
    return {
        name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
        for name in ("train_operators.py", "operator_models.py", "operator_data.py", "instance.py")
    }


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["identity"] != identity():
        raise ValueError("operator source identity mismatch")
    model = MIMO() if ck["method"] == "mimo" else RBDINO()
    model = model.to(dtype=torch.float64, device=device)
    model.load_state_dict(ck["model"])
    if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError("nonfinite model")
    return model, ck


def pca(values):
    mean = values.mean(0)
    _, _, v = np.linalg.svd(values - mean, full_matrices=False)
    q = v.T
    scale = np.sqrt(np.mean(((values - mean) @ q) ** 2, axis=0))
    if np.min(scale) < 1e-12:
        raise ValueError("rank-deficient training family")
    return mean, q, scale


def online_audit(model, data):
    """Classical BFGS over control coefficients, learned state held fixed.

    Check every accepted iterate with the true PDE and true reduced gradient.
    Online validation never selects an offline model in this driver.
    """
    model = copy.deepcopy(model).cpu().eval().requires_grad_(False)
    rows = []
    for i in range(min(8, len(data["ids"]))):
        f, yd = data["f"][i], data["yd"][i]
        trace = []
        ustar = data["u"][i]
        gradient0 = 0.25 * ((f / EIGENVALUES - yd) / EIGENVALUES)

        def fun(u):
            z = torch.tensor(f + u, dtype=torch.float64, requires_grad=True)
            y = model(z[None])[0]
            ut = z - torch.tensor(f, dtype=torch.float64)
            loss = 0.125 * (
                (y - torch.tensor(yd, dtype=torch.float64)).square().sum()
                + ALPHA * ut.square().sum()
            )
            grad = torch.autograd.grad(loss, z)[0]
            return float(loss.detach()), grad.detach().numpy()

        def record(u):
            y = (f + u) / EIGENVALUES
            control_gradient = 0.25 * ((y - yd) / EIGENVALUES + ALPHA * u)
            jp, gp = fun(u)
            trace.append(
                dict(
                    J_true=float(0.125 * (np.sum((y - yd) ** 2) + ALPHA * np.sum(u * u))),
                    J_surrogate=jp,
                    true_gradient_relative=float(
                        np.linalg.norm(control_gradient) / np.linalg.norm(gradient0)
                    ),
                    gradient_discrepancy_relative_to_initial=float(
                        np.linalg.norm(control_gradient - gp) / np.linalg.norm(gradient0)
                    ),
                    input_latent_max=float(
                        model.encode(torch.tensor((f + u)[None], dtype=torch.float64)).abs().max()
                    ),
                )
            )

        record(np.zeros_like(f))
        opt = minimize(
            fun,
            np.zeros_like(f),
            jac=True,
            method="BFGS",
            callback=record,
            options=dict(gtol=1e-10, maxiter=1000),
        )
        record(opt.x)
        jstar = float(0.125 * (np.sum((data["y"][i] - yd) ** 2) + ALPHA * np.sum(ustar**2)))
        rows.append(
            dict(
                id=str(data["ids"][i]),
                optimizer_success=bool(opt.success),
                message=str(opt.message),
                nit=int(opt.nit),
                control_relative_l2=float(np.linalg.norm(opt.x - ustar) / np.linalg.norm(ustar)),
                true_objective_gap=trace[-1]["J_true"] - jstar,
                trajectory=trace,
            )
        )
    return dict(
        role="validation",
        selection_use=False,
        method="classical BFGS with fixed learned state; exact PDE audit",
        mean_control_relative_l2=float(np.mean([r["control_relative_l2"] for r in rows])),
        rows=rows,
    )


def train(method, output, steps=2000, seed=0, device="cuda"):
    if method not in ("mimo", "rb_dino", "rb_state") or steps < 1:
        raise ValueError("invalid run")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 444)
    tr, va = generate("train", 256, seed), generate("validation", 64, seed)
    hashes = validate(tr, va)
    for name, data in [("train", tr), ("validation", va)]:
        np.savez(out / (name + ".npz"), **data)
    config = dict(
        method=method,
        steps=steps,
        seed=seed,
        device=device,
        train=256,
        validation=64,
        alpha=ALPHA,
        modes=MODES.tolist(),
        jacobian_weight=1.0 if method == "rb_dino" else 0.0,
        dtype="float64",
        test_evaluations=0,
    )
    write_json(out / "config.json", config)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
    if method == "mimo":
        model = MIMO().to(dtype=torch.float64, device=device)
        grid = np.linspace(0, 1, 17)
        xx, yy = np.meshgrid(grid, grid, indexing="ij")
        sensors = np.stack([xx.ravel(), yy.ravel()], 1)
        bs = basis(sensors)
        xq, wq = quadrature(40)
        xv, wv = quadrature(72)
        bq, bv = basis(xq), basis(xv)
        prepared = []
        for split, b in [(tr, bq), (va, bv)]:
            prepared.append(
                tuple(tensor(split[k] @ bs.T) for k in ("f", "yd"))
                + tuple(tensor(split[k] @ b.T) for k in ("y", "u"))
            )
        tf, td, ty, tu = prepared[0]
        vf, vd, vy, vu = prepared[1]
        with torch.no_grad():
            model.input_mean.copy_(torch.stack([tf.mean(0), td.mean(0)]))
            model.input_scale.copy_(
                torch.stack(
                    [
                        (tf - model.input_mean[0]).square().mean().sqrt(),
                        (td - model.input_mean[1]).square().mean().sqrt(),
                    ]
                )
            )
            model.output_scale.copy_(
                torch.stack(
                    [
                        (ty.square() * tensor(wq.T)).sum(1).mean().sqrt(),
                        (tu.square() * tensor(wq.T)).sum(1).mean().sqrt(),
                    ]
                )
            )
        tq, tv, tw, tvw = tensor(xq), tensor(xv), tensor(wq.T), tensor(wv.T)

        def loss_fn(ids):
            y, u, _ = model(tf[ids], td[ids], tq)
            return ((y - ty[ids]) ** 2 * tw).sum(1).mean() / model.output_scale[0] ** 2 + (
                (u - tu[ids]) ** 2 * tw
            ).sum(1).mean() / model.output_scale[1] ** 2

        def metrics():
            with torch.no_grad():
                y, u, p = model(vf, vd, tv)
                rel = lambda a, b: float(
                    (((a - b) ** 2 * tvw).sum(1) / (b * b * tvw).sum(1)).sqrt().mean()
                )
                ry, ru = rel(y, vy), rel(u, vu)
            return dict(
                validation_score=ry + ru,
                state_relative_l2_mean=ry,
                control_relative_l2_mean=ru,
                physical_optimality_max=float((ALPHA * u + p).abs().max()),
            )

        def audit_prediction(m):
            with torch.no_grad():
                return m(vf[:4], vd[:4], tv)[1].detach().cpu().numpy()

    else:
        model = RBDINO().to(dtype=torch.float64, device=device)
        with torch.no_grad():
            for prefix, key, bkey in [("input", "rhs", "Psi"), ("output", "state", "Phi")]:
                mean, b, scale = pca(tr[key])
                getattr(model, prefix + "_mean").copy_(tensor(mean))
                getattr(model, bkey).copy_(tensor(b))
                getattr(model, prefix + "_scale").copy_(tensor(scale))
        tx, ty, vx, vy = [tensor(v) for v in (tr["rhs"], tr["state"], va["rhs"], va["state"])]
        normalized_target = ((ty - model.output_mean) @ model.Phi) / model.output_scale
        physical_j = tensor(np.diag(1 / EIGENVALUES))
        jr = (
            (model.Phi.T @ physical_j @ model.Psi)
            * model.input_scale[None, :]
            / model.output_scale[:, None]
        )

        def loss_fn(ids):
            z = model.encode(tx[ids])
            prediction = model.latent_net(z)
            loss = (prediction - normalized_target[ids]).square().mean()
            if method == "rb_dino":
                j = torch.func.vmap(torch.func.jacfwd(model.latent_net))(z)
                loss = loss + (j - jr).square().sum((1, 2)).mean() / jr.square().sum()
            return loss

        def metrics():
            with torch.no_grad():
                rel = float(((model(vx) - vy).square().sum(1) / vy.square().sum(1)).sqrt().mean())
            jl = model.jacobian(vx).detach()
            normalized = float(((jl - jr).square().sum((1, 2)) / jr.square().sum()).sqrt().mean())
            jp = model.physical_jacobian(vx).detach()
            physical = float(
                ((jp - physical_j).square().sum((1, 2)) / physical_j.square().sum()).sqrt().mean()
            )
            return dict(
                validation_score=rel + normalized,
                state_relative_l2_mean=rel,
                normalized_jacobian_relative_mean=normalized,
                physical_jacobian_relative_mean=physical,
            )

        def audit_prediction(m):
            with torch.no_grad():
                return m(vx[:4]).detach().cpu().numpy()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-4)
    best = float("inf")
    best_step = 0
    start = time.perf_counter()
    src = identity()

    def checkpoint(step):
        return dict(
            schema="ch04-smooth-operator-v1",
            method=method,
            config=config,
            identity=src,
            data_hashes=hashes,
            step=step,
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        )

    def record(step, training_loss=None):
        nonlocal best, best_step
        model.eval()
        m = metrics()
        row = dict(step=step, seconds=time.perf_counter() - start, training_loss=training_loss, **m)
        with (out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if m["validation_score"] < best:
            best = m["validation_score"]
            best_step = step
            torch.save(checkpoint(step), out / "best.pt")
            write_json(out / "metrics.json", row)
        torch.save(checkpoint(step), out / "last.pt")
        print(json.dumps(row, allow_nan=False), flush=True)

    try:
        record(0)
        for step in range(1, steps + 1):
            model.train()
            ids = rng.choice(len(tr["ids"]), 32, replace=False)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(ids)
            loss.backward()
            if not bool(torch.isfinite(loss)) or any(
                p.grad is not None and not bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
            ):
                raise FloatingPointError("nonfinite operator update")
            optimizer.step()
            scheduler.step()
            if step % 100 == 0 or step == steps:
                record(step, float(loss.detach()))
        selected, ck = load(out / "best.pt", device)
        model.load_state_dict(ck["model"])
        pred = audit_prediction(model)
        reloaded = audit_prediction(selected)
        if not np.array_equal(pred, reloaded):
            raise RuntimeError("reload prediction mismatch")
        np.savez(out / "reload_prediction.npz", prediction=pred)
        if method != "mimo":
            write_json(out / "online_validation.json", online_audit(selected, va))
        write_json(
            out / "run.json",
            dict(
                status="completed",
                updates=steps,
                best_step=best_step,
                seconds=time.perf_counter() - start,
                identity=src,
                test_evaluations=0,
                reload_max_abs=float(abs(pred - reloaded).max()),
                parameter_count=sum(p.numel() for p in model.parameters()),
            ),
        )
    except Exception as e:
        write_json(out / "run.json", dict(status="failed", error=str(e)))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["mimo", "rb_dino", "rb_state"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    train(a.method, a.output, a.steps, device=a.device)
