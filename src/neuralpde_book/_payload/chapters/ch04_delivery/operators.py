"""Shared-data, three-seed MIMO and RB-DINO/state training for final delivery.

MIMO retains four multiplicative branches and the shared spatial trunk. Modal
normalization and a declared 17-mode output projection fix the scaling mismatch.
No linear head polishing is applied to these neural checkpoints.
"""

from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from torch import nn
from chapters.ch04_adaptation.operator_models import MIMO, RBDINO
from chapters.ch04_adaptation.operator_data import generate, validate, basis, EIGENVALUES, ALPHA
from chapters.ch04_adaptation.train_operators import pca
from .frozen_jobs import write


def identity():
    paths = [
        Path(__file__),
        Path("chapters/ch04_adaptation/operator_models.py"),
        Path("chapters/ch04_adaptation/operator_data.py"),
        Path("chapters/ch04_adaptation/train_operators.py"),
    ]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class ModalMIMO(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MIMO().double()
        grid = np.linspace(0, 1, 17)
        x, y = np.meshgrid(grid, grid, indexing="ij")
        bs = basis(np.stack([x.ravel(), y.ravel()], 1))
        z, w = np.polynomial.legendre.leggauss(32)
        x, y = np.meshgrid((z + 1) / 2, (z + 1) / 2, indexing="ij")
        xy = np.stack([x.ravel(), y.ravel()], 1)
        weights = np.outer(w / 2, w / 2).ravel()
        for name, value in [
            ("sensors", bs),
            ("points", xy),
            ("project", 4 * basis(xy).T * weights),
            ("input_mean", np.zeros((2, 17))),
            ("input_scale", np.ones((2, 17))),
            ("output_mean", np.zeros((2, 17))),
            ("output_scale", np.ones((2, 17))),
        ]:
            self.register_buffer(name, torch.tensor(value, dtype=torch.float64))

    def standardized(self, z):
        sensors = (z @ self.sensors.T) / np.sqrt(17)
        t = self.project @ self.net.basis(self.points)
        by = self.net.branches[0](sensors[:, 0]) * self.net.branches[1](sensors[:, 1])
        bu = self.net.branches[2](sensors[:, 0]) * self.net.branches[3](sensors[:, 1])
        return torch.stack([by @ t.T, bu @ t.T], 1)

    def forward(self, x):
        return self.output_mean + self.output_scale * self.standardized(
            (x - self.input_mean) / self.input_scale
        )


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["schema"] != "ch04-delivery-operator-v1" or ck["source_identity"] != identity():
        raise ValueError("checkpoint identity mismatch")
    net = ModalMIMO() if ck["method"] == "mimo" else RBDINO()
    net = net.double().to(device)
    net.load_state_dict(ck["model"])
    if not all(bool(torch.isfinite(v).all()) for v in net.state_dict().values()):
        raise ValueError("nonfinite model")
    return net, ck


def train(method, seed, output, updates=4000, device="cuda"):
    if method not in ("mimo", "rb_dino", "rb_state") or seed not in (0, 1, 2) or updates < 1:
        raise ValueError("config")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    tr, va = generate("train", 256, 0), generate("validation", 64, 0)
    hashes = validate(tr, va)
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    config = dict(
        method=method,
        seed=seed,
        updates=updates,
        batch=64,
        train_data_seed=0,
        train=256,
        validation=64,
        learning_rate=1e-3,
        min_learning_rate=1e-5,
        tangent_weight=0.1 if method == "mimo" else 1.0,
        dtype="float64",
        device=device,
        test_evaluations=0,
    )
    write(out / "config.json", config)
    write(out / "data_hashes.json", hashes)
    rng = np.random.default_rng(seed + 9081)
    if method == "mimo":
        model = ModalMIMO().to(device)
        tx = tensor(np.stack([tr["f"], tr["yd"]], 1))
        ty = tensor(np.stack([tr["y"], tr["u"]], 1))
        vx = tensor(np.stack([va["f"], va["yd"]], 1))
        vy = tensor(np.stack([va["y"], va["u"]], 1))
        with torch.no_grad():
            model.input_mean.copy_(tx.mean(0))
            model.input_scale.copy_(tx.std(0, unbiased=False))
            model.output_mean.copy_(ty.mean(0))
            model.output_scale.copy_(ty.std(0, unbiased=False))
        z = (tx - model.input_mean) / model.input_scale
        t = (ty - model.output_mean) / model.output_scale

        def loss_fn(ids, step):
            loss = (model.standardized(z[ids]) - t[ids]).square().mean()
            if step > 200:
                a = ids[:8]
                b = tensor(rng.integers(0, len(tx), 8)).long()
                _, j = torch.autograd.functional.jvp(
                    model.standardized, z[a], z[b] - z[a], create_graph=True
                )
                loss = loss + 0.1 * (j - t[b] + t[a]).square().mean()
            return loss

        def metrics():
            with torch.no_grad():
                pred = model(vx)
                error = ((pred - vy).norm(dim=-1) / vy.norm(dim=-1)).mean(0)
                state = (EIGENVALUES_tensor * pred[:, 0] - pred[:, 1] - vx[:, 0]).norm(dim=1) / vx[
                    :, 0
                ].norm(dim=1)
                adj = (-ALPHA * EIGENVALUES_tensor * pred[:, 1] - pred[:, 0] + vx[:, 1]).norm(
                    dim=1
                ) / vx[:, 1].norm(dim=1)
            return dict(
                validation_score=float(error.mean()),
                state_relative_l2=float(error[0]),
                control_relative_l2=float(error[1]),
                state_residual=float(state.mean()),
                adjoint_residual=float(adj.mean()),
            )

        EIGENVALUES_tensor = tensor(EIGENVALUES)
    else:
        model = RBDINO().double().to(device)
        with torch.no_grad():
            for prefix, key, bkey in [("input", "rhs", "Psi"), ("output", "state", "Phi")]:
                mean, b, scale = pca(tr[key])
                getattr(model, prefix + "_mean").copy_(tensor(mean))
                getattr(model, bkey).copy_(tensor(b))
                getattr(model, prefix + "_scale").copy_(tensor(scale))
        tx, ty, vx, vy = map(tensor, [tr["rhs"], tr["state"], va["rhs"], va["state"]])
        target = ((ty - model.output_mean) @ model.Phi) / model.output_scale
        j = (
            (model.Phi.T @ tensor(np.diag(1 / EIGENVALUES)) @ model.Psi)
            * model.input_scale[None, :]
            / model.output_scale[:, None]
        )

        def loss_fn(ids, step):
            z = model.encode(tx[ids])
            loss = (model.latent_net(z) - target[ids]).square().mean()
            if method == "rb_dino":
                jac = torch.func.vmap(torch.func.jacfwd(model.latent_net))(z[:16])
                loss = loss + (jac - j).square().sum((1, 2)).mean() / j.square().sum()
            return loss

        def metrics():
            with torch.no_grad():
                error = float(((model(vx) - vy).norm(dim=1) / vy.norm(dim=1)).mean())
            jac = model.jacobian(vx).detach()
            derivative = float(((jac - j).square().sum((1, 2)) / j.square().sum()).sqrt().mean())
            return dict(
                validation_score=error + derivative,
                state_relative_l2=error,
                normalized_jacobian_relative=derivative,
            )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=1e-5)
    best = float("inf")
    best_step = -1
    start = time.perf_counter()

    def checkpoint(step, full=False):
        ck = dict(
            schema="ch04-delivery-operator-v1",
            method=method,
            config=config,
            source_identity=identity(),
            data_hashes=hashes,
            step=step,
            best_step=best_step,
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        )
        if full:
            ck.update(
                optimizer=opt.state_dict(),
                scheduler=scheduler.state_dict(),
                numpy_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),
            )
        return ck

    def record(step, loss=None):
        nonlocal best, best_step
        model.eval()
        m = metrics()
        row = dict(
            step=step, updates=step, seconds=time.perf_counter() - start, training_loss=loss, **m
        )
        with (out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if m["validation_score"] < best:
            best = m["validation_score"]
            best_step = step
            torch.save(checkpoint(step), out / "best.pt")
            write(out / "metrics.json", row)
        torch.save(checkpoint(step, True), out / "last.pt")
        print(method, seed, step, m, flush=True)

    record(0)
    for step in range(1, updates + 1):
        model.train()
        ids = torch.tensor(rng.choice(256, 64, replace=False), device=device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(ids, step)
        loss.backward()
        if not bool(torch.isfinite(loss)) or any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        ):
            raise FloatingPointError("nonfinite update")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        if step % 100 == 0 or step == updates:
            record(step, float(loss.detach()))
    restored, ck = load(out / "best.pt", device)
    with torch.no_grad():
        prediction = restored(vx).cpu().numpy()
    np.savez_compressed(out / "validation_prediction.npz", prediction=prediction, ids=va["ids"])
    write(
        out / "run.json",
        dict(
            status="completed",
            updates=updates,
            best_step=best_step,
            seconds=time.perf_counter() - start,
            parameter_count=sum(p.numel() for p in model.parameters()),
            source_identity=identity(),
            test_evaluations=0,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["mimo", "rb_dino", "rb_state"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--updates", type=int, default=4000)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    a = p.parse_args()
    train(a.method, a.seed, a.output, a.updates, a.device)
