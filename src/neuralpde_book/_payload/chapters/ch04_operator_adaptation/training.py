"""Train/val adaptation, exact budget accounting, and train-only head polish.

Book/code notation:
Book 4.4.2: state-role and KKT-role physics losses use the same positive spectral Laplacian. physical_prediction denotes decoded outputs, not the physical adjoint p.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, hashlib, json, math, time
import numpy as np
import torch
from chapters.ch04_operator_learning.data import load as load_data, arrays
from chapters.ch04_operator_learning.training import write, metrics
from chapters.ch04_operator_learning.reference import Poisson
from .models import AdaptedOperator


@dataclass
class Config:
    architecture: str = "fno"
    role: str = "state"
    epochs: int = 1600
    seed: int = 0
    batch_size: int = 0
    device: str = "cuda"
    tangent_weight: float = 0.1
    physics_weight: float = 0.1
    tangent_warmup: int = 200
    tangent_samples: int = 8
    small: bool = False

    def validate(self):
        if self.architecture not in ("fno", "deeponet", "fedonet") or self.role not in (
            "state",
            "kkt",
        ):
            raise ValueError("architecture/role")
        for k in ("epochs", "seed", "batch_size", "tangent_warmup", "tangent_samples"):
            v = getattr(self, k)
            if type(v) is not int or v < (1 if k in ("epochs", "tangent_samples") else 0):
                raise ValueError(k)
        if any(not math.isfinite(v) or v < 0 for v in (self.tangent_weight, self.physics_weight)):
            raise ValueError("loss weights")
        return self

    @property
    def batch(self):
        return self.batch_size or (128 if self.architecture == "fno" else 256)


def identity():
    root = Path(__file__).resolve().parents[1]
    names = ["ch04_operator_adaptation/" + f for f in ("models.py", "training.py")]
    names += [
        "ch04_operator_learning/" + f
        for f in ("models.py", "data.py", "reference.py", "training.py")
    ]
    names += ["ch04_adaptation/operator_data.py"]
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}


def build(cfg):
    cfg.validate()
    torch.manual_seed(cfg.seed)
    return AdaptedOperator(cfg.architecture, cfg.role, small=cfg.small).to(cfg.device)


def load(path, device="cpu", dtype=torch.float64):
    if dtype != torch.float64:
        raise ValueError("adaptation checkpoints require float64")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("schema") != "ch04-operator-adaptation-v2" or ck["source_identity"] != identity():
        raise ValueError("checkpoint schema/source mismatch")
    cfg = Config(**ck["config"])
    cfg.device = device
    model = build(cfg)
    model.load_state_dict(ck["model"])
    if any(not bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError("nonfinite checkpoint")
    return model, cfg, ck


def predict(model, x, batch=32):
    model.eval()
    result = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(x), batch):
            result.append(
                model(torch.tensor(x[i : i + batch], device=device, dtype=torch.float64))
                .cpu()
                .numpy()
            )
    return np.concatenate(result)


def physics_loss(model, c, standardized_prediction):
    physical_prediction = model.output_mean + model.output_scale * standardized_prediction
    eigenvalues = model.eigenvalues
    if model.role == "state":
        residual = (eigenvalues * physical_prediction[:, 0] - c) / (
            eigenvalues * model.output_scale[0]
        )
        return residual.square().mean()
    f, yd = c.reshape(-1, 2, 17).unbind(1)
    y, u = physical_prediction.unbind(1)
    sf, sd = model.input_scale.reshape(2, 17)
    sy, su = model.output_scale
    rs = (eigenvalues * y - u - f) / torch.sqrt((eigenvalues * sy) ** 2 + su**2 + sf**2)
    ra = (-0.001 * eigenvalues * u - y + yd) / torch.sqrt(
        (0.001 * eigenvalues * su) ** 2 + sy**2 + sd**2
    )
    return 0.5 * (rs.square().mean() + ra.square().mean())


def tangent_loss(model, z, direction, target):
    # For this declared LINEAR PDE/KKT map, differences of two independent
    # training solutions are exact directional derivative labels. Never use
    # model-generated finite differences as the reference derivative.
    _, jvp = torch.autograd.functional.jvp(model.standardized, z, direction, create_graph=True)
    return (jvp - target).square().mean()


def train(cfg, data_path, output):
    cfg.validate()
    torch.set_num_threads(2)
    tr, va, manifest = load_data(data_path)
    if manifest["n"] != 32 or manifest["alpha"] != 0.001:
        raise ValueError("family/config mismatch")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    model = build(cfg)
    ref = Poisson()
    tx, ty = arrays(tr, cfg.role)
    vx, vy = arrays(va, cfg.role)
    x = torch.tensor(tx, device=cfg.device, dtype=torch.float64)
    y = torch.tensor(ty, device=cfg.device, dtype=torch.float64)
    model.set_normalization(x, y)
    with torch.no_grad():
        c = model.inputs(x)
        z = (c - model.input_mean) / model.input_scale
        t = (model.encode_fields(y) - model.output_mean) / model.output_scale
    opt = torch.optim.AdamW(
        model.parameters(), lr=1e-3 if cfg.architecture == "fno" else 5e-4, weight_decay=1e-6
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    rng = torch.Generator().manual_seed(cfg.seed + 8801)
    updates = 0
    tangent_updates = 0
    best = float("inf")
    best_epoch = -1
    start = time.perf_counter()
    write(out / "config.json", asdict(cfg))
    write(out / "data_manifest.json", manifest)
    write(
        out / "protocol.json",
        dict(
            dtype="float64",
            selection="earliest minimum validation L2 (state) or mean y/u L2 (kkt)",
            tangent_labels="train solution differences, valid for this declared linear problem only",
            head_polish="train-only linear SVD fit after freezing selected neural features; separate result",
            test_evaluations=0,
        ),
    )

    def checkpoint(epoch, full=False, stage="neural"):
        obj = dict(
            schema="ch04-operator-adaptation-v2",
            config=asdict(cfg),
            source_identity=identity(),
            data_hashes=manifest["hashes"],
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            epoch=epoch,
            updates=updates,
            tangent_updates=tangent_updates,
            best_epoch=best_epoch,
            best_score=best,
            stage=stage,
        )
        if full:
            obj.update(
                optimizer=opt.state_dict(),
                scheduler=scheduler.state_dict(),
                permutation_rng=rng.get_state(),
                torch_rng=torch.get_rng_state(),
            )
        return obj

    def record(epoch, losses=None):
        nonlocal best, best_epoch
        m = metrics(predict(model, vx), vy, vx, cfg.role, ref)
        row = dict(
            epoch=epoch,
            updates=updates,
            tangent_updates=tangent_updates,
            seconds=time.perf_counter() - start,
            lr=opt.param_groups[0]["lr"],
            losses=losses,
            **m,
        )
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        if m["validation_score"] < best:
            best = m["validation_score"]
            best_epoch = epoch
            torch.save(checkpoint(epoch), out / "neural_best.pt")
            write(out / "neural_metrics.json", row)
        if epoch % 20 == 0 or epoch == cfg.epochs:
            torch.save(checkpoint(epoch, True), out / "last.pt")
            print(cfg.architecture, cfg.role, epoch, m["validation_score"], flush=True)

    try:
        record(0)
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            perm = torch.randperm(len(x), generator=rng)
            sums = np.zeros(3)
            for ix in perm.split(cfg.batch):
                ix = ix.to(cfg.device)
                opt.zero_grad(set_to_none=True)
                standardized_prediction = model.standardized(z[ix])
                data = (standardized_prediction - t[ix]).square().mean()
                physics = physics_loss(model, c[ix], standardized_prediction)
                tangent = data.new_zeros(())
                if epoch > cfg.tangent_warmup and cfg.tangent_weight:
                    j = ix[: cfg.tangent_samples]
                    other = torch.randint(len(x), (len(j),), generator=rng).to(cfg.device)
                    tangent = tangent_loss(model, z[j], z[other] - z[j], t[other] - t[j])
                    tangent_updates += 1
                loss = data + cfg.physics_weight * physics + cfg.tangent_weight * tangent
                loss.backward()
                if not bool(torch.isfinite(loss)) or any(
                    p.grad is not None and not bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()
                ):
                    raise FloatingPointError("nonfinite update")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                updates += 1
                sums += np.array([float(a.detach()) for a in (data, physics, tangent)]) * len(ix)
            scheduler.step()
            record(epoch, dict(zip(("data", "physics", "tangent"), (sums / len(x)).tolist())))
        torch.save(checkpoint(cfg.epochs, True), out / "last.pt")
        model, _, selected = load(out / "neural_best.pt", cfg.device)
        neural_pred = predict(model, vx)
        np.savez_compressed(
            out / "neural_validation_prediction.npz", prediction=neural_pred, ids=va["ids"]
        )
        write(out / "neural_train_metrics.json", metrics(predict(model, tx), ty, tx, cfg.role, ref))
        diagnostic = model.fit_head(x, y)
        write(out / "head_fit.json", diagnostic)
        # Selected backbone epoch is retained; full accounting is in run/last.
        head_ck = dict(selected)
        head_ck["model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        head_ck["stage"] = "head_polished"
        head_ck["head_fit"] = diagnostic
        torch.save(head_ck, out / "best.pt")
        pred = predict(model, vx)
        write(out / "metrics.json", metrics(pred, vy, vx, cfg.role, ref))
        write(out / "train_metrics.json", metrics(predict(model, tx), ty, tx, cfg.role, ref))
        np.savez_compressed(out / "validation_prediction.npz", prediction=pred, ids=va["ids"])
        reloaded, _, _ = load(out / "best.pt", cfg.device)
        np.testing.assert_allclose(predict(reloaded, vx), pred, rtol=1e-11, atol=1e-12)
        write(
            out / "run.json",
            dict(
                status="completed",
                epochs=cfg.epochs,
                updates=updates,
                tangent_updates=tangent_updates,
                best_neural_epoch=best_epoch,
                full_budget_completed=cfg.epochs == 1600 and not cfg.small,
                parameter_count=sum(p.numel() for p in model.parameters()),
                seconds=time.perf_counter() - start,
                source_identity=identity(),
                test_evaluations=0,
            ),
        )
    except Exception as exc:
        write(out / "run.json", dict(status="failed", updates=updates, error=str(exc)))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    train(Config(**json.loads(Path(a.config).read_text())), a.data, a.output)
