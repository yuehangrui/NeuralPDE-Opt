"""Shared six-way training protocol, validation selection, source-bound reload."""

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
import torch
from .models import Operator
from .data import load as load_data, arrays
from .reference import Poisson


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def identity():
    folder = Path(__file__).parent
    paths = [folder / name for name in ("models.py", "reference.py", "data.py", "training.py")]
    paths.append(folder.parent / "ch04_adaptation/operator_data.py")
    return {
        str(p.relative_to(folder.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }


@dataclass
class Config:
    architecture: str = "fno"
    role: str = "kkt"
    epochs: int = 1600
    seed: int = 0
    device: str = "cuda"
    batch_size: int = 0
    n: int = 32
    alpha: float = 0.001

    def validate(self):
        if self.architecture not in ("fno", "deeponet", "fedonet") or self.role not in (
            "state",
            "kkt",
        ):
            raise ValueError("invalid architecture/role")
        if (
            type(self.epochs) is not int
            or self.epochs < 1
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise ValueError("invalid epochs/seed")
        if type(self.batch_size) is not int or self.batch_size < 0:
            raise ValueError("invalid batch size")
        if self.n != 32 or self.alpha != 0.001:
            raise ValueError("this version is fixed to n=32, alpha=0.001")
        return self

    @property
    def batch(self):
        return self.batch_size or (128 if self.architecture == "fno" else 1024)


def build(cfg):
    cfg.validate()
    torch.manual_seed(cfg.seed)
    return Operator(cfg.architecture, cfg.role, cfg.n, cfg.alpha).to(
        device=cfg.device, dtype=torch.float32
    )


def load(path, device="cpu", dtype=torch.float32):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("schema") != "ch04-grid-operators-v1" or ck["source_identity"] != identity():
        raise ValueError("checkpoint schema/source mismatch")
    cfg = Config(**ck["config"])
    cfg.device = device
    model = build(cfg)
    model.load_state_dict(ck["model"])
    if any(not bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError("nonfinite weights")
    return model.to(dtype=dtype), cfg, ck


def predict(model, inputs, batch=32):
    param = next(model.parameters())
    result = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(inputs), batch):
            result.append(
                model(
                    torch.as_tensor(inputs[i : i + batch], device=param.device, dtype=param.dtype)
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(result)


def metrics(pred, target, inputs, role, ref):
    rel = lambda a, b: ref.norm(a - b) / np.maximum(ref.norm(b), 1e-14)
    ys = rel(pred[:, 0], target[:, 0])
    result = dict(state_relative_l2_mean=float(ys.mean()), state_relative_l2_max=float(ys.max()))
    result["validation_score"] = result["state_relative_l2_mean"]
    if role == "kkt":
        y, u = pred[:, 0], pred[:, 1]
        p = -ref.alpha * u
        us = rel(u, target[:, 1])
        r = ref.apply(y) - inputs[:, 0] - u
        adj = ref.apply(p) - y + inputs[:, 1]
        _, _, g, j = ref.recover(inputs[:, 0], inputs[:, 1], u)
        _, _, g0, _ = ref.recover(inputs[:, 0], inputs[:, 1], np.zeros_like(u))
        result.update(
            control_relative_l2_mean=float(us.mean()),
            control_relative_l2_max=float(us.max()),
            adjoint_residual_l2_mean=float(ref.norm(adj).mean()),
            recovered_gradient_relative_mean=float(np.mean(ref.norm(g) / ref.norm(g0))),
            true_objective_mean=float(j.mean()),
        )
        result["validation_score"] = 0.5 * (
            result["state_relative_l2_mean"] + result["control_relative_l2_mean"]
        )
    else:
        r = ref.apply(pred[:, 0]) - inputs[:, 0] - inputs[:, 1]
    result["state_residual_relative_mean"] = float(
        np.mean(ref.norm(r) / ref.norm(inputs.sum(1) if role == "state" else inputs[:, 0]))
    )
    if not all(np.isfinite(v) for v in result.values()):
        raise FloatingPointError("nonfinite validation metrics")
    return result


def train(cfg, data_path, output):
    cfg.validate()
    torch.set_num_threads(2)
    tr, va, manifest = load_data(data_path)
    if manifest["n"] != cfg.n or manifest["alpha"] != cfg.alpha:
        raise ValueError("data/config mismatch")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    model = build(cfg)
    ref = Poisson(cfg.n, cfg.alpha)
    tx, ty = arrays(tr, cfg.role)
    vx, vy = arrays(va, cfg.role)
    xt = torch.tensor(tx, device=cfg.device, dtype=torch.float32)
    yt = torch.tensor(ty, device=cfg.device, dtype=torch.float32)
    with torch.no_grad():
        model.input_scale.copy_(
            xt[:, :, 1:-1, 1:-1].square().mean((0, 2, 3), keepdim=True).sqrt().clamp_min(1e-10)
        )
        model.output_scale.copy_(
            yt[:, :, 1:-1, 1:-1].square().mean((0, 2, 3), keepdim=True).sqrt().clamp_min(1e-10)
        )
    if cfg.architecture == "fno":
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=30, factor=0.5)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    write(root / "config.json", asdict(cfg))
    write(root / "data_manifest.json", manifest)
    src = identity()
    rng = torch.Generator().manual_seed(cfg.seed + 7711)
    updates = 0
    best = float("inf")
    best_epoch = -1
    start = time.perf_counter()

    def checkpoint(epoch, full=False):
        c = dict(
            schema="ch04-grid-operators-v1",
            config=asdict(cfg),
            source_identity=src,
            data_hashes=manifest["hashes"],
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            epoch=epoch,
            updates=updates,
            best_epoch=best_epoch,
            best_score=best,
        )
        if full:
            c.update(
                optimizer=opt.state_dict(),
                scheduler=scheduler.state_dict(),
                permutation_rng=rng.get_state(),
                torch_rng=torch.get_rng_state(),
            )
        return c

    def record(epoch, train_loss=None):
        nonlocal best, best_epoch
        model.eval()
        pred = predict(model, vx)
        m = metrics(pred, vy, vx, cfg.role, ref)
        row = dict(
            epoch=epoch,
            updates=updates,
            seconds=time.perf_counter() - start,
            train_loss=train_loss,
            lr=opt.param_groups[0]["lr"],
            **m,
        )
        with (root / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if m["validation_score"] < best:
            best = m["validation_score"]
            best_epoch = epoch
            torch.save(checkpoint(epoch), root / "best.pt")
            write(root / "metrics.json", row)
        if epoch % 20 == 0 or epoch == cfg.epochs:
            torch.save(checkpoint(epoch, True), root / "last.pt")
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in ("epoch", "updates", "state_relative_l2_mean", "validation_score")
                    }
                ),
                flush=True,
            )
        return m

    try:
        record(0)
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            perm = torch.randperm(len(tx), generator=rng)
            loss_sum = 0.0
            for ids in perm.split(cfg.batch):
                ids = ids.to(cfg.device)
                opt.zero_grad(set_to_none=True)
                pred = model(xt[ids])
                error = (pred - yt[ids]) / model.output_scale
                loss = error[:, :, 1:-1, 1:-1].square().mean()
                loss.backward()
                if not bool(torch.isfinite(loss)) or any(
                    p.grad is not None and not bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()
                ):
                    raise FloatingPointError("nonfinite training update")
                if cfg.architecture != "fno":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                updates += 1
                loss_sum += float(loss.detach()) * len(ids)
            m = record(epoch, loss_sum / len(tx))
            if cfg.architecture == "fno":
                scheduler.step(m["validation_score"])
            else:
                scheduler.step()
        # Persist scheduler after the final step; checkpoints are not an automatic resume API.
        torch.save(checkpoint(cfg.epochs, True), root / "last.pt")
        selected, _, ck = load(root / "best.pt", cfg.device)
        pred = predict(selected, vx)
        check = metrics(pred, vy, vx, cfg.role, ref)
        original = json.loads((root / "metrics.json").read_text())
        for k, v in check.items():
            if not np.isclose(v, original[k], rtol=1e-6, atol=1e-8):
                raise AssertionError("reload metrics mismatch")
        np.savez_compressed(root / "validation_prediction.npz", prediction=pred, ids=va["ids"])
        write(
            root / "run.json",
            dict(
                status="completed",
                epochs=cfg.epochs,
                updates=updates,
                best_epoch=best_epoch,
                source_identity=src,
                seconds=time.perf_counter() - start,
                parameter_count=sum(p.numel() for p in model.parameters()),
                test_evaluations=0,
                nominal_full_epochs=1600,
                full_budget_completed=cfg.epochs >= 1600,
            ),
        )
    except Exception as e:
        write(root / "run.json", dict(status="failed", error=str(e), updates=updates))
        raise
