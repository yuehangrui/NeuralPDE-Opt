"""Weighted quadrature and frequency enrichment for direct/ALM controls.

No exact solution is used by the loss. All tensor sine modes 1..m are included,
not just the manufactured solution's frequencies. The old Tanh field remains.
For residual E=-Delta(y)-u-f, this ALM multiplier tends to -p, NOT p.

Book/code notation:
Book 4.4.1 and (4.25)-(4.26): ALM uses multiplier for the state residual. The code penalty is mu*||r||^2, so book beta=2*mu and the multiplier update uses 2*mu*r.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from torch import nn
from chapters.ch04.models import Fields, boundary_factor, laplace
from chapters.ch04.problem import Problem
from chapters.ch04.evaluation import evaluate, sample_points
from chapters.ch04.fem import Reference
from chapters.ch04.training import write_json


def quadrature(n):
    if type(n) is not int or n < 2:
        raise ValueError("quadrature order must be an integer >=2")
    z, w = np.polynomial.legendre.leggauss(n)
    z, w = (z + 1) / 2, w / 2
    x, y = np.meshgrid(z, z, indexing="ij")
    return np.stack([x.ravel(), y.ravel()], 1), np.outer(w, w).reshape(-1, 1)


class EnrichedFields(Fields):
    def __init__(self, method, alpha=1e-3, width=64, depth=6, modes=12):
        super().__init__(method, alpha, width, depth)
        k = torch.cartesian_prod(torch.arange(1, modes + 1), torch.arange(1, modes + 1))
        self.register_buffer("frequencies", k.to(torch.float64) * torch.pi)
        self.register_buffer("eigenvalues", (self.frequencies**2).sum(1))
        self.state_head = nn.Parameter(torch.zeros(modes * modes, dtype=torch.float64))
        self.control_head = nn.Parameter(torch.zeros(modes * modes, dtype=torch.float64))

    def basis(self, x):
        return torch.sin(x[:, 0:1] * self.frequencies[:, 0]) * torch.sin(
            x[:, 1:2] * self.frequencies[:, 1]
        )

    def forward(self, x):
        y, u, _ = super().forward(x)
        b = self.basis(x)
        return (
            y + (b @ (self.state_head / self.eigenvalues))[:, None],
            u + (b @ self.control_head)[:, None],
            None,
        )

    def state_residual(self, x, problem):
        # Autograd only for the Tanh component, exact derivatives of sine features.
        y0 = boundary_factor(x) * self.y_net(x)
        u = self(x)[1]
        return -laplace(y0, x) + (self.basis(x) @ self.state_head)[:, None] - u - problem.f(x)


@dataclass
class InstanceConfig:
    method: str = "direct"
    architecture: str = "enriched"
    sampling: str = "gauss"
    seed: int = 0
    modes: int = 12
    width: int = 64
    depth: int = 6
    quadrature: int = 48
    random_points: int = 2304
    adam_steps: int = 1000
    lbfgs_steps: int = 600
    outer_steps: int = 6
    penalty: float = 1.0
    mu: float = 0.01
    loss_scale: float = 1.0
    validation_mesh: int = 128
    device: str = "cuda"

    def validate(self):
        if (
            self.method not in ("direct", "alm")
            or self.architecture not in ("tanh", "enriched")
            or self.sampling not in ("gauss", "random")
        ):
            raise ValueError("invalid method/architecture/sampling")
        for key in (
            "modes",
            "width",
            "depth",
            "quadrature",
            "random_points",
            "outer_steps",
            "validation_mesh",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(key)
        if self.quadrature < 2 * self.modes + 4 or self.validation_mesh < 64:
            raise ValueError("underresolved quadrature/validation")
        for key in ("adam_steps", "lbfgs_steps", "seed"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(key)
        for key in ("penalty", "mu", "loss_scale"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(key)
        if self.adam_steps + self.lbfgs_steps == 0:
            raise ValueError("empty budget")
        return self


def build(cfg):
    cfg.validate()
    torch.manual_seed(cfg.seed)
    if cfg.architecture == "enriched":
        model = EnrichedFields(cfg.method, width=cfg.width, depth=cfg.depth, modes=cfg.modes)
    else:
        model = Fields(cfg.method, width=cfg.width, depth=cfg.depth)
    return model.to(device=cfg.device, dtype=torch.float64)


def weighted_loss(model, problem, x, w, penalty, multiplier):
    y, u, _ = model(x)
    r = (
        model.state_residual(x, problem)
        if hasattr(model, "state_residual")
        else -laplace(y, x) - u - problem.f(x)
    )
    j = (w * (0.5 * (y - problem.yd(x)) ** 2 + 0.5 * problem.alpha * u**2)).sum()
    return j + (w * (penalty * r * r + multiplier * r)).sum(), j, r


def source_identity():
    paths = [Path(__file__), Path(__file__).parent / "__init__.py"]
    paths += sorted((Path(__file__).parent.parent / "ch04").glob("*.py"))
    return {
        str(p.relative_to(Path(__file__).parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def load(path, device="cpu"):
    ck = torch.load(path, weights_only=True, map_location="cpu")
    if ck["source_identity"] != source_identity():
        raise ValueError("source identity mismatch")
    cfg = InstanceConfig(**ck["config"])
    cfg.device = device
    model = build(cfg)
    model.load_state_dict(ck["model"])
    if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError("nonfinite checkpoint")
    return model, cfg, ck


def train(cfg, output):
    cfg.validate()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    problem = Problem()
    model = build(cfg)
    start = time.perf_counter()
    if cfg.sampling == "gauss":
        xn, wn = quadrature(cfg.quadrature)
    else:
        xn = sample_points(cfg.seed, 1, cfg.random_points)
        wn = np.full((len(xn), 1), 1 / len(xn))
    vn = sample_points(cfg.seed, 2, 2048)
    xbase = torch.tensor(xn, device=cfg.device, dtype=torch.float64)
    w = torch.tensor(wn, device=cfg.device, dtype=torch.float64)
    multiplier = torch.zeros_like(w)
    penalty = cfg.mu if cfg.method == "alm" else cfg.penalty
    ref = Reference(problem, cfg.validation_mesh, 4)
    state_scale = float(np.sqrt(np.sum(wn * problem.f(xn) ** 2)))
    identity = source_identity()
    updates = 0
    calls = 0
    best = float("inf")
    best_progress = None
    hashes = {
        k: hashlib.sha256(v.tobytes()).hexdigest()
        for k, v in [("train", xn), ("weights", wn), ("validation", vn)]
    }
    np.savez(out / "sampling.npz", train=xn, weights=wn, validation=vn)
    write_json(out / "config.json", asdict(cfg))

    def ck(progress):
        return dict(
            schema="ch04-adaptation-instance-v1",
            config=asdict(cfg),
            source_identity=identity,
            sample_hashes=hashes,
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            multiplier=multiplier.detach().cpu(),
            penalty=penalty,
            progress=progress,
        )

    def record(progress):
        nonlocal best, best_progress
        metrics = evaluate(model, problem, ref, vn, state_scale)
        x = xbase.detach().requires_grad_(True)
        loss, j, r = weighted_loss(model, problem, x, w, penalty, multiplier)
        gs = torch.autograd.grad(loss, tuple(model.parameters()))
        row = dict(
            progress=progress,
            updates=updates,
            closure_calls=calls,
            seconds=time.perf_counter() - start,
            train_objective=float(j.detach()),
            train_residual_rms=float((w * r * r).sum().sqrt().detach()),
            parameter_gradient_norm=float(sum((g * g).sum() for g in gs).sqrt()),
            **metrics,
        )
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        if row["validation_score"] < best:
            best = row["validation_score"]
            best_progress = progress
            torch.save(ck(progress), out / "best.pt")
            write_json(out / "metrics.json", row)
        torch.save(ck(progress), out / "last.pt")
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "progress",
                        "updates",
                        "validation_score",
                        "relative_l2_u_raw",
                        "pde_relative",
                        "parameter_gradient_norm",
                    )
                }
            ),
            flush=True,
        )

    try:
        record("initial")
        for outer in range(cfg.outer_steps if cfg.method == "alm" else 1):
            if outer:
                x = xbase.detach().requires_grad_(True)
                _, _, r = weighted_loss(model, problem, x, w, penalty, multiplier)
                multiplier = (multiplier + 2 * penalty * r.detach()).detach()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for step in range(cfg.adam_steps):
                opt.zero_grad(set_to_none=True)
                loss, _, _ = weighted_loss(
                    model, problem, xbase.detach().requires_grad_(True), w, penalty, multiplier
                )
                (cfg.loss_scale * loss).backward()
                if not bool(torch.isfinite(loss)) or any(
                    p.grad is not None and not bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()
                ):
                    raise FloatingPointError("nonfinite Adam update")
                opt.step()
                updates += 1
            record(f"outer{outer}:adam")
            if cfg.lbfgs_steps:
                opt = torch.optim.LBFGS(
                    model.parameters(),
                    lr=1.0,
                    max_iter=cfg.lbfgs_steps,
                    history_size=100,
                    tolerance_grad=1e-11,
                    tolerance_change=1e-14,
                    line_search_fn="strong_wolfe",
                )

                def closure():
                    nonlocal calls
                    calls += 1
                    opt.zero_grad(set_to_none=True)
                    loss, _, _ = weighted_loss(
                        model, problem, xbase.detach().requires_grad_(True), w, penalty, multiplier
                    )
                    (cfg.loss_scale * loss).backward()
                    if not bool(torch.isfinite(loss)) or any(
                        p.grad is not None and not bool(torch.isfinite(p.grad).all())
                        for p in model.parameters()
                    ):
                        raise FloatingPointError("nonfinite LBFGS update")
                    return cfg.loss_scale * loss

                opt.step(closure)
                updates += int(opt.state[next(model.parameters())].get("n_iter", 0))
                record(f"outer{outer}:lbfgs")
        write_json(
            out / "run.json",
            dict(
                status="completed",
                updates=updates,
                closure_calls=calls,
                best_progress=best_progress,
                seconds=time.perf_counter() - start,
                source_identity=identity,
                test_evaluations=0,
            ),
        )
    except Exception as e:
        write_json(out / "run.json", dict(status="failed", error=str(e), updates=updates))
        raise
