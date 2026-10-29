"""Training adapters for the three migrated instance-wise methods.

Fresh output directories, fixed train/validation points, common label-free
selection criterion, and separate best/last checkpoints. No automatic test run.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import scipy
import torch
from .problem import Problem
from .models import Fields, loss_terms, residuals, multiplier_update
from .fem import Reference
from .evaluation import sample_points, evaluate

HERE = Path(__file__).resolve().parent


@dataclass
class Config:
    method: str = "kkt"
    alpha: float = 1e-3
    control_frequency: int = 8
    seed: int = 0
    width: int = 64
    depth: int = 6
    dtype: str = "float64"
    device: str = "cpu"
    train_points: int = 10000
    validation_points: int = 2048
    validation_mesh: int = 64
    quadrature_order: int = 4
    adam_steps: int = 10000
    lbfgs_steps: int = 1000
    learning_rate: float = 1e-3
    evaluate_every: int = 250
    penalty: float = 1.0
    mu_init: float = 0.01
    mu_growth: float = 1.0
    mu_max: float = 1.0
    outer_steps: int = 15
    inner_steps: int = 2000
    feasibility_tol: float = 1e-4
    optimality_tol: float = 1e-4
    smoke: bool = False

    def validate(self):
        Problem(self.alpha, self.control_frequency)
        if self.method not in ("direct", "kkt", "alm") or self.dtype not in ("float32", "float64"):
            raise ValueError("invalid method or dtype")
        for key in ("seed", "adam_steps", "lbfgs_steps", "outer_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(key + " must be a nonnegative integer")
        for key in (
            "width",
            "depth",
            "train_points",
            "validation_points",
            "validation_mesh",
            "quadrature_order",
            "evaluate_every",
            "inner_steps",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(key + " must be a positive integer")
        for key in (
            "learning_rate",
            "penalty",
            "mu_init",
            "mu_growth",
            "mu_max",
            "feasibility_tol",
            "optimality_tol",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(key + " must be finite and positive")
        if self.mu_growth < 1 or self.mu_max < self.mu_init:
            raise ValueError("invalid ALM penalty schedule")
        if self.validation_mesh < 2 * self.control_frequency or self.quadrature_order < 2:
            raise ValueError("under-resolved validation mesh or quadrature")
        if self.method != "alm" and self.adam_steps + self.lbfgs_steps == 0:
            raise ValueError("training budget is empty")
        return self


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob("*.py"))}


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def build(config):
    config.validate()
    torch.manual_seed(config.seed)
    if str(config.device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("requested CUDA is unavailable")
    model = Fields(config.method, config.alpha, config.width, config.depth)
    return model.to(device=config.device, dtype=getattr(torch, config.dtype))


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema") != "ch04-instance-v1"
        or checkpoint.get("source_hashes") != source_hashes()
    ):
        raise ValueError("checkpoint schema or source identity mismatch")
    config = Config(**checkpoint["config"])
    config.device = device
    model = build(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
        raise ValueError("checkpoint contains nonfinite weights")
    return model, config, checkpoint


def tolerances_met(metrics, config):
    return (
        metrics["pde_relative"] <= config.feasibility_tol
        and metrics["reduced_gradient_relative"] <= config.optimality_tol
    )


def train(config, output):
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    problem = Problem(config.alpha, config.control_frequency)
    model = build(config)
    train_np = sample_points(config.seed, 1, config.train_points)
    val_np = sample_points(config.seed, 2, config.validation_points)
    state_scale = max(float(np.sqrt(np.mean(problem.f(train_np) ** 2))), 1.0)
    fingerprints = {
        key: hashlib.sha256(value.tobytes()).hexdigest()
        for key, value in [("train", train_np), ("validation", val_np)]
    }
    ref = Reference(problem, config.validation_mesh, config.quadrature_order)
    parameter = next(model.parameters())
    x_base = torch.as_tensor(train_np, dtype=parameter.dtype, device=parameter.device)
    multiplier = torch.zeros((len(x_base), 1), dtype=parameter.dtype, device=parameter.device)
    mu = config.mu_init
    identity = source_hashes()
    status = dict(
        schema="ch04-instance-v1",
        status="running",
        config=asdict(config),
        source_hashes=identity,
        sample_hashes=fingerprints,
        state_scale=state_scale,
        gradient_scale=ref.gradient_scale,
        selection="sqrt((independent PDE RMS/state_scale)^2+(FEM recovered gradient/gradient_scale)^2)",
        versions={
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        run_kind="smoke" if config.smoke else "training",
        completed_updates=0,
    )
    write_json(output / "run.json", status)
    write_json(output / "config.json", asdict(config))
    write_json(output / "reference.json", ref.report())
    np.savez(output / "sampling.npz", train=train_np, validation=val_np)
    best, best_progress, best_metrics = float("inf"), None, None
    updates = 0
    start = time.perf_counter()

    def checkpoint(progress):
        return dict(
            schema="ch04-instance-v1",
            config=asdict(config),
            source_hashes=identity,
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            progress=progress,
            sample_hashes=fingerprints,
            state_scale=state_scale,
            multiplier=multiplier.detach().cpu().clone() if config.method == "alm" else None,
            mu=mu,
        )

    def record(progress):
        nonlocal best, best_progress, best_metrics
        metrics = evaluate(model, problem, ref, val_np, state_scale)
        row = dict(
            progress=progress,
            updates=updates,
            seconds=time.perf_counter() - start,
            mu=mu if config.method == "alm" else None,
            **metrics,
        )
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if metrics["validation_score"] < best:  # keep earliest exact tie
            best, best_progress, best_metrics = metrics["validation_score"], progress, metrics
            torch.save(checkpoint(progress), output / "best.pt")
            write_json(output / "metrics.json", dict(selection_progress=progress, **metrics))
        torch.save(checkpoint(progress), output / "last.pt")
        print(
            json.dumps(
                dict(
                    progress=progress,
                    score=metrics["validation_score"],
                    pde=metrics["pde_relative"],
                    gradient=metrics["reduced_gradient_relative"],
                ),
                allow_nan=False,
            ),
            flush=True,
        )
        return metrics

    def adam(count, penalty, stage, record_intermediate):
        nonlocal updates
        opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        for step in range(1, count + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            x = x_base.detach().requires_grad_(True)
            loss, _, _ = loss_terms(model, problem, x, penalty, multiplier)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            if any(
                p.grad is not None and not bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
            ):
                raise FloatingPointError("nonfinite gradient")
            opt.step()
            updates += 1
            if record_intermediate and (step % config.evaluate_every == 0 or step == count):
                record(f"{stage}:{step}")

    try:
        record("initial")
        if config.method == "alm":
            for outer in range(config.outer_steps + 1):
                if outer:
                    x = x_base.detach().requires_grad_(True)
                    r, _ = residuals(model, problem, x)
                    multiplier = multiplier_update(multiplier, r, mu)  # previous penalty
                    mu = min(mu * config.mu_growth, config.mu_max)
                adam(config.inner_steps, mu, f"alm:{outer}", False)
                metrics = record(f"alm:{outer}")
                if tolerances_met(metrics, config):
                    break
        else:
            adam(config.adam_steps, config.penalty, "adam", True)
            if config.lbfgs_steps:
                model.train()
                opt = torch.optim.LBFGS(
                    model.parameters(),
                    lr=1.0,
                    max_iter=config.lbfgs_steps,
                    tolerance_grad=1e-9,
                    tolerance_change=1e-9,
                    history_size=100,
                    line_search_fn="strong_wolfe",
                )

                def closure():
                    opt.zero_grad(set_to_none=True)
                    x = x_base.detach().requires_grad_(True)
                    loss, _, _ = loss_terms(model, problem, x, config.penalty, multiplier)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite LBFGS loss")
                    loss.backward()
                    if any(
                        p.grad is not None and not bool(torch.isfinite(p.grad).all())
                        for p in model.parameters()
                    ):
                        raise FloatingPointError("nonfinite LBFGS gradient")
                    return loss

                opt.step(closure)
                actual = int(opt.state[next(iter(model.parameters()))].get("n_iter", 0))
                updates += actual
                record(f"lbfgs:{actual}")
        status.update(
            status="completed",
            completed_updates=updates,
            best_progress=best_progress,
            optimization_converged=tolerances_met(best_metrics, config),
            seconds=time.perf_counter() - start,
        )
        # A finished budget is not a claim of optimality, particularly in smoke runs.
        write_json(output / "run.json", status)
        return status
    except Exception as exc:
        status.update(status="failed", error=str(exc), completed_updates=updates)
        write_json(output / "run.json", status)
        raise
