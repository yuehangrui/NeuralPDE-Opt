"""Exact convex least-squares solve of the enriched linear output heads.

Freeze the nonlinear Tanh features; fit BOTH state/control heads together.
This removes head optimization error without reading optimal-solution labels.
It does not claim a global optimum over the nonlinear network parameters.
"""

from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from .instance import load
from chapters.ch04.models import boundary_factor, laplace
from chapters.ch04.problem import Problem
from chapters.ch04.fem import Reference
from chapters.ch04.evaluation import evaluate
from chapters.ch04.training import write_json


def fit_heads(model, problem, x, w, penalty, multiplier):
    if penalty <= 0 or not np.isfinite(penalty):
        raise ValueError("invalid penalty")
    y0 = boundary_factor(x) * model.y_net(x)
    u0 = model.second_net(x)
    r0 = -laplace(y0, x) - u0 - problem.f(x)
    b = model.basis(x).detach().cpu().numpy()
    eigenvalues = model.eigenvalues.detach().cpu().numpy()
    w = np.asarray(w).reshape(-1, 1)
    multiplier = np.asarray(multiplier).reshape(-1, 1)
    if (
        len(w) != len(x)
        or (w <= 0).any()
        or not np.isfinite(w).all()
        or not np.isfinite(multiplier).all()
    ):
        raise ValueError("invalid quadrature/multiplier")
    z = np.zeros_like(b)
    bs = b / eigenvalues
    design = np.concatenate(
        [
            np.sqrt(0.5 * w) * np.concatenate([bs, z], 1),
            np.sqrt(0.5 * problem.alpha * w) * np.concatenate([z, b], 1),
            np.sqrt(penalty * w) * np.concatenate([b, -b], 1),
        ]
    )
    rhs = np.concatenate(
        [
            np.sqrt(0.5 * w) * (problem.yd(x) - y0).detach().cpu().numpy(),
            -np.sqrt(0.5 * problem.alpha * w) * u0.detach().cpu().numpy(),
            -np.sqrt(penalty * w) * (r0.detach().cpu().numpy() + multiplier / (2 * penalty)),
        ]
    )
    solution, _, rank, svals = np.linalg.lstsq(design, rhs, rcond=None)
    if rank != design.shape[1] or not np.isfinite(solution).all():
        raise ValueError("rank deficient linear head fit")
    with torch.no_grad():
        model.state_head.copy_(
            torch.as_tensor(solution[: b.shape[1], 0], device=x.device, dtype=x.dtype)
        )
        model.control_head.copy_(
            torch.as_tensor(solution[b.shape[1] :, 0], device=x.device, dtype=x.dtype)
        )
    return dict(
        rank=int(rank),
        condition=float(svals[0] / svals[-1]),
        normal_equation_gradient_norm=float(
            np.linalg.norm(2 * design.T @ (design @ solution - rhs))
        ),
    )


def run(source, output):
    torch.set_num_threads(2)
    src = Path(source)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    model, cfg, parent = load(src / "best.pt", "cuda")
    if cfg.architecture != "enriched" or cfg.sampling != "gauss":
        raise ValueError("expected enriched Gauss parent")
    data = np.load(src / "sampling.npz")
    xn, wn, vn = data["train"], data["weights"], data["validation"]
    device = next(model.parameters()).device
    x = torch.tensor(xn, dtype=torch.float64, device=device, requires_grad=True)
    multiplier = parent["multiplier"].numpy().copy()
    mu = parent["penalty"]
    problem = Problem()
    ref = Reference(problem, 128, 4)
    scale = float(np.sqrt(np.sum(wn * problem.f(xn) ** 2)))
    best = float("inf")
    start = time.perf_counter()
    protocol = dict(
        parent=str(src),
        parent_checkpoint_sha256=hashlib.sha256((src / "best.pt").read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        head_solves=1 if cfg.method == "direct" else 5,
        nonlinear_parameters_frozen=True,
        selection="same validation_score as parent",
        test_evaluations=0,
    )
    write_json(out / "protocol.json", protocol)
    write_json(out / "config.json", vars(cfg))
    np.savez(out / "sampling.npz", train=xn, weights=wn, validation=vn)

    def record(progress, diagnostic=None):
        nonlocal best
        m = evaluate(model, problem, ref, vn, scale)
        row = dict(
            progress=progress, seconds=time.perf_counter() - start, head_diagnostics=diagnostic, **m
        )
        with (out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        ck = dict(parent)
        ck.update(
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            multiplier=torch.tensor(multiplier),
            progress=progress,
            head_polish=protocol,
        )
        if m["validation_score"] < best:
            best = m["validation_score"]
            torch.save(ck, out / "best.pt")
            write_json(out / "metrics.json", row)
        torch.save(ck, out / "last.pt")
        print(
            progress,
            "control_percent",
            100 * m["relative_l2_u_raw"],
            "score",
            m["validation_score"],
            flush=True,
        )

    record("parent_best")
    for step in range(protocol["head_solves"]):
        if step and cfg.method == "alm":
            r = model.state_residual(x, problem).detach().cpu().numpy()
            multiplier = multiplier + 2 * mu * r
        diagnostic = fit_heads(model, problem, x, wn, mu, multiplier)
        record(f"head_solve:{step+1}", diagnostic)
    write_json(
        out / "run.json",
        dict(
            status="completed",
            seconds=time.perf_counter() - start,
            head_solves=protocol["head_solves"],
            test_evaluations=0,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.source, a.output)
