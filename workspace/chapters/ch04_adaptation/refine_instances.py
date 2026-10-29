"""Continue selected train/validation models with warm L-BFGS inner solves.

No repeated Adam reset after the fitted model. Fixed saved quadrature and
multiplier, fresh output, parent checkpoint hash, explicit iteration accounting.
"""

from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from .instance import load, weighted_loss
from chapters.ch04.problem import Problem
from chapters.ch04.fem import Reference
from chapters.ch04.evaluation import evaluate
from chapters.ch04.training import write_json


def run(source, output):
    torch.set_num_threads(2)
    src = Path(source)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    model, cfg, parent = load(src / "best.pt", "cuda")
    problem = Problem()
    device = next(model.parameters()).device
    data = np.load(src / "sampling.npz")
    xn, wn, vn = data["train"], data["weights"], data["validation"]
    if cfg.sampling != "gauss":
        raise ValueError("refinement only after quadrature adaptation")
    xbase = torch.tensor(xn, dtype=torch.float64, device=device)
    w = torch.tensor(wn, dtype=torch.float64, device=device)
    multiplier = parent["multiplier"].to(device).clone()
    penalty = parent["penalty"]
    ref = Reference(problem, cfg.validation_mesh, 4)
    state_scale = float(np.sqrt(np.sum(wn * problem.f(xn) ** 2)))
    parent_hash = hashlib.sha256((src / "best.pt").read_bytes()).hexdigest()
    protocol = dict(
        parent=str(src),
        parent_checkpoint_sha256=parent_hash,
        method=cfg.method,
        blocks=4,
        lbfgs_per_block=500 if cfg.method == "direct" else 600,
        adam_steps=0,
        multiplier_updates=0 if cfg.method == "direct" else 4,
        selection="same validation_score as parent",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_evaluations=0,
    )
    write_json(out / "protocol.json", protocol)
    write_json(out / "config.json", vars(cfg))
    np.savez(out / "sampling.npz", train=xn, weights=wn, validation=vn)
    start = time.perf_counter()
    updates = 0
    calls = 0
    best = float("inf")
    best_progress = None

    def record(progress):
        nonlocal best, best_progress
        m = evaluate(model, problem, ref, vn, state_scale)
        row = dict(
            progress=progress,
            additional_updates=updates,
            closure_calls=calls,
            seconds=time.perf_counter() - start,
            **m,
        )
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        ck = dict(parent)
        ck.update(
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            multiplier=multiplier.detach().cpu().clone(),
            progress=progress,
            refinement=protocol,
        )
        if m["validation_score"] < best:
            best = m["validation_score"]
            best_progress = progress
            torch.save(ck, out / "best.pt")
            write_json(out / "metrics.json", row)
        torch.save(ck, out / "last.pt")
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "progress",
                        "additional_updates",
                        "relative_l2_u_raw",
                        "pde_relative",
                        "validation_score",
                    )
                }
            ),
            flush=True,
        )

    try:
        record("parent_best")
        opt = None
        for block in range(4):
            if cfg.method == "alm":
                _, _, r = weighted_loss(
                    model, problem, xbase.detach().requires_grad_(True), w, penalty, multiplier
                )
                multiplier = (multiplier + 2 * penalty * r.detach()).detach()
            if opt is None or cfg.method == "alm":
                opt = torch.optim.LBFGS(
                    model.parameters(),
                    lr=1.0,
                    max_iter=protocol["lbfgs_per_block"],
                    history_size=100,
                    tolerance_grad=1e-11,
                    tolerance_change=1e-15,
                    line_search_fn="strong_wolfe",
                )
            before = int(opt.state[next(model.parameters())].get("n_iter", 0))

            def closure():
                nonlocal calls
                calls += 1
                opt.zero_grad(set_to_none=True)
                loss, _, _ = weighted_loss(
                    model, problem, xbase.detach().requires_grad_(True), w, penalty, multiplier
                )
                loss.backward()
                if not bool(torch.isfinite(loss)) or any(
                    p.grad is not None and not bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()
                ):
                    raise FloatingPointError("nonfinite refinement")
                return loss

            opt.step(closure)
            updates += int(opt.state[next(model.parameters())]["n_iter"]) - before
            record(f"warm_lbfgs:{block+1}")
        write_json(
            out / "run.json",
            dict(
                status="completed",
                additional_updates=updates,
                closure_calls=calls,
                best_progress=best_progress,
                seconds=time.perf_counter() - start,
                test_evaluations=0,
            ),
        )
    except Exception as e:
        write_json(out / "run.json", dict(status="failed", error=str(e)))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.source, a.output)
