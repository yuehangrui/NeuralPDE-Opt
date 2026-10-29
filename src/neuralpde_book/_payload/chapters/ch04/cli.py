"""Run with python -B chapters/ch04/cli.py --help from any working directory."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

torch.set_num_threads(2)
from ch04.training import Config, train, load_checkpoint, write_json
from ch04.problem import Problem
from ch04.fem import Reference
from ch04.evaluation import sample_points, evaluate, predict


def main():
    parser = argparse.ArgumentParser(description="Chapter 4 instance-wise control migration")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--config", type=Path)
    p.add_argument("--method", choices=["direct", "kkt", "alm"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--mesh", type=int, default=128)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("reference")
    p.add_argument("--mesh", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--frequency", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("check")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        if not args.config and not args.smoke:
            parser.error("a formal run requires --config; use --smoke for migration verification")
        cfg = (
            Config(**json.loads(args.config.read_text(encoding="utf-8")))
            if args.config
            else Config()
        )
        if args.method:
            cfg.method = args.method
        if args.device:
            cfg.device = args.device
        if args.seed is not None:
            cfg.seed = args.seed
        if args.smoke:
            cfg.width, cfg.depth = 16, 2
            cfg.train_points, cfg.validation_points = 96, 80
            cfg.validation_mesh, cfg.quadrature_order = max(16, 2 * cfg.control_frequency), 3
            cfg.adam_steps, cfg.lbfgs_steps, cfg.evaluate_every = 4, 3, 2
            cfg.outer_steps, cfg.inner_steps, cfg.smoke = 1, 4, True
        result = train(cfg, args.output)
    elif args.command == "reference":
        args.output.mkdir(parents=True, exist_ok=False)
        problem = Problem(args.alpha, args.frequency)
        result = {
            "problem": asdict(problem),
            "runs": [Reference(problem, n, 6).report() for n in args.mesh],
        }
        write_json(args.output / "convergence.json", result)
    elif args.command == "evaluate":
        import numpy as np

        model, cfg, checkpoint = load_checkpoint(args.checkpoint, args.device)
        args.output.mkdir(parents=True, exist_ok=False)
        problem = Problem(cfg.alpha, cfg.control_frequency)
        ref = Reference(problem, args.mesh, 6)
        val = sample_points(cfg.seed, 2, cfg.validation_points)
        result = evaluate(model, problem, ref, val, checkpoint["state_scale"])
        result.update(
            evaluation_kind="single-instance diagnostic, not a held-out instance test",
            run_kind="smoke" if cfg.smoke else "training",
            checkpoint=str(args.checkpoint),
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            selected_progress=checkpoint["progress"],
            reference=ref.report(),
        )
        write_json(args.output / "metrics.json", result)
        y, u, p = predict(model, ref.nodes)
        arrays = dict(
            nodes=ref.nodes,
            elements=ref.elements,
            y_raw=y,
            u_raw=u,
            y_reference=ref.y,
            u_reference=ref.u,
            p_reference=ref.p,
        )
        if p is not None:
            arrays["p_raw"] = p
        np.savez_compressed(args.output / "fields.npz", **arrays)
    else:
        import unittest

        args.output.mkdir(parents=True, exist_ok=False)
        suite = unittest.defaultTestLoader.loadTestsFromName("ch04.test_ch04")
        with (args.output / "tests.log").open("w", encoding="utf-8") as stream:
            run = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        result = dict(
            tests=run.testsRun,
            failures=len(run.failures),
            errors=len(run.errors),
            passed=run.wasSuccessful(),
        )
        write_json(args.output / "summary.json", result)
        if not run.wasSuccessful():
            print((args.output / "tests.log").read_text(encoding="utf-8"))
            raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
