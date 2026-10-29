"""Explicit bounded development and frozen formal delivery entry points."""

import argparse
from pathlib import Path
import json
import platform
import torch
import scipy
import numpy as np
from .common import write_json
from .training import defaults, train_fem, train_uzawa, validate_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "finish", "verify"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--method",
        choices=["uzawa", "fem_clean", "fem_noisy", "fem_regularized", "all"],
        default="all",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action != "train":
        from .assessment import finish, verify

        return (finish if args.action == "finish" else verify)(args.root)
    if args.root.exists():
        raise FileExistsError("choose a fresh output root; existing data are never overwritten")
    methods = (
        ["uzawa", "fem_clean", "fem_noisy", "fem_regularized"]
        if args.method == "all"
        else [args.method]
    )
    overrides = json.loads(args.config.read_text()) if args.config else {}
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("duplicate training seeds")
    jobs = []
    for method in methods:
        for seed in args.seeds:
            config = defaults(method, seed, args.pilot)
            config.update(overrides.get(method, {}))
            if config["method"] != method or config["seed"] != seed:
                raise ValueError("overrides cannot change method or seed")
            validate_config(config)
            jobs.append(config)
    args.root.mkdir(parents=True)
    write_json(
        args.root / "environment.json",
        dict(
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            scipy=scipy.__version__,
            platform=platform.platform(),
            torch_threads=torch.get_num_threads(),
        ),
    )
    write_json(
        args.root / "protocol.json",
        dict(
            pilot=args.pilot,
            jobs=jobs,
            selection="earliest minimal validation score",
            test_created=False,
            scope="single physical problem; independent spatial/observation validation, not unseen-problem generalization",
        ),
    )
    for config in jobs:
        name = f"{config['method']}_seed{config['seed']}"
        print("START", name, flush=True)
        score = (train_uzawa if config["method"] == "uzawa" else train_fem)(
            config, args.root / "training" / name
        )
        print("DONE", name, "validation", score, flush=True)


if __name__ == "__main__":
    main()
