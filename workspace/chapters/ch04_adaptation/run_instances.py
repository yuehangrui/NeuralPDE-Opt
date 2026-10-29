"""A bounded, predeclared train/validation ablation. Never reads final/test data."""

from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
from .instance import InstanceConfig, train
from chapters.ch04.training import write_json

CASES = {
    "direct_tanh_gauss": dict(method="direct", architecture="tanh"),
    "direct_enriched_gauss": dict(method="direct", architecture="enriched"),
    "direct_enriched_random": dict(method="direct", architecture="enriched", sampling="random"),
    "alm_tanh_gauss": dict(
        method="alm", architecture="tanh", adam_steps=250, lbfgs_steps=250, outer_steps=6
    ),
    "alm_enriched_gauss": dict(
        method="alm", architecture="enriched", adam_steps=250, lbfgs_steps=250, outer_steps=6
    ),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--case", choices=CASES)
    a = p.parse_args()
    root = Path(a.output)
    if a.case:
        import torch

        torch.set_num_threads(2)
        train(InstanceConfig(**CASES[a.case]), root / a.case)
        return
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "protocol.json",
        dict(cases={k: vars(InstanceConfig(**v)) for k, v in CASES.items()}, test_evaluations=0),
    )
    pending = list(CASES)
    active = {}
    completed = []
    while pending or active:
        while pending and len(active) < 2:
            name = pending.pop(0)
            log = (root / (name + ".log")).open("w")
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "chapters.ch04_adaptation.run_instances",
                    "--output",
                    str(root),
                    "--case",
                    name,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active[name] = (proc, log)
        for name, (proc, log) in list(active.items()):
            code = proc.poll()
            if code is not None:
                log.close()
                del active[name]
                completed.append(dict(name=name, exit_code=code))
                if code:
                    pending = []  # allow other already-started run to finish, no silent retries
        write_json(
            root / "queue_status.json",
            dict(
                completed=completed,
                active={k: p.pid for k, (p, _) in active.items()},
                pending=pending,
            ),
        )
        if active:
            time.sleep(10)
    if all(c["exit_code"] == 0 for c in completed) and len(completed) == len(CASES):
        (root / "TRAIN_VAL_COMPLETE").write_text(
            "All declared budgets completed; see metrics for accuracy.\n"
        )
    else:
        raise RuntimeError("instance ablation incomplete")


if __name__ == "__main__":
    main()
