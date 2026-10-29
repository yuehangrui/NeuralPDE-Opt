"""CPU queue for remaining formal configurations; stops after failures."""

import argparse
from pathlib import Path
import subprocess
import sys
import os
import time
from .frozen_jobs import write


def run(root, concurrency=2):
    root = Path(root)
    jobs = []
    for seed in (0, 1, 2):
        for method in ("mimo", "rb_state", "rb_dino"):
            name = f"{method}_s{seed}"
            jobs.append(
                (
                    name,
                    [
                        "chapters.ch04_delivery.operators",
                        "--method",
                        method,
                        "--seed",
                        str(seed),
                        "--device",
                        "cpu",
                        "--output",
                        str(root / "training" / name),
                    ],
                )
            )
        for method in ("bilevel", "deep_uzawa", "ismo", "fem_nn"):
            name = f"{method}_s{seed}"
            jobs.append(
                (
                    name,
                    [
                        "chapters.ch04_delivery.hybrids",
                        "--method",
                        method,
                        "--seed",
                        str(seed),
                        "--output",
                        str(root / "training" / name),
                    ],
                )
            )
    jobs.append(
        (
            "classical",
            [
                "chapters.ch04_delivery.hybrids",
                "--method",
                "classical",
                "--output",
                str(root / "classical"),
            ],
        )
    )
    write(
        root / "additional_jobs.json",
        dict(jobs=[dict(name=n, arguments=a) for n, a in jobs], concurrency=concurrency),
    )
    active = {}
    done = []
    failed = False
    started = time.time()
    while jobs or active:
        while jobs and len(active) < concurrency and not failed:
            name, args = jobs.pop(0)
            log = (root / "logs" / (name + ".log")).open("w")
            proc = subprocess.Popen(
                [sys.executable, "-B", "-u", "-m", *args],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
            )
            active[name] = (proc, log)
        for name, (proc, log) in list(active.items()):
            if proc.poll() is not None:
                log.close()
                del active[name]
                done.append(dict(name=name, exit_code=proc.returncode))
                failed |= bool(proc.returncode)
        write(
            root / "additional_status.json",
            dict(
                stage="failed" if failed else "training",
                active={n: p.pid for n, (p, _) in active.items()},
                pending=[n for n, _ in jobs],
                done=done,
                seconds=time.time() - started,
            ),
        )
        if failed and not active:
            break
        if active:
            time.sleep(5)
    if failed:
        raise RuntimeError("additional training failed; outputs preserved")
    (root / "ADDITIONAL_TRAINING_COMPLETE").write_text(
        "All explicit budgets/stopping conditions evaluated.\n"
    )
    write(
        root / "additional_status.json",
        dict(stage="complete", active={}, pending=[], done=done, seconds=time.time() - started),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--concurrency", type=int, default=2)
    a = p.parse_args()
    run(a.root, a.concurrency)
