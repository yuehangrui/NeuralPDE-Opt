"""One entry point: train, freeze, independent assessment, audit and report."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from . import frozen_jobs, additional_jobs, protocol, assessment, audit, report

TEST_MODULES = [
    "chapters.ch04.test_ch04",
    "chapters.ch04_adaptation.test_adaptation",
    "chapters.ch04_operator_learning.test_operators",
    "chapters.ch04_operator_adaptation.test_adaptation",
    "chapters.ch04_completion.test_completion",
    "chapters.ch04_delivery.test_delivery",
]


def train(root, reuse_seed0=False):
    frozen_jobs.prepare(root, reuse_seed0)
    base = [sys.executable, "-B", "-u", "-m"]
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    with (Path(root) / "tests.log").open("w") as log:
        subprocess.run(
            base + ["unittest", *TEST_MODULES, "-v"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )
    p1 = subprocess.Popen(
        base + ["chapters.ch04_delivery.frozen_jobs", "--root", str(root), "--concurrency", "4"],
        env=env,
    )
    p2 = subprocess.Popen(
        base
        + ["chapters.ch04_delivery.additional_jobs", "--root", str(root), "--concurrency", "2"],
        env=env,
    )
    codes = [p1.wait(), p2.wait()]
    if any(codes):
        raise RuntimeError("training queue failed; retained results, no automatic overwrite")


def finish(root, resume=False):
    root = Path(root)
    if not (root / "SELECTION_FROZEN.json").exists():
        audit.pretest(root)
        protocol.freeze(root)
    if not (root / "heldout").exists():
        protocol.generate_holdout(root)
    if not (root / "INDEPENDENT_ASSESSMENT_COMPLETE").exists():
        assessment.run(root, resume=resume)
    audit.run(root)
    report.run(root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "finish", "all", "verify", "report"])
    p.add_argument("--root", required=True)
    p.add_argument("--reuse-seed0", action="store_true")
    p.add_argument("--resume-assessment", action="store_true")
    a = p.parse_args()
    if a.command in ("train", "all"):
        train(a.root, a.reuse_seed0)
    if a.command in ("finish", "all"):
        finish(a.root, a.resume_assessment)
    if a.command == "verify":
        audit.run(a.root)
    if a.command == "report":
        report.run(a.root)


if __name__ == "__main__":
    main()
