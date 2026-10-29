"""Unified Chapter 5 lifecycle; explicit fresh training or frozen-result verification."""

import argparse, os, subprocess, sys
from pathlib import Path
from . import queue, protocol, assessment, audit, report

TESTS = ["chapters.ch05.test_ch05", "chapters.ch05_delivery.test_delivery"]


def checks(log):
    with Path(log).open("w") as f:
        subprocess.run(
            [sys.executable, "-B", "-m", "unittest", *TESTS, "-v"],
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
            check=True,
        )


def finish(root, resume=False, device="cuda"):
    root = Path(root)
    if not (root / "TRAINING_COMPLETE").exists():
        raise RuntimeError("training incomplete")
    if not (root / "SELECTION_FROZEN.json").exists():
        checks(root / "tests.log")
        protocol.pretest(root)
        protocol.freeze(root)
    if not (root / "heldout").exists():
        protocol.generate_holdout(root)
    if not (root / "INDEPENDENT_ASSESSMENT_COMPLETE").exists():
        assessment.run(root, device, resume)
    audit.run(root)
    report.run(root)
    (root / "DELIVERY_COMPLETE").write_text(
        "Complete declared training, independent assessments, numerical audit and report; accuracy/convergence remain separate outcomes.\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "finish", "all", "verify", "report", "check"])
    p.add_argument("--root", required=True)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume-assessment", action="store_true")
    a = p.parse_args()
    root = Path(a.root)
    if a.command in ("train", "all"):
        queue.prepare(root, device=a.device)
        checks(root / "pretrain_tests.log")
        queue.run(root, a.concurrency)
    if a.command in ("finish", "all"):
        finish(root, a.resume_assessment, a.device)
    if a.command == "verify":
        audit.run(root)
    if a.command == "report":
        report.run(root)
    if a.command == "check":
        checks(root)


if __name__ == "__main__":
    main()
