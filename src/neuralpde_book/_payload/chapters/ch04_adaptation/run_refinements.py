"""Declared bounded refinements after the completed ablation."""

from pathlib import Path
import subprocess
import sys
from chapters.ch04.training import write_json


def main():
    base = Path("results/ch04/adaptation_v1")
    if not (base / "instances/TRAIN_VAL_COMPLETE").exists():
        raise RuntimeError("parent ablation not complete")
    root = base / "refinement"
    root.mkdir(exist_ok=False)
    active = []
    for method in ("direct", "alm"):
        log = (root / (method + ".log")).open("w")
        p = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-u",
                "-m",
                "chapters.ch04_adaptation.refine_instances",
                "--source",
                str(base / "instances" / (method + "_enriched_gauss")),
                "--output",
                str(root / method),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        active.append((method, p, log))
    write_json(root / "queue_status.json", dict(active={name: p.pid for name, p, _ in active}))
    done = []
    for name, p, log in active:
        code = p.wait()
        log.close()
        done.append(dict(method=name, exit_code=code))
    write_json(root / "queue_status.json", dict(completed=done, active={}))
    if any(row["exit_code"] for row in done):
        raise RuntimeError("refinement failed")
    (root / "TRAIN_VAL_COMPLETE").write_text("Warm refinement budgets complete.\n")


if __name__ == "__main__":
    main()
