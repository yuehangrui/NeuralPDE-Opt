"""Finish linear-head conditioning checks, after the declared refinements."""

from pathlib import Path
import os
import subprocess
import sys
from chapters.ch04.training import write_json


def main():
    base = Path("results/ch04/adaptation_v1")
    if not (base / "refinement/TRAIN_VAL_COMPLETE").exists():
        raise RuntimeError("parent refinement incomplete")
    root = base / "head_polish"
    root.mkdir(exist_ok=False)
    done = []
    for method in ("direct", "alm"):
        with (root / (method + ".log")).open("w") as log:
            p = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    "-m",
                    "chapters.ch04_adaptation.head_polish",
                    "--source",
                    str(base / "refinement" / method),
                    "--output",
                    str(root / method),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "2"},
            )
        done.append(dict(method=method, exit_code=p.returncode))
        write_json(root / "queue_status.json", dict(completed=done))
        if p.returncode:
            raise RuntimeError("head solve failed")
    (root / "TRAIN_VAL_COMPLETE").write_text("Head conditioning checks complete.\n")


if __name__ == "__main__":
    main()
