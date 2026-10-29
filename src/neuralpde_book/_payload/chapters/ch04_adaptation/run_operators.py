"""Run three declared non-PINN validation budgets, two at a time."""

from pathlib import Path
import subprocess
import sys
import time
from chapters.ch04.training import write_json


def main():
    root = Path("results/ch04/adaptation_v1/operators")
    root.mkdir(parents=True, exist_ok=False)
    methods = ["mimo", "rb_state", "rb_dino"]
    pending = methods.copy()
    active = {}
    done = []
    write_json(
        root / "protocol.json",
        dict(methods=methods, steps=2000, seed=0, train=256, validation=64, test_evaluations=0),
    )
    while pending or active:
        while pending and len(active) < 2:
            method = pending.pop(0)
            log = (root / (method + ".log")).open("w")
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    "-m",
                    "chapters.ch04_adaptation.train_operators",
                    "--method",
                    method,
                    "--output",
                    str(root / method),
                    "--steps",
                    "2000",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active[method] = (p, log)
        for method, (p, log) in list(active.items()):
            if p.poll() is not None:
                log.close()
                done.append(dict(method=method, exit_code=p.returncode))
                del active[method]
                if p.returncode:
                    pending = []
        write_json(
            root / "queue_status.json",
            dict(
                completed=done, active={k: p.pid for k, (p, _) in active.items()}, pending=pending
            ),
        )
        if active:
            time.sleep(10)
    if len(done) != 3 or any(row["exit_code"] for row in done):
        raise RuntimeError("operator validation failed")
    (root / "TRAIN_VAL_COMPLETE").write_text(
        "Declared adaptation budgets complete; no test evaluation.\n"
    )


if __name__ == "__main__":
    main()
