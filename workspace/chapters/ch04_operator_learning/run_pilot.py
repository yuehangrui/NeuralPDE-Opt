"""Six full-architecture, 200-epoch migration checks; not full training."""

from pathlib import Path
import subprocess
import sys
import time
from .data import create, load
from .training import write


def main():
    data = Path("data/ch04/operator_learning_v1")
    root = Path("results/ch04/operator_learning_v1")
    root.mkdir(parents=True, exist_ok=False)
    if data.exists():
        load(data)
    else:
        create(data)
    pending = [a + "_" + r for a in ("fno", "deeponet", "fedonet") for r in ("state", "kkt")]
    active = {}
    done = []
    write(
        root / "protocol.json",
        dict(
            cases=pending.copy(),
            epochs=200,
            nominal_full_epochs=1600,
            seed=0,
            data=str(data),
            test_evaluations=0,
        ),
    )
    while pending or active:
        while pending and len(active) < 2:
            name = pending.pop(0)
            log = (root / (name + ".log")).open("w")
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    "-m",
                    "chapters.ch04_operator_learning.cli",
                    "train",
                    "--config",
                    "chapters/ch04_operator_learning/configs/" + name + ".json",
                    "--data",
                    str(data),
                    "--output",
                    str(root / name),
                    "--epochs",
                    "200",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active[name] = (p, log)
        for name, (p, log) in list(active.items()):
            if p.poll() is not None:
                log.close()
                done.append(dict(name=name, exit_code=p.returncode))
                del active[name]
                if p.returncode:
                    pending = []
        write(
            root / "queue_status.json",
            dict(
                completed=done, active={k: p.pid for k, (p, _) in active.items()}, pending=pending
            ),
        )
        if active:
            time.sleep(10)
    if len(done) != 6 or any(row["exit_code"] for row in done):
        raise RuntimeError("migration pilot incomplete")
    (root / "TRAIN_VAL_COMPLETE").write_text(
        "Six 200-epoch full-architecture pilot budgets completed; full 1600-epoch budgets not run.\n"
    )


if __name__ == "__main__":
    main()
