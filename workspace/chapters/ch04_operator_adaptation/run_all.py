"""Run the six explicit full configurations, after preflight tests/smokes."""

from pathlib import Path
import argparse, json, os, subprocess, sys, time
from chapters.ch04_operator_learning.training import write


def run(data, output):
    root = Path(output)
    root.mkdir(exist_ok=False, parents=True)
    pending = [a + "_" + r for a in ("fno", "deeponet", "fedonet") for r in ("state", "kkt")]
    active = {}
    done = []
    start = time.perf_counter()
    source = Path(__file__).parent
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
                    "chapters.ch04_operator_adaptation.training",
                    "--config",
                    str(source / "configs" / (name + ".json")),
                    "--data",
                    str(data),
                    "--output",
                    str(root / name),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
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
                stage="training",
                active={n: p.pid for n, (p, _) in active.items()},
                pending=pending,
                done=done,
                seconds=time.perf_counter() - start,
            ),
        )
        if active:
            time.sleep(10)
    if len(done) != 6 or any(r["exit_code"] for r in done):
        raise RuntimeError("training failed; no automatic retry")
    for row in done:
        info = json.loads((root / row["name"] / "run.json").read_text())
        assert info["epochs"] == 1600 and info["full_budget_completed"]
    (root / "TRAINING_COMPLETE").write_text("Six full budgets; independent validation pending.\n")
    write(
        root / "queue_status.json",
        dict(
            stage="training_complete_validation_pending",
            active={},
            pending=[],
            done=done,
            seconds=time.perf_counter() - start,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.data, a.output)
