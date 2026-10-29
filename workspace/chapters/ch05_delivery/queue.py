"""Bounded, explicit queues; failures preserve every run, never auto-retrain."""

import argparse, os, subprocess, sys, time, json
from dataclasses import asdict
from pathlib import Path
from .common import METHODS, Config, configuration, write
from .data import create_train_val


def validate_config_data(config, metadata):
    config.validate()
    if vars(config.problem()) != metadata["problem"] or config.grid != metadata["grid"]:
        raise ValueError(
            "This delivery requires the same declared problem and grid as its dataset."
        )


def prepare(root, pilot=False, device="cuda"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    create_train_val(root / "data", counts=(256, 32) if pilot else (1024, 128))
    order = METHODS if pilot else ("admm", "osnn", "cnp", "mimo", "pdol", "iuzawa_f", "iuzawa_s")
    tasks = [
        dict(method=m, seed=s, pilot=pilot, config=asdict(configuration(m, s, pilot)))
        for m in order
        for s in ((0,) if pilot else (0, 1, 2))
    ]
    for task in tasks:
        task["config"]["device"] = device
    write(root / "jobs.json", tasks)
    write(
        root / "protocol.json",
        dict(
            version="ch05-unified-delivery-v1",
            stage="train_val_only",
            pilot=pilot,
            seeds=[0] if pilot else [0, 1, 2],
            training_samples=256 if pilot else 1024,
            validation_samples=32 if pilot else 128,
            heldout_policy="No test before all configs/models frozen; generate a separate 128-case split only afterward.",
            scope="7 migrated Chapter-5 methods, fixed box-constrained elliptic problem + separate CNP state-cap problem",
        ),
    )


def run(root, concurrency=3):
    if concurrency < 1:
        raise ValueError("positive concurrency required")
    root = Path(root)
    pending = json.loads((root / "jobs.json").read_text())
    active = {}
    done = []
    failed = False
    while pending or active:
        while pending and len(active) < concurrency and not failed:
            task = pending.pop(0)
            name = task["method"] + "_s" + str(task["seed"])
            log = (root / "logs" / (name + ".log")).open("w")
            cmd = [
                sys.executable,
                "-B",
                "-u",
                "-m",
                "chapters.ch05_delivery.queue",
                "--root",
                str(root),
                "--worker",
                name,
            ]
            p = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
            )
            active[name] = (p, log)
        for name, (p, log) in list(active.items()):
            if p.poll() is not None:
                log.close()
                del active[name]
                done.append(dict(name=name, exit_code=p.returncode))
                failed |= bool(p.returncode)
        write(
            root / "queue_status.json",
            dict(
                stage="failed" if failed else "training",
                active={n: p.pid for n, (p, _) in active.items()},
                pending=pending,
                done=done,
            ),
        )
        if failed and not active:
            raise RuntimeError("training failed; no automatic retry")
        if active:
            time.sleep(5)
    write(
        root / "queue_status.json",
        dict(stage="training_complete", active={}, pending=[], done=done),
    )
    (root / "TRAINING_COMPLETE").write_text(
        "Declared budgets complete; convergence and independent assessment remain separate.\n"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--worker")
    p.add_argument("--device", default="cuda")
    p.add_argument("--concurrency", type=int, default=3)
    a = p.parse_args()
    if a.prepare:
        prepare(a.root, a.pilot, a.device)
    if a.worker:
        from .training import train

        root = Path(a.root)
        task = next(
            t
            for t in json.loads((root / "jobs.json").read_text())
            if t["method"] + "_s" + str(t["seed"]) == a.worker
        )
        c = (
            Config(**task["config"])
            if "config" in task
            else configuration(task["method"], task["seed"], task["pilot"])
        )
        validate_config_data(c, json.loads((root / "data/manifest.json").read_text()))
        if c.method != task["method"] or c.seed != task["seed"]:
            raise ValueError("job/config identity mismatch")
        train(c, root / "training" / a.worker, root / "data")
    else:
        run(a.root, a.concurrency)


if __name__ == "__main__":
    main()
