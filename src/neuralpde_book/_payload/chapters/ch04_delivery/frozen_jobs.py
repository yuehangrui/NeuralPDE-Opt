"""Supplement existing frozen implementations without changing their source.

Run this stage before opening any held-out problem data. Existing seed zero
experiments are copied with recorded provenance, never retrained implicitly.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, reuse_seed0=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    training = root / "training"
    training.mkdir()
    (root / "logs").mkdir()
    tasks, adopted = [], []
    source = Path("results/ch04")

    def adopt(name, pieces):
        out = training / name
        if len(pieces) == 1 and pieces[0][0] == ".":
            shutil.copytree(pieces[0][1], out)
        else:
            out.mkdir()
            for dest, src in pieces:
                shutil.copytree(src, out / dest)
        provenance = {
            str(src): {str(p.relative_to(src)): digest(p) for p in src.rglob("*") if p.is_file()}
            for _, src in pieces
        }
        for _, src in pieces:
            info = json.loads((src / "run.json").read_text())
            if info["status"] != "completed":
                raise ValueError("cannot adopt incomplete run")
        write(
            out / "adoption.json",
            dict(
                original_files=provenance,
                purpose="reuse completed seed0 within unchanged configuration",
            ),
        )
        adopted.append(name)

    if not Path("data/ch04/operator_learning_v1/manifest.json").exists():
        from chapters.ch04_operator_learning.data import create

        create("data/ch04/operator_learning_v1")
    if reuse_seed0:
        for method in ("direct", "alm"):
            adopt(
                method + "_s0",
                [
                    (stage, source / "adaptation_v1" / folder / item)
                    for stage, folder, item in [
                        ("base", "instances", method + "_enriched_gauss"),
                        ("refine", "refinement", method),
                        ("head", "head_polish", method),
                    ]
                ],
            )
        adopt("kkt_s0", [(".", source / "full_training_v1/kkt")])
        for arch in ("fno", "deeponet", "fedonet"):
            for role in ("state", "kkt"):
                adopt(
                    arch + "_" + role + "_s0",
                    [(".", source / "operator_adaptation_v2/training" / (arch + "_" + role))],
                )
    seeds = (1, 2) if reuse_seed0 else (0, 1, 2)
    # Start the longest independent PINN jobs first.
    tasks += [dict(name=f"kkt_s{s}", kind="kkt", seed=s) for s in seeds]
    tasks += [
        dict(name=f"{a}_{r}_s{s}", kind="operator", architecture=a, role=r, seed=s)
        for s in seeds
        for a in ("fno", "deeponet", "fedonet")
        for r in ("state", "kkt")
    ]
    tasks += [
        dict(name=f"{m}_s{s}", kind="instance", method=m, seed=s)
        for s in seeds
        for m in ("direct", "alm")
    ]
    write(root / "frozen_jobs.json", tasks)
    write(root / "adopted_jobs.json", adopted)
    write(
        root / "delivery_protocol.json",
        dict(
            version="ch04-unified-delivery-v1",
            seeds=[0, 1, 2],
            population="fixed 17-mode linear family; fixed-instance methods evaluated separately",
            training_data="data/ch04/operator_learning_v1, common seed0 train256/validation64",
            test_policy="Generate fresh holdout only after all selections and configs are frozen; no test tuning.",
            deterministic_algorithms="mesh refinement instead of artificial seed repetitions",
            stages=[
                "supplement_frozen_training",
                "adapted_and_hybrid_training",
                "freeze_all",
                "independent_assessment",
                "audit_and_archive",
            ],
        ),
    )


def worker(root, name):
    root = Path(root)
    task = next(t for t in json.loads((root / "frozen_jobs.json").read_text()) if t["name"] == name)
    out = root / "training" / name
    import torch

    torch.set_num_threads(2)
    if task["kind"] == "operator":
        from chapters.ch04_operator_adaptation.training import Config, train

        train(
            Config(architecture=task["architecture"], role=task["role"], seed=task["seed"]),
            "data/ch04/operator_learning_v1",
            out,
        )
    elif task["kind"] == "kkt":
        from chapters.ch04.training import Config, train

        cfg = Config(**json.loads(Path("chapters/ch04/configs/elliptic_kkt.json").read_text()))
        cfg.seed = task["seed"]
        train(cfg, out)
    else:
        from chapters.ch04_adaptation.instance import InstanceConfig, train
        from chapters.ch04_adaptation.refine_instances import run as refine
        from chapters.ch04_adaptation.head_polish import run as polish

        method = task["method"]
        cfg = InstanceConfig(method=method, seed=task["seed"])
        if method == "alm":
            cfg.adam_steps = 250
            cfg.lbfgs_steps = 250
        out.mkdir(parents=True, exist_ok=False)
        train(cfg, out / "base")
        refine(out / "base", out / "refine")
        polish(out / "refine", out / "head")


def queue(root, concurrency=4):
    root = Path(root)
    tasks = json.loads((root / "frozen_jobs.json").read_text())
    pending = [t["name"] for t in tasks]
    active, done, failed = {}, [], False
    started = time.time()
    while pending or active:
        while pending and len(active) < concurrency and not failed:
            name = pending.pop(0)
            log = (root / "logs" / (name + ".log")).open("w")
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    "-m",
                    "chapters.ch04_delivery.frozen_jobs",
                    "--root",
                    str(root),
                    "--worker",
                    name,
                ],
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
                failed = failed or bool(p.returncode)
        write(
            root / "queue_status.json",
            dict(
                stage="failed" if failed else "training",
                active={n: p.pid for n, (p, _) in active.items()},
                pending=pending,
                done=done,
                seconds=time.time() - started,
            ),
        )
        if failed and not active:
            break
        if active:
            time.sleep(5)
    if failed:
        raise RuntimeError("worker failed; retained artifacts and no automatic overwrite/retry")
    (root / "FROZEN_IMPLEMENTATIONS_TRAINED").write_text(
        "Supplemental training complete; held-out assessment not yet opened.\n"
    )
    write(
        root / "queue_status.json",
        dict(
            stage="supplemental_training_complete",
            active={},
            pending=[],
            done=done,
            seconds=time.time() - started,
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--worker")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--fresh", action="store_true")
    a = p.parse_args()
    if a.prepare:
        prepare(a.root, reuse_seed0=not a.fresh)
    if a.worker:
        worker(a.root, a.worker)
    else:
        queue(a.root, a.concurrency)


if __name__ == "__main__":
    main()
