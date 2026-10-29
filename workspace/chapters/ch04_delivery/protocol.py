"""Fail-closed selection freeze and a genuinely new held-out problem split."""

from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from chapters.ch04_adaptation.operator_data import generate, kkt, EIGENVALUES, ALPHA
from .frozen_jobs import write

METHODS = (
    "direct",
    "kkt",
    "alm",
    "fno_state",
    "fno_kkt",
    "deeponet_state",
    "deeponet_kkt",
    "fedonet_state",
    "fedonet_kkt",
    "mimo",
    "rb_state",
    "rb_dino",
    "bilevel",
    "deep_uzawa",
    "ismo",
    "fem_nn",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_files():
    files = []
    for module in (
        "ch04",
        "ch04_adaptation",
        "ch04_operator_learning",
        "ch04_operator_adaptation",
        "ch04_completion",
        "ch04_delivery",
    ):
        files += list((Path("chapters") / module).rglob("*.py"))
    return sorted(files)


def history(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def verify_history(folder, checkpoint, kind):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if kind == "grid_operator":
        hist = history(folder / "history.jsonl")
        best = min(hist, key=lambda r: r["validation_score"])
        run = json.loads((folder / "run.json").read_text())
        if run["epochs"] != 1600 or len(hist) != 1601 or not run["full_budget_completed"]:
            raise ValueError("incomplete full operator budget")
        if ck["epoch"] != best["epoch"] or ck["best_epoch"] != best["epoch"]:
            raise ValueError("operator selection mismatch")
    elif kind in ("pinn", "adapted_pinn"):
        hist = history(folder / "history.jsonl")
        best = min(hist, key=lambda r: r["validation_score"])
        if ck["progress"] != best["progress"]:
            raise ValueError("PINN selection mismatch")
    elif kind == "modal_operator":
        hist = history(folder / "history.jsonl")
        best = min(hist, key=lambda r: r["validation_score"])
        run = json.loads((folder / "run.json").read_text())
        if run["updates"] != 4000 or hist[-1]["step"] != 4000:
            raise ValueError("incomplete modal operator budget")
        if ck["step"] != best["step"]:
            raise ValueError("modal operator selection mismatch")
    else:
        metrics = json.loads((folder / "metrics.json").read_text())
        hist = metrics["history"]
        method = ck["method"]
        if method in ("bilevel", "deep_uzawa"):
            scores = [r["validation_score"] for r in hist]
            if metrics["best_index"] != int(np.argmin(scores)):
                raise ValueError("hybrid selection mismatch")
        elif method == "fem_nn":
            scores = [r["validation_observation_relative_l2"] for r in hist]
            if metrics["best_index"] != int(np.argmin(scores)):
                raise ValueError("inverse-model selection mismatch")
        else:
            if not 1 <= len(hist) <= 12:
                raise ValueError("ISMO round budget")
            for record in hist:
                if record["updates"] != 1600 or len(record["validation_history"]) != 1600:
                    raise ValueError("ISMO training budget")
                if record["selected_epoch"] != int(np.argmin(record["validation_history"])):
                    raise ValueError("ISMO selection mismatch")
    return True


def build_registry(root):
    root = Path(root)
    entries = []
    for method in METHODS:
        for seed in (0, 1, 2):
            folder = root / "training" / f"{method}_s{seed}"
            if method in ("direct", "alm"):
                variants = [
                    ("neural", folder / "refine", "adapted_pinn", "best.pt"),
                    ("head_selected", folder / "head", "adapted_pinn", "best.pt"),
                ]
            elif method == "kkt":
                variants = [("neural", folder, "pinn", "best.pt")]
            elif method.startswith(("fno_", "deeponet_", "fedonet_")):
                variants = [
                    ("neural", folder, "grid_operator", "neural_best.pt"),
                    ("head_polished", folder, "grid_operator", "best.pt"),
                ]
            elif method in ("mimo", "rb_state", "rb_dino"):
                variants = [("neural", folder, "modal_operator", "best.pt")]
            else:
                variants = [("declared_trial", folder, "hybrid", "best.pt")]
            for variant, directory, kind, filename in variants:
                if json.loads((directory / "run.json").read_text())["status"] != "completed":
                    raise ValueError("incomplete " + str(directory))
                path = directory / filename
                verify_history(directory, path, kind)
                entries.append(
                    dict(
                        key=f"{method}_{variant}_s{seed}",
                        method=method,
                        seed=seed,
                        variant=variant,
                        kind=kind,
                        directory=str(directory),
                        checkpoint=str(path),
                        sha256=sha(path),
                        track=(
                            "problem_family"
                            if kind in ("grid_operator", "modal_operator")
                            else "fixed_instance"
                        ),
                    )
                )
    return entries


def freeze(root):
    root = Path(root)
    if (root / "SELECTION_FROZEN.json").exists():
        raise FileExistsError("selection already frozen")
    for marker in ("FROZEN_IMPLEMENTATIONS_TRAINED", "ADDITIONAL_TRAINING_COMPLETE"):
        if not (root / marker).is_file():
            raise RuntimeError("all declared training must finish before freeze: " + marker)
    if json.loads((root / "classical/run.json").read_text())["status"] != "completed":
        raise ValueError("classical study incomplete")
    entries = build_registry(root)
    configs = {str(p): sha(p) for p in (root / "training").rglob("config.json")}
    sources = {str(p): sha(p) for p in source_files()}
    data = {str(p): sha(p) for p in Path("data/ch04/operator_learning_v1").iterdir() if p.is_file()}
    value = dict(
        version="ch04-unified-delivery-v1",
        timestamp_unix=time.time(),
        entries=entries,
        config_hashes=configs,
        source_hashes=sources,
        training_data_hashes=data,
        test_protocol=dict(count=64, seed_sequence=[20261028, 444, 20261029, 1], online_count=8),
        fixed_instance_protocol=dict(gauss_order=96, fem_meshes=[128, 256], boundary_points=129),
        distinction="same delivery checks, method-specific budgets and representations; no architecture ranking",
    )
    write(root / "SELECTION_FROZEN.json", value)
    return value


def verify_freeze(root):
    root = Path(root)
    frozen = json.loads((root / "SELECTION_FROZEN.json").read_text())
    for collection in ("source_hashes", "config_hashes", "training_data_hashes"):
        for path, expected in frozen[collection].items():
            if sha(path) != expected:
                raise ValueError("post-freeze mutation: " + path)
    for entry in frozen["entries"]:
        if sha(entry["checkpoint"]) != entry["sha256"]:
            raise ValueError("post-freeze model mutation")
    return frozen


def generate_holdout(root):
    root = Path(root)
    frozen = verify_freeze(root)
    directory = root / "heldout"
    directory.mkdir(exist_ok=False)
    spec = frozen["test_protocol"]
    rng = np.random.default_rng(np.random.SeedSequence(spec["seed_sequence"]))
    count = spec["count"]
    d = len(EIGENVALUES)
    f0 = np.zeros(d)
    yd0 = np.zeros(d)
    f0[0] = 2 * np.pi**2
    f0[-1] = 1 / (ALPHA * EIGENVALUES[-1])
    yd0[0] = 1
    yd0[-1] = -1
    f = f0 + 0.5 * rng.standard_normal((count, d))
    yd = yd0 + 0.15 * rng.standard_normal((count, d))
    rhs = f0 + 4 * rng.standard_normal((count, d))
    y, u, p = kkt(f, yd)
    values = dict(
        f=f,
        yd=yd,
        rhs=rhs,
        state=rhs / EIGENVALUES,
        y=y,
        u=u,
        p=p,
        ids=np.array([f"ch04-unified-heldout:{i}" for i in range(count)]),
    )

    def rows(data):
        return {
            hashlib.sha256(
                np.concatenate([data[k][i] for k in ("f", "yd", "rhs")]).tobytes()
            ).hexdigest()
            for i in range(len(data["ids"]))
        }

    heldout_hashes = rows(values)
    for role, n in [("train", 256), ("validation", 64)]:
        if heldout_hashes & rows(generate(role, n, 0)):
            raise ValueError("cross-split overlap")
    if len(heldout_hashes) != count:
        raise ValueError("duplicate held-out samples")
    np.savez_compressed(directory / "problems.npz", **values)
    manifest = dict(
        schema="ch04-unified-heldout-v1",
        created_unix=time.time(),
        role="test",
        count=count,
        selection_sha256=sha(root / "SELECTION_FROZEN.json"),
        file_sha256=sha(directory / "problems.npz"),
        seed_sequence=spec["seed_sequence"],
        overlap_train_validation=False,
    )
    write(directory / "manifest.json", manifest)
    return values


def load_holdout(root):
    root = Path(root)
    verify_freeze(root)
    manifest = json.loads((root / "heldout/manifest.json").read_text())
    if manifest["selection_sha256"] != sha(root / "SELECTION_FROZEN.json") or manifest[
        "file_sha256"
    ] != sha(root / "heldout/problems.npz"):
        raise ValueError("held-out identity mismatch")
    with np.load(root / "heldout/problems.npz", allow_pickle=False) as archive:
        return dict(archive)
