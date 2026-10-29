"""Shared train/validation grid data and fail-closed array identities."""

from pathlib import Path
import hashlib
import json
import numpy as np
from chapters.ch04_adaptation.operator_data import generate, basis
from .reference import Poisson

FIELDS = ("f", "yd", "y", "u", "p", "state_u", "state_y")


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def make_split(role, count, n=32, seed=0):
    ref = Poisson(n)
    raw = generate(role, count, seed)
    b = basis(ref.points.reshape(-1, 2))
    coeff = {k: raw[k] for k in ("f", "yd", "y", "u", "p")}
    coeff.update(state_u=raw["rhs"] - raw["f"], state_y=raw["state"])
    result = {k: (c @ b.T).reshape(count, n + 1, n + 1) * ref.mask for k, c in coeff.items()}
    result["ids"] = raw["ids"]
    check_split(result, n)
    return result


def check_split(data, n):
    if set(data) != set(FIELDS) | {"ids"}:
        raise ValueError("unexpected data fields")
    count = len(data["ids"])
    ref = Poisson(n)
    if count == 0 or len(set(data["ids"])) != count:
        raise ValueError("invalid or duplicated ids")
    for k in FIELDS:
        if data[k].shape != (count, n + 1, n + 1) or not np.isfinite(data[k]).all():
            raise ValueError("invalid " + k)
        if np.max(abs(data[k] * (1 - ref.mask))) > 1e-12:
            raise ValueError("noncanonical boundary nodes")
    for residual in (
        ref.apply(data["y"]) - data["u"] - data["f"],
        ref.apply(data["p"]) - data["y"] + data["yd"],
        ref.alpha * data["u"] + data["p"],
        ref.apply(data["state_y"]) - data["state_u"] - data["f"],
    ):
        if np.max(abs(residual)) > 1e-8:
            raise ValueError("state/KKT labels fail reference residual")
    return {k: sha(data[k]) for k in FIELDS + ("ids",)}


def create(path, n=32, seed=0, train_count=256, validation_count=64):
    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)
    tr = make_split("train", train_count, n, seed)
    va = make_split("validation", validation_count, n, seed)
    if set(tr["ids"]) & set(va["ids"]):
        raise ValueError("split overlap")
    if {sha(a) for a in tr["f"]} & {sha(a) for a in va["f"]}:
        raise ValueError("duplicate cross-split source")
    hashes = {}
    for role, d in [("train", tr), ("validation", va)]:
        np.savez_compressed(root / (role + ".npz"), **d)
        hashes[role] = check_split(d, n)
    manifest = dict(
        schema="ch04-grid-operators-v1",
        n=n,
        seed=seed,
        counts=dict(train=train_count, validation=validation_count),
        family="17 sine modes, same linear family as adaptation_v1; state controls sampled independently of optimality",
        alpha=0.001,
        hashes=hashes,
        test_evaluations=0,
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(path):
    root = Path(path)
    m = json.loads((root / "manifest.json").read_text())
    if m["schema"] != "ch04-grid-operators-v1" or m["test_evaluations"] != 0:
        raise ValueError("unsupported data protocol")
    values = []
    for role in ("train", "validation"):
        d = dict(np.load(root / (role + ".npz"), allow_pickle=False))
        if check_split(d, m["n"]) != m["hashes"][role] or len(d["ids"]) != m["counts"][role]:
            raise ValueError("data identity mismatch")
        values.append(d)
    if set(values[0]["ids"]) & set(values[1]["ids"]):
        raise ValueError("split overlap")
    return *values, m


def arrays(data, role):
    if role == "kkt":
        return np.stack([data["f"], data["yd"]], 1), np.stack([data["y"], data["u"]], 1)
    if role == "state":
        return np.stack([data["f"], data["state_u"]], 1), data["state_y"][:, None]
    raise ValueError("invalid role")
