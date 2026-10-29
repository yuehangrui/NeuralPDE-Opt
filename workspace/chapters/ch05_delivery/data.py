"""Reconstructible cosine inputs, checked FD labels, deferred held-out generation."""

from pathlib import Path
import json
import numpy as np
from chapters.ch05.reference import Grid
from chapters.ch05.problem import Problem
from .common import write, sha

MODES = np.array([(i, j) for i in range(4) for j in range(4)])
KEYS = (
    "f",
    "yd",
    "u",
    "y",
    "p",
    "v",
    "sv",
    "w",
    "sw",
    "coeff_f",
    "coeff_yd",
    "coeff_v",
    "coeff_w",
    "sample_id",
)


def basis(points):
    return np.cos(np.pi * points[..., 0, None] * MODES[:, 0]) * np.cos(
        np.pi * points[..., 1, None] * MODES[:, 1]
    )


def inputs(coeff, n, problem=None):
    problem_data = problem or Problem()
    grid = Grid(n)
    b = basis(grid.points)
    result = {k: np.einsum("bd,xyd->bxy", coeff["coeff_" + k], b) for k in ("f", "yd", "v", "w")}
    result["f"] += problem_data.f(grid.points)[..., 0]
    result["yd"] += problem_data.yd(grid.points)[..., 0]
    return result


def make(split, count, n=32):
    if split not in ("train", "val", "test") or type(count) is not int or count < 1:
        raise ValueError("split/count")
    stream = {"train": 0, "val": 1, "test": 20261026}[split]
    rng = np.random.default_rng(np.random.SeedSequence([20261025, 5051, stream]))
    grid = Grid(n)
    b = basis(grid.points)
    coeff = {}
    for k, scale in [("f", 0.3), ("yd", 0.05), ("v", 1.0), ("w", 0.1)]:
        raw = rng.normal(size=(count, 16)) / (1 + (MODES**2).sum(1)) ** 1.5
        rms = np.sqrt(np.mean(np.einsum("bd,xyd->bxy", raw, b) ** 2, axis=(1, 2)))
        coeff["coeff_" + k] = raw * (rng.uniform(0.25, 1.5, count) * scale / rms)[:, None]
    fields = inputs(coeff, n)
    ref = grid.optimal_control(fields["f"], fields["yd"], Problem(), tol=5e-10)
    arrays = dict(
        **fields,
        **coeff,
        **{k: ref[k] for k in ("y", "u", "p")},
        sv=grid.solve(fields["v"]),
        sw=grid.solve(fields["w"]),
        sample_id=np.array([f"ch05-delivery:{split}:{i}" for i in range(count)]),
    )
    return arrays, dict(
        iterations=ref["iterations"],
        max_projected_residual=float(np.max(ref["projected_residual"])),
    )


def validate(data, n=32):
    if set(data) != set(KEYS):
        raise ValueError("keys")
    count = len(data["sample_id"])
    grid = Grid(n)
    problem = Problem()
    if count < 1 or len(np.unique(data["sample_id"])) != count:
        raise ValueError("IDs")
    for k, a in data.items():
        expected = (
            (count,)
            if k == "sample_id"
            else ((count, 16) if k.startswith("coeff_") else (count, n - 1, n - 1))
        )
        if a.shape != expected or (
            k != "sample_id" and (a.dtype != np.float64 or not np.isfinite(a).all())
        ):
            raise ValueError(k)
    for k, v in inputs(data, n).items():
        np.testing.assert_allclose(data[k], v, atol=2e-13, rtol=2e-13)
    d = grid.diagnostics(data["u"], data["f"], data["yd"], problem)
    if np.max(d["bound_violation"]) > 1e-12 or np.max(d["projected_residual"]) > 2e-9:
        raise ValueError("reference optimality")
    for key, rhs in [
        ("y", data["u"] + data["f"]),
        ("p", data["y"] - data["yd"]),
        ("sv", data["v"]),
        ("sw", data["w"]),
    ]:
        np.testing.assert_allclose(data[key], grid.solve(rhs), atol=1e-11, rtol=1e-11)
    return True


def fingerprints(data):
    import hashlib

    c = np.concatenate([data["coeff_" + k] for k in ("f", "yd", "v", "w")], 1)
    return {hashlib.sha256(row.tobytes()).hexdigest() for row in c}


def create_train_val(root, counts=(1024, 128), n=32):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    meta = dict(
        schema="ch05-reconstructible-v1",
        grid=n,
        problem=vars(Problem()),
        distribution="base manufactured f/yd + independent cosine(0:3)^2 perturbations; train-grid RMS scales .3/.05; v/w scales 1/.1; amplitude Uniform(.25,1.5)",
        splits={},
    )
    prior = set()
    for role, count in zip(("train", "val"), counts):
        a, ref = make(role, count, n)
        validate(a, n)
        ids = fingerprints(a)
        if len(ids) != count or prior & ids:
            raise ValueError("duplicate samples")
        prior |= ids
        np.savez_compressed(root / (role + ".npz"), **a)
        meta["splits"][role] = dict(count=count, sha256=sha(root / (role + ".npz")), reference=ref)
    write(root / "manifest.json", meta)
    return meta


def load(root, split):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    if m["schema"] != "ch05-reconstructible-v1" or m["problem"] != vars(Problem()):
        raise ValueError("schema/problem")
    if split not in m["splits"]:
        raise ValueError("split unavailable")
    if sha(root / (split + ".npz")) != m["splits"][split]["sha256"]:
        raise ValueError("hash")
    with np.load(root / (split + ".npz"), allow_pickle=False) as a:
        data = dict(a)
    if len(data["sample_id"]) != m["splits"][split]["count"]:
        raise ValueError("count")
    validate(data, m["grid"])
    return data, m
