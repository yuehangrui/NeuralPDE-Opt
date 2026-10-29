"""New, disjoint ch05 data. Labels checked against projected optimality.

Distribution is an explicit small Fourier family around the legacy example,
NOT the original paper GRF. Training loads train/val only.
"""

from pathlib import Path
import numpy as np
from .common import write_json, sha
from .reference import Grid
from .problem import Problem


def random_field(rng, points, count, scale):
    out = np.zeros((count,) + points.shape[:2])
    for i in range(4):
        for j in range(4):
            basis = np.cos(np.pi * i * points[..., 0]) * np.cos(np.pi * j * points[..., 1])
            out += rng.normal(size=(count, 1, 1)) * basis / (1 + i * i + j * j) ** 1.5
    rms = np.sqrt(np.mean(out**2, axis=(1, 2), keepdims=True))
    return out / np.maximum(rms, 1e-12) * rng.uniform(0.25, 1.5, (count, 1, 1)) * scale


def generate(out, n=32, counts=(1024, 128, 128), seed=0, problem=None):
    problem = problem or Problem()
    if len(counts) != 3 or any(type(c) is not int or c < 1 for c in counts):
        raise ValueError("invalid split counts")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be nonnegative integer")
    grid = Grid(n)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        schema="ch05-interior-fd-v1",
        n=n,
        seed=seed,
        problem=vars(problem),
        distribution="base manufactured f,yd + independent cosine(0:3)^2 fields; perturbation RMS f=.3, yd=.05",
        discretization="five-point FD, homogeneous Dirichlet, h^2 I quadrature on interior grid",
        splits={},
    )
    for i, (split, count) in enumerate(zip(("train", "val", "test"), counts)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, i, 505]))
        f = problem.f(grid.points)[..., 0] + random_field(rng, grid.points, count, 0.3)
        yd = problem.yd(grid.points)[..., 0] + random_field(rng, grid.points, count, 0.05)
        ref = grid.optimal_control(f, yd, problem)
        v = random_field(rng, grid.points, count, 1.0)
        w = random_field(rng, grid.points, count, 0.1)
        values = dict(
            f=f,
            yd=yd,
            u=ref["u"],
            y=ref["y"],
            p=ref["p"],
            v=v,
            sv=grid.solve(v),
            w=w,
            sw=grid.solve(w),
            sample_id=np.arange(count, dtype=np.int64) + i * 10**9,
        )
        file = out / (split + ".npz")
        np.savez_compressed(file, **values)
        metadata["splits"][split] = dict(
            count=count,
            sha256=sha(file),
            reference_iterations=ref["iterations"],
            max_projected_residual=float(np.max(ref["projected_residual"])),
        )
    write_json(out / "manifest.json", metadata)
    return metadata


def load_split(root, split, problem=None, n=None):
    import json

    if split not in ("train", "val", "test"):
        raise ValueError("unknown split")
    root = Path(root)
    meta = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if meta["schema"] != "ch05-interior-fd-v1":
        raise ValueError("unknown dataset schema")
    if problem is not None and meta["problem"] != vars(problem):
        raise ValueError("problem/data mismatch")
    if n is not None and meta["n"] != n:
        raise ValueError("grid/data mismatch")
    file = root / (split + ".npz")
    if sha(file) != meta["splits"][split]["sha256"]:
        raise ValueError("dataset hash mismatch")
    with np.load(file, allow_pickle=False) as blob:
        data = {k: blob[k] for k in blob.files}
    count = meta["splits"][split]["count"]
    expected = (count, meta["n"] - 1, meta["n"] - 1)
    if set(data) != set(("f", "yd", "u", "y", "p", "v", "sv", "w", "sw", "sample_id")):
        raise ValueError("data keys")
    for key, value in data.items():
        shape = (count,) if key == "sample_id" else expected
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("invalid array " + key)
    if data["sample_id"].dtype != np.int64 or len(np.unique(data["sample_id"])) != count:
        raise ValueError("invalid sample IDs")
    prob = Problem(**meta["problem"])
    grid = Grid(meta["n"])
    d = grid.diagnostics(data["u"], data["f"], data["yd"], prob)
    if np.max(d["bound_violation"]) > 1e-12 or np.max(d["projected_residual"]) > 2e-9:
        raise ValueError("invalid optimal control labels")
    for name, rhs in (
        ("y", data["u"] + data["f"]),
        ("p", data["y"] - data["yd"]),
        ("sv", data["v"]),
        ("sw", data["w"]),
    ):
        if np.max(np.abs(grid.solve(rhs) - data[name])) > 1e-10:
            raise ValueError("invalid PDE label " + name)
    return data, meta
