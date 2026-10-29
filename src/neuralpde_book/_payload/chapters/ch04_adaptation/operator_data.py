"""A declared finite smooth family containing the existing Chapter 4 instance.

Tensor sine modes 1..4 plus (8,8); coefficients have L2 mass 1/4 I.
This is a linear spectral-Galerkin validation family, not the original
semilinear legacy benchmark and not arbitrary-field generalization.
Only train and validation data are generated. No held-out test is created.

Book/code notation:
Book 4.4.2: 17-mode sine coefficients represent f, yd, y, u. EIGENVALUES are positive Laplacian eigenvalues, not multipliers; ALPHA is the physical control penalty.
"""

import hashlib
import numpy as np

ALPHA = 1e-3
MODES = np.array([(i, j) for i in range(1, 5) for j in range(1, 5)] + [(8, 8)], dtype=int)
EIGENVALUES = np.pi**2 * (MODES**2).sum(1)


def basis(x):
    return np.sin(np.pi * x[:, 0:1] * MODES[:, 0]) * np.sin(np.pi * x[:, 1:2] * MODES[:, 1])


def kkt(f, yd):
    y = (ALPHA * EIGENVALUES * f + yd) / (1 + ALPHA * EIGENVALUES**2)
    u = (EIGENVALUES * yd - f) / (1 + ALPHA * EIGENVALUES**2)
    return y, u, -ALPHA * u


def generate(role, count, seed=0):
    if role not in ("train", "validation") or type(count) is not int or count < 1:
        raise ValueError("only explicit train/validation splits are supported")
    rng = np.random.default_rng(
        np.random.SeedSequence([20261025, 444, seed, 1 if role == "train" else 2])
    )
    d = len(MODES)
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
    values = dict(f=f, yd=yd, y=y, u=u, p=p, rhs=rhs, state=rhs / EIGENVALUES)
    for key, value in values.items():
        if value.shape != (count, d) or not np.isfinite(value).all():
            raise ValueError("invalid generated " + key)
    values["ids"] = np.array([f"{role}:{seed}:{i}" for i in range(count)])
    return values


def validate(train, val):
    if set(train["ids"]) & set(val["ids"]):
        raise ValueError("overlapping sample ids")
    hashes = []
    for split in (train, val):
        hh = []
        for i in range(len(split["ids"])):
            row = np.concatenate([split[k][i] for k in ("f", "yd", "rhs")])
            hh.append(hashlib.sha256(row.tobytes()).hexdigest())
        if len(set(hh)) != len(hh):
            raise ValueError("duplicate samples")
        hashes.append(hh)
        y, u, p = split["y"], split["u"], split["p"]
        if not all(
            np.allclose(a, 0, atol=1e-11, rtol=0)
            for a in (
                EIGENVALUES * y - u - split["f"],
                EIGENVALUES * p - y + split["yd"],
                ALPHA * u + p,
            )
        ):
            raise ValueError("nonconverged KKT labels")
    if set(hashes[0]) & set(hashes[1]):
        raise ValueError("train/validation data overlap")
    return dict(train=hashes[0], validation=hashes[1])
