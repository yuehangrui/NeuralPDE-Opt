"""Semantic data checks supplement historical byte certificates."""

import hashlib
import numpy as np
from integrity import require


def field_signatures(c):
    return [
        hashlib.sha256(np.ascontiguousarray(v, dtype=np.float64).tobytes()).hexdigest() for v in c
    ]


def validate_fields(data, groups=None, forcings=None, prefix=None):
    required = {"c", "y", "sample_id"}
    require(required <= set(data), "Missing data fields")
    c, y, ids = data["c"], data["y"], data["sample_id"]
    require(
        c.ndim == 3 and c.shape[1:] == (65, 65) and len(c) > 0 and y.shape == c.shape,
        "Invalid public shape",
    )
    require(
        ids.shape == (len(c),) and ids.dtype.kind in "US" and len(set(ids)) == len(c),
        "Invalid sample IDs",
    )
    require(prefix is None or all(str(s).startswith(prefix) for s in ids), "Wrong split ID")
    require(
        np.isfinite(c).all() and np.isfinite(y).all() and c.min() >= 0.5 and c.max() <= 1.5,
        "Invalid physical values",
    )
    require(
        max(abs(y[:, [0, -1], :]).max(), abs(y[:, :, [0, -1]]).max()) <= 1e-14, "Nonzero boundary"
    )
    require(
        np.all(np.linalg.norm(y.reshape(len(y), -1), axis=1) > 0),
        "Zero target invalidates relative metrics",
    )
    if "f" in data:
        require(data["f"].shape == c.shape and np.isfinite(data["f"]).all(), "Invalid source")
    if "coefficient_group" in data:
        labels = data["coefficient_group"]
        require(labels.shape == ids.shape and labels.dtype.kind in "US", "Invalid group labels")
        unique = list(dict.fromkeys(labels))
        require(groups is None or len(unique) == groups, "Wrong group count")
        seen = set()
        for group in unique:
            ix = np.flatnonzero(labels == group)
            require(forcings is None or len(ix) == forcings, "Wrong forcings per group")
            require(
                all(str(ids[j]).startswith(str(group) + ":f") for j in ix), "Sample/group mismatch"
            )
            signatures = field_signatures(c[ix])
            require(
                len(set(signatures)) == 1 and signatures[0] not in seen,
                "Group/field mapping is not one-to-one",
            )
            seen.add(signatures[0])
        if "f" in data:
            pairs = [
                hashlib.sha256(np.concatenate([a.ravel(), b.ravel()]).tobytes()).hexdigest()
                for a, b in zip(c, data["f"])
            ]
            require(len(set(pairs)) == len(pairs), "Duplicate (c,f) pair")
    else:
        require(len(set(field_signatures(c))) == len(c), "Duplicate coefficient")
    return set(field_signatures(c))


def disjoint(*sets):
    seen = set()
    for group in sets:
        require(not seen.intersection(group), "Coefficient leakage across partitions")
        seen.update(group)
