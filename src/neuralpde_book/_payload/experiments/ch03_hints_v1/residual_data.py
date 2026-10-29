"""HINTS load residual r_h=b_h-A_h*y_h is encoded by Phi_I^T*r_h. Do not apply a second mass matrix: r_h is already a load vector, not a nodal source field."""

import numpy as np
from scipy.sparse.linalg import splu
from common import dump, certify, verify


def generate(directory, source, basis, system, config):
    directory.mkdir(parents=True, exist_ok=False)
    ii = system.interior
    phi = basis[ii]
    mass_phi = system.mass @ basis
    reports = {}
    for split, d in source.items():
        rows = {
            k: []
            for k in [
                "c",
                "z",
                "alpha",
                "denominator",
                "projection_squared",
                "r",
                "y",
                "sample_id",
                "source_index",
                "jacobi_step",
            ]
        }
        worst = 0.0
        selected = []
        for group in dict.fromkeys(d["coefficient_group"]):
            indices = np.flatnonzero(d["coefficient_group"] == group)[
                : config[split + "_forcings_per_group"]
            ]
            first = int(indices[0])
            a_full, _ = system.assemble(d["c"][first])
            a = a_full[ii][:, ii].tocsr()
            factor = splu(a.tocsc())
            diag = a.diagonal()
            for local_index, j in enumerate(indices):
                j = int(j)
                selected.append(j)
                b = (system.mass @ d["f"][j].ravel())[ii]
                truth = factor.solve(b)
                xy = system.mesh.nodes[ii]
                y = (
                    np.zeros(len(ii))
                    if local_index % 2 == 0
                    else 0.05 * np.sin(20 * np.pi * xy[:, 0]) * np.sin(17 * np.pi * xy[:, 1])
                )
                for step in range(max(config["jacobi_steps"]) + 1):
                    r = b - a @ y
                    if step in config["jacobi_steps"]:
                        e = np.zeros(system.size)
                        e[ii] = truth - y
                        den = float(e @ (system.mass @ e))
                        alpha = e @ mass_phi
                        proj = e - basis @ alpha
                        if den <= 1e-24:
                            raise RuntimeError("Degenerate correction sample")
                        defect = float(
                            np.linalg.norm(a @ e[ii] - r) / max(np.linalg.norm(r), 1e-30)
                        )
                        worst = max(worst, defect)
                        if defect > 1e-9:
                            raise RuntimeError("Residual-correction identity failed")
                        values = {
                            "c": d["c"][j],
                            "z": phi.T @ r,
                            "alpha": alpha,
                            "denominator": den,
                            "projection_squared": float(proj @ (system.mass @ proj)),
                            "r": r,
                            "y": e.reshape(system.n + 1, system.n + 1),
                            "sample_id": str(d["sample_id"][j]) + f":jacobi:{step:04d}",
                            "source_index": j,
                            "jacobi_step": step,
                        }
                        for key, value in values.items():
                            rows[key].append(value)
                    if step < max(config["jacobi_steps"]):
                        y += config["omega"] * r / diag
        np.savez_compressed(
            directory / (split + ".npz"), **{k: np.asarray(v) for k, v in rows.items()}
        )
        reports[split] = {
            "count": len(rows["sample_id"]),
            "selected_source_indices": selected,
            "max_residual_equation_defect": worst,
        }
    dump(directory / "quality.json", {"passed": True, "splits": reports, "test_evaluations": 0})
    dump(directory / "config.json", config)
    certify(directory, "RESIDUAL_DATA_COMPLETE")


def load(directory):
    verify(directory, "RESIDUAL_DATA_COMPLETE")
    data = {}
    arrays = {}
    for split in ["train", "val"]:
        with np.load(directory / (split + ".npz"), allow_pickle=False) as z:
            data[split] = {k: z[k].copy() for k in z.files}
        d = data[split]
        if (
            not all(
                np.isfinite(d[k]).all()
                for k in ["c", "z", "alpha", "denominator", "projection_squared", "r", "y"]
            )
            or np.min(d["denominator"]) <= 0
        ):
            raise ValueError("Invalid residual data")
        if len(set(d["sample_id"])) != len(d["sample_id"]) or any(
            f":{split}:" not in str(s) for s in d["sample_id"]
        ):
            raise ValueError("Invalid residual identity")
        arrays[split] = {k: d[k] for k in ["c", "z", "alpha", "denominator", "projection_squared"]}
    return data, arrays
