"""Read-only audit of all training budgets, selected weights and data semantics."""

import math
import numpy as np
import torch
from bootstrap import ROOT
from integrity import read, require, sha, verify, dump, inventory, check_manifest
from registry import runs
from validation import validate_fields, disjoint
from predictors import array_file, predict
from neuralpde_opt.chapter3.operator_training import P1Geometry, evaluate_arrays


def audit_run(row, reload=False):
    path = ROOT / row["path"]
    records = verify(path, strict=False)
    # Baseline v1 used a flat manifest, with source snapshots certified separately.
    extras = set(inventory(path, ["manifest.json", "COMPLETE"])) - set(records)
    for name in extras:
        require(name.startswith("source_snapshot/"), "Unexpected historical artifact: " + name)
        original = ROOT / name.removeprefix("source_snapshot/")
        require(
            original.is_file() and sha(path / name) == sha(original),
            "Historical source snapshot changed",
        )
    saved = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
    last = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
    history = read(path / "history.json")
    count = row["epochs"]
    require(last["epoch"] == count and len(history) == count, row["id"] + ": incomplete epochs")
    require([h["epoch"] for h in history] == list(range(1, count + 1)), "History epoch gap")
    require(
        all(np.isfinite(h["val_l2"]) and h["val_l2"] >= 0 for h in history),
        "Nonfinite validation score",
    )
    best = min(history, key=lambda r: r["val_l2"])
    require(
        best["epoch"] == saved["best_epoch"] == last["best_epoch"],
        "Earliest validation selection mismatch",
    )
    require(abs(best["val_l2"] - saved["best_score"]) < 1e-12, "Saved selection score mismatch")
    require(saved["identity"] == last["identity"], "Checkpoint identity mismatch")
    steps = count * math.ceil(row["samples"] / row["batch"])
    actual = last.get("optimizer_steps", last.get("steps"))
    optimizer = last.get("optimizer", last.get("optimizer_state"))
    require(
        actual == steps and {int(v["step"]) for v in optimizer["state"].values()} == {steps},
        "Optimizer budget mismatch",
    )
    visits = last.get("sample_visits", last.get("input_visits", last.get("visits")))
    require(visits == count * row["samples"], "Sample exposure mismatch")
    key = "best_model_state" if row["family"] == "baseline" else "best_model"
    selected = saved["best_model_state"] if row["family"] == "baseline" else saved["model"]
    require(
        set(selected) == set(last[key])
        and all(torch.equal(selected[k], last[key][k]) for k in selected),
        "best.pt differs from last selected weights",
    )
    metric = read(path / "metrics.json")
    require(metric["test_evaluations"] == 0, "Development checkpoint used test")
    if (path / "verification.json").exists():
        require(
            read(path / "verification.json")["passed"] is True, "Failed historical verification"
        )
    result = dict(
        **row,
        steps=steps,
        visits=visits,
        best_epoch=best["epoch"],
        val_l2=best["val_l2"],
        best_sha256=sha(path / "best.pt"),
    )
    if reload and row["family"] != "hints":
        directory = "reaction_v1" if row["dataset"] == "fixed" else "varmion_v1"
        data = array_file(ROOT / f"data/ch03/{directory}/val.npz")
        geometry = P1Geometry(ROOT / f"data/ch03/{directory}/observation_geometry.npz")
        prediction = predict(row, data, geometry.mass, "cuda", 4)
        value = evaluate_arrays(prediction, data["y"], geometry, data["sample_id"])["mean"][
            "relative_l2"
        ]
        # Device/batch change affects float32 accumulation, not checkpoint selection.
        require(abs(value - best["val_l2"]) < 3e-6, "Independent inference/selection mismatch")
        result["reloaded_val_l2"] = value
    return result


def audit_data():
    result = {}
    for name in ["reaction_v1", "varmion_v1"]:
        check_manifest(ROOT / f"data/ch03/{name}", read(ROOT / f"data/ch03/{name}/manifest.json"))
        sets = []
        for split in ["train", "val"]:
            data = array_file(ROOT / f"data/ch03/{name}/{split}.npz")
            config = read(ROOT / f"data/ch03/{name}/config.json")
            groups = config.get(split + "_coefficient_groups")
            forcings = config.get(split + "_forcings_per_group")
            sets.append(validate_fields(data, groups, forcings))
        disjoint(*sets)
        result[name] = dict(train_groups=len(sets[0]), val_groups=len(sets[1]), disjoint=True)
    # Residual metadata must refer to the exact source and Jacobi state, not just valid shapes.
    from physics import P1System

    system = P1System(64, 0.01)
    ii = system.interior
    prepared = ROOT / "results/ch03/varmion_v1"
    check_manifest(prepared, read(prepared / "prepared_manifest.json"))
    phi = array_file(prepared / "basis.npz")["basis"]
    check_manifest(ROOT / "data/ch03/hints_v1", read(ROOT / "data/ch03/hints_v1/manifest.json"))
    worst = 0.0
    total = 0
    for split in ["train", "val"]:
        source = array_file(ROOT / f"data/ch03/varmion_v1/{split}.npz")
        residual = array_file(ROOT / f"data/ch03/hints_v1/{split}.npz")
        require(
            set(read(prepared / "normalization.json")["fit_sample_ids"])
            == set(array_file(ROOT / "data/ch03/varmion_v1/train.npz")["sample_id"]),
            "Normalization fit IDs mismatch",
        )
        indices = residual["source_index"]
        require(
            indices.shape == (len(residual["y"]),)
            and indices.dtype.kind in "iu"
            and indices.min() >= 0
            and indices.max() < len(source["y"]),
            "Invalid residual source indices",
        )
        cached = {}
        for j, index in enumerate(indices):
            require(
                np.array_equal(residual["c"][j], source["c"][index]),
                "Residual coefficient/source mismatch",
            )
            expected = str(source["sample_id"][index]) + f':jacobi:{residual["jacobi_step"][j]:04d}'
            require(
                str(residual["sample_id"][j]) == expected
                and int(residual["jacobi_step"][j]) in [0, 4, 16, 64, 256],
                "Residual state identity mismatch",
            )
            group = str(source["coefficient_group"][index])
            if group not in cached:
                cached[group] = system.assemble(source["c"][index])[0][ii][:, ii]
            error = residual["y"][j].ravel()[ii]
            r = residual["r"][j]
            defect = float(
                np.linalg.norm(cached[group] @ error - r) / max(np.linalg.norm(r), 1e-30)
            )
            require(
                defect < 1e-9 and np.max(abs(phi[ii].T @ r - residual["z"][j])) < 1e-10,
                "Residual equation/load encoding mismatch",
            )
            worst = max(worst, defect)
            total += 1
    result["residuals"] = dict(count=total, max_equation_defect=worst, metadata_verified=True)
    return result


def run(output, reload=True):
    torch.set_num_threads(2)
    data = audit_data()
    records = []
    for row in runs():
        records.append(audit_run(row, reload))
        print("AUDITED", row["id"], flush=True)
    require(len(records) == 57, "Scope mismatch")
    dump(
        output,
        dict(passed=True, runs=records, count=len(records), data=data, test_arrays_read=False),
    )
