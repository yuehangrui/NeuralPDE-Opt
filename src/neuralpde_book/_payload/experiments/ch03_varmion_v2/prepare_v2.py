"""Reuse certified labels/basis; extra TRAIN coefficients supply operators only."""

import json
import numpy as np
from bootstrap import ROOT
from common import sha, dump
from data import load, parameters, CFField
from physics import P1System
from neuralpde_opt.chapter3.operator_training import P1Geometry


def prepare(out, config):
    source = ROOT / "results/ch03/varmion_v1"
    dataset = ROOT / "data/ch03/varmion_v1"
    old_config = json.loads((source / "config.json").read_text())
    for name, digest in json.loads((source / "prepared_manifest.json").read_text()).items():
        if sha(source / name) != digest:
            raise ValueError("Changed v1 prepared artifact")
    data = load(dataset)
    geometry = P1Geometry(dataset / "observation_geometry.npz")
    with np.load(source / "basis.npz", allow_pickle=False) as a:
        phi, k0 = a["basis"].copy(), a["k0"].copy()
        np.testing.assert_array_equal(a["fit_sample_ids"], data["train"]["sample_id"])
    if phi.shape[1] != config["basis_rank"]:
        raise ValueError("Rank mismatch")
    norm = json.loads((source / "normalization.json").read_text())
    if norm["fit_split"] != "train":
        raise ValueError("Normalization leakage")
    arrays = {}
    for split, d in data.items():
        with np.load(source / (split + "_coordinates.npz"), allow_pickle=False) as a:
            np.testing.assert_array_equal(a["c"], d["c"])
            arrays[split] = {
                k: a[k].copy() for k in ["z", "alpha", "projection_squared", "denominator"]
            }
        arrays[split]["sensors"] = d["c"][:, ::4, ::4].reshape(len(d["c"]), -1)
        if np.min(arrays[split]["denominator"]) <= 0:
            raise ValueError("Invalid target norms")
    system = P1System(64, 0.01)
    base = json.loads((ROOT / "configs/ch03_reaction_v1.json").read_text())
    c_fields, cc_fields, ids, ar_all, b_all = [], [], [], [], []
    l0 = np.linalg.cholesky(k0)
    train_unique = data["train"]["c"][:: old_config["train_forcings_per_group"]]
    val_hashes = {sha_array(c) for c in data["val"]["c"]}
    for group in range(config["augmented_train_coefficient_groups"]):
        cc, ff = parameters(old_config, base, "train", group, 0)
        c = CFField(cc, ff, base["reaction"]).reaction(system.mesh.nodes).reshape(65, 65)
        if group < len(train_unique):
            np.testing.assert_array_equal(c, train_unique[group])
        if sha_array(c) in val_hashes:
            raise ValueError("Operator training input overlaps validation")
        a, _ = system.assemble(c)
        ar = phi.T @ (a @ phi)
        b = l0.T @ ar @ l0
        np.linalg.cholesky(b)
        c_fields.append(c)
        cc_fields.append(cc)
        ids.append(f"cf-v1:train:c{group:03d}")
        ar_all.append(ar)
        b_all.append(b)
    if len({sha_array(c) for c in c_fields}) != len(c_fields):
        raise ValueError("Duplicate training coefficient fields")
    c_fields = np.asarray(c_fields)
    pool = {"sensors": c_fields[:, ::4, ::4].reshape(len(c_fields), -1), "B": np.asarray(b_all)}
    np.savez_compressed(
        out / "operator_pool.npz",
        **pool,
        c=c_fields,
        c_coefficients=np.asarray(cc_fields),
        Ar=np.asarray(ar_all),
        coefficient_group=np.asarray(ids),
    )
    np.savez_compressed(
        out / "basis.npz", basis=phi, k0=k0, fit_sample_ids=data["train"]["sample_id"]
    )
    dump(out / "normalization.json", norm)
    for split, d in arrays.items():
        np.savez_compressed(out / (split + "_coordinates.npz"), **d)
    dump(
        out / "preparation.json",
        dict(
            labeled_train=256,
            labeled_val=64,
            existing_train_coefficients=32,
            augmented_train_coefficients=len(c_fields),
            additional_fine_grid_labels=0,
            basis_fitted_on="existing v1 train only",
            normalization_fitted_on="existing v1 train only",
            operator_pool_split="train",
            validation_coefficient_overlap=False,
            test_evaluations=0,
            source_data_manifest=sha(dataset / "manifest.json"),
            source_prepared_manifest=sha(source / "prepared_manifest.json"),
        ),
    )
    names = [
        "operator_pool.npz",
        "basis.npz",
        "normalization.json",
        "train_coordinates.npz",
        "val_coordinates.npz",
        "preparation.json",
    ]
    dump(out / "prepared_manifest.json", {name: sha(out / name) for name in names})
    return data, geometry, phi, k0, norm, arrays, pool


def sha_array(a):
    import hashlib

    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def reload_prepared(out):
    for name, digest in json.loads((out / "prepared_manifest.json").read_text()).items():
        if sha(out / name) != digest:
            raise ValueError("Prepared v2 artifact changed")
    data = load(ROOT / "data/ch03/varmion_v1")
    geometry = P1Geometry(ROOT / "data/ch03/varmion_v1/observation_geometry.npz")
    with np.load(out / "basis.npz", allow_pickle=False) as a:
        phi, k0 = a["basis"].copy(), a["k0"].copy()
    norm = json.loads((out / "normalization.json").read_text())
    arrays = {}
    for split in ["train", "val"]:
        with np.load(out / (split + "_coordinates.npz"), allow_pickle=False) as a:
            arrays[split] = {k: a[k].copy() for k in a.files}
    with np.load(out / "operator_pool.npz", allow_pickle=False) as a:
        pool = {k: a[k].copy() for k in ["sensors", "B"]}
    return data, geometry, phi, k0, norm, arrays, pool
