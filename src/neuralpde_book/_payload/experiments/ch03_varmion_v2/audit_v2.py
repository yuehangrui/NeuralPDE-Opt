"""Independent quadrature assembly, checkpoint/budget and physical-norm audit."""

import argparse
import json
import subprocess
import sys
import time
import numpy as np
import torch
from scipy import sparse
from bootstrap import ROOT, HERE
from common import sha, dump, verify, certify
from model_v2 import MatrixMap
from neuralpde_opt.mesh import make_mesh


def load(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    out = ROOT / "results/ch03/varmion_v2"
    cfg = json.loads((out / "config.json").read_text())
    for name, digest in json.loads((out / "identity.json").read_text())["sources"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Source changed")
    for name, digest in json.loads((out / "prepared_manifest.json").read_text()).items():
        if sha(out / name) != digest:
            raise RuntimeError("Prepared artifact changed")
    mesh = make_mesh(2, 64)
    elements = mesh.elements
    area = mesh.determinants / 2
    boundary = np.setdiff1d(np.arange(len(mesh.nodes)), mesh.interior)
    _, weights, shapes = mesh.quadrature(4)
    rr = np.broadcast_to(elements[:, :, None], (len(elements), 3, 3)).ravel()
    cc = np.broadcast_to(elements[:, None, :], (len(elements), 3, 3)).ravel()
    phi = load(out / "basis.npz")["basis"]
    k0 = load(out / "basis.npz")["k0"]
    l0 = np.linalg.cholesky(k0)
    np.testing.assert_array_equal(phi[boundary], 0)
    local_mass = np.einsum("eq,qa,qb->eab", weights, shapes, shapes)
    mass = sparse.coo_matrix(
        (local_mass.ravel(), (rr, cc)), shape=(len(mesh.nodes), len(mesh.nodes))
    ).tocsr()
    np.testing.assert_allclose(phi.T @ (mass @ phi), np.eye(64), atol=1e-10)
    base_stiffness = 0.01 * mesh.stiffness()
    op = load(out / "operator_pool.npz")
    worst = 0.0
    for j, c in enumerate(op["c"]):
        cq = np.einsum("ea,qa->eq", c.ravel()[elements], shapes)
        local = np.einsum("eq,eq,qa,qb->eab", weights, cq, shapes, shapes)
        a = base_stiffness + sparse.coo_matrix((local.ravel(), (rr, cc)), shape=mass.shape).tocsr()
        ar = phi.T @ (a @ phi)
        b = l0.T @ ar @ l0
        np.testing.assert_allclose(ar, op["Ar"][j], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(b, op["B"][j], rtol=1e-11, atol=1e-11)
        worst = max(worst, float(np.max(abs(ar - op["Ar"][j]))))
        if np.linalg.eigvalsh(b).min() <= 0:
            raise RuntimeError("Nonpositive target operator")
    data = {s: load(ROOT / f"data/ch03/varmion_v1/{s}.npz") for s in ["train", "val"]}
    from prepare_v2 import sha_array

    if {sha_array(c) for c in op["c"]} & {sha_array(c) for c in data["val"]["c"]}:
        raise RuntimeError("Validation input used in operator training")
    np.testing.assert_array_equal(op["c"][:32], data["train"]["c"][::8])
    if list(op["coefficient_group"]) != [f"cf-v1:train:c{j:03d}" for j in range(128)]:
        raise RuntimeError("Operator pool provenance")
    for split, d in data.items():
        a = load(out / (split + "_coordinates.npz"))
        np.testing.assert_allclose(
            a["alpha"], d["y"].reshape(len(d["y"]), -1) @ (mass @ phi), atol=1e-12
        )
        np.testing.assert_allclose(
            a["z"], d["f"].reshape(len(d["f"]), -1) @ (mass @ phi), atol=1e-12
        )
        np.testing.assert_array_equal(a["sensors"], d["c"][:, ::4, ::4].reshape(len(d["c"]), -1))
    report = dict(
        passed=True,
        operator_inputs_audited=len(op["c"]),
        max_operator_entry_difference=worst,
        independent_quadrature_order=4,
        train_only_operator_pool=True,
        test_evaluations=0,
    )
    if args.preflight:
        tests = subprocess.run(
            [sys.executable, "-B", str(HERE / "test_v2.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        (out / "tests.log").write_text(tests.stdout + "\n" + tests.stderr)
        if tests.returncode:
            report["passed"] = False
            dump(out / "preflight.json", report)
            print(tests.stdout + tests.stderr)
            raise RuntimeError("Preflight tests failed")
        report["test_methods"] = 6
        dump(out / "preflight.json", report)
        print("V2_PREFLIGHT_PASSED", json.dumps(report), flush=True)
        return

    def l2sq(v):
        v = np.asarray(v).reshape(-1, len(mesh.nodes))[:, elements]
        return ((v.sum(-1) ** 2 + (v * v).sum(-1)) * area / 12).sum(1)

    def h1sq(v):
        v = np.asarray(v).reshape(-1, len(mesh.nodes))[:, elements]
        g = np.einsum("nea,eak->nek", v, mesh.gradients)
        return (g * g).sum(2) @ area

    norm = json.loads((out / "normalization.json").read_text())
    audited = []
    aggregate = json.loads((out / "aggregate.json").read_text())
    selection = json.loads((out / "selection.json").read_text())
    eligible = [row for row in selection["screen"] if row["variant"] != "dense_data"]
    if min(eligible, key=lambda row: row["val_l2"])["variant"] != selection["selected"]:
        raise RuntimeError("Pilot selection mismatch")
    expected = {(name, 0) for name in cfg["variants"]}
    expected |= {(name, seed) for name in ["dense_data", selection["selected"]] for seed in [1, 2]}
    if {(r["variant"], r["seed"]) for r in aggregate["rows"]} != expected:
        raise RuntimeError("Missing or unexpected runs")
    for row in aggregate["rows"]:
        variant, seed = row["variant"], row["seed"]
        path = out / variant / f"seed_{seed}"
        verify(path, "COMPLETE")
        last = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        best = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
        steps = cfg["epochs"] * 8
        assert (
            last["steps"] == steps
            and last["visits"] == 256 * cfg["epochs"]
            and len(last["history"]) == cfg["epochs"]
        )
        assert {int(v["step"]) for v in last["optimizer"]["state"].values()} == {steps}
        assert last["operator_visits"] == (
            steps * 32 if cfg["variants"][variant]["operator_groups"] else 0
        )
        selected = min(last["history"], key=lambda h: h["val_l2"])
        assert selected["epoch"] == best["best_epoch"] == last["best_epoch"]
        assert selected["val_l2"] == best["best_score"] == last["best_score"]
        assert (
            best["identity"] == last["identity"] == json.loads((path / "identity.json").read_text())
        )
        for key in best["model"]:
            assert torch.equal(best["model"][key], last["best_model"][key])
        model = (
            MatrixMap(cfg["variants"][variant]["kind"], norm["c_mean"], norm["c_std"], k0)
            .float()
            .eval()
        )
        model.load_state_dict(best["model"])
        metrics = json.loads((path / "metrics.json").read_text())
        verified = {}
        for split, d in data.items():
            saved = load(path / (split + "_predictions.npz"))
            prediction = saved["prediction"]
            target = d["y"]
            np.testing.assert_array_equal(saved["sample_id"], d["sample_id"])
            np.testing.assert_array_equal(prediction.reshape(len(prediction), -1)[:, boundary], 0)
            l2 = np.sqrt(l2sq(prediction - target) / l2sq(target))
            h1 = np.sqrt(h1sq(prediction - target) / h1sq(target))
            np.testing.assert_allclose(
                l2,
                [r["relative_l2"] for r in metrics["metrics"][split]["per_sample"]],
                rtol=1e-10,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                h1,
                [r["relative_h1_seminorm"] for r in metrics["metrics"][split]["per_sample"]],
                rtol=1e-10,
                atol=1e-12,
            )
            coords = load(out / (split + "_coordinates.npz"))
            with torch.no_grad():
                alpha = model(
                    torch.tensor(coords["sensors"], dtype=torch.float32),
                    torch.tensor(coords["z"], dtype=torch.float32),
                ).numpy()
            np.testing.assert_allclose(alpha, saved["coefficients"], rtol=1e-4, atol=2e-6)
            np.testing.assert_allclose(
                prediction, (saved["coefficients"] @ phi.T).reshape(target.shape), atol=1e-13
            )
            verified[split] = dict(l2=float(l2.mean()), h1=float(h1.mean()))
        np.testing.assert_allclose(verified["val"]["l2"], row["val_l2"], atol=1e-12)
        audited.append(dict(variant=variant, seed=seed, metrics=verified))
    frozen = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/verify_frozen_release.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if frozen.returncode:
        raise RuntimeError("Frozen code changed")
    report.update(
        runs=len(audited),
        audited=audited,
        frozen_code=json.loads(frozen.stdout),
        auditor_sha256=sha(Path(__file__)),
    )
    dump(out / "independent_audit.json", report)
    dump(
        out / "status.json",
        dict(status="complete", runs=len(audited), selected=selection["selected"]),
    )
    print(
        "V2_FINAL_AUDIT_PASSED",
        json.dumps({"runs": len(audited), "selected": selection["selected"]}),
        flush=True,
    )


from pathlib import Path

if __name__ == "__main__":
    main()
