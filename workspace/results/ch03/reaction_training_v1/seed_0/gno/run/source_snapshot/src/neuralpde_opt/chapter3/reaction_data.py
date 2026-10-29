"""Gated reaction-only train/val dataset; no legacy-schema conversion or test access."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from .reaction import (
    SCHEMA,
    validate_config,
    draw_reaction,
    ReactionField,
    ReactionFEM,
    ReactionManufactured,
)
from .reference import compare_p1, analytic_errors, save_geometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_split(directory, split):
    if split not in ("train", "val"):
        raise ValueError("Reaction development loader accepts train/val only")
    d = Path(directory)
    config = json.loads((d / "config.json").read_text())
    validate_config(config)
    with np.load(d / (split + ".npz"), allow_pickle=False) as z:
        if set(z.files) != {"c", "y", "sample_id"}:
            raise ValueError("Expected explicit reaction c/y schema")
        data = {k: z[k].copy() for k in z.files}
    n = config["observation_n"] + 1
    shape = (config["splits"][split], n, n)
    if any(data[k].shape != shape or not np.isfinite(data[k]).all() for k in ["c", "y"]):
        raise ValueError("Invalid public arrays")
    if data["sample_id"].tolist() != [f"reaction-v1:{split}:{i:04d}" for i in range(shape[0])]:
        raise ValueError("Invalid sample IDs")
    s = config["reaction"]
    if data["c"].min() < s["lower"] or data["c"].max() > s["upper"]:
        raise ValueError("Reaction outside configured range")
    if (
        np.max(np.abs(data["y"][:, [0, -1], :])) > 1e-14
        or np.max(np.abs(data["y"][:, :, [0, -1]])) > 1e-14
    ):
        raise ValueError("Nonzero boundary labels")
    return data


def verify_dataset(directory):
    d = Path(directory)
    cert = json.loads((d / "COMPLETE").read_text())
    if cert.get("schema_version") != SCHEMA or cert.get("manifest_sha256") != sha(
        d / "manifest.json"
    ):
        raise ValueError("Reaction certificate mismatch")
    manifest = json.loads((d / "manifest.json").read_text())
    required = {
        "train.npz",
        "val.npz",
        "train_reference.npz",
        "val_reference.npz",
        "config.json",
        "quality.json",
        "reload_audit.json",
        "observation_geometry.npz",
        "source_identity.json",
    }
    if not required.issubset(manifest):
        raise ValueError("Missing certified dataset files")
    for name, h in manifest.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(d / name) != h:
            raise ValueError("Reaction dataset checksum mismatch: " + name)
    config = validate_config(json.loads((d / "config.json").read_text()))
    quality = json.loads((d / "quality.json").read_text())
    audit = json.loads((d / "reload_audit.json").read_text())
    if (
        quality.get("passed") is not True
        or audit.get("passed") is not True
        or audit.get("failures") != []
    ):
        raise ValueError("Reaction quality certificate did not pass")
    if audit.get("independent_weak_residual_samples") != sum(config["splits"].values()):
        raise ValueError("Incomplete independent audit coverage")
    data = {s: read_split(d, s) for s in ["train", "val"]}
    if set(data["train"]["sample_id"]) & set(data["val"]["sample_id"]):
        raise ValueError("Overlapping IDs")
    signatures = []
    for s in ["train", "val"]:
        signatures.extend(hashlib.sha256(c.tobytes()).hexdigest() for c in data[s]["c"])
    if len(set(signatures)) != len(signatures):
        raise ValueError("Repeated input fields across dataset")
    return data


def generate(config, output):
    validate_config(config)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "config.json", config)
    # Snapshot all solver dependencies without changing or copying the frozen Ch2 package.
    root = Path(__file__).resolve().parents[3]
    sources = [
        "src/neuralpde_opt/mesh.py",
        "src/neuralpde_opt/chapter3/reaction.py",
        "src/neuralpde_opt/chapter3/reaction_data.py",
        "src/neuralpde_opt/chapter3/fields.py",
        "src/neuralpde_opt/chapter3/reference.py",
    ]
    identity = {p: sha(root / p) for p in sources}
    for p in sources:
        dest = out / "source_snapshot" / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / p, dest)
    dump(out / "source_identity.json", identity)
    eps = config["epsilon"]
    q = config["quadrature_order"]
    qn = config["audit_quadrature_order"]
    on, rn, an = [config[k] for k in ["observation_n", "reference_n", "audit_n"]]
    gates = config["quality_gates"]
    failures = []
    records = []
    mms = []
    for variable in [False, True]:
        field = ReactionManufactured(eps, variable)
        errors = [
            analytic_errors(ReactionFEM(n, eps, q).solve(field), field.exact, field.gradient)
            for n in [8, 16, 32]
        ]
        l2rate = np.log2(errors[-2]["relative_l2"] / errors[-1]["relative_l2"])
        h1rate = np.log2(errors[-2]["relative_h1_seminorm"] / errors[-1]["relative_h1_seminorm"])
        mms.append(
            {
                "variable_reaction": variable,
                "grids": [8, 16, 32],
                "errors": errors,
                "l2_rate": l2rate,
                "h1_rate": h1rate,
            }
        )
        if l2rate < 1.8 or h1rate < 0.9:
            failures.append("manufactured convergence")
    coarse, ref, audit, higher = [
        ReactionFEM(n, eps, order) for n, order in [(on, q), (rn, q), (an, q), (rn, qn)]
    ]
    save_geometry(out / "observation_geometry.npz", on)

    def gate(name, value, sid):
        if not np.isfinite(value) or value > gates[name]:
            failures.append(f"{sid}: {name}={value}")

    for split, count in config["splits"].items():
        public = {k: [] for k in ["c", "y", "sample_id"]}
        parameters = []
        references = []
        for index in range(count):
            sid = f"reaction-v1:{split}:{index:04d}"
            coeff = draw_reaction(config, split, index)
            field = ReactionField(coeff, config["reaction"])
            sol = ref.solve(field)
            c = field.reaction(coarse.mesh.nodes).reshape(on + 1, on + 1)
            y = sol.values.reshape(rn + 1, rn + 1)[:: rn // on, :: rn // on].copy()
            public["c"].append(c)
            public["y"].append(y)
            public["sample_id"].append(sid)
            parameters.append(coeff)
            references.append(sol.values.reshape(rn + 1, rn + 1))
            rec = {"sample_id": sid, **sol.diagnostics}
            for key in [
                "relative_algebraic_residual",
                "relative_energy_identity_defect",
                "boundary_max",
            ]:
                gate(key, rec[key], sid)
            reconstruction = compare_p1(y.ravel(), on, sol)
            rec["observation_reconstruction"] = reconstruction
            gate("observation_reconstruction_l2", reconstruction["relative_l2"], sid)
            if index < config["audit_samples"][split]:
                fine = audit.solve(field)
                low = coarse.solve(field)
                high = higher.solve(field)
                r = compare_p1(sol.values, rn, fine)
                l = compare_p1(low.values, on, fine)
                quad = compare_p1(sol.values, rn, high)
                rec["reference_vs_audit"] = r
                rec["refinement_ratios"] = {k: r[k] / l[k] for k in r}
                rec["quadrature_l2"] = quad["relative_l2"]
                for name, value in [
                    ("reference_vs_audit_l2", r["relative_l2"]),
                    ("reference_vs_audit_h1", r["relative_h1_seminorm"]),
                    ("l2_refinement_ratio", r["relative_l2"] / l["relative_l2"]),
                    ("h1_refinement_ratio", r["relative_h1_seminorm"] / l["relative_h1_seminorm"]),
                    ("quadrature_l2", quad["relative_l2"]),
                ]:
                    gate(name, value, sid)
            records.append(rec)
            if index % 8 == 0:
                print(split, index + 1, count, "failures", len(failures), flush=True)
        np.savez_compressed(out / (split + ".npz"), **{k: np.array(v) for k, v in public.items()})
        np.savez_compressed(
            out / (split + "_reference.npz"),
            c_coefficients=np.array(parameters),
            y=np.array(references),
            sample_id=np.array(public["sample_id"]),
        )
    # Reload every serialized sample and independently evaluate weak form + observation identity.
    reload_records = []
    for split, count in config["splits"].items():
        public = read_split(out, split)
        with np.load(out / (split + "_reference.npz"), allow_pickle=False) as z:
            for i in range(count):
                sid = str(public["sample_id"][i])
                coeff = z["c_coefficients"][i]
                yr = z["y"][i]
                field = ReactionField(coeff, config["reaction"])
                if not np.array_equal(coeff, draw_reaction(config, split, i)):
                    failures.append(sid + ": parameter stream")
                if not np.array_equal(
                    public["c"][i], field.reaction(coarse.mesh.nodes).reshape(on + 1, on + 1)
                ):
                    failures.append(sid + ": c reload")
                if not np.array_equal(public["y"][i], yr[:: rn // on, :: rn // on]):
                    failures.append(sid + ": y reload")
                residual = ref.independent_weak_residual(field, yr.ravel())
                gate("independent_weak_residual", residual, sid)
                reload_records.append({"sample_id": sid, "independent_weak_residual": residual})
    quality = {
        "passed": not failures,
        "failures": failures,
        "mms": mms,
        "samples": records,
        "test_evaluations": 0,
    }
    dump(out / "quality.json", quality)
    dump(
        out / "reload_audit.json",
        {
            "passed": not failures,
            "failures": failures,
            "independent_weak_residual_samples": len(reload_records),
            "samples": reload_records,
        },
    )
    if failures:
        raise RuntimeError("Reaction gates failed; no COMPLETE certificate: " + str(failures[:10]))
    manifest = {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}
    dump(out / "manifest.json", manifest)
    dump(
        out / "COMPLETE",
        {
            "schema_version": SCHEMA,
            "manifest_sha256": sha(out / "manifest.json"),
            "training": "not run",
            "test": "not generated",
        },
    )
    verify_dataset(out)
    print("REACTION_DATA_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate(json.loads(Path(args.config).read_text()), args.output)


if __name__ == "__main__":
    main()
