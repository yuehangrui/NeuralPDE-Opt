"""Grouped train/val data for the continuous (c,f)->y reaction equation."""

import hashlib, json
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import splu
from neuralpde_opt.chapter3.fields import cosine_series, spectral_weights
from neuralpde_opt.chapter3.reaction import draw_reaction, ReactionField, ReactionFEM
from neuralpde_opt.chapter3.reference import compare_p1, save_geometry

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class CFField:
    def __init__(self, c_coeff, f_coeff, c_spec):
        self.c = ReactionField(c_coeff, c_spec)
        self.f = np.asarray(f_coeff)

    def reaction(self, points):
        return self.c.reaction(points)

    def source(self, points):
        return cosine_series(points, self.f)


def parameters(config, base, split, group, index):
    if split not in ["train", "val"] or min(group, index) < 0:
        raise ValueError("Train/val parameters only")
    copy = {**base, "seed": config["seed"]}
    c = draw_reaction(copy, split, group)
    code = 731 if split == "train" else 947
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], code, group, index, 811]))
    f = rng.normal(size=(config["source_modes"], config["source_modes"])) * spectral_weights(
        config["source_modes"], config["source_length_scale"], config["source_exponent"], True
    )
    return c, f


def generate(config, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    base = json.loads((ROOT / "configs/ch03_reaction_v1.json").read_text())
    dump(out / "config.json", config)
    dump(out / "reaction_field_spec.json", base["reaction"])
    ref = ReactionFEM(config["reference_n"], config["epsilon"], config["quadrature_order"])
    fine = ReactionFEM(config["audit_n"], config["epsilon"], config["quadrature_order"])
    obs = ReactionFEM(config["observation_n"], config["epsilon"], config["quadrature_order"])
    save_geometry(out / "observation_geometry.npz", config["observation_n"])
    ii = ref.mesh.interior
    stride = config["reference_n"] // config["observation_n"]
    records = []
    refinement = []
    for split in ["train", "val"]:
        count = config[split + "_coefficient_groups"]
        nf = config[split + "_forcings_per_group"]
        public = {k: [] for k in ["c", "f", "y", "sample_id", "coefficient_group"]}
        raw = {k: [] for k in ["c_coefficients", "f_coefficients", "y_reference", "sample_id"]}
        for group in range(count):
            cc, fc = parameters(config, base, split, group, 0)
            field = CFField(cc, fc, base["reaction"])
            a, _, _ = ref.assemble(field)
            factor = splu(a[ii][:, ii].tocsc())
            c_public = field.reaction(obs.mesh.nodes).reshape(config["observation_n"] + 1, -1)
            for index in range(nf):
                cc, fc = parameters(config, base, split, group, index)
                field = CFField(cc, fc, base["reaction"])
                fq = field.source(ref.points)
                local = np.einsum("eq,eq,qa->ea", ref.weights, fq, ref.phi)
                b = np.zeros(len(ref.mesh.nodes))
                np.add.at(b, ref.mesh.elements.ravel(), local.ravel())
                y = np.zeros_like(b)
                y[ii] = factor.solve(b[ii])
                sid = f"cf-v1:{split}:c{group:03d}:f{index:03d}"
                residual = float(np.linalg.norm((a @ y - b)[ii]) / np.linalg.norm(b[ii]))
                independent = ref.independent_weak_residual(field, y)
                if max(residual, independent) > 1e-9:
                    raise RuntimeError("Reference residual gate: " + sid)
                yr = y.reshape(config["reference_n"] + 1, -1)
                yp = yr[::stride, ::stride].copy()
                public["c"].append(c_public)
                public["f"].append(
                    field.source(obs.mesh.nodes).reshape(config["observation_n"] + 1, -1)
                )
                public["y"].append(yp)
                public["sample_id"].append(sid)
                public["coefficient_group"].append(f"cf-v1:{split}:c{group:03d}")
                raw["c_coefficients"].append(cc)
                raw["f_coefficients"].append(fc)
                raw["y_reference"].append(yr)
                raw["sample_id"].append(sid)
                records.append(
                    {
                        "sample_id": sid,
                        "relative_residual": residual,
                        "independent_weak_residual": independent,
                    }
                )
                if group < 2 and index == 0:
                    high = fine.solve(field)
                    comparison = compare_p1(y, config["reference_n"], high)
                    if (
                        comparison["relative_l2"] > 0.003
                        or comparison["relative_h1_seminorm"] > 0.05
                    ):
                        raise RuntimeError("Reference refinement gate: " + sid)
                    refinement.append({"sample_id": sid, **comparison})
            if group % 8 == 0:
                print("CF_DATA_GROUP", split, group, count, flush=True)
        np.savez_compressed(out / (split + ".npz"), **{k: np.asarray(v) for k, v in public.items()})
        np.savez_compressed(
            out / (split + "_reference.npz"), **{k: np.asarray(v) for k, v in raw.items()}
        )
    # Reload every public label and compare it with its serialized fine-grid reference.
    for split in ["train", "val"]:
        with (
            np.load(out / (split + ".npz"), allow_pickle=False) as z,
            np.load(out / (split + "_reference.npz"), allow_pickle=False) as raw,
        ):
            if not np.array_equal(z["sample_id"], raw["sample_id"]) or not np.array_equal(
                z["y"], raw["y_reference"][:, ::stride, ::stride]
            ):
                raise RuntimeError("Serialized label mismatch")
            for j, sid in enumerate(z["sample_id"]):
                field = CFField(
                    raw["c_coefficients"][j], raw["f_coefficients"][j], base["reaction"]
                )
                if not np.array_equal(
                    z["c"][j].ravel(), field.reaction(obs.mesh.nodes)
                ) or not np.array_equal(z["f"][j].ravel(), field.source(obs.mesh.nodes)):
                    raise RuntimeError("Serialized field mismatch: " + str(sid))
    dump(
        out / "quality.json",
        {
            "passed": True,
            "reference_records": records,
            "refinement": refinement,
            "reloaded_public_samples": len(records),
            "test_evaluations": 0,
        },
    )
    dump(
        out / "source_identity.json",
        {
            "data.py": sha(Path(__file__)),
            "reaction.py": sha(ROOT / "src/neuralpde_opt/chapter3/reaction.py"),
            "fields.py": sha(ROOT / "src/neuralpde_opt/chapter3/fields.py"),
        },
    )
    dump(out / "manifest.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    dump(
        out / "CF_DATA_COMPLETE",
        {"schema_version": "ch03-cf-v1", "manifest_sha256": sha(out / "manifest.json")},
    )
    load(out)
    print("CF_DATA_COMPLETE", flush=True)


def load(output):
    out = Path(output)
    cert = json.loads((out / "CF_DATA_COMPLETE").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    if cert["schema_version"] != "ch03-cf-v1" or cert["manifest_sha256"] != sha(
        out / "manifest.json"
    ):
        raise ValueError("Data certificate mismatch")
    for name, digest in manifest.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(out / name) != digest:
            raise ValueError("Changed dataset file")
    config = json.loads((out / "config.json").read_text())
    quality = json.loads((out / "quality.json").read_text())
    if quality["passed"] is not True or config["test_evaluations"] != 0:
        raise ValueError("Invalid data protocol")
    data = {}
    groups = {}
    signatures = []
    for split in ["train", "val"]:
        with np.load(out / (split + ".npz"), allow_pickle=False) as z:
            data[split] = {k: z[k].copy() for k in z.files}
        d = data[split]
        count = config[split + "_coefficient_groups"] * config[split + "_forcings_per_group"]
        shape = (count, config["observation_n"] + 1, config["observation_n"] + 1)
        if set(d) != {"c", "f", "y", "sample_id", "coefficient_group"} or any(
            d[k].shape != shape or not np.isfinite(d[k]).all() for k in ["c", "f", "y"]
        ):
            raise ValueError("Invalid public data")
        if (
            d["c"].min() < 0.5
            or d["c"].max() > 1.5
            or len(set(d["sample_id"])) != count
            or any(not str(s).startswith("cf-v1:" + split + ":") for s in d["sample_id"])
        ):
            raise ValueError("Invalid sample identity/coefficient")
        if max(abs(d["y"][:, [0, -1], :]).max(), abs(d["y"][:, :, [0, -1]]).max()) > 1e-14:
            raise ValueError("Invalid boundary")
        groups[split] = {hashlib.sha256(c.tobytes()).hexdigest() for c in d["c"]}
        if len(groups[split]) != config[split + "_coefficient_groups"]:
            raise ValueError("Incorrect coefficient groups")
        signatures.extend(
            hashlib.sha256(c.tobytes() + f.tobytes()).hexdigest() for c, f in zip(d["c"], d["f"])
        )
    if groups["train"] & groups["val"] or len(set(signatures)) != len(signatures):
        raise ValueError("Overlapping coefficient groups or duplicated pairs")
    return data
