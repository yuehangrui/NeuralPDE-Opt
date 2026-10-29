"""Predeclared unseen reaction fields; test generation cannot select checkpoints."""

import numpy as np
from scipy.sparse.linalg import splu
from bootstrap import ROOT
from integrity import require, read, dump, sha, certify, verify, check_manifest
from registry import runs, source_manifest
from predictors import array_file
from validation import validate_fields, field_signatures, disjoint
from neuralpde_opt.chapter3.fields import spectral_weights
from neuralpde_opt.chapter3.reaction import ReactionFEM
from neuralpde_opt.chapter3.reference import compare_p1, save_geometry
from data import CFField


def register(directory):
    """Must run before generation, with all selections and numerical controls fixed."""
    directory.mkdir(parents=True, exist_ok=False)
    audit = read(ROOT / "results/ch03/final_v1_support/training_audit.json")
    require(audit["passed"] is True and audit["count"] == 57, "Training audit required")
    protocol = dict(
        schema="ch03-final-test-v1",
        seed=2026102803,
        split_code=2813,
        fixed_count=128,
        cf_groups=32,
        cf_forcings=4,
        hints_forcings=2,
        observation_n=64,
        reference_n=128,
        audit_n=256,
        quadrature_order=4,
        epsilon=0.01,
        refinement_first_groups=2,
        relative_reference_l2_gate=0.003,
        relative_reference_h1_gate=0.05,
        relative_weak_residual_gate=1e-9,
        model_selection="Already fixed by train/val; every registered checkpoint is reported",
        future_use="After this evaluation these test sets are exposed; no tuning on them",
        hints=dict(
            period=10, eta=0.9, omega=2 / 3, rtol=1e-8, atol=1e-12, maxiter=10000, repetitions=3
        ),
        sources=source_manifest(),
        checkpoints={
            r["id"]: dict(path=r["path"] + "/best.pt", sha256=sha(ROOT / r["path"] / "best.pt"))
            for r in runs()
        },
        runs=runs(),
    )
    dump(directory / "protocol.json", protocol)
    (directory / "REGISTERED").write_text(sha(directory / "protocol.json") + "\n", encoding="ascii")
    return protocol


def protocol_at(directory):
    require(
        (directory / "REGISTERED").read_text().strip() == sha(directory / "protocol.json"),
        "Protocol changed",
    )
    p = read(directory / "protocol.json")
    check_manifest(ROOT, p["sources"])
    require(p["sources"] == source_manifest(), "Source inventory changed after registration")
    for value in p["checkpoints"].values():
        require(sha(ROOT / value["path"]) == value["sha256"], "Selected weights changed")
    return p


def parameters(protocol, kind, group, forcing, reaction_spec, cf_config):
    family = 1 if kind == "fixed" else 2
    key = [protocol["seed"], protocol["split_code"], family, group]
    rng = np.random.default_rng(np.random.SeedSequence(key + [23]))
    c = rng.normal(size=(reaction_spec["modes"],) * 2) * spectral_weights(
        reaction_spec["modes"],
        reaction_spec["length_scale"],
        reaction_spec["exponent"],
        reaction_spec["include_constant"],
    )
    if kind == "fixed":
        f = np.zeros((cf_config["source_modes"],) * 2)
        # cosine_series uses an orthonormal constant basis of value 1.
        f[0, 0] = 1.0
    else:
        rng = np.random.default_rng(np.random.SeedSequence(key + [forcing, 811]))
        f = rng.normal(size=(cf_config["source_modes"],) * 2) * spectral_weights(
            cf_config["source_modes"],
            cf_config["source_length_scale"],
            cf_config["source_exponent"],
            True,
        )
    return c, f


def generate(directory):
    p = protocol_at(directory)
    out = ROOT / "data/ch03/final_test_v1"
    if (out / "COMPLETE").exists():
        verify(out)
        require(
            read(out / "identity.json")["protocol_sha256"] == sha(directory / "protocol.json"),
            "Different test protocol",
        )
        return out
    # A failed incomplete generation is kept for diagnosis and is never overwritten.
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "identity.json", dict(protocol_sha256=sha(directory / "protocol.json")))
    base = read(ROOT / "configs/ch03_reaction_v1.json")
    cf = read(ROOT / "experiments/ch03_varmion_v1/config.json")
    reference = ReactionFEM(p["reference_n"], p["epsilon"], p["quadrature_order"])
    observation = ReactionFEM(p["observation_n"], p["epsilon"], p["quadrature_order"])
    high = ReactionFEM(p["audit_n"], p["epsilon"], p["quadrature_order"])
    ii = reference.mesh.interior
    save_geometry(out / "observation_geometry.npz", p["observation_n"])
    forbidden = set()
    for old in ["reaction_v1", "varmion_v1"]:
        for split in ["train", "val"]:
            forbidden.update(
                field_signatures(array_file(ROOT / f"data/ch03/{old}/{split}.npz")["c"])
            )
    # Also exclude the extra coefficient-only training inputs used by VarMiON v2.
    from data import parameters as old_parameters

    for group in range(128):
        cc, fc = old_parameters(cf, base, "train", group, 0)
        c = CFField(cc, fc, base["reaction"]).reaction(observation.mesh.nodes).reshape(65, 65)
        forbidden.update(field_signatures(c[None]))
    quality = []
    refinements = []
    for kind, groups, nf in [
        ("fixed", p["fixed_count"], 1),
        ("cf", p["cf_groups"], p["cf_forcings"]),
    ]:
        values = {k: [] for k in ["c", "f", "y", "sample_id", "coefficient_group"]}
        raw = {k: [] for k in ["c_coefficients", "f_coefficients", "y_reference", "sample_id"]}
        for group in range(groups):
            cc, fc = parameters(p, kind, group, 0, base["reaction"], cf)
            field = CFField(cc, fc, base["reaction"])
            matrix, _, _ = reference.assemble(field)
            factor = splu(matrix[ii][:, ii].tocsc())
            cp = field.reaction(observation.mesh.nodes).reshape(65, 65)
            require(
                not set(field_signatures(cp[None])) & forbidden,
                "New test overlaps training/development",
            )
            for forcing in range(nf):
                cc, fc = parameters(p, kind, group, forcing, base["reaction"], cf)
                field = CFField(cc, fc, base["reaction"])
                local = np.einsum(
                    "eq,eq,qa->ea", reference.weights, field.source(reference.points), reference.phi
                )
                b = np.zeros(len(reference.mesh.nodes))
                np.add.at(b, reference.mesh.elements.ravel(), local.ravel())
                y = np.zeros_like(b)
                y[ii] = factor.solve(b[ii])
                weak = reference.independent_weak_residual(field, y)
                residual = float(np.linalg.norm((matrix @ y - b)[ii]) / np.linalg.norm(b[ii]))
                require(
                    max(weak, residual) < p["relative_weak_residual_gate"],
                    "Reference residual gate",
                )
                group_id = f"final-v1:{kind}:test:c{group:03d}"
                sid = group_id + f":f{forcing:03d}"
                yr = y.reshape(129, 129)
                items = dict(
                    c=cp,
                    f=field.source(observation.mesh.nodes).reshape(65, 65),
                    y=yr[::2, ::2].copy(),
                    sample_id=sid,
                    coefficient_group=group_id,
                )
                for k, v in items.items():
                    values[k].append(v)
                for k, v in dict(
                    c_coefficients=cc, f_coefficients=fc, y_reference=yr, sample_id=sid
                ).items():
                    raw[k].append(v)
                quality.append(
                    dict(sample_id=sid, relative_residual=residual, independent_weak_residual=weak)
                )
                if group < p["refinement_first_groups"] and forcing == 0:
                    comparison = compare_p1(y, 128, high.solve(field))
                    require(
                        comparison["relative_l2"] < p["relative_reference_l2_gate"]
                        and comparison["relative_h1_seminorm"] < p["relative_reference_h1_gate"],
                        "Refinement gate",
                    )
                    refinements.append(dict(sample_id=sid, **comparison))
            if group % 16 == 0:
                print("TEST_DATA", kind, group, groups, flush=True)
        np.savez_compressed(out / (kind + ".npz"), **{k: np.asarray(v) for k, v in values.items()})
        np.savez_compressed(
            out / (kind + "_reference.npz"), **{k: np.asarray(v) for k, v in raw.items()}
        )
        loaded = array_file(out / (kind + ".npz"))
        refs = array_file(out / (kind + "_reference.npz"))
        current = validate_fields(loaded, groups, nf, f"final-v1:{kind}:test:")
        disjoint(forbidden, current)
        forbidden.update(current)
        require(
            np.array_equal(loaded["y"], refs["y_reference"][:, ::2, ::2])
            and np.array_equal(loaded["sample_id"], refs["sample_id"]),
            "Serialized labels/IDs mismatch",
        )
        for j in range(len(loaded["y"])):
            field = CFField(refs["c_coefficients"][j], refs["f_coefficients"][j], base["reaction"])
            require(
                np.array_equal(loaded["c"][j].ravel(), field.reaction(observation.mesh.nodes))
                and np.array_equal(loaded["f"][j].ravel(), field.source(observation.mesh.nodes)),
                "Serialized fields mismatch",
            )
        if kind == "fixed":
            require(np.array_equal(loaded["f"], np.ones_like(loaded["f"])), "Fixed forcing drift")
    dump(
        out / "quality.json",
        dict(
            passed=True,
            samples=len(quality),
            records=quality,
            refinement=refinements,
            disjoint_from_all_development=True,
        ),
    )
    certify(out)
    return out
