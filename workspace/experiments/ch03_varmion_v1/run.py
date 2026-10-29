import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, csv, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.append(str(ROOT / "experiments/ch03_physics_v1"))
from physics import P1System
from data import generate, load
from model import fit_uncentered_pod, model_for, structure_checks
from common import sha, dump, certify, verify
from train_coeff import fit, evaluate, tensor_data
from neuralpde_opt.chapter3.operator_training import P1Geometry, evaluate_arrays, exclusive_run_lock


def identities():
    return {p.name: sha(p) for p in HERE.glob("*.py")}


def prepare(out, data, geometry, config):
    rank = config["basis_rank"]
    basis, values = fit_uncentered_pod(data["train"]["y"], geometry.mass, rank)
    c = data["train"]["c"]
    mean = float(np.mean(c.reshape(len(c), -1) @ geometry.weights))
    std = float(np.sqrt(np.mean(geometry.l2_squared(c - mean))))
    norm = {
        "c_mean": mean,
        "c_std": std,
        "fit_sample_ids": list(map(str, data["train"]["sample_id"])),
        "fit_split": "train",
        "source_centered": False,
        "output_centered": False,
    }
    system = P1System(config["observation_n"], config["epsilon"])
    a, _ = system.assemble(np.ones(geometry.grid_shape))
    k0 = np.linalg.inv(basis.T @ (a @ basis))
    np.savez_compressed(
        out / "basis.npz",
        basis=basis,
        eigenvalues=values,
        k0=k0,
        fit_sample_ids=data["train"]["sample_id"],
    )
    dump(out / "normalization.json", norm)
    reports = {}
    arrays = {}
    cache = {}
    for split, d in data.items():
        alpha = d["y"].reshape(len(d["y"]), -1) @ (geometry.mass @ basis)
        z = d["f"].reshape(len(d["f"]), -1) @ (geometry.mass @ basis)
        projection = (alpha @ basis.T).reshape(d["y"].shape)
        galerkin = []
        for j, group in enumerate(d["coefficient_group"]):
            if str(group) not in cache:
                a, _ = system.assemble(d["c"][j])
                cache[str(group)] = basis.T @ (a @ basis)
            galerkin.append(basis @ np.linalg.solve(cache[str(group)], z[j]))
        galerkin = np.array(galerkin).reshape(d["y"].shape)
        reports[split] = {
            "projection": evaluate_arrays(projection, d["y"], geometry, d["sample_id"]),
            "reduced_galerkin": evaluate_arrays(galerkin, d["y"], geometry, d["sample_id"]),
        }
        arrays[split] = {
            "c": d["c"],
            "z": z,
            "alpha": alpha,
            "projection_squared": geometry.l2_squared(projection - d["y"]),
            "denominator": geometry.l2_squared(d["y"]),
        }
        np.savez_compressed(out / (split + "_coordinates.npz"), **arrays[split])
    dump(out / "representation.json", reports)
    return basis, k0, norm, arrays


def finish(path, state, name, config, data, geometry, basis, k0, norm, arrays, device):
    saved = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
    if saved["identity"] != state["identity"] or saved["best_epoch"] != state["best_epoch"]:
        raise RuntimeError("Saved identity mismatch")
    model = model_for(name, norm, k0).to(device)
    model.load_state_dict(saved["model"])
    reference = model_for(name, norm, k0).to(device)
    reference.load_state_dict(state["best_model"])
    tensors = tensor_data(arrays, device)
    system = P1System(config["observation_n"], config["epsilon"])
    metrics = {}
    checks = {}
    for split, d in data.items():
        alpha, score = evaluate(model, tensors[split])
        again, _ = evaluate(reference, tensors[split])
        if not np.array_equal(alpha, again):
            raise RuntimeError("Checkpoint reload changed predictions")
        prediction = (alpha @ basis.T).reshape(d["y"].shape)
        metric = evaluate_arrays(prediction, d["y"], geometry, d["sample_id"])
        error = geometry.l2_squared(prediction - d["y"])
        coefficient = ((alpha - arrays[split]["alpha"]) ** 2).sum(1)
        defect = float(np.max(abs(error - arrays[split]["projection_squared"] - coefficient)))
        if (
            defect > 1e-10
            or abs(score - metric["mean"]["relative_l2"]) > 1e-10
            or metric["max"]["boundary_max"] > 1e-12
        ):
            raise RuntimeError("Projection/metric audit failed")
        local = (prediction - d["y"]).reshape(len(prediction), -1)[:, geometry.elements]
        local_y = d["y"].reshape(len(prediction), -1)[:, geometry.elements]

        def norm2(v):
            return ((v.sum(-1) ** 2 + (v * v).sum(-1)) * geometry.areas / 12).sum(1)

        independent = np.sqrt(norm2(local) / norm2(local_y))
        np.testing.assert_allclose(
            independent, [v["relative_l2"] for v in metric["per_sample"]], atol=1e-12
        )
        metric["physics"] = system.diagnostics(d["c"], prediction, d["f"])
        metrics[split] = metric
        checks[split] = {
            "reload_identical": True,
            "pythagoras_squared_defect": defect,
            "independent_l2_verified": True,
        }
        np.savez_compressed(
            path / (split + "_predictions.npz"),
            prediction=prediction,
            coefficients=alpha,
            sample_id=d["sample_id"],
        )
    d = tensors["val"]
    linear = structure_checks(model, d["c"], d["z"], d["z"].roll(1, 0))
    if name == "varmion" and (
        linear["zero_load_max"] != 0
        or linear["relative_superposition_max"] > 2e-6
        or linear["relative_homogeneity_max"] > 2e-6
    ):
        raise RuntimeError("Source linearity failed")
    result = {
        "model": name,
        "seed": state["identity"]["seed"],
        "epochs": state["epoch"],
        "optimizer_steps": state["steps"],
        "sample_visits": state["visits"],
        "best_epoch": state["best_epoch"],
        "parameter_count": state["parameter_count"],
        "elapsed_seconds": state["elapsed_seconds"],
        "metrics": metrics,
        "structure": linear,
        "test_evaluations": 0,
    }
    dump(path / "metrics.json", result)
    dump(path / "verification.json", {"passed": True, "checks": checks, "structure": linear})
    dump(path / "status.json", {"status": "complete"})
    certify(path, "COMPLETE")
    print(
        "VARMION_RUN_COMPLETE",
        name,
        state["identity"]["seed"],
        metrics["val"]["mean"]["relative_l2"],
        flush=True,
    )
    return result


def summarize(out, rows, reps):
    groups = {}
    for name in ["varmion", "unstructured"]:
        group = [r for r in rows if r["model"] == name]
        if group:
            groups[name] = {
                key: {
                    "mean": float(np.mean([r[key] for r in group])),
                    "std_population": float(np.std([r[key] for r in group])),
                }
                for key in ["train_l2", "val_l2", "val_h1", "val_residual", "elapsed_seconds"]
            }
    dump(
        out / "aggregate.json",
        {"groups": groups, "rows": rows, "completed_runs": len(rows), "test_evaluations": 0},
    )
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [
        "# VarMiON结构化对照",
        "",
        f"完成{len(rows)}/6组。train256/val64按反应系数组划分；每组1600epoch。",
        "",
        f'验证POD投影L2：{100*reps["val"]["projection"]["mean"]["relative_l2"]:.4f}%；精确约化Galerkin L2：{100*reps["val"]["reduced_galerkin"]["mean"]["relative_l2"]:.4f}%。',
        "",
        "| 实现 | train L2 | val L2，均值±总体std | val H1半范 |",
        "|---|---:|---:|---:|",
    ]
    for name, g in groups.items():
        lines.append(
            f'| {name} | {100*g["train_l2"]["mean"]:.4f}% | {100*g["val_l2"]["mean"]:.4f}% ± {100*g["val_l2"]["std_population"]:.4f}% | {100*g["val_h1"]["mean"]:.4f}% |'
        )
    lines += [
        "",
        "VarMiON为书稿公式的固定基底结构化实现：Phi K(c) Phi^T M f。两种学习实现共享物理初值K0；参数量近似但未严格匹配。仅train构造非中心化POD与输入归一化。val参与选模，test未生成。",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/ch03/varmion_v1")
    parser.add_argument("--data", type=Path, default=ROOT / "data/ch03/varmion_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    dataset = args.data.resolve()
    config = json.loads((HERE / "config.json").read_text())
    if (
        not (ROOT / "results/ch03/neural_cg_v1/CG_COMPLETE").is_file()
        or not json.loads((ROOT / "results/ch03/neural_cg_v1/verification.json").read_text())[
            "passed"
        ]
    ):
        raise RuntimeError("Step 3 must finish before step 4")
    frozen = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/verify_frozen_release.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if frozen.returncode:
        raise RuntimeError("Frozen baseline changed")
    if config["test_evaluations"] != 0 or config["schema_version"] != "ch03-varmion-v1":
        raise ValueError("Protocol")
    if not dataset.exists():
        generate(config, dataset)
    elif json.loads((dataset / "config.json").read_text()) != config:
        raise ValueError("Dataset config mismatch")
    data = load(dataset)
    geometry = P1Geometry(dataset / "observation_geometry.npz")
    identity = {
        "sources": identities(),
        "data_manifest": sha(dataset / "manifest.json"),
        "config": config,
        "p1_physics": sha(ROOT / "experiments/ch03_physics_v1/physics.py"),
        "torch": str(torch.__version__),
    }
    if args.resume:
        if json.loads((out / "identity.json").read_text()) != identity:
            raise ValueError("Source/data/config changed")
        if (out / "VARMION_COMPLETE").exists():
            verify(out, "VARMION_COMPLETE")
            print("VERIFIED_COMPLETE_NO_TRAINING")
            return
    else:
        out.mkdir(parents=True, exist_ok=False)
        dump(out / "identity.json", identity)
        dump(out / "config.json", config)
    with exclusive_run_lock(out):
        if not args.resume:
            basis, k0, norm, arrays = prepare(out, data, geometry, config)
            shutil.copytree(
                HERE, out / "source_snapshot", ignore=shutil.ignore_patterns("__pycache__")
            )
            dump(
                out / "prepared_manifest.json",
                {
                    name: sha(out / name)
                    for name in [
                        "basis.npz",
                        "normalization.json",
                        "train_coordinates.npz",
                        "val_coordinates.npz",
                        "representation.json",
                    ]
                },
            )
        else:
            for name, h in json.loads((out / "prepared_manifest.json").read_text()).items():
                if sha(out / name) != h:
                    raise ValueError("Prepared artifact changed")
            with np.load(out / "basis.npz", allow_pickle=False) as z:
                basis = z["basis"].copy()
                k0 = z["k0"].copy()
            norm = json.loads((out / "normalization.json").read_text())
            arrays = {}
            for split in ["train", "val"]:
                with np.load(out / (split + "_coordinates.npz"), allow_pickle=False) as z:
                    arrays[split] = {k: z[k].copy() for k in z.files}
        rows = []
        reps = json.loads((out / "representation.json").read_text())
        for name in config["models"]:
            for seed in config["seeds"]:
                path = out / name / f"seed_{seed}"
                dump(
                    out / "status.json",
                    {"status": "running", "model": name, "seed": seed, "completed_runs": len(rows)},
                )
                if (path / "COMPLETE").is_file():
                    verify(path, "COMPLETE")
                    result = json.loads((path / "metrics.json").read_text())
                    if json.loads((path / "identity.json").read_text()) != {
                        **identity,
                        "model": name,
                        "basis": sha(out / "basis.npz"),
                        "config": config,
                        "seed": seed,
                        "device": "cuda",
                    }:
                        raise ValueError("Completed run identity changed")
                else:
                    state = fit(
                        path,
                        config,
                        seed,
                        lambda: model_for(name, norm, k0),
                        arrays,
                        {**identity, "model": name, "basis": sha(out / "basis.npz")},
                        resume=args.resume and path.exists(),
                    )
                    result = finish(
                        path, state, name, config, data, geometry, basis, k0, norm, arrays, "cuda"
                    )
                m = result["metrics"]
                rows.append(
                    {
                        "model": name,
                        "seed": seed,
                        "parameter_count": result["parameter_count"],
                        "best_epoch": result["best_epoch"],
                        "train_l2": m["train"]["mean"]["relative_l2"],
                        "val_l2": m["val"]["mean"]["relative_l2"],
                        "val_h1": m["val"]["mean"]["relative_h1_seminorm"],
                        "val_residual": float(
                            np.mean(
                                [v["relative_lumped_dual_residual"] for v in m["val"]["physics"]]
                            )
                        ),
                        "superposition": result["structure"]["relative_superposition_max"],
                        "zero_load_max": result["structure"]["zero_load_max"],
                        "elapsed_seconds": result["elapsed_seconds"],
                    }
                )
                summarize(out, rows, reps)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for name in config["models"]:
            for seed in config["seeds"]:
                h = json.loads((out / name / f"seed_{seed}/history.json").read_text())
                axes[0].semilogy(
                    [v["epoch"] for v in h], [v["val_l2"] for v in h], label=f"{name} {seed}"
                )
        axes[0].axhline(
            reps["val"]["projection"]["mean"]["relative_l2"],
            color="black",
            ls="--",
            label="POD projection",
        )
        axes[0].set(xlabel="Epoch", ylabel="Validation P1 L2")
        axes[0].legend(fontsize=7)
        names = config["models"]
        axes[1].bar(
            names, [np.mean([r["superposition"] for r in rows if r["model"] == n]) for n in names]
        )
        axes[1].set(yscale="log", ylabel="Relative superposition defect")
        fig.savefig(out / "varmion_comparison.png", dpi=170)
        fig.savefig(out / "varmion_comparison.svg")
        plt.close(fig)
        if identities() != identity["sources"]:
            raise RuntimeError("Source drift")
        dump(
            out / "verification.json",
            {
                "passed": True,
                "runs": len(rows),
                "train_only_basis": True,
                "all_runs_reloaded_and_audited": True,
                "test_evaluations": 0,
                "frozen_baseline": json.loads(frozen.stdout),
            },
        )
        dump(out / "status.json", {"status": "complete", "runs": len(rows)})
        certify(out, "VARMION_COMPLETE")
        verify(out, "VARMION_COMPLETE")
        print("ALL_VARMION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
