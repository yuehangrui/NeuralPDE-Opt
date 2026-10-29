"""POD experiment queue; isolated from the frozen Chapter 3 code and artifacts."""

import argparse
import csv
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from pod import fit_pod, coordinates, reconstruct, prepare_arrays
from train import sha, dump, train_one, finish_one, validate_completed
from neuralpde_opt.chapter3.reaction_data import verify_dataset
from neuralpde_opt.chapter3.reaction_training import public_data_only
from neuralpde_opt.chapter3.reaction_operators import fit_reaction_normalization
from neuralpde_opt.chapter3.operator_training import P1Geometry, evaluate_arrays, exclusive_run_lock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def validate_config(c):
    if c.get("schema_version") != "ch03-pod-v1" or c.get("test_evaluations") != 0:
        raise ValueError("Unsupported experiment schema/test policy")
    if (
        c.get("center_output") is not True
        or c.get("device") != "cpu"
        or c.get("activation") != "GELU"
    ):
        raise ValueError("This protocol requires centered POD, CPU and GELU")
    if (
        c.get("objective") != "mean_per_sample_relative_p1_l2_squared"
        or c.get("selection") != "earliest_min_val_mean_relative_p1_l2"
    ):
        raise ValueError("Loss or selection protocol changed")
    for key in ["ranks", "seeds"]:
        if not c[key] or len(set(c[key])) != len(c[key]) or any(type(v) is not int for v in c[key]):
            raise ValueError("Invalid " + key)
    if c["ranks"] != sorted(c["ranks"]) or min(c["ranks"]) < 1 or min(c["seeds"]) < 0:
        raise ValueError("Ranks must be increasing and seeds nonnegative")
    for key in [
        "epochs",
        "batch_size",
        "checkpoint_every",
        "cpu_threads",
        "width",
        "depth",
        "sensor_stride",
    ]:
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError("Invalid " + key)
    for key in ["learning_rate", "min_lr", "weight_decay", "grad_clip"]:
        if not np.isfinite(c[key]) or c[key] < 0:
            raise ValueError("Invalid " + key)
    if c["learning_rate"] <= 0 or c["min_lr"] > c["learning_rate"] or c["grad_clip"] <= 0:
        raise ValueError("Invalid optimizer scales")
    return c


def source_identity():
    return {p.name: sha(p) for p in sorted(HERE.glob("*.py"))}


def prepare(out, config, data, geometry, opened, data_dir, identity):
    basis = fit_pod(data["train"], geometry, max(config["ranks"]))
    np.savez_compressed(out / "basis.npz", **basis)
    norm = fit_reaction_normalization(data["train"], geometry)
    dump(out / "normalization.json", norm)
    projection = {
        "fit_split": "train",
        "fit_sample_ids": list(map(str, basis["fit_sample_ids"])),
        "centered": True,
        "mass_orthogonality_max_defect": float(basis["orthogonality_defect"]),
        "total_centered_train_energy": float(sum(basis["eigenvalues"])),
        "eigenvalues": basis["eigenvalues"].tolist(),
        "ranks": {},
        "mean_baseline": {},
    }
    for split, d in data.items():
        mean_field = np.broadcast_to(basis["mean"].reshape(geometry.grid_shape), d["y"].shape)
        projection["mean_baseline"][split] = evaluate_arrays(
            mean_field, d["y"], geometry, d["sample_id"]
        )
    for rank in config["ranks"]:
        entry = {
            "centered_train_energy_fraction": float(
                sum(basis["eigenvalues"][:rank]) / sum(basis["eigenvalues"])
            )
        }
        for split, d in data.items():
            alpha = coordinates(d["y"], basis, geometry, rank)
            projected = reconstruct(alpha, basis, rank)
            entry[split] = evaluate_arrays(projected, d["y"], geometry, d["sample_id"])
            if np.max(abs(projected.reshape(len(projected), -1)[:, geometry.boundary])) > 1e-12:
                raise RuntimeError("Projection violated boundary")
        projection["ranks"][str(rank)] = entry
    dump(out / "projection_report.json", projection)
    dump(out / "config.json", config)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for p in sorted(HERE.glob("*")):
        if p.is_file():
            shutil.copyfile(p, snapshot / p.name)
    meta = {
        "identity": identity,
        "dataset": str(data_dir),
        "opened_dataset_arrays": sorted(set(opened)),
        "epsilon": json.loads((data_dir / "config.json").read_text())["epsilon"],
        "prepared_files": {
            n: sha(out / n)
            for n in ["basis.npz", "normalization.json", "projection_report.json", "config.json"]
        },
        "train_samples": len(data["train"]["y"]),
        "val_samples": len(data["val"]["y"]),
        "test_evaluations": 0,
    }
    dump(out / "preparation.json", meta)
    return basis, norm, projection, meta


def load_prepared(out, identity):
    meta = json.loads((out / "preparation.json").read_text())
    if meta["identity"] != identity:
        raise ValueError("Source/config/data/environment identity changed")
    for name, digest in meta["prepared_files"].items():
        if sha(out / name) != digest:
            raise ValueError("Preparation artifact changed: " + name)
    with np.load(out / "basis.npz", allow_pickle=False) as z:
        basis = {k: z[k].copy() for k in z.files}
    return (
        basis,
        json.loads((out / "normalization.json").read_text()),
        json.loads((out / "projection_report.json").read_text()),
        meta,
    )


def summarize(out, config, rows, projection):
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    aggregate = {}
    for rank in config["ranks"]:
        group = [r for r in rows if r["rank"] == rank]
        if group:
            aggregate[str(rank)] = {
                key: {
                    "mean": float(np.mean([r[key] for r in group])),
                    "std_population": float(np.std([r[key] for r in group])),
                }
                for key in ["train_l2", "val_l2", "val_h1", "coefficient_val_l2", "elapsed_seconds"]
            }
    dump(
        out / "aggregate.json",
        {"completed_runs": len(rows), "ranks": aggregate, "rows": rows, "test_evaluations": 0},
    )
    text = "# POD/PCA + 系数网络：反应算例\n\n"
    text += f'已完成{len(rows)}/{len(config["ranks"])*len(config["seeds"])}组。train128/val32；训练基与均值仅由train拟合。无test评估。\n\n'
    text += "| r | train投影L2 | val投影L2 | 网络train L2 | 网络val L2（均值±总体std） | 网络val H1 |\n|---|---:|---:|---:|---:|---:|\n"
    for rank in config["ranks"]:
        p = projection["ranks"][str(rank)]
        a = aggregate.get(str(rank))
        if a:
            text += f'| {rank} | {100*p["train"]["mean"]["relative_l2"]:.4f}% | {100*p["val"]["mean"]["relative_l2"]:.4f}% | {100*a["train_l2"]["mean"]:.4f}% | {100*a["val_l2"]["mean"]:.4f}% ± {100*a["val_l2"]["std_population"]:.4f}% | {100*a["val_h1"]["mean"]:.4f}% |\n'
    text += "\n投影误差仅是当前固定仿射输出空间在公共网格L2下的误差下限；不等于连续真误差下界。标准POD最小化绝对L2训练投影误差；网络使用逐样本相对L2平方，优化时省略与参数无关的投影常数。\n"
    text += "\n逐样本L2误差平方分解经过独立核验，不能把表中的平均百分比直接相加；H1没有同样的正交分解。验证集参与选模，维数比较属于开发实验，不能称独立测试或架构家族排名。\n"
    text += f'\n预算：每组{config["epochs"]}epoch、batch{config["batch_size"]}，3×{config["width"]} GELU MLP、17×17输入传感器、AdamW+cosine。与冻结四架构匹配样本访问预算，未匹配参数量/计算量。所有配置在训练前固定。\n'
    (out / "summary.md").write_text(text)
    return aggregate


def plots(out, config, rows, projection, basis, data, geometry):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = config["ranks"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    eig = np.asarray(projection["eigenvalues"])
    positive = eig[eig > eig.max() * 1e-14]
    ax[0].semilogy(np.arange(1, len(positive) + 1), positive / eig.sum(), "o-", ms=3)
    ax[0].set(
        xlabel="Mode",
        ylabel="Fraction of centered training energy",
        title="Mass-weighted POD spectrum",
    )
    for split, marker in [("train", "o"), ("val", "s")]:
        ax[1].semilogy(
            ranks,
            [100 * projection["ranks"][str(r)][split]["mean"]["relative_l2"] for r in ranks],
            marker + "-",
            label=split + " projection",
        )
    for split, marker in [("train", "^"), ("val", "d")]:
        values = [[100 * x[split + "_l2"] for x in rows if x["rank"] == r] for r in ranks]
        ax[1].errorbar(
            ranks,
            [np.mean(v) for v in values],
            yerr=[np.std(v) for v in values],
            marker=marker,
            capsize=3,
            label=split + " network",
        )
    ax[1].set(
        xlabel="POD rank",
        ylabel="Mean relative P1 L2 (%)",
        title="Representation and learned error",
    )
    ax[1].legend(fontsize=8)
    fig.savefig(out / "pod_errors.png", dpi=170)
    fig.savefig(out / "pod_errors.svg")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, rank in zip(axes.ravel(), ranks):
        for seed in config["seeds"]:
            history = json.loads((out / f"rank_{rank}/seed_{seed}/history.json").read_text())
            ax.semilogy(
                [v["epoch"] for v in history],
                [v["val_l2"] for v in history],
                label=f"val seed {seed}",
            )
        ax.axhline(
            projection["ranks"][str(rank)]["val"]["mean"]["relative_l2"],
            color="black",
            linestyle="--",
            label="val projection",
        )
        ax.set(title=f"Rank {rank}", xlabel="Epoch", ylabel="Relative P1 L2")
        ax.legend(fontsize=7)
    fig.savefig(out / "pod_training.png", dpi=170)
    fig.savefig(out / "pod_training.svg")
    plt.close(fig)
    rank = max(ranks)
    seed = config["seeds"][0]
    with np.load(out / f"rank_{rank}/seed_{seed}/val_predictions.npz", allow_pickle=False) as z:
        prediction = z["prediction"][0]
    alpha = coordinates(data["val"]["y"][:1], basis, geometry, rank)
    projection_field = reconstruct(alpha, basis, rank)[0]
    target = data["val"]["y"][0]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    for ax, field, title in zip(
        axes,
        [target, projection_field, prediction, prediction - target],
        ["Reference label", "POD projection", "Coefficient network", "Signed error"],
    ):
        kw = (
            {"vmin": 0, "vmax": float(target.max()), "cmap": "viridis"}
            if title != "Signed error"
            else {
                "cmap": "RdBu_r",
                "vmin": -float(abs(prediction - target).max()),
                "vmax": float(abs(prediction - target).max()),
            }
        )
        im = ax.imshow(field.T, origin="lower", extent=[0, 1, 0, 1], **kw)
        ax.set(title=title, xlabel="x", ylabel="y")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Fixed validation sample 0; rank {rank}, seed {seed}")
    fig.savefig(out / "pod_example.png", dpi=170)
    fig.savefig(out / "pod_example.svg")
    plt.close(fig)


def verify_root(out):
    complete = json.loads((out / "POD_COMPLETE").read_text())
    if complete["manifest_sha256"] != sha(out / "manifest.json"):
        raise ValueError("Root manifest changed")
    for name, digest in json.loads((out / "manifest.json").read_text()).items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(out / name) != digest:
            raise ValueError("Frozen result changed: " + name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    out = args.output.resolve()
    data_dir = args.data.resolve()
    config = validate_config(json.loads(args.config.read_text()))
    baseline = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/verify_frozen_release.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if baseline.returncode:
        raise RuntimeError("Frozen baseline failed verification: " + baseline.stdout)
    identity = {
        "sources": source_identity(),
        "config": config,
        "dataset_manifest_sha256": sha(data_dir / "manifest.json"),
        "frozen_release_sha256": sha(ROOT / "releases/ch03-reaction-v1.0-20261022/release.json"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
    }
    if args.resume:
        if not out.is_dir():
            raise ValueError("No experiment to resume")
        if (out / "POD_COMPLETE").exists():
            if json.loads((out / "preparation.json").read_text())["identity"] != identity:
                raise ValueError("Completed experiment identity changed")
            verify_root(out)
            print("VERIFIED_COMPLETE_NO_TRAINING")
            return
    else:
        out.mkdir(parents=True, exist_ok=False)
    with exclusive_run_lock(out):
        with public_data_only(data_dir) as opened:
            data = verify_dataset(data_dir)
            geometry = P1Geometry(data_dir / "observation_geometry.npz")
        if args.resume:
            basis, norm, projection, meta = load_prepared(out, identity)
        else:
            basis, norm, projection, meta = prepare(
                out, config, data, geometry, opened, data_dir, identity
            )
        if args.prepare_only:
            dump(out / "status.json", {"status": "prepared", "test_evaluations": 0})
            print(
                "POD_PREPARED",
                json.dumps(
                    {r: p["val"]["mean"]["relative_l2"] for r, p in projection["ranks"].items()}
                ),
            )
            return
        rows = []
        run_identity = {
            **identity,
            "basis_sha256": sha(out / "basis.npz"),
            "normalization_sha256": sha(out / "normalization.json"),
        }
        for rank in config["ranks"]:
            arrays = prepare_arrays(data, basis, geometry, norm, rank, config["sensor_stride"])
            for seed in config["seeds"]:
                run = out / f"rank_{rank}/seed_{seed}"
                dump(
                    out / "status.json",
                    {
                        "status": "running",
                        "rank": rank,
                        "seed": seed,
                        "completed_runs": len(rows),
                        "total_runs": len(config["ranks"]) * len(config["seeds"]),
                    },
                )
                if (run / "COMPLETE").exists():
                    if not args.resume:
                        raise ValueError("Run already exists")
                    result = validate_completed(run)
                    if json.loads((run / "identity.json").read_text()) != {
                        **run_identity,
                        "config": config,
                        "rank": rank,
                        "seed": seed,
                    }:
                        raise ValueError("Completed run identity mismatch")
                else:
                    state = train_one(
                        run,
                        config,
                        arrays,
                        np.sqrt(basis["eigenvalues"][:rank]),
                        rank,
                        seed,
                        run_identity,
                        resume=run.exists() and args.resume,
                    )
                    result = finish_one(run, state, config, arrays, basis, rank, data, geometry)
                rows.append(
                    {
                        "rank": rank,
                        "seed": seed,
                        "best_epoch": result["best_epoch"],
                        "epochs": result["epochs"],
                        "optimizer_steps": result["optimizer_steps"],
                        "parameter_count": result["parameter_count"],
                        "train_l2": result["best"]["train"]["mean"]["relative_l2"],
                        "val_l2": result["best"]["val"]["mean"]["relative_l2"],
                        "val_h1": result["best"]["val"]["mean"]["relative_h1_seminorm"],
                        "coefficient_val_l2": result["best"]["val"]["coefficient_mean_relative_l2"],
                        "elapsed_seconds": result["elapsed_seconds"],
                    }
                )
                summarize(out, config, rows, projection)
        plots(out, config, rows, projection, basis, data, geometry)
        if (
            source_identity() != identity["sources"]
            or sha(data_dir / "manifest.json") != identity["dataset_manifest_sha256"]
        ):
            raise RuntimeError("Identity drift during experiment")
        dump(
            out / "verification.json",
            {
                "passed": True,
                "completed_runs": len(rows),
                "projection_fit_split": "train",
                "normalization_fit_split": "train",
                "mass_orthogonality_max_defect": float(basis["orthogonality_defect"]),
                "opened_dataset_arrays": meta["opened_dataset_arrays"],
                "test_evaluations": 0,
                "all_runs_individually_audited": True,
                "frozen_baseline_check": json.loads(baseline.stdout),
            },
        )
        dump(
            out / "status.json",
            {"status": "complete", "completed_runs": len(rows), "test_evaluations": 0},
        )
        dump(
            out / "manifest.json",
            {
                p.relative_to(out).as_posix(): sha(p)
                for p in out.rglob("*")
                if p.is_file()
                and p != out / "manifest.json"
                and p.name not in {".run.lock", "POD_COMPLETE"}
            },
        )
        dump(
            out / "POD_COMPLETE",
            {"schema_version": "ch03-pod-v1", "manifest_sha256": sha(out / "manifest.json")},
        )
        verify_root(out)
        print("ALL_POD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
