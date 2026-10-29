"""Predeclared train/validation study of FNO data, energy and mixed objectives."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from physics import P1System, TorchP1, objective
from neuralpde_opt.chapter3.operator_training import (
    P1Geometry,
    evaluate_arrays,
    exclusive_run_lock,
    capture_rng,
    restore_rng,
    _cpu_copy,
)
from neuralpde_opt.chapter3.reaction_training import public_data_only, seed_everything
from neuralpde_opt.chapter3.reaction_operators import make_reaction_model
from neuralpde_opt.chapter3.reaction_data import verify_dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def sources():
    return {p.name: sha(p) for p in sorted(HERE.glob("*.py"))}


def certify(path, marker="COMPLETE"):
    path = Path(path)
    dump(
        path / "manifest.json",
        {
            p.relative_to(path).as_posix(): sha(p)
            for p in path.rglob("*")
            if p.is_file() and p != path / "manifest.json" and p.name not in {marker, ".run.lock"}
        },
    )
    dump(
        path / marker,
        {"schema_version": "ch03-physics-v1", "manifest_sha256": sha(path / "manifest.json")},
    )


def verify(path, marker="COMPLETE"):
    path = Path(path)
    cert = json.loads((path / marker).read_text())
    if cert["manifest_sha256"] != sha(path / "manifest.json"):
        raise ValueError("Changed manifest")
    for name, h in json.loads((path / "manifest.json").read_text()).items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(path / name) != h:
            raise ValueError("Changed artifact: " + name)


def validate_config(c):
    if (
        c["schema_version"] != "ch03-physics-v1"
        or c["test_evaluations"] != 0
        or c["selection"] != "earliest_min_val_mean_relative_p1_l2"
    ):
        raise ValueError("Unsupported protocol")
    for key in [
        "epochs",
        "batch_size",
        "eval_batch_size",
        "cpu_threads",
        "checkpoint_every",
        "train_eval_every",
    ]:
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(key)
    if (
        not c["seeds"]
        or any(type(s) is not int or s < 0 for s in c["seeds"])
        or len(set(c["seeds"])) != len(c["seeds"])
    ):
        raise ValueError("Invalid seeds")
    if c["output_scale"] != 1.0:
        raise ValueError("Fixed physics output scale required")
    if (
        not 0 < c["learning_rate"]
        or not 0 <= c["min_lr"] <= c["learning_rate"]
        or c["grad_clip"] <= 0
    ):
        raise ValueError("Invalid optimization")
    for key in ["learning_rate", "min_lr", "weight_decay", "grad_clip"]:
        if not np.isfinite(c[key]) or c[key] < 0:
            raise ValueError(key)
    names = []
    for case in c["cases"]:
        names.append(case["name"])
        if (
            type(case["labels"]) is not int
            or case["labels"] < 0
            or not np.isfinite(case["physics_weight"])
            or case["physics_weight"] < 0
        ):
            raise ValueError("Invalid case")
        if case["labels"] == 0 and case["physics_weight"] == 0:
            raise ValueError("Empty objective")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate cases")
    return c


def normalize_inputs(data, geometry):
    c = data["train"]["c"]
    mean = float(np.mean(c.reshape(len(c), -1) @ geometry.weights))
    std = float(np.sqrt(np.mean(geometry.l2_squared(c - mean))))
    if std <= 1e-12:
        raise ValueError("Degenerate c scale")
    return {
        "schema_version": "ch03-reaction-normalization-v1",
        "fit_split": "train_inputs_only",
        "fit_sample_ids": list(map(str, data["train"]["sample_id"])),
        "c_mean": mean,
        "c_std": std,
        "y_rms": 1.0,
        "output_scale_source": "fixed dimensionless value 1; no training y labels",
        "second_channel": "fixed forcing encoded as zeros",
    }


def label_mask(count, labels, seed):
    if not 0 <= labels <= count:
        raise ValueError("Invalid label count")
    mask = np.zeros(count, dtype=bool)
    mask[np.random.default_rng(seed).permutation(count)[:labels]] = True
    return mask


def predict(model, c, device, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(c), batch_size):
            y_prediction = model(
                torch.tensor(c[start : start + batch_size], device=device, dtype=torch.float32)
            )
            if not torch.isfinite(y_prediction).all():
                raise RuntimeError("Nonfinite prediction")
            out.append(y_prediction.cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def score(prediction, target, geometry):
    return float(
        np.sqrt(geometry.l2_squared(prediction - target) / geometry.l2_squared(target)).mean()
    )


def train_one(
    path,
    config,
    case,
    seed,
    data,
    geometry,
    system,
    norm,
    identity,
    device="cuda",
    resume=False,
    stop_after=None,
):
    path = Path(path)
    run_identity = {**identity, "config": config, "case": case, "seed": seed, "device": device}
    if resume:
        if json.loads((path / "identity.json").read_text()) != run_identity:
            raise ValueError("Changed run identity")
    else:
        path.mkdir(parents=True, exist_ok=False)
        dump(path / "identity.json", run_identity)
    seed_everything(seed, config["cpu_threads"])
    rng = np.random.default_rng(seed)
    model = make_reaction_model(config["model"], norm, system.epsilon).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config["epochs"], eta_min=config["min_lr"]
    )
    mask = label_mask(len(data["train"]["c"]), case["labels"], config["subset_seed"])
    state = {
        "identity": run_identity,
        "normalization": norm,
        "epoch": 0,
        "optimizer_steps": 0,
        "input_visits": 0,
        "supervised_label_visits": 0,
        "physics_input_visits": 0,
        "best_epoch": 0,
        "best_score": float("inf"),
        "best_model": None,
        "history": [],
        "elapsed_seconds": 0.0,
        "label_ids": list(map(str, data["train"]["sample_id"][mask])),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    if resume:
        state = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        if state["identity"] != run_identity:
            raise ValueError("Changed checkpoint identity")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"], rng)
    c = torch.tensor(data["train"]["c"], device=device, dtype=torch.float32)
    target = (
        torch.tensor(data["train"]["y"], device=device, dtype=torch.float32)
        if case["labels"]
        else None
    )
    selected = torch.tensor(mask, device=device)
    kernel = TorchP1(system, device)
    local = kernel.local(data["train"]["c"])
    before = state["elapsed_seconds"]
    start = time.perf_counter()

    def checkpoint():
        state.update(
            model=_cpu_copy(model.state_dict()),
            optimizer=_cpu_copy(opt.state_dict()),
            scheduler=scheduler.state_dict(),
            rng=capture_rng(rng),
        )
        save(path / "last.pt", state)
        if state["best_model"] is not None:
            save(
                path / "best.pt",
                {
                    "model": state["best_model"],
                    "identity": run_identity,
                    "normalization": norm,
                    "best_epoch": state["best_epoch"],
                    "best_score": state["best_score"],
                },
            )
        dump(path / "history.json", state["history"])

    if not resume:
        checkpoint()
    end = config["epochs"] if stop_after is None else stop_after
    if not state["epoch"] <= end <= config["epochs"]:
        raise ValueError("Invalid stopping epoch")
    for epoch in range(state["epoch"] + 1, end + 1):
        model.train()
        order = rng.permutation(len(c))
        loss_mean = 0.0
        energy_mean = 0.0
        data_mean = 0.0
        for begin in range(0, len(c), config["batch_size"]):
            indices = order[begin : begin + config["batch_size"]]
            ix = torch.tensor(indices, device=device)
            opt.zero_grad(set_to_none=True)
            pred = model(c[ix])
            loss, supervised, energy = objective(
                pred,
                None if target is None else target[ix],
                selected[ix],
                local[ix],
                kernel,
                case["physics_weight"],
                case["labels"],
                len(c),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["grad_clip"], error_if_nonfinite=True
            )
            opt.step()
            w = len(ix) / len(c)
            loss_mean += float(loss.detach()) * w
            data_mean += float(supervised.detach()) * w
            energy_mean += float(energy.detach()) * w
            state["optimizer_steps"] += 1
            state["input_visits"] += len(ix)
            state["supervised_label_visits"] += int(mask[indices].sum())
            state["physics_input_visits"] += len(ix) if case["physics_weight"] else 0
        val_pred = predict(model, data["val"]["c"], device, config["eval_batch_size"])
        val_score = score(val_pred, data["val"]["y"], geometry)
        train_score = None
        if epoch == 1 or epoch % config["train_eval_every"] == 0 or epoch == end:
            train_score = score(
                predict(model, data["train"]["c"], device, config["eval_batch_size"]),
                data["train"]["y"],
                geometry,
            )
        if val_score < state["best_score"]:
            state.update(
                best_score=val_score, best_epoch=epoch, best_model=_cpu_copy(model.state_dict())
            )
        row = {
            "epoch": epoch,
            "objective": loss_mean,
            "supervised_objective": data_mean,
            "twice_energy_objective": energy_mean,
            "train_l2": train_score,
            "val_l2": val_score,
            "learning_rate": float(opt.param_groups[0]["lr"]),
        }
        state["history"].append(row)
        scheduler.step()
        state["epoch"] = epoch
        state["elapsed_seconds"] = before + time.perf_counter() - start
        if epoch % config["checkpoint_every"] == 0 or epoch == end:
            checkpoint()
        if epoch == 1 or epoch % 25 == 0 or epoch == end:
            dump(
                path / "status.json",
                {
                    "status": "training" if epoch < config["epochs"] else "trained_pending_audit",
                    "epoch": epoch,
                    "best_epoch": state["best_epoch"],
                    "best_val_l2": state["best_score"],
                    "elapsed_seconds": state["elapsed_seconds"],
                },
            )
            print(
                "PHYSICS_PROGRESS",
                case["name"],
                seed,
                epoch,
                val_score,
                state["elapsed_seconds"],
                flush=True,
            )
    return state


def finish_one(path, state, config, case, data, geometry, system, device):
    expected = config["epochs"] * (
        (len(data["train"]["c"]) + config["batch_size"] - 1) // config["batch_size"]
    )
    if (
        state["epoch"] != config["epochs"]
        or state["optimizer_steps"] != expected
        or sorted({int(v["step"]) for v in state["optimizer"]["state"].values()}) != [expected]
    ):
        raise RuntimeError("Incomplete updates")
    if state["supervised_label_visits"] != config["epochs"] * case["labels"] or state[
        "input_visits"
    ] != config["epochs"] * len(data["train"]["c"]):
        raise RuntimeError("Visit counts")
    if min(state["history"], key=lambda v: v["val_l2"])["epoch"] != state["best_epoch"]:
        raise RuntimeError("Wrong checkpoint selection")
    saved = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
    if saved["identity"] != state["identity"] or saved["best_epoch"] != state["best_epoch"]:
        raise RuntimeError("Checkpoint identity")
    model = make_reaction_model(config["model"], state["normalization"], system.epsilon).to(device)
    model.load_state_dict(saved["model"])
    reference = make_reaction_model(config["model"], state["normalization"], system.epsilon).to(
        device
    )
    reference.load_state_dict(state["best_model"])
    metrics = {}
    checks = {}
    for split, d in data.items():
        prediction = predict(model, d["c"], device, config["eval_batch_size"])
        again = predict(reference, d["c"], device, config["eval_batch_size"])
        if not np.array_equal(prediction, again):
            raise RuntimeError("Reload difference")
        metrics[split] = evaluate_arrays(prediction, d["y"], geometry, d["sample_id"])
        diagnostics = system.diagnostics(d["c"], prediction)
        metrics[split]["physics_mean"] = {
            key: float(np.mean([x[key] for x in diagnostics])) for key in diagnostics[0]
        }
        metrics[split]["physics_per_sample"] = diagnostics
        if metrics[split]["max"]["boundary_max"] > 1e-12:
            raise RuntimeError("Boundary violation")
        local = (prediction - d["y"]).reshape(len(prediction), -1)[:, system.elements]
        local_y = d["y"].reshape(len(prediction), -1)[:, system.elements]

        def independent(v):
            return ((v.sum(-1) ** 2 + (v * v).sum(-1)) * system.area / 12).sum(1)

        rel = np.sqrt(independent(local) / independent(local_y))
        reported = np.array([x["relative_l2"] for x in metrics[split]["per_sample"]])
        diff = float(abs(rel - reported).max())
        if diff > 1e-12:
            raise RuntimeError("Independent L2 mismatch")
        checks[split] = {
            "reload_max_difference": 0.0,
            "independent_l2_max_difference": diff,
            "csr_physics_recomputed": True,
        }
        np.savez_compressed(
            path / (split + "_predictions.npz"), prediction=prediction, sample_id=d["sample_id"]
        )
    if abs(metrics["val"]["mean"]["relative_l2"] - state["best_score"]) > 1e-12:
        raise RuntimeError("Selection score differs")
    result = {
        "case": case,
        "seed": state["identity"]["seed"],
        "epochs": state["epoch"],
        "optimizer_steps": expected,
        "input_visits": state["input_visits"],
        "supervised_label_visits": state["supervised_label_visits"],
        "physics_input_visits": state["physics_input_visits"],
        "best_epoch": state["best_epoch"],
        "elapsed_seconds": state["elapsed_seconds"],
        "parameter_count": state["parameter_count"],
        "metrics": metrics,
        "test_evaluations": 0,
    }
    dump(path / "metrics.json", result)
    dump(
        path / "verification.json",
        {
            "passed": True,
            "checks": checks,
            "optimizer_selection_verified": True,
            "test_evaluations": 0,
        },
    )
    dump(path / "status.json", {"status": "complete", "epoch": state["epoch"]})
    certify(path)
    print(
        "PHYSICS_RUN_COMPLETE",
        case["name"],
        state["identity"]["seed"],
        metrics["val"]["mean"]["relative_l2"],
        flush=True,
    )
    return result


def summarize(out, rows, config):
    aggregate = {}
    for case in config["cases"]:
        group = [r for r in rows if r["case"] == case["name"]]
        if group:
            aggregate[case["name"]] = {
                k: {
                    "mean": float(np.mean([r[k] for r in group])),
                    "std_population": float(np.std([r[k] for r in group])),
                }
                for k in ["train_l2", "val_l2", "val_h1", "val_residual", "elapsed_seconds"]
            }
    dump(
        out / "aggregate.json",
        {"completed_runs": len(rows), "groups": aggregate, "rows": rows, "test_evaluations": 0},
    )
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# FNO物理训练首轮受控实验",
        "",
        f'完成{len(rows)}/{len(config["cases"])*len(config["seeds"])}组；每组{config["epochs"]}epoch。train128/val32；test未评估。',
        "",
        "| 设置 | train L2 | val L2，均值±总体std | val H1半范 | val弱残差相对值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, a in aggregate.items():
        lines.append(
            f'| {name} | {100*a["train_l2"]["mean"]:.4f}% | {100*a["val_l2"]["mean"]:.4f}% ± {100*a["val_l2"]["std_population"]:.4f}% | {100*a["val_h1"]["mean"]:.4f}% | {a["val_residual"]["mean"]:.6g} |'
        )
    lines += [
        "",
        "弱残差采用内部负载向量的lumped质量对偶范数。物理系统使用公共输入的P1插值系数；与129节点参考限制标签的差异见preparation.json。",
        "",
        "纯能量组0个训练标签进入优化目标/归一化；所有组仍用32个验证标签选模。32标签子集预先固定，监督minibatch使用128/n_labels权重。",
        "",
        "这15组共享架构、初始化种子、批次输入顺序、400epoch预算与cosine学习率。旧1600epoch基线预算及scheduler不同，不作为本受控比较的同预算结果。验证不是独立测试；不作架构家族排名。",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines))


def plots(out, rows, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [c["name"] for c in config["cases"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, key, label, scale in zip(
        axes,
        ["val_l2", "val_h1", "val_residual"],
        ["Relative P1 L2 (%)", "Relative H1 seminorm (%)", "Relative lumped-dual residual"],
        [100, 100, 1],
    ):
        values = [[r[key] * scale for r in rows if r["case"] == name] for name in names]
        ax.bar(
            range(len(names)),
            [np.mean(x) for x in values],
            yerr=[np.std(x) for x in values],
            capsize=3,
        )
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.set_ylabel(label)
        ax.set_yscale("log")
    fig.savefig(out / "physics_comparison.png", dpi=170)
    fig.savefig(out / "physics_comparison.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(names), figsize=(18, 3.7), constrained_layout=True, sharey=True)
    for ax, name in zip(axes, names):
        for seed in config["seeds"]:
            h = json.loads((out / name / f"seed_{seed}/history.json").read_text())
            ax.semilogy([v["epoch"] for v in h], [v["val_l2"] for v in h], label=f"seed {seed}")
        ax.set(title=name, xlabel="Epoch")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Validation mean relative P1 L2")
    fig.savefig(out / "physics_training.png", dpi=170)
    fig.savefig(out / "physics_training.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = validate_config(json.loads(args.config.read_text()))
    out = args.output.resolve()
    dataset = args.data.resolve()
    check = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/verify_frozen_release.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if check.returncode:
        raise RuntimeError("Frozen baseline changed: " + check.stdout)
    identity = {
        "sources": sources(),
        "config": config,
        "dataset_manifest_sha256": sha(dataset / "manifest.json"),
        "frozen_release_sha256": sha(ROOT / "releases/ch03-reaction-v1.0-20261022/release.json"),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        },
    }
    if args.resume:
        if json.loads((out / "preparation.json").read_text())["identity"] != identity:
            raise ValueError("Changed source/data/config/runtime")
        if (out / "PHYSICS_COMPLETE").exists():
            verify(out, "PHYSICS_COMPLETE")
            print("VERIFIED_COMPLETE_NO_TRAINING")
            return
    else:
        out.mkdir(parents=True, exist_ok=False)
    with exclusive_run_lock(out):
        with public_data_only(dataset) as opened:
            data = verify_dataset(dataset)
            geometry = P1Geometry(dataset / "observation_geometry.npz")
        epsilon = json.loads((dataset / "config.json").read_text())["epsilon"]
        system = P1System(geometry.grid_shape[0] - 1, epsilon)
        norm = normalize_inputs(data, geometry)
        if not args.resume:
            differences = {}
            label_residuals = {}
            for split, d in data.items():
                exact = np.stack([system.solve(c) for c in d["c"]])
                differences[split] = evaluate_arrays(exact, d["y"], geometry, d["sample_id"])
                label_residuals[split] = system.diagnostics(d["c"], d["y"])
                np.savez_compressed(
                    out / (split + "_discrete_solutions.npz"),
                    prediction=exact,
                    sample_id=d["sample_id"],
                )
            dump(out / "config.json", config)
            dump(out / "normalization.json", norm)
            shutil.copytree(
                HERE, out / "source_snapshot", ignore=shutil.ignore_patterns("__pycache__")
            )
            dump(
                out / "preparation.json",
                {
                    "identity": identity,
                    "opened_dataset_arrays": sorted(set(opened)),
                    "epsilon": epsilon,
                    "discrete_vs_public_label": differences,
                    "public_label_residuals": label_residuals,
                    "label_subsets": {
                        c["name"]: list(
                            map(
                                str,
                                data["train"]["sample_id"][
                                    label_mask(
                                        len(data["train"]["c"]), c["labels"], config["subset_seed"]
                                    )
                                ],
                            )
                        )
                        for c in config["cases"]
                    },
                    "test_evaluations": 0,
                    "prepared_files": {
                        n: sha(out / n)
                        for n in [
                            "config.json",
                            "normalization.json",
                            "train_discrete_solutions.npz",
                            "val_discrete_solutions.npz",
                        ]
                    },
                },
            )
        else:
            meta = json.loads((out / "preparation.json").read_text())
            for name, h in meta["prepared_files"].items():
                if sha(out / name) != h:
                    raise ValueError("Preparation changed: " + name)
            if norm != json.loads((out / "normalization.json").read_text()):
                raise ValueError("Normalization drift")
        if args.prepare_only:
            dump(out / "status.json", {"status": "prepared", "test_evaluations": 0})
            print("PHYSICS_PREPARED")
            return
        rows = []
        for case in config["cases"]:
            for seed in config["seeds"]:
                path = out / case["name"] / f"seed_{seed}"
                dump(
                    out / "status.json",
                    {
                        "status": "running",
                        "case": case["name"],
                        "seed": seed,
                        "completed_runs": len(rows),
                        "total_runs": len(config["cases"]) * len(config["seeds"]),
                    },
                )
                if (path / "COMPLETE").exists():
                    verify(path)
                    result = json.loads((path / "metrics.json").read_text())
                    if json.loads((path / "identity.json").read_text()) != {
                        **identity,
                        "config": config,
                        "case": case,
                        "seed": seed,
                        "device": args.device,
                    }:
                        raise ValueError("Run identity changed")
                else:
                    state = train_one(
                        path,
                        config,
                        case,
                        seed,
                        data,
                        geometry,
                        system,
                        norm,
                        identity,
                        args.device,
                        resume=args.resume and path.exists(),
                    )
                    result = finish_one(
                        path, state, config, case, data, geometry, system, args.device
                    )
                m = result["metrics"]
                rows.append(
                    {
                        "case": case["name"],
                        "seed": seed,
                        "labels": case["labels"],
                        "best_epoch": result["best_epoch"],
                        "epochs": result["epochs"],
                        "optimizer_steps": result["optimizer_steps"],
                        "train_l2": m["train"]["mean"]["relative_l2"],
                        "val_l2": m["val"]["mean"]["relative_l2"],
                        "val_h1": m["val"]["mean"]["relative_h1_seminorm"],
                        "val_residual": m["val"]["physics_mean"]["relative_lumped_dual_residual"],
                        "elapsed_seconds": result["elapsed_seconds"],
                    }
                )
                summarize(out, rows, config)
        plots(out, rows, config)
        if sources() != identity["sources"]:
            raise RuntimeError("Source drift")
        dump(
            out / "verification.json",
            {
                "passed": True,
                "runs": len(rows),
                "all_runs_reloaded_and_audited": True,
                "test_evaluations": 0,
                "frozen_baseline": json.loads(check.stdout),
            },
        )
        dump(
            out / "status.json",
            {"status": "complete", "completed_runs": len(rows), "test_evaluations": 0},
        )
        certify(out, "PHYSICS_COMPLETE")
        verify(out, "PHYSICS_COMPLETE")
        print("ALL_PHYSICS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
