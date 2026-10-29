import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, csv, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import spsolve
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from solver import solve, residual_coordinates

sys.path.insert(0, str(ROOT / "experiments/ch03_varmion_v1"))
sys.path.append(str(ROOT / "experiments/ch03_physics_v1"))
from common import sha, dump, certify, verify
from model import model_for, structure_checks
from train_coeff import fit, evaluate, tensor_data
from data import load as load_cf
from physics import P1System
from residual_data import generate, load
from neuralpde_opt.chapter3.operator_training import P1Geometry, evaluate_arrays, exclusive_run_lock


def finish(path, state, factory, arrays, data, basis, geometry):
    best = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
    if best["identity"] != state["identity"] or best["best_epoch"] != state["best_epoch"]:
        raise RuntimeError("Checkpoint mismatch")
    model = factory().cuda()
    model.load_state_dict(best["model"])
    reference = factory().cuda()
    reference.load_state_dict(state["best_model"])
    tensors = tensor_data(arrays, "cuda")
    metrics = {}
    defects = {}
    for split, d in data.items():
        alpha, score = evaluate(model, tensors[split])
        same, _ = evaluate(reference, tensors[split])
        if not np.array_equal(alpha, same):
            raise RuntimeError("Reload mismatch")
        prediction = (alpha @ basis.T).reshape(d["y"].shape)
        metrics[split] = evaluate_arrays(prediction, d["y"], geometry, d["sample_id"])
        defect = float(
            np.max(
                abs(
                    geometry.l2_squared(prediction - d["y"])
                    - d["projection_squared"]
                    - ((alpha - d["alpha"]) ** 2).sum(1)
                )
            )
        )
        if (
            defect > 1e-10
            or abs(score - metrics[split]["mean"]["relative_l2"]) > 1e-10
            or metrics[split]["max"]["boundary_max"] > 1e-12
        ):
            raise RuntimeError("Residual-map decomposition/metric audit")
        defects[split] = defect
        np.savez_compressed(
            path / (split + "_predictions.npz"),
            prediction=prediction,
            coefficients=alpha,
            sample_id=d["sample_id"],
        )
    d = tensors["val"]
    checks = structure_checks(model, d["c"], d["z"], d["z"].roll(1, 0))
    if checks["zero_load_max"] != 0 or checks["relative_superposition_max"] > 2e-6:
        raise RuntimeError("Residual linearity failed")
    result = {
        "seed": state["identity"]["seed"],
        "epochs": state["epoch"],
        "optimizer_steps": state["steps"],
        "sample_visits": state["visits"],
        "best_epoch": state["best_epoch"],
        "elapsed_seconds": state["elapsed_seconds"],
        "metrics": metrics,
        "structure": checks,
        "test_evaluations": 0,
    }
    dump(path / "metrics.json", result)
    dump(
        path / "verification.json",
        {
            "passed": True,
            "reload_identical": True,
            "pythagoras_defects": defects,
            "structure": checks,
        },
    )
    certify(path, "COMPLETE")
    print(
        "RESIDUAL_MAP_COMPLETE",
        state["identity"]["seed"],
        metrics["val"]["mean"]["relative_l2"],
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/ch03/hints_v1")
    parser.add_argument("--data", type=Path, default=ROOT / "data/ch03/hints_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    dataset = args.data.resolve()
    previous = ROOT / "results/ch03/varmion_v1"
    config = json.loads((HERE / "config.json").read_text())
    verify(previous, "VARMION_COMPLETE")
    if not json.loads((previous / "verification.json").read_text())["passed"]:
        raise RuntimeError("Step 4 must pass before step 5")
    frozen = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/verify_frozen_release.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if frozen.returncode:
        raise RuntimeError("Frozen baseline failed")
    source = load_cf(ROOT / "data/ch03/varmion_v1")
    geometry = P1Geometry(ROOT / "data/ch03/varmion_v1/observation_geometry.npz")
    system = P1System(64, 0.01)
    with np.load(previous / "basis.npz", allow_pickle=False) as z:
        basis = z["basis"].copy()
        k0 = z["k0"].copy()
    norm = json.loads((previous / "normalization.json").read_text())
    ii = system.interior
    phi = basis[ii]
    if not dataset.exists():
        generate(dataset, source, basis, system, config)
    elif json.loads((dataset / "config.json").read_text()) != config:
        raise ValueError("Residual data config changed")
    data, arrays = load(dataset)
    quality = json.loads((dataset / "quality.json").read_text())
    # Independent mass-load conversion identity on a real stored residual.
    residual = data["train"]["r"][0]
    f = spsolve(system.mass[ii][:, ii], residual)
    if (
        np.max(abs(phi.T @ (system.mass[ii][:, ii] @ f) - residual_coordinates(phi, residual)))
        > 1e-10
    ):
        raise RuntimeError("Residual transfer mismatch")
    identity = {
        "sources": {p.name: sha(p) for p in HERE.glob("*.py")},
        "shared": {
            name: sha(ROOT / "experiments/ch03_varmion_v1" / name)
            for name in ["model.py", "train_coeff.py", "common.py", "data.py"]
        },
        "p1_physics": sha(ROOT / "experiments/ch03_physics_v1/physics.py"),
        "config": config,
        "basis_sha256": sha(previous / "basis.npz"),
        "residual_data_manifest": sha(dataset / "manifest.json"),
        "pretrained_models": {
            str(seed): sha(previous / "varmion" / f"seed_{seed}/best.pt")
            for seed in config["seeds"]
        },
        "torch": str(torch.__version__),
    }
    if args.resume:
        if json.loads((out / "identity.json").read_text()) != identity:
            raise ValueError("Identity changed")
        if (out / "HINTS_COMPLETE").exists():
            verify(out, "HINTS_COMPLETE")
            print("VERIFIED_COMPLETE_NO_TRAINING")
            return
    else:
        out.mkdir(parents=True, exist_ok=False)
        dump(out / "identity.json", identity)
        dump(out / "config.json", config)
        shutil.copytree(HERE, out / "source_snapshot", ignore=shutil.ignore_patterns("__pycache__"))
    with exclusive_run_lock(out):
        training = []
        for seed in config["seeds"]:
            path = out / "residual_models" / f"seed_{seed}"

            def factory():
                model = model_for("varmion", norm, k0)
                ck = torch.load(
                    previous / "varmion" / f"seed_{seed}/best.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                model.load_state_dict(ck["model"])
                return model

            dump(
                out / "status.json",
                {"status": "training_residual_map", "seed": seed, "completed_runs": len(training)},
            )
            if (path / "COMPLETE").exists():
                verify(path, "COMPLETE")
                result = json.loads((path / "metrics.json").read_text())
                if json.loads((path / "identity.json").read_text()) != {
                    **identity,
                    "config": config,
                    "seed": seed,
                    "device": "cuda",
                }:
                    raise ValueError("Completed residual run identity")
            else:
                state = fit(
                    path,
                    config,
                    seed,
                    factory,
                    arrays,
                    identity,
                    resume=args.resume and path.exists(),
                )
                result = finish(path, state, factory, arrays, data, basis, geometry)
            training.append(result)
        dump(out / "residual_training.json", training)
        # Conditional matrices are reused during a solve, valid because the learned map is linear in r.
        models = {}
        for seed in config["seeds"]:
            model = model_for("varmion", norm, k0)
            model.load_state_dict(
                torch.load(
                    out / "residual_models" / f"seed_{seed}/best.pt",
                    map_location="cpu",
                    weights_only=True,
                )["model"]
            )
            models[seed] = model.eval()
        selected = quality["splits"]["val"]["selected_source_indices"]
        val = source["val"]
        records = []
        solutions = []
        solution_keys = []
        for position, j in enumerate(selected):
            start = time.perf_counter()
            full, bfull = system.assemble(val["c"][j], val["f"][j])
            a = full[ii][:, ii].tocsr()
            b = bfull[ii]
            assembly = time.perf_counter() - start
            exact = spsolve(a, b)
            cases = [("jacobi", None)] + [
                (mode, seed) for mode in ["fixed", "safeguarded"] for seed in config["seeds"]
            ]
            for mode, seed in cases:
                durations = []
                first = None
                matrix_time = 0.0
                for repeat in range(3):
                    start = time.perf_counter()
                    corrector = None
                    if seed is not None:
                        with torch.no_grad():
                            k = (
                                models[seed]
                                .matrix(torch.tensor(val["c"][j : j + 1], dtype=torch.float32))[0]
                                .numpy()
                                .astype(np.float64)
                            )
                        corrector = lambda r: phi @ (k @ residual_coordinates(phi, r))
                    conditioned = time.perf_counter() - start
                    y, record = solve(
                        a,
                        b,
                        corrector,
                        mode,
                        period=config["period"],
                        eta=config["eta"],
                        omega=config["omega"],
                        rtol=config["rtol"],
                        atol=config["atol"],
                        maxiter=config["maxiter"],
                        trace=False,
                    )
                    durations.append(time.perf_counter() - start)
                    matrix_time += conditioned / 3
                    if first is None:
                        first = record
                        solution = y.copy()
                    elif record["iterations"] != first["iterations"] or not np.array_equal(
                        solution, y
                    ):
                        raise RuntimeError("Repeated hybrid solve changed")
                e = np.zeros(system.size)
                e[ii] = solution - exact
                truth = np.zeros(system.size)
                truth[ii] = exact
                error = float(np.sqrt((e @ (system.mass @ e)) / (truth @ (system.mass @ truth))))
                if first["converged"] and error > 1e-6:
                    raise RuntimeError("Hybrid solution verification failed")
                if (
                    mode == "safeguarded"
                    and first["max_accepted_neural_ratio"] > config["eta"] + 1e-12
                ):
                    raise RuntimeError("Safeguard failed")
                if (
                    first["max_classical_observed_ratio"]
                    > first["classical_contraction_bound"] + 1e-5
                ):
                    raise RuntimeError("Classical contraction violated")
                key = f"{mode}:seed_{seed}:source_{j}"
                entry = {
                    "key": key,
                    "mode": mode,
                    "seed": seed,
                    "source_index": j,
                    "sample_id": str(val["sample_id"][j]),
                    **first,
                    "relative_discrete_l2": error,
                    "conditional_matrix_seconds": matrix_time,
                    "assembly_seconds": assembly,
                    "online_seconds": float(np.median(durations)),
                    "total_seconds": assembly + float(np.median(durations)),
                    "timed_repetitions": durations,
                }
                records.append(entry)
                v = np.zeros(system.size)
                v[ii] = solution
                solutions.append(v.reshape(65, 65))
                solution_keys.append(key)
                if position == 0 and seed in [None, 0]:
                    _, trace = solve(
                        a,
                        b,
                        corrector,
                        mode,
                        period=config["period"],
                        eta=config["eta"],
                        omega=config["omega"],
                        rtol=config["rtol"],
                        atol=config["atol"],
                        maxiter=config["maxiter"],
                        trace=True,
                    )
                    dump(out / (mode + "_trace.json"), trace)
            dump(
                out / "status.json",
                {
                    "status": "evaluating_hybrid",
                    "completed_sources": position + 1,
                    "total_sources": len(selected),
                },
            )
            print("HINTS_SOURCE_COMPLETE", position + 1, len(selected), flush=True)
        np.savez_compressed(
            out / "solutions.npz",
            prediction=np.array(solutions),
            sample_key=np.array(solution_keys),
        )
        dump(out / "per_sample.json", records)
        summary = {}
        for mode in ["jacobi", "fixed", "safeguarded"]:
            group = [r for r in records if r["mode"] == mode]
            summary[mode] = {
                "count": len(group),
                "converged": sum(r["converged"] for r in group),
                **{
                    key: float(np.mean([r[key] for r in group]))
                    for key in [
                        "iterations",
                        "neural_calls",
                        "accepted_neural",
                        "rejected_neural",
                        "classical_steps",
                        "total_seconds",
                        "online_seconds",
                    ]
                },
                "max_relative_residual": float(max(r["relative_residual"] for r in group)),
                "max_converged_discrete_l2": float(
                    max([r["relative_discrete_l2"] for r in group if r["converged"]] or [0.0])
                ),
            }
        for mode, s in summary.items():
            s["speedup_vs_jacobi"] = (
                summary["jacobi"]["total_seconds"] / s["total_seconds"]
                if s["count"] == s["converged"]
                else None
            )
        dump(out / "summary.json", summary)
        flat = [
            {
                k: v
                for k, v in r.items()
                if k
                not in [
                    "residual_history",
                    "neural_events",
                    "scaled_eigenvalue_bounds",
                    "timed_repetitions",
                ]
            }
            for r in records
        ]
        with (out / "metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0]))
            w.writeheader()
            w.writerows(flat)
        lines = [
            "# HINTS固定混合与保护混合",
            "",
            "残差校正器使用640个train/160个val迭代残差训练；原方程求解用32个开发验证源项。每方法/种子/样本计时重复3次。",
            "",
            "| 方法 | 收敛/总数 | 平均迭代 | 神经接受 | 神经拒绝 | 总ms | 总时间加速比 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for mode, s in summary.items():
            lines.append(
                f'| {mode} | {s["converged"]}/{s["count"]} | {s["iterations"]:.1f} | {s["accepted_neural"]:.1f} | {s["rejected_neural"]:.1f} | {1000*s["total_seconds"]:.2f} | {s["speedup_vs_jacobi"]} |'
            )
        lines += [
            "",
            "残差采用D^-1加权范数，所有方法相同atol+rtol*||b||停止阈值。每步重算残差，保护版只接受残差降至0.9倍以内的校正。以对角占优界验证Jacobi分支收缩。固定混合的失败不隐藏。",
            "",
            "编码Phi_I^T r等价于先求M_II^(-1)r再进行源项负载编码；不能把r直接作为节点源项。总成本包含条件矩阵构造、残差转换/输出重构、经典步、接受/回退核验及共同系统装配；模型加载为离线一次性开销。",
            "",
            "这是当前公共网格的线性残差校正器实验。val已用于选模，结果不是独立test，也不证明任意PDE/残差分布下有效。",
            "",
        ]
        (out / "summary.md").write_text("\n".join(lines))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for mode in summary:
            r = json.loads((out / (mode + "_trace.json")).read_text())
            axes[0].semilogy(
                np.arange(len(r["residual_history"])),
                np.array(r["residual_history"]) / r["rhs_norm"],
                label=mode,
            )
        axes[0].set(
            xlabel="Iteration",
            ylabel="True relative D^-1 residual",
            title="Fixed validation source 0",
        )
        axes[0].legend()
        axes[1].bar(list(summary), [summary[k]["total_seconds"] * 1000 for k in summary])
        axes[1].set_ylabel("Mean total latency (ms)")
        fig.savefig(out / "hints_comparison.png", dpi=170)
        fig.savefig(out / "hints_comparison.svg")
        plt.close(fig)
        if {p.name: sha(p) for p in HERE.glob("*.py")} != identity["sources"]:
            raise RuntimeError("Source drift")
        safe = summary["safeguarded"]
        passed = (
            safe["count"] == safe["converged"]
            and summary["jacobi"]["count"] == summary["jacobi"]["converged"]
        )
        dump(
            out / "verification.json",
            {
                "passed": passed,
                "all_accepted_neural_steps_checked": True,
                "classical_contraction_certified": True,
                "mass_load_conversion_checked": True,
                "test_evaluations": 0,
                "frozen_baseline": json.loads(frozen.stdout),
            },
        )
        dump(
            out / "status.json",
            {
                "status": "complete" if passed else "completed_with_solver_failures",
                "solves": len(records),
            },
        )
        certify(out, "HINTS_COMPLETE")
        verify(out, "HINTS_COMPLETE")
        print("ALL_HINTS_COMPLETE", passed, flush=True)


if __name__ == "__main__":
    main()
