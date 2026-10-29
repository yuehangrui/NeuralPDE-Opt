"""One frozen held-out evaluation, resumable only with identical inputs/weights."""

import time
import numpy as np
import torch
from scipy.sparse.linalg import spsolve
from bootstrap import ROOT
from integrity import read, require, dump, sha, certify, verify
from heldout import protocol_at
from predictors import array_file, predict, build
from hybrid import solve, residual_coordinates
from physics import P1System
from neuralpde_opt.chapter3.operator_training import P1Geometry, evaluate_arrays, exclusive_run_lock
from neuralpde_opt.chapter3.reaction_training import seed_everything


def checked_metrics(prediction, data, geometry):
    result = evaluate_arrays(prediction, data["y"], geometry, data["sample_id"])
    error = (prediction - data["y"]).reshape(len(prediction), -1)[:, geometry.elements]
    target = data["y"].reshape(len(prediction), -1)[:, geometry.elements]

    def norm(local):
        return ((local.sum(-1) ** 2 + (local * local).sum(-1)) * geometry.areas / 12).sum(1)

    independent = np.sqrt(norm(error) / norm(target))
    require(
        np.max(abs(independent - np.array([r["relative_l2"] for r in result["per_sample"]])))
        < 1e-12,
        "Independent L2 metric disagreement",
    )
    require(result["max"]["boundary_max"] <= 1e-12, "Boundary violation")
    return result


def test_models(directory, protocol, data, geometry):
    rows = []
    for row in protocol["runs"]:
        if row["family"] == "hints":
            continue
        target = directory / "models" / row["id"]
        d = data[row["dataset"]]
        if (target / "COMPLETE").exists():
            verify(target)
            record = read(target / "metrics.json")
        else:
            target.mkdir(parents=True, exist_ok=True)
            prediction = predict(row, d, geometry.mass, "cuda", 4)
            record = dict(
                **row,
                metrics=checked_metrics(prediction, d, geometry),
                checkpoint_sha256=protocol["checkpoints"][row["id"]]["sha256"],
            )
            # A saved prediction is independently reloaded and measured before certification.
            np.savez_compressed(
                target / "predictions.npz", prediction=prediction, sample_id=d["sample_id"]
            )
            reloaded = array_file(target / "predictions.npz")
            require(
                np.array_equal(reloaded["sample_id"], d["sample_id"])
                and np.array_equal(reloaded["prediction"], prediction),
                "Prediction serialization mismatch",
            )
            require(
                checked_metrics(reloaded["prediction"], d, geometry) == record["metrics"],
                "Metric serialization mismatch",
            )
            dump(target / "metrics.json", record)
            certify(target)
        require(
            record["checkpoint_sha256"] == protocol["checkpoints"][row["id"]]["sha256"],
            "Wrong completed model",
        )
        rows.append(
            dict(
                id=row["id"],
                family=row["family"],
                variant=row["variant"],
                seed=row["seed"],
                dataset=row["dataset"],
                l2=record["metrics"]["mean"]["relative_l2"],
                h1=record["metrics"]["mean"]["relative_h1_seminorm"],
            )
        )
        print("TEST_MODEL_COMPLETE", row["id"], rows[-1]["l2"], flush=True)
    groups = {}
    for row in rows:
        key = row["family"] + "/" + row["variant"]
        selected = [
            r for r in rows if r["family"] == row["family"] and r["variant"] == row["variant"]
        ]
        groups[key] = dict(
            dataset=row["dataset"],
            seeds=len(selected),
            **{
                k: dict(
                    mean=float(np.mean([r[k] for r in selected])),
                    std_population=float(np.std([r[k] for r in selected])),
                )
                for k in ["l2", "h1"]
            },
        )
    dump(directory / "model_summary.json", dict(rows=rows, groups=groups))
    return groups


def test_hints(directory, protocol, data, geometry):
    system = P1System(64, 0.01)
    ii = system.interior
    models = {}
    phi = None
    for row in protocol["runs"]:
        if row["family"] == "hints":
            model, basis = build(row, "cpu")
            models[row["seed"]] = model
            phi = basis[ii]
    selected = [
        j for j in range(len(data["y"])) if j % protocol["cf_forcings"] < protocol["hints_forcings"]
    ]
    records = []
    settings = {k: v for k, v in protocol["hints"].items() if k != "repetitions"}
    for j in selected:
        start = time.perf_counter()
        full, load = system.assemble(data["c"][j], data["f"][j])
        a = full[ii][:, ii].tocsr()
        b = load[ii]
        assembly = time.perf_counter() - start
        exact = spsolve(a, b)
        for mode, seed in [("jacobi", None)] + [
            (m, s) for m in ["fixed", "safeguarded"] for s in models
        ]:
            key = f"source_{j:03d}__{mode}__s{seed}"
            target = directory / "hints" / key
            if (target / "COMPLETE").exists():
                verify(target)
                record = read(target / "metrics.json")
            else:
                target.mkdir(parents=True, exist_ok=True)
                durations = []
                solution = None
                for repeat in range(protocol["hints"]["repetitions"]):
                    start = time.perf_counter()
                    corrector = None
                    if seed is not None:
                        with torch.no_grad():
                            k = (
                                models[seed]
                                .matrix(torch.as_tensor(data["c"][j : j + 1], dtype=torch.float32))[
                                    0
                                ]
                                .double()
                                .numpy()
                            )
                        corrector = lambda r: phi @ (k @ residual_coordinates(phi, r))
                    y, info = solve(a, b, corrector, mode=mode, **settings)
                    durations.append(time.perf_counter() - start)
                    if solution is None:
                        solution = y.copy()
                        first = info
                    else:
                        require(
                            np.array_equal(solution, y)
                            and info["iterations"] == first["iterations"],
                            "Repeated solve differs",
                        )
                full_y = np.zeros(system.size)
                full_y[ii] = solution
                truth = np.zeros(system.size)
                truth[ii] = exact
                error = full_y - truth
                relative = float(
                    np.sqrt(error @ (system.mass @ error) / (truth @ (system.mass @ truth)))
                )
                # Algebraic error and public-reference error answer different questions.
                public_error = checked_metrics(
                    full_y.reshape(1, 65, 65), {k: v[j : j + 1] for k, v in data.items()}, geometry
                )
                if first["converged"]:
                    require(relative < 1e-6, "Converged flag contradicts direct solve")
                if mode == "safeguarded":
                    require(
                        first["max_accepted_neural_ratio"] <= settings["eta"] + 1e-12,
                        "Safeguard violation",
                    )
                require(
                    first["max_classical_observed_ratio"]
                    <= first["classical_contraction_bound"] + 1e-5,
                    "Contraction violation",
                )
                record = dict(
                    key=key,
                    source_index=j,
                    sample_id=str(data["sample_id"][j]),
                    mode=mode,
                    seed=seed,
                    **first,
                    relative_discrete_l2=relative,
                    public_l2=public_error["mean"]["relative_l2"],
                    public_h1=public_error["mean"]["relative_h1_seminorm"],
                    online_seconds=float(np.median(durations)),
                    assembly_seconds=assembly,
                    total_seconds=assembly + float(np.median(durations)),
                    timed_repetitions=durations,
                )
                np.savez_compressed(target / "solution.npz", prediction=full_y.reshape(65, 65))
                dump(target / "metrics.json", record)
                certify(target)
            records.append(record)
        print("TEST_HINTS_SOURCE", j, len(selected), flush=True)
    summary = {}
    for mode in ["jacobi", "fixed", "safeguarded"]:
        group = [r for r in records if r["mode"] == mode]
        summary[mode] = dict(
            count=len(group),
            converged=sum(r["converged"] for r in group),
            **{
                k: float(np.mean([r[k] for r in group]))
                for k in ["iterations", "total_seconds", "online_seconds", "public_l2", "public_h1"]
            },
            max_relative_discrete_l2=max(r["relative_discrete_l2"] for r in group),
            max_relative_residual=max(r["relative_residual"] for r in group),
        )
    for value in summary.values():
        value["speedup_vs_jacobi"] = (
            summary["jacobi"]["total_seconds"] / value["total_seconds"]
            if value["count"] == value["converged"]
            else None
        )
    dump(
        directory / "hints_summary.json",
        dict(groups=summary, solves=len(records), sources=len(selected), records=records),
    )
    return summary


def representation(directory, data, geometry):
    from pod import coordinates, reconstruct

    basis = array_file(ROOT / "results/ch03/pod_v1/basis.npz")
    result = {}
    for rank in [4, 8, 16, 32]:
        pred = reconstruct(coordinates(data["fixed"]["y"], basis, geometry, rank), basis, rank)
        result[f"pod_rank_{rank}"] = checked_metrics(pred, data["fixed"], geometry)["mean"]
    basis = array_file(ROOT / "results/ch03/varmion_v1/basis.npz")["basis"]
    d = data["cf"]
    alpha = d["y"].reshape(len(d["y"]), -1) @ (geometry.mass @ basis)
    result["cf_projection"] = checked_metrics((alpha @ basis.T).reshape(-1, 65, 65), d, geometry)[
        "mean"
    ]
    system = P1System(64, 0.01)
    load_coordinates = d["f"].reshape(len(d["f"]), -1) @ (geometry.mass @ basis)
    predictions = []
    for j in range(len(d["y"])):
        a, _ = system.assemble(d["c"][j])
        predictions.append(basis @ np.linalg.solve(basis.T @ (a @ basis), load_coordinates[j]))
    result["cf_reduced_galerkin"] = checked_metrics(
        np.array(predictions).reshape(-1, 65, 65), d, geometry
    )["mean"]
    dump(directory / "representation.json", result)


def run(directory):
    protocol = protocol_at(directory)
    if (directory / "COMPLETE").exists():
        verify(directory)
        print("VERIFIED_COMPLETE_NO_EVALUATION")
        return
    dataset = ROOT / "data/ch03/final_test_v1"
    verify(dataset)
    require(
        read(dataset / "identity.json")["protocol_sha256"] == sha(directory / "protocol.json"),
        "Dataset identity mismatch",
    )
    seed_everything(0, 2)
    geometry = P1Geometry(dataset / "observation_geometry.npz")
    data = {name: array_file(dataset / (name + ".npz")) for name in ["fixed", "cf"]}
    with exclusive_run_lock(directory):
        models = test_models(directory, protocol, data, geometry)
        hints = test_hints(directory, protocol, data["cf"], geometry)
        representation(directory, data, geometry)
        protocol_at(directory)
        dump(
            directory / "verification.json",
            dict(
                passed=True,
                model_evaluations=54,
                hints_solves=sum(v["count"] for v in hints.values()),
                all_solvers_converged=all(v["count"] == v["converged"] for v in hints.values()),
                test_exposed=True,
                weights_changed=False,
            ),
        )
        lines = [
            "# 第三章最终固定测试",
            "",
            "所有检查点、数据生成规则和HINTS参数均在生成新测试集前锁定。固定源项128样本；变源项32个新反应系数组×4源项。测试后不选模、不调参。",
            "",
            "| 组别 | 种子数 | test相对L2 | test相对H1半范 |",
            "|---|---:|---:|---:|",
        ]
        for name, value in models.items():
            lines.append(
                f'| {name} | {value["seeds"]} | {100*value["l2"]["mean"]:.4f}% ± {100*value["l2"]["std_population"]:.4f}% | {100*value["h1"]["mean"]:.4f}% |'
            )
        lines += [
            "",
            "上述±是种子间总体标准差，不是统计置信区间；同一系数组内源项相关，不能把128个变源项样本当作128个独立系数。不同数据、预算、物理信息和输出空间不构成架构家族排名。",
            "",
            "| HINTS模式 | 收敛/总数 | 平均迭代 | 总耗时ms | 相对Jacobi加速 | 最大离散L2误差 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, value in hints.items():
            lines.append(
                f'| {name} | {value["converged"]}/{value["count"]} | {value["iterations"]:.1f} | {1000*value["total_seconds"]:.2f} | {value["speedup_vs_jacobi"]} | {value["max_relative_discrete_l2"]:.3e} |'
            )
        lines += [
            "",
            "HINTS离散误差相对同一65×65 P1系统的直接解；连续PDE参考误差另存于hints_summary.json。时间包含装配、条件矩阵、残差转换和接受/回退；模型加载不计入。每例重复3次取中位数，只与Jacobi作同机比较。",
            "",
            "POD投影与精确约化Galerkin仅是数值诊断，使用参考标签的投影不是可部署预测模型。所有参考解经过残差、序列化及预定网格加密检查。",
            "",
        ]
        (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        certify(directory)
        print("FINAL_TEST_COMPLETE", flush=True)
