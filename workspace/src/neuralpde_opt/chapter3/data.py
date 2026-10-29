"""Small shared-data pilot with mandatory quality gates and immutable output.

Run with python -m neuralpde_opt.chapter3.data generate|audit ...
No architecture, rank, target derivative, or model training is selected here.
"""

import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
from ..mesh import make_mesh
from .fields import FieldPair, draw_parameters
from .reference import Manufactured, analytic_errors, compare_p1, save_geometry, solve_fields

SPLIT_CODES = {"train": 0, "val": 1, "test": 2}


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(c):
    if c["purpose"] != "pilot_only_no_training":
        raise ValueError("This entry point is restricted to an explicit pilot configuration")
    if set(c["splits"]) != set(SPLIT_CODES) or any(
        not isinstance(n, int) or n < 1 for n in c["splits"].values()
    ):
        raise ValueError("Need positive train/val/test counts")
    if sum(c["splits"].values()) > 64:
        raise ValueError("Pilot cap is 64; design a reviewed production workflow first")
    ns = [c[k] for k in ("observation_n", "reference_n", "audit_n")]
    if any(not isinstance(n, int) or n < 4 for n in ns) or ns[1] != 2 * ns[0] or ns[2] != 2 * ns[1]:
        raise ValueError("Pilot needs nested observation/reference/audit n, 2n, 4n")
    if ns[-1] > 256:
        raise ValueError("Pilot reference audit is capped at 256 intervals")
    for name in ("a", "f"):
        s = c["fields"][name]
        if not isinstance(s["modes"], int) or not 2 <= s["modes"] <= ns[0] // 2:
            raise ValueError("Field modes must resolve on the observation grid")
        if s["length_scale"] <= 0 or s["exponent"] <= 0:
            raise ValueError("Positive spectral length parameter and exponent required")
    a, f = c["fields"]["a"], c["fields"]["f"]
    if not 0 < a["lower"] < a["upper"] or a["latent_scale"] <= 0 or f["rms_in_expectation"] <= 0:
        raise ValueError("Invalid field amplitudes")
    if not 2 <= c["quadrature_order"] < c["audit_quadrature_order"] <= 8:
        raise ValueError("Need a strictly enriched quadrature audit")


def manufactured_audit():
    result = {}
    for variable in (False, True):
        field = Manufactured(variable)
        rows = []
        for n in (8, 16, 32, 64):
            sol = solve_fields(field, n, 5)
            rows.append(
                dict(**analytic_errors(sol, field.exact, field.gradient), **sol.diagnostics)
            )
        slopes = {
            key: float(np.log2(rows[-2][key] / rows[-1][key]))
            for key in ("relative_l2", "relative_h1_seminorm")
        }
        passed = slopes["relative_l2"] > 1.8 and slopes["relative_h1_seminorm"] > 0.9
        result["variable_a" if variable else "constant_a"] = {
            "rows": rows,
            "last_refinement_orders": slopes,
            "passed": passed,
        }
    return result


def sample_gates(row, config):
    g = config["quality_gates"]
    checks = {}
    for key in ("relative_algebraic_residual", "relative_energy_identity_defect", "boundary_max"):
        checks[key] = all(s[key] <= g[key] for s in row["solves"])
    checks["reference_l2"] = row["reference_vs_audit"]["relative_l2"] <= g["reference_vs_audit_l2"]
    checks["reference_h1"] = (
        row["reference_vs_audit"]["relative_h1_seminorm"] <= g["reference_vs_audit_h1"]
    )
    checks["l2_refinement"] = row["refinement_ratio"]["relative_l2"] <= g["l2_refinement_ratio"]
    checks["h1_refinement"] = (
        row["refinement_ratio"]["relative_h1_seminorm"] <= g["h1_refinement_ratio"]
    )
    checks["quadrature"] = row["quadrature_comparison"]["relative_l2"] <= g["quadrature_l2"]
    checks["observation_reconstruction"] = (
        row["observation_vs_audit"]["relative_l2"] <= g["observation_reconstruction_l2"]
    )
    a = config["fields"]["a"]
    checks["ellipticity"] = all(
        a["lower"] <= s["a_quadrature_min"] <= s["a_quadrature_max"] <= a["upper"]
        for s in row["solves"]
    )
    checks["cross_grid_input"] = row["cross_grid_input_max"] <= 1e-13
    return checks


def generate(config_path, output):
    c = json.loads(config_path.read_text())
    validate_config(c)
    if shutil.disk_usage(output.parent).free < 512 * 1024**2:
        raise RuntimeError("Need at least 512 MiB free for this pilot")
    output.mkdir()  # Refuse overwriting or silently resuming a different dataset.
    (output / "audit_solutions").mkdir()
    (output / "source_snapshot").mkdir()
    dump_json(output / "config.json", c)
    n0, n1, n2 = [c[k] for k in ("observation_n", "reference_n", "audit_n")]
    meshes = {n: make_mesh(2, n) for n in (n0, n1, n2)}
    save_geometry(output / "observation_geometry.npz", n0)
    save_geometry(output / "reference_geometry.npz", n1)
    # Audit topology is exactly the same make_mesh convention, with n=audit_n.
    code_files = list(Path(__file__).parent.glob("*.py")) + [Path(__file__).parents[1] / "mesh.py"]
    for p in code_files:
        shutil.copyfile(p, output / "source_snapshot" / p.name)
    environment = {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "solver": "SciPy sparse direct spsolve; float64; no iterative tolerance",
        "generator": "NumPy PCG64 + SeedSequence([seed, split, index, channel])",
        "source_sha256": {p.name: sha256(p) for p in code_files},
    }
    dump_json(output / "environment.json", environment)
    qa = {
        "manufactured": manufactured_audit(),
        "samples": [],
        "scope": "Finite pilot checks; mesh differences are estimates, not continuum error bounds.",
    }
    for split, count in c["splits"].items():
        arrays = {
            k: []
            for k in (
                "a",
                "f",
                "y",
                "a_reference",
                "f_reference",
                "y_reference",
                "a_coefficients",
                "f_coefficients",
                "sample_id",
            )
        }
        for index in range(count):
            sample_id = f'ch03-p{c["seed"]}-{split}-{index:06d}'
            parameters = draw_parameters(c, SPLIT_CODES[split], index)
            fields = FieldPair(parameters, c)
            solutions = [
                solve_fields(fields, n, c["quadrature_order"], meshes[n]) for n in (n0, n1, n2)
            ]
            coarse, ref, fine = solutions
            enriched = solve_fields(fields, n1, c["audit_quadrature_order"], meshes[n1])
            coarse_ref = compare_p1(coarse.values, n0, ref)
            ref_fine = compare_p1(ref.values, n1, fine)
            ratio = {k: ref_fine[k] / coarse_ref[k] for k in ref_fine}
            grid0 = meshes[n0].nodes
            grid1 = meshes[n1].nodes
            observed_a = fields.coefficient(grid0).reshape(n0 + 1, n0 + 1)
            observed_f = fields.source(grid0).reshape(n0 + 1, n0 + 1)
            reference_a = fields.coefficient(grid1).reshape(n1 + 1, n1 + 1)
            reference_f = fields.source(grid1).reshape(n1 + 1, n1 + 1)
            y = ref.values.reshape(n1 + 1, n1 + 1)[::2, ::2].copy()
            row = {
                "sample_id": sample_id,
                "split": split,
                "index": index,
                "solves": [s.diagnostics for s in solutions] + [enriched.diagnostics],
                "coarse_vs_reference": coarse_ref,
                "reference_vs_audit": ref_fine,
                "refinement_ratio": ratio,
                "quadrature_comparison": compare_p1(ref.values, n1, enriched),
                "observation_vs_audit": compare_p1(y.ravel(), n0, fine),
                "cross_grid_input_max": float(
                    max(
                        np.max(np.abs(observed_a - reference_a[::2, ::2])),
                        np.max(np.abs(observed_f - reference_f[::2, ::2])),
                    )
                ),
            }
            row["checks"] = sample_gates(row, c)
            row["passed"] = all(row["checks"].values())
            qa["samples"].append(row)
            for key, value in dict(
                a=observed_a,
                f=observed_f,
                y=y,
                a_reference=reference_a,
                f_reference=reference_f,
                y_reference=ref.values.reshape(n1 + 1, n1 + 1),
                sample_id=sample_id,
                **parameters,
            ).items():
                arrays[key].append(value)
            np.savez_compressed(
                output / "audit_solutions" / f"{sample_id}.npz",
                y_audit=fine.values.reshape(n2 + 1, n2 + 1),
            )
            print(
                f'{sample_id}: pass={row["passed"]} '
                f'L2={ref_fine["relative_l2"]:.3e} '
                f'H1={ref_fine["relative_h1_seminorm"]:.3e}',
                flush=True,
            )
            dump_json(output / "quality.json", qa)
        np.savez_compressed(
            output / f"{split}.npz", **{k: np.asarray(v) for k, v in arrays.items()}
        )
    qa["passed"] = all(r["passed"] for r in qa["samples"]) and all(
        r["passed"] for r in qa["manufactured"].values()
    )
    dump_json(output / "quality.json", qa)
    write_summary(output, c, qa)
    manifest = {
        str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    dump_json(output / "manifest.json", manifest)
    report = audit_dataset(output)
    dump_json(output / "reload_audit.json", report)
    if not qa["passed"] or not report["passed"]:
        raise RuntimeError(
            "Pilot quality failed: inspect quality.json and reload_audit.json; no COMPLETE marker"
        )
    (output / "COMPLETE").write_text(
        "Pilot quality and serialized-data audit passed. No models trained.\n"
    )
    print("COMPLETE: pilot quality and reload audit passed; no model training.", flush=True)


def audit_dataset(output):
    """Read-only audit of checksums, raw fields, IDs and stored labels."""
    c = json.loads((output / "config.json").read_text())
    validate_config(c)
    qa = json.loads((output / "quality.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    for path, expected in manifest.items():
        p = output / path
        check(p.is_file() and sha256(p) == expected, "checksum: " + path)
    for name, expected in json.loads((output / "environment.json").read_text())[
        "source_sha256"
    ].items():
        check(sha256(output / "source_snapshot" / name) == expected, "source: " + name)
    n0, n1 = c["observation_n"], c["reference_n"]
    geo0 = np.load(output / "observation_geometry.npz", allow_pickle=False)
    geo1 = np.load(output / "reference_geometry.npz", allow_pickle=False)
    for geo in (geo0, geo1):
        check(abs(geo["lumped_mass_weights"].sum() - 1) < 1e-13, "mass sum")
        check(np.min(geo["lumped_mass_weights"]) > 0, "mass positivity")
        check(np.min(geo["quadrature_reference_weights"]) > 0, "quadrature positivity")
    seen_ids, seen_fields = set(), set()
    stats = {}
    rows = {r["sample_id"]: r for r in qa["samples"]}
    for split, count in c["splits"].items():
        with np.load(output / f"{split}.npz", allow_pickle=False) as data:
            check(len(data["sample_id"]) == count, f"{split}: count")
            for key in ("a", "f", "y", "a_reference", "f_reference", "y_reference"):
                n = n1 if key.endswith("_reference") else n0
                check(data[key].shape == (count, n + 1, n + 1), f"{split}/{key}: shape")
                check(
                    data[key].dtype == np.float64 and np.isfinite(data[key]).all(),
                    f"{split}/{key}: precision/finite",
                )
            for index, sid in enumerate(data["sample_id"]):
                sid = str(sid)
                check(sid == f'ch03-p{c["seed"]}-{split}-{index:06d}', "ID convention")
                check(sid not in seen_ids, "duplicate ID: " + sid)
                seen_ids.add(sid)
                digest = hashlib.sha256(
                    data["a"][index].tobytes() + data["f"][index].tobytes()
                ).hexdigest()
                check(digest not in seen_fields, "duplicate input: " + sid)
                seen_fields.add(digest)
                params = draw_parameters(c, SPLIT_CODES[split], index)
                for key in params:
                    check(np.array_equal(params[key], data[key][index]), sid + ": parameter stream")
                field = FieldPair(params, c)
                for suffix, geo in (("", geo0), ("_reference", geo1)):
                    for key, fn in (("a", field.coefficient), ("f", field.source)):
                        check(
                            np.max(np.abs(fn(geo["coordinates"]) - data[key + suffix][index]))
                            < 1e-13,
                            sid + ": raw field reconstruction",
                        )
                ref = data["y_reference"][index]
                check(np.array_equal(data["y"][index], ref[::2, ::2]), sid + ": observation labels")
                check(
                    np.max(np.abs(np.concatenate((ref[0], ref[-1], ref[:, 0], ref[:, -1]))))
                    <= c["quality_gates"]["boundary_max"],
                    sid + ": boundary",
                )
                check(
                    sid in rows and all(sample_gates(rows[sid], c).values()),
                    sid + ": quality gates",
                )
            weights = geo0["lumped_mass_weights"]
            stats[split] = {
                "count": count,
                "a_min": float(data["a"].min()),
                "a_max": float(data["a"].max()),
                "f_min": float(data["f"].min()),
                "f_max": float(data["f"].max()),
                "f_rms_min": float(np.sqrt(np.sum(data["f"] ** 2 * weights, axis=(1, 2))).min()),
                "f_rms_max": float(np.sqrt(np.sum(data["f"] ** 2 * weights, axis=(1, 2))).max()),
            }
    check(len(rows) == sum(c["splits"].values()), "quality row count")
    check(all(r["passed"] for r in qa["manufactured"].values()), "manufactured convergence")
    return {
        "passed": not failures,
        "failures": failures,
        "unique_ids": len(seen_ids),
        "unique_input_pairs": len(seen_fields),
        "manifest_files": len(manifest),
        "split_statistics": stats,
    }


def write_summary(output, c, qa):
    rows = qa["samples"]
    maximum = lambda key, sub: max(r[key][sub] for r in rows)
    lines = [
        "# 第三章公共数据小批量质量报告",
        "",
        f'质量门槛：{"通过" if qa["passed"] else "未通过"}。仅为试验批次，无模型训练。',
        "",
        f'划分：{c["splits"]}；公共/参考/加密核查网格：'
        f'{c["observation_n"]+1} / {c["reference_n"]+1} / {c["audit_n"]+1} 个节点/轴。',
        "",
        "| 检查 | 全样本最大值 |",
        "|---|---:|",
        f'| 参考解 vs 加密解：相对 L2 | {maximum("reference_vs_audit", "relative_l2"):.6e} |',
        f'| 参考解 vs 加密解：相对 H1 半范数 | {maximum("reference_vs_audit", "relative_h1_seminorm"):.6e} |',
        f'| 公共网格 P1 重构 vs 加密解：相对 L2 | {maximum("observation_vs_audit", "relative_l2"):.6e} |',
        f'| 积分加密导致的解变化：相对 L2 | {maximum("quadrature_comparison", "relative_l2"):.6e} |',
        f'| 输入跨网格一致性 | {max(r["cross_grid_input_max"] for r in rows):.6e} |',
        f'| 相对代数残差 | {max(s["relative_algebraic_residual"] for r in rows for s in r["solves"]):.6e} |',
        f'| 边界最大值 | {max(s["boundary_max"] for r in rows for s in r["solves"]):.6e} |',
        "",
        "解析制造解独立核查（常系数和线性变系数，8/16/32/64 单元/轴）：",
        "",
    ]
    for name, item in qa["manufactured"].items():
        orders = item["last_refinement_orders"]
        lines.append(
            f'- {name}：L2 阶 {orders["relative_l2"]:.4f}，H1 半范数阶 {orders["relative_h1_seminorm"]:.4f}；通过={item["passed"]}。'
        )
    lines += [
        "",
        "上述 L2/H1 由单元内部积分计算，不使用纯节点误差。网格差是经验精度估计，不能充当连续问题严格误差界。",
        "公共 y 是参考 P1 解在公共节点的取值；公共网格 P1 重构另有表示误差，不能混同为参考解误差。",
        "全部参数为试验值，尚未批准正式分布、数据规模、模型架构、导数标签或迭代残差扩展。",
        "具体概率分布和字段语义见 docs/CH03_DATA_ZH.md；逐样本记录见 quality.json。",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--config", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        generate(args.config, args.output)
    else:
        report = audit_dataset(args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
