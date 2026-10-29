"""Transparent three-seed tables and scientific figures; no architecture ranking."""

from pathlib import Path
import csv, json
import numpy as np
from .common import write, METHODS
from .protocol import verify_freeze

LABELS = {
    "osnn": "OSNN",
    "admm": "ADMM-PINN",
    "cnp": "CNP",
    "mimo": "Projected KKT-FNO",
    "pdol": "PDOL",
    "iuzawa_f": "iUzawa-Net-F",
    "iuzawa_s": "iUzawa-Net-S",
}


def stats(values):
    v = np.asarray(values, dtype=float)
    return dict(
        mean=float(np.mean(v)),
        std=float(np.std(v, ddof=1)),
        max=float(np.max(v)),
        values=v.tolist(),
    )


def run(root):
    root = Path(root)
    verify_freeze(root)
    if json.loads((root / "DELIVERY_AUDIT.json").read_text())["status"] != "passed":
        raise RuntimeError("audit required")
    rows = json.loads((root / "assessment/summary.json").read_text())["rows"]
    table = []
    pretest = {r["key"]: r for r in json.loads((root / "PRETEST_AUDIT.json").read_text())["checks"]}
    for method in METHODS:
        r = sorted([v for v in rows if v["method"] == method], key=lambda v: v["seed"])
        if [v["seed"] for v in r] != [0, 1, 2]:
            raise ValueError("missing seeds")
        statuses = [
            json.loads((root / "training" / v["key"] / "status.json").read_text()) for v in r
        ]
        item = dict(
            method=method,
            label=LABELS[method],
            training_seconds=stats([s["elapsed_seconds"] for s in statuses]),
            parameter_accounting=pretest[r[0]["key"]]["parameters"],
            selected_training_loss=stats([pretest[v["key"]]["selected_training_loss"] for v in r]),
            parameters_real=[s["parameters_real"] for s in statuses],
            counters=[s["counters"] for s in statuses],
            checkpoint_bytes=[(root / "training" / v["key"] / "best.pt").stat().st_size for v in r],
            fixed_grid128_control=stats(
                [
                    (
                        v["fixed"]["grid128"]["control_relative_l2"]
                        if method == "cnp"
                        else v["fixed"]["grid128"]["control_continuous_relative"]
                    )
                    for v in r
                ]
            ),
        )
        if method == "cnp":
            item.update(
                gauss_objective=stats([v["fixed"]["quadrature"]["objective"] for v in r]),
                quadrature_difference_from_validation=stats(
                    [
                        v["fixed"]["quadrature"]["objective"] - v["validation_selection_score"]
                        for v in r
                    ]
                ),
                state_violation=stats([v["fixed"]["grid128"]["state_violation"] for v in r]),
                recovered_violation=stats(
                    [v["fixed"]["grid128"]["recovered_state_violation"] for v in r]
                ),
                reference_objective=r[0]["fixed"]["grid128"]["reference_objective"],
            )
        else:
            item["fixed_grid128_projected_residual"] = stats(
                [v["fixed"]["grid128"]["projected_residual_l2"] for v in r]
            )
            if method in ("osnn", "admm"):
                item["gauss_control"] = stats(
                    [v["fixed"]["quadrature"]["control_relative_l2"] for v in r]
                )
            else:
                item.update(
                    test_control=stats(
                        [v["heldout"]["metrics"]["control_relative_l2"]["mean"] for v in r]
                    ),
                    test_objective_gap=stats(
                        [v["heldout"]["metrics"]["objective_gap_relative"]["mean"] for v in r]
                    ),
                    test_projected_residual=stats(
                        [v["heldout"]["metrics"]["projected_residual_l2"]["mean"] for v in r]
                    ),
                    test_active_mismatch=stats(
                        [v["heldout"]["metrics"]["active_mismatch_0.001"]["mean"] for v in r]
                    ),
                    test_control_sample_max=max(
                        v["heldout"]["metrics"]["control_relative_l2"]["max"] for v in r
                    ),
                    fine128_control=stats(
                        [
                            v["heldout"]["refinement128"]["metrics"]["control_relative_l2"]["mean"]
                            for v in r
                        ]
                    ),
                )
        table.append(item)
    write(
        root / "comparison.json",
        dict(
            rows=table,
            dispersion="sample standard deviation across three seed-level means; within-test metrics also saved per sample",
        ),
    )
    flat = []
    for r in rows:
        accounting = pretest[r["key"]]["parameters"]
        f = dict(
            method=r["method"],
            seed=r["seed"],
            fixed_grid128_control=r["fixed"]["grid128"].get(
                "control_continuous_relative", r["fixed"]["grid128"]["control_relative_l2"]
            ),
            nominal_real_parameters=accounting["nominal_real_parameters"],
            known_inactive_real_parameters=accounting["structurally_inactive_real_parameters"],
        )
        if "heldout" in r:
            f.update({f"test_{k}_mean": v["mean"] for k, v in r["heldout"]["metrics"].items()})
        if r["method"] == "cnp":
            f["gauss_objective"] = r["fixed"]["quadrature"]["objective"]
        flat.append(f)
    columns = sorted({k for row in flat for k in row})
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(flat)

    def display(s, scale=1):
        return f"{s['mean']*scale:.5g} ± {s['std']*scale:.3g}"

    lines = [
        "# 第五章统一交付 v1",
        "",
        "7条迁移流程 × seed0/1/2，共21组正式训练和21个验证集选定模型。实例方法使用共同训练积分和验证点；问题族使用同一train1024/val128。",
        "全部选模和源码冻结后，才生成新128例test；12个问题族模型使用相同全部样本，并在相同前8例上检查64/128加密网格。",
        "所有模型另做原始算例空间复核；OSNN、ADMM、CNP使用96阶Gauss、129点边界及32/64/128差分网格。CNP是独立状态约束问题，不与控制盒约束例横向排名。",
        "",
        "## 固定控制约束算例：三个种子均值 ± 样本标准差",
        "",
        "| 方法 | Gauss控制相对L2误差（%） | FD128点上对连续制造解控制误差（%） | FD128真实投影残差L2 |",
        "|---|---:|---:|---:|",
    ]
    for t in table:
        if t["method"] == "cnp":
            continue
        gauss = (
            display(t["gauss_control"], 100)
            if "gauss_control" in t
            else "固定网格算子，无连续网络Gauss值"
        )
        lines.append(
            f"| {t['label']} | {gauss} | {display(t['fixed_grid128_control'],100)} | {display(t['fixed_grid128_projected_residual'])} |"
        )
    lines += [
        "",
        "算子在细网格上的控制由原网格作边缘延拓的双线性插值得到；这是交付控制的独立数值检查，不是网络分辨率迁移。",
        "",
        "## 新留出128例：相同问题、独立真实PDE重求",
        "",
        "| 方法 | 控制相对L2均值（%，三种子平均 ± 标准差） | 真实目标相对差均值（%） | 真实投影残差L2均值 | 活动集错分均值（%，tol=1e-3） |",
        "|---|---:|---:|---:|---:|",
    ]
    for t in table:
        if "test_control" in t:
            lines.append(
                f"| {t['label']} | {display(t['test_control'],100)} | {display(t['test_objective_gap'],100)} | {display(t['test_projected_residual'])} | {display(t['test_active_mismatch'],100)} |"
            )
    t = next(t for t in table if t["method"] == "cnp")
    lines += [
        "",
        "## CNP状态上界算例",
        "",
        f"96阶独立积分目标：{display(t['gauss_objective'])}；FD128约束QP参考目标：{t['reference_objective']:.10g}。",
        f"96阶目标减去选模时80阶目标的差：{display(t['quadrature_difference_from_validation'])}。",
        f"对FD128参考控制的相对L2误差：{display(t['fixed_grid128_control'],100)}%；神经状态上界违反量：{display(t['state_violation'])}；真实PDE重求状态的违反量：{display(t['recovered_violation'])}。",
        "不同积分/离散空间的目标差保留符号；参数化可行不等于原连续问题已最优。离散参考使用独立状态空间凸QP，其可行性、非负乘子、驻点与互补性均复核。",
        "",
        "## 训练、评测与解释边界",
        "",
        "- training/保存每组配置、初始/最佳/最后权重、完整选模历史和实际Adam/L-BFGS计数。ADMM另保存可重建任意点分裂变量的全部历史。",
        "- pure projected KKT-FNO、PDOL、iUzawa F/S均保留其实际架构；本批没有训练后线性头求解或测试集拟合。mimo是历史入口名，不是原乘法MIMOONet的复刻。",
        "- PDOL的代理固定点残差、真实PDE投影残差分开；raw p为A_theta(S_theta(u+f)-yd)的诊断重构，不是观测空间对偶d。学习作用的配对、线性性与JVP误差单列。",
        "- iUzawa采用精确盒投影、固定物理尺度和半正定结构Q_B；非线性权重共享不提供真实伴随保证，经验诊断也不是收敛证明。",
        "- 各方法表示、训练数据和工作量不同，计时分列，不作架构排名或端到端加速比。",
        "- DELIVERY_COMPLETE表示预算、选模、独立评测和审计齐全，不代表所有模型满足同一精度或严格最优性。",
        "- 本轮test已查看；以后使用它属于复测，不得据此调整本版本。旧章节、迁移版本和开发验证保留，5.3/5.4及其他方程扩展继续后置。",
        "",
        "统一入口：chapters/ch05_delivery/cli.py。冻结证据：PRETEST_AUDIT.json、SELECTION_FROZEN.json、heldout/manifest.json、DELIVERY_AUDIT.json。",
        "",
        "![固定算例控制误差](fixed_control_comparison.png)",
        "",
        "![留出问题控制误差](heldout_control_comparison.png)",
        "",
        "![各流程验证历史](training_histories.png)",
        "",
        "![CNP固定seed0状态与控制对照](cnp_fields_seed0.png)",
        "",
        "![四个算子固定seed0与相同第一例test](heldout_fields_seed0.png)",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figures(root, table)
    return table


def figures(root, table):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for field, name, title in [
        (
            "fixed_grid128_control",
            "fixed_control_comparison",
            "Original box-control case: FD128 checks",
        ),
        ("test_control", "heldout_control_comparison", "128 held-out box-control problems"),
    ]:
        chosen = [t for t in table if field in t and t["method"] != "cnp"]
        fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
        for i, t in enumerate(chosen):
            ax.scatter(
                100 * np.array(t[field]["values"]), np.full(3, i), s=40, color="#2563a6", alpha=0.7
            )
            ax.scatter(100 * t[field]["mean"], i, s=65, color="#bd4939", marker="D")
        ax.set_yticks(range(len(chosen)), [t["label"] for t in chosen])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("Relative control L2 error (%)")
        ax.set_title(title + "\nBlue: individual seeds; red: seed mean")
        ax.grid(axis="x", alpha=0.2)
        fig.savefig(root / (name + ".png"), dpi=180)
        fig.savefig(root / (name + ".svg"))
        plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), layout="constrained")
    for ax, method in zip(axes.flat, METHODS):
        for seed in (0, 1, 2):
            h = json.loads((root / "training" / f"{method}_s{seed}" / "history.json").read_text())
            ax.plot(
                [v["optimizer_updates"] for v in h],
                [max(v["selection_score"], 1e-16) for v in h],
                label=f"seed {seed}",
            )
        ax.set_title(LABELS[method])
        ax.set_yscale("log")
        ax.set_xlabel("Recorded optimizer iterations")
        ax.set_ylabel("Method-specific validation score")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(METHODS) :]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.savefig(root / "training_histories.png", dpi=180)
    fig.savefig(root / "training_histories.svg")
    plt.close(fig)
    field_figures(root)


def field_figures(root):
    import matplotlib.pyplot as plt

    with np.load(root / "classical/grid128.npz", allow_pickle=False) as a:
        reference = dict(a)
    with np.load(root / "assessment/cnp_s0/fixed_fields.npz", allow_pickle=False) as a:
        prediction = dict(a)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout="constrained")
    for row, (field, label) in enumerate([("y", "State"), ("u", "Control")]):
        true = reference["cnp_" + field]
        pred = prediction["grid128_" + field]
        error = pred - true
        low = min(true.min(), pred.min())
        high = max(true.max(), pred.max())
        bound = max(abs(error).max(), 1e-14)
        for col, (v, title) in enumerate(
            [
                (true, "FD128 constrained reference"),
                (pred, "CNP, fixed seed 0"),
                (error, "Prediction - reference"),
            ]
        ):
            ax = axes[row, col]
            im = ax.imshow(
                v.T,
                origin="lower",
                extent=(1 / 128, 127 / 128, 1 / 128, 127 / 128),
                cmap="RdBu_r" if col == 2 else "viridis",
                vmin=-bound if col == 2 else low,
                vmax=bound if col == 2 else high,
                rasterized=True,
            )
            ax.set_title(label + ": " + title, fontsize=10)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(root / "cnp_fields_seed0.png", dpi=180)
    fig.savefig(root / "cnp_fields_seed0.svg")
    plt.close(fig)
    with np.load(root / "heldout/test.npz", allow_pickle=False) as a:
        true = a["u"][0]
    methods = ["mimo", "pdol", "iuzawa_f", "iuzawa_s"]
    predictions = []
    for method in methods:
        with np.load(
            root / "assessment" / f"{method}_s0" / "heldout_predictions.npz", allow_pickle=False
        ) as a:
            predictions.append(a["u"][0])
    errors = [v - true for v in predictions]
    bound = max(max(abs(v).max() for v in errors), 1e-14)
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), layout="constrained")
    for col, (method, pred, error) in enumerate(zip(methods, predictions, errors)):
        for row, (value, label) in enumerate(
            [(true, "Reference"), (pred, "Prediction"), (error, "Error")]
        ):
            ax = axes[row, col]
            im = ax.imshow(
                value.T,
                origin="lower",
                extent=(1 / 32, 31 / 32, 1 / 32, 31 / 32),
                cmap="RdBu_r" if row == 2 else "viridis",
                vmin=-bound if row == 2 else 0.3,
                vmax=bound if row == 2 else 1.0,
                rasterized=True,
            )
            ax.set_title(LABELS[method] + "\n" + label if row == 0 else label, fontsize=10)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Same first held-out case; seed 0 chosen in advance", fontsize=12)
    fig.savefig(root / "heldout_fields_seed0.png", dpi=180)
    fig.savefig(root / "heldout_fields_seed0.svg")
    plt.close(fig)
