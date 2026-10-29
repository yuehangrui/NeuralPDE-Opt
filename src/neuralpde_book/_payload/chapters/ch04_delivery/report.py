"""Unified tables and figures with explicit method/variant/seed separation."""

from collections import defaultdict
from pathlib import Path
import csv
import json
import numpy as np
from .frozen_jobs import write


def run(root):
    root = Path(root)
    if not (root / "DELIVERY_COMPLETE").is_file():
        raise RuntimeError("audit first")
    rows = json.loads((root / "assessment/summary.json").read_text())["rows"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["variant"])].append(row)
    table = []
    for (method, variant), group in groups.items():
        if sorted(r["seed"] for r in group) != [0, 1, 2]:
            raise ValueError("seed group incomplete")
        errors = np.array([r["fixed_instance"]["control_relative_l2"] for r in group])
        gradients = np.array(
            [r["fixed_instance"]["fem"][-1]["reduced_gradient_relative"] for r in group]
        )
        record = dict(
            method=method,
            variant=variant,
            seeds=3,
            fixed_control_mean=float(errors.mean()),
            fixed_control_std=float(errors.std(ddof=1)),
            fixed_control_max=float(errors.max()),
            fem256_gradient_mean=float(gradients.mean()),
        )
        if group[0]["track"] == "problem_family":
            fam = np.array(
                [r["heldout"]["matched_control_8"]["control_relative_l2"]["mean"] for r in group]
            )
            record.update(
                matched_8_control_mean=float(fam.mean()),
                matched_8_control_std=float(fam.std(ddof=1)),
            )
            if "state" in group[0]["heldout"]:
                record["heldout64_state_mean"] = float(
                    np.mean([r["heldout"]["state"]["state_relative_l2"]["mean"] for r in group])
                )
            else:
                record["heldout64_direct_control_mean"] = float(
                    np.mean(
                        [
                            r["heldout"]["direct_control"]["control_relative_l2"]["mean"]
                            for r in group
                        ]
                    )
                )
        table.append(record)
    write(root / "comparison.json", table)
    columns = sorted({k for row in table for k in row})
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    lines = [
        "# 第四章统一交付 v1",
        "",
        "已完成16类随机训练流程 × 3个种子，共48组训练交付；复用的9组seed0结果保留来源与字节哈希。",
        "纯神经/输出头修正等阶段分开，共72个冻结检查点版本。经典算法另做32/64/128/256网格研究。",
        "所有选模冻结后才生成新的64例test，45个问题族模型版本完成一次正式评测；状态代理在相同前8例上做在线控制。",
        "全部72个版本另在同一原始算例、96阶Gauss积分和128/256网格FEM上复核。",
        "",
        "固定正弦特征、训练集线性头、完整非线性网络均明确区分。这是相同交付标准，不是等参数量/等训练代价的架构排名。",
        "双层、Deep Uzawa、FEM–NN仍采用已声明的固定正弦特征与可训练头；其三种子检验实际不同初始化，不代表一般深层非凸训练。",
        "",
        "## 同一原始算例：3种子平均 ± 样本标准差",
        "",
        "| 方法 | 模型阶段 | 控制L2误差（%） | 最差种子（%） | FEM256约化梯度相对范数 |",
        "|---|---|---:|---:|---:|",
    ]
    for r in table:
        lines.append(
            f"| {r['method']} | {r['variant']} | {100*r['fixed_control_mean']:.4g} ± {100*r['fixed_control_std']:.3g} | {100*r['fixed_control_max']:.4g} | {r['fem256_gradient_mean']:.4g} |"
        )
    lines += [
        "",
        "FEM梯度含离散误差，接近精确谱解的模型仍可能有非零FEM诊断值。不同控制试探空间对P1参考的目标差保留符号，不据此宣称优于连续最优解。",
        "",
        "## 新留出问题：相同8例控制比较",
        "",
        "| 方法 | 模型阶段 | 控制L2误差（%，3种子平均 ± 标准差） |",
        "|---|---|---:|",
    ]
    for r in table:
        if "matched_8_control_mean" in r:
            lines.append(
                f"| {r['method']} | {r['variant']} | {100*r['matched_8_control_mean']:.4g} ± {100*r['matched_8_control_std']:.3g} |"
            )
    lines += [
        "",
        "完整64例状态/直接控制指标、各样本目标值和梯度、在线迭代轨迹见assessment；上表严格使用同一8例。",
        "",
        "## 交付与边界",
        "",
        "- 配置/预算/历史曲线/检查点：training/；选模锁定与源码身份：SELECTION_FROZEN.json。",
        "- 新测试集及无重叠证据：heldout/；独立预测和物理指标：assessment/；复核：DELIVERY_AUDIT.json。",
        "- 统一入口：chapters/ch04_delivery/cli.py；从零训练和显式复用seed0均受支持。",
        "- DELIVERY_COMPLETE表示规定训练、评测、审计齐全，不表示每个方法都达到同一精度或严格最优性。",
        "- 旧模型、数据、结果及冻结源码继续保留；半线性/抛物型扩展后置。",
        "",
        "![统一原始算例控制误差](fixed_control_comparison.png)",
        "",
        "![同一8例留出控制误差](heldout_control_comparison.png)",
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for filename, column, title in [
        ("fixed_control_comparison", "fixed_control_mean", "Original instance: control error"),
        (
            "heldout_control_comparison",
            "matched_8_control_mean",
            "Same 8 held-out cases: control error",
        ),
    ]:
        values = [r for r in table if column in r]
        fig, ax = plt.subplots(figsize=(11, max(5, len(values) * 0.3)))
        labels = [r["method"] + " / " + r["variant"] for r in values]
        ax.barh(labels, [max(r[column], 1e-15) for r in values])
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_xlabel("Relative L2 error (ratio), mean over 3 seeds")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(root / (filename + ".png"), dpi=160)
        fig.savefig(root / (filename + ".pdf"))
        plt.close(fig)
    return table
