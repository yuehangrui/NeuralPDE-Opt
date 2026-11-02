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
        "# Chapter 4: unified delivery v1",
        "",
        "Completed 16 stochastic training workflows x 3 seeds, totaling 48 runs; the 9 reused seed0 runs retain provenance and byte hashes.",
        "Pure-neural and output-head-corrected stages are reported separately, with 72 frozen checkpoint versions. Classical algorithms also undergo a 32/64/128/256-grid study.",
        "A new 64-instance test set was generated only after all model selections were frozen. All 45 problem-family model versions underwent one formal evaluation; state surrogates perform online control on the same first 8 instances.",
        "All 72 versions were additionally checked on the same original instance using order-96 Gauss quadrature and FEM grids 128/256.",
        "",
        "Fixed sine features, training-only linear heads, and fully nonlinear networks are distinguished explicitly. A common delivery standard does not imply equal parameter counts or training costs, or an architecture ranking.",
        "Bilevel, Deep Uzawa, and FEM-NN baselines retain the declared fixed sine features and trainable heads. Their three seeds test distinct initializations, not general deep nonconvex training.",
        "",
        "## Same original instance: three-seed mean +/- sample standard deviation",
        "",
        "| Method | Model stage | Control L2 error (%) | Worst seed (%) | FEM256 relative reduced-gradient norm |",
        "|---|---|---:|---:|---:|",
    ]
    for r in table:
        lines.append(
            f"| {r['method']} | {r['variant']} | {100*r['fixed_control_mean']:.4g} ± {100*r['fixed_control_std']:.3g} | {100*r['fixed_control_max']:.4g} | {r['fem256_gradient_mean']:.4g} |"
        )
    lines += [
        "",
        "FEM gradients include discretization error; models close to the exact spectral solution can still have nonzero FEM diagnostics. Objective differences against the P1 reference retain their signs across different control trial spaces and do not establish superiority to the continuous optimum.",
        "",
        "## New held-out problems: control comparison on the same 8 instances",
        "",
        "| Method | Model stage | Control L2 error (%, three-seed mean +/- std) |",
        "|---|---|---:|",
    ]
    for r in table:
        if "matched_8_control_mean" in r:
            lines.append(
                f"| {r['method']} | {r['variant']} | {100*r['matched_8_control_mean']:.4g} ± {100*r['matched_8_control_std']:.3g} |"
            )
    lines += [
        "",
        "See assessment for state/direct-control metrics over all 64 cases, per-sample objectives and gradients, and online iteration trajectories. The table above uses exactly the same 8 cases.",
        "",
        "## Delivery and limitations",
        "",
        "- Configurations/budgets/histories/checkpoints: training/; frozen selection and source identity: SELECTION_FROZEN.json.",
        "- New test set and nonoverlap evidence: heldout/; independent predictions and physical metrics: assessment/; audit: DELIVERY_AUDIT.json.",
        "- Unified entry point: chapters/ch04_delivery/cli.py; both from-scratch training and explicit seed0 reuse are supported.",
        "- DELIVERY_COMPLETE means the required training, evaluation, and audit are complete, not that every method reaches the same accuracy or strict optimality.",
        "- Previous models, data, results, and frozen sources are retained; semilinear/parabolic extensions remain deferred.",
        "",
        "![Control errors on the shared original instance](fixed_control_comparison.png)",
        "",
        "![Control errors on the same 8 held-out cases](heldout_control_comparison.png)",
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
