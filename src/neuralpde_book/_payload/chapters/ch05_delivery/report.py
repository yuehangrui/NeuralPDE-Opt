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
        "# Chapter 5: unified delivery v1",
        "",
        "Seven ported workflows x seed0/1/2: 21 formal training runs and 21 validation-selected models. Instance methods share training quadrature and validation points; problem-family methods share train1024/val128.",
        "A new 128-instance test set was generated only after model selection and sources were frozen. All 12 problem-family models use every test sample, with grid-64/128 refinement checks on the same first 8 cases.",
        "All models also undergo spatial checks on the original instance. OSNN, ADMM, and CNP use order-96 Gauss quadrature, 129 boundary points, and finite-difference grids 32/64/128. CNP is a separate state-constrained problem and is not ranked against the box-constrained control example.",
        "",
        "## Fixed control-constrained example: three-seed mean +/- sample standard deviation",
        "",
        "| Method | Gauss relative control L2 error (%) | Control error against the continuous manufactured solution at FD128 points (%) | FD128 true projected-residual L2 |",
        "|---|---:|---:|---:|",
    ]
    for t in table:
        if t["method"] == "cnp":
            continue
        gauss = (
            display(t["gauss_control"], 100)
            if "gauss_control" in t
            else "Fixed-grid operator; no continuous-network Gauss value"
        )
        lines.append(
            f"| {t['label']} | {gauss} | {display(t['fixed_grid128_control'],100)} | {display(t['fixed_grid128_projected_residual'])} |"
        )
    lines += [
        "",
        "Fine-grid controls for operators use bilinear interpolation with edge extension from the original grid. This independently checks the delivered control numerically; it is not network resolution transfer.",
        "",
        "## New 128-instance holdout: shared problems and independent true-PDE solves",
        "",
        "| Method | Mean relative control L2 (%, three-seed mean +/- std) | Mean relative true-objective difference (%) | Mean true projected-residual L2 | Mean active-set misclassification (%, tol=1e-3) |",
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
        "## CNP state-upper-bound example",
        "",
        f"Order-96 independent-quadrature objective: {display(t['gauss_objective'])}; FD128 constrained-QP reference objective: {t['reference_objective']:.10g}.",
        f"Order-96 objective minus the order-80 objective used during selection: {display(t['quadrature_difference_from_validation'])}.",
        f"Relative L2 error against the FD128 reference control: {display(t['fixed_grid128_control'],100)}%; neural-state upper-bound violation: {display(t['state_violation'])}; violation after a true-PDE state solve: {display(t['recovered_violation'])}.",
        "Objective differences between quadrature/discrete spaces retain their signs. Feasibility of the parameterization does not imply optimality for the original continuous problem. The discrete reference is an independent convex QP in state space; feasibility, nonnegative multipliers, stationarity, and complementarity are checked.",
        "",
        "## Training, evaluation, and interpretation limits",
        "",
        "- training/ stores configurations, initial/best/last weights, full selection histories, and actual Adam/L-BFGS counts. ADMM also saves complete histories for reconstructing split variables at arbitrary points.",
        "- Pure projected KKT-FNO, PDOL, and iUzawa F/S retain their actual architectures. This batch uses no post-training linear-head solve or test-set fitting. mimo is a historical entry-point name, not a replication of the original multiplicative MIMOONet.",
        "- PDOL surrogate fixed-point residuals and true-PDE projected residuals are separate. Raw p is the diagnostic reconstruction A_theta(S_theta(u+f)-yd), not the observation-space dual d. Pairing, linearity, and JVP errors of learned actions are reported separately.",
        "- iUzawa uses exact box projection, fixed physical scaling, and the positive-semidefinite structure Q_B. Nonlinear weight sharing does not guarantee a true adjoint, and empirical diagnostics are not convergence proofs.",
        "- Representations, training data, and workloads differ between methods. Timings are reported separately; no architecture ranking or end-to-end speedup is claimed.",
        "- DELIVERY_COMPLETE means budgets, selection, independent evaluation, and audits are complete, not that all models meet one accuracy target or strict optimality.",
        "- This test set has now been viewed; future use is a retest and must not guide changes to this version. Previous chapters, ports, and development validation are retained. Sections 5.3/5.4 and other equation extensions remain deferred.",
        "",
        "Unified entry point: chapters/ch05_delivery/cli.py. Frozen evidence: PRETEST_AUDIT.json, SELECTION_FROZEN.json, heldout/manifest.json, DELIVERY_AUDIT.json.",
        "",
        "![Control errors on the fixed example](fixed_control_comparison.png)",
        "",
        "![Control errors on held-out problems](heldout_control_comparison.png)",
        "",
        "![Validation histories for all workflows](training_histories.png)",
        "",
        "![CNP state and control comparison for fixed seed0](cnp_fields_seed0.png)",
        "",
        "![Four operators with fixed seed0 on the same first test case](heldout_fields_seed0.png)"
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
