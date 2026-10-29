"""CSV tables and publication-friendly figures from saved runs."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.ticker import NullFormatter
from .models import CoordinateMLP
from .problems import Poisson
from .training import save_json

LABELS = {"pinn": "PINN", "ritz": "Deep Ritz", "vpinn": "VPINN", "fem": "P1 FEM"}
COLORS = {"pinn": "#0072B2", "ritz": "#D55E00", "vpinn": "#009E73", "fem": "#666666"}


def load_model(folder, row):
    model = CoordinateMLP(Poisson(row["dim"]), row["width"], row["depth"]).double()
    model.load_state_dict(torch.load(folder/"model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict(model, points):
    with torch.no_grad():
        return model(torch.as_tensor(points, dtype=torch.float64)).numpy().reshape(-1)


def savefig(fig, folder, name):
    fig.savefig(folder/f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(folder/f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate(output):
    output = Path(output)
    folder = output/"figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.constrained_layout.use": True})
    pairs = [(p.parent, json.loads(p.read_text())) for p in sorted(output.glob("poisson*d_*_seed*/metrics.json"))]
    fem = [json.loads(p.read_text()) for p in sorted((output/"fem").glob("*.json"))]
    rows = [row for _, row in pairs]
    columns = ["problem", "method", "seed", "l2_rel", "h1_semi_rel", "boundary_max", "energy_gap",
               "strong_residual_l2", "weak_residual_enriched", "training_seconds", "objective_evaluations",
               "n_params", "training_points", "quadrature_l2_relative_change", "quadrature_h1_relative_change"]
    with (output/"metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if fem:
        with (output/"fem_metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=sorted(fem[0]))
            writer.writeheader()
            writer.writerows(fem)
    aggregate = []
    for dim in (1, 2):
        for method in ("pinn", "ritz", "vpinn"):
            subset = [r for r in rows if r["dim"] == dim and r["method"] == method]
            if not subset:
                continue
            result = dict(dim=dim, method=method, seeds=len(subset))
            for key in ("l2_rel", "h1_semi_rel", "training_seconds", "strong_residual_l2", "weak_residual_enriched"):
                vals = [r[key] for r in subset]
                result[key+"_mean"] = float(np.mean(vals))
                result[key+"_std"] = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.
            aggregate.append(result)
    save_json(output/"aggregate.json", aggregate)

    # Independent solution errors against accumulated optimization wall time.
    for dim in (1, 2):
        subset = [(p, r) for p, r in pairs if r["dim"] == dim]
        if not subset:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        seen = set()
        for p, row in subset:
            hist = json.loads((p/"history.json").read_text())
            method = row["method"]
            label = LABELS[method] if method not in seen else None
            for ax, key, title in zip(axes, ("l2_rel", "h1_semi_rel"), ("Relative L2 error", "Relative H1 seminorm error")):
                ax.semilogy([h["training_seconds"] for h in hist], [h[key] for h in hist],
                            color=COLORS[method], alpha=.8, linewidth=1.2, label=label)
                ax.set(xlabel="Optimization time (s)", ylabel=title)
                ax.grid(alpha=.2)
            seen.add(method)
        axes[0].legend()
        fig.suptitle(f"{dim}D Poisson: independent validation (each line is one seed)")
        savefig(fig, folder, f"poisson{dim}d_training")

    # Representative seed is fixed at 0, never selected by smallest error.
    selected = { (r["dim"], r["method"]): (p, r) for p, r in pairs if r["seed"] == 0 }
    if any(dim == 1 for dim, _ in selected):
        x = np.linspace(0, 1, 801)[:, None]
        exact = Poisson(1).exact(x).ravel()
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].plot(x[:, 0], exact, "k--", label="Exact", linewidth=2)
        for method in ("pinn", "ritz", "vpinn"):
            if (1, method) not in selected:
                continue
            p, row = selected[1, method]
            pred = predict(load_model(p, row), x)
            axes[0].plot(x[:, 0], pred, color=COLORS[method], label=LABELS[method])
            axes[1].plot(x[:, 0], pred-exact, color=COLORS[method], label=LABELS[method])
        frows = [r for r in fem if r["dim"] == 1]
        if frows:
            best = max(frows, key=lambda r: r["n"])
            data = np.load(output/"fem"/f"poisson1d_n{best['n']}.npz")
            pred = np.interp(x[:, 0], data["nodes"][:, 0], data["values"])
            axes[0].plot(x[:, 0], pred, color=COLORS["fem"], linestyle=":", label=f"P1 FEM n={best['n']}")
            axes[1].plot(x[:, 0], pred-exact, color=COLORS["fem"], linestyle=":")
        axes[0].set(xlabel="x", ylabel="y(x)", title="Solutions, seed 0")
        axes[1].set(xlabel="x", ylabel="Prediction minus exact", title="Pointwise error")
        axes[0].legend()
        for ax in axes:
            ax.grid(alpha=.2)
        savefig(fig, folder, "poisson1d_solutions")

    if any(dim == 2 for dim, _ in selected):
        t = np.linspace(0, 1, 121)
        xx, yy = np.meshgrid(t, t, indexing="ij")
        x = np.stack((xx.ravel(), yy.ravel()), -1)
        exact = Poisson(2).exact(x).reshape(xx.shape)
        predictions = []
        for method in ("pinn", "ritz", "vpinn"):
            if (2, method) in selected:
                p, row = selected[2, method]
                predictions.append((LABELS[method], predict(load_model(p, row), x).reshape(xx.shape)))
        frows = [r for r in fem if r["dim"] == 2]
        if frows:
            best = max(frows, key=lambda r: r["n"])
            data = np.load(output/"fem"/f"poisson2d_n{best['n']}.npz")
            triang = mtri.Triangulation(data["nodes"][:, 0], data["nodes"][:, 1], data["elements"])
            interp = mtri.LinearTriInterpolator(triang, data["values"])
            predictions.append((f"P1 FEM n={best['n']}", np.asarray(interp(xx, yy))))
        fig, axes = plt.subplots(2, len(predictions), figsize=(3.2*len(predictions), 6), squeeze=False)
        max_error = max(float(np.abs(pred-exact).max()) for _, pred in predictions)
        for j, (label, pred) in enumerate(predictions):
            im = axes[0, j].pcolormesh(xx, yy, pred, shading="auto", cmap="viridis", vmin=0, vmax=1)
            err = axes[1, j].pcolormesh(xx, yy, np.abs(pred-exact), shading="auto", cmap="magma", vmin=0, vmax=max_error)
            axes[0, j].set_title(label)
            axes[1, j].set_title(f"Absolute error; max={np.abs(pred-exact).max():.1e}")
            for ax in axes[:, j]:
                ax.set(xlabel="x1", ylabel="x2", aspect="equal")
        fig.colorbar(im, ax=list(axes[0]), shrink=.8)
        fig.colorbar(err, ax=list(axes[1]), shrink=.8)
        fig.suptitle("2D Poisson: solutions and errors, neural seed 0")
        savefig(fig, folder, "poisson2d_solutions")

    if fem:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for dim, ax in zip((1, 2), axes):
            subset = sorted([r for r in fem if r["dim"] == dim], key=lambda r: r["n"])
            if not subset:
                continue
            h = np.array([1/r["n"] for r in subset])
            e = np.array([r["l2_abs"] for r in subset])
            g = np.array([r["h1_semi_abs"] for r in subset])
            ax.loglog(h, e, "o-", label="L2 error")
            ax.loglog(h, g, "s-", label="H1 seminorm error")
            ax.loglog(h, e[-1]*(h/h[-1])**2, "k:", label="Order 2")
            ax.loglog(h, g[-1]*(h/h[-1]), "k--", label="Order 1")
            ax.set(xlabel="h", ylabel="Absolute error", title=f"{dim}D P1 FEM convergence")
            ax.set_xticks(h, [f"1/{r['n']}" for r in subset])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.legend()
            ax.grid(alpha=.2)
        savefig(fig, folder, "fem_convergence")

    text = ["# Chapter 2 experiment results", "", f"Completed neural runs: {len(rows)}.", "",
            "| Problem | Method | Seeds | Relative L2 (mean +/- std) | Relative H1 seminorm (mean +/- std) | Training seconds (mean) |",
            "|---|---|---:|---:|---:|---:|"]
    for r in aggregate:
        text.append(f"| {r['dim']}D | {LABELS[r['method']]} | {r['seeds']} | {r['l2_rel_mean']:.3e} +/- {r['l2_rel_std']:.2e} | "
                    f"{r['h1_semi_rel_mean']:.3e} +/- {r['h1_semi_rel_std']:.2e} | {r['training_seconds_mean']:.1f} |")
    text.extend(["", "## FEM reference", "", "| Problem | Intervals per axis | Interior DOFs | Relative L2 | Relative H1 seminorm | Assembly + solve seconds |",
                 "|---|---:|---:|---:|---:|---:|"])
    for r in sorted(fem, key=lambda r: (r["dim"], r["n"])):
        text.append(f"| {r['dim']}D | {r['n']} | {r['dofs']} | {r['l2_rel']:.3e} | {r['h1_semi_rel']:.3e} | {r['solve_seconds']:.4f} |")
    if rows:
        text.extend(["", "## Validation", "",
                     f"Maximum boundary mismatch: {max(r['boundary_max'] for r in rows):.3e}.",
                     f"Maximum relative L2 change after quadrature refinement: {max(r['quadrature_l2_relative_change'] for r in rows):.3e}.",
                     f"Maximum relative H1 change after quadrature refinement: {max(r['quadrature_h1_relative_change'] for r in rows):.3e}.",
                     f"Maximum energy-identity defect: {max(r['energy_identity_defect'] for r in rows):.3e}."])
    text.extend(["", "## Interpretation", "",
                 "Neural runs use equal architectures, seeds, quadrature and nominal optimization budgets within each dimension. The cost of a step differs by formulation; actual objective evaluations and synchronized optimization wall time are recorded. Timing excludes validation, checkpoint I/O and setup. FEM timing includes assembly and solve on CPU; neural training uses the recorded device. This is not an equal-hardware speedup claim.", "",
                 "The VPINN objective is the finite-test dual norm r^T G^{-1} r with P1 hat test functions. It is a norm-aware weak-residual variant, not the unweighted original VPINN loss. Additional test functions and finer quadrature are used for validation. A small finite-test loss is not a standalone continuum error certificate.", "",
                 "Errors are computed against analytic solutions. H1 denotes the gradient seminorm. The energy-gap identity applies because all trial functions satisfy the same essential boundary data. The pointwise strong residual is reported for neural fields only; elementwise Laplacians are not a valid global strong-residual measure for P1 FEM.", "",
                 "Solution-field figures always use seed 0. Standard deviations use ddof=1 when multiple seeds exist. No claim of a global optimum, exact theoretical error decomposition or generic superiority is made.", ""])
    (output/"summary.md").write_text("\n".join(text), encoding="utf-8")
    print(f"REPORT {output/'summary.md'}", flush=True)
