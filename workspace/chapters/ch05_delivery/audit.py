"""Reload selected weights, reproduce predictions and recompute physical diagnostics."""

from pathlib import Path
import json
import numpy as np
import torch
from chapters.ch05.problem import Problem
from chapters.ch05.reference import Grid
from .common import write, sha, quadrature
from .protocol import verify_freeze, load_holdout, old_sources, normalization, check_disjoint
from .training import load
from .data import load as load_data, inputs
from .assessment import instance_fields, predict, action_diagnostics
from .metrics import control, aggregate, prolong


def close(a, b, rtol=1e-8, atol=1e-10):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def references(root):
    problem = Problem()
    records = []
    summary = json.loads((Path(root) / "classical/summary.json").read_text())
    for row in summary:
        n = row["grid"]
        grid = Grid(n)
        with np.load(Path(root) / "classical" / f"grid{n}.npz", allow_pickle=False) as f:
            a = dict(f)
        d = grid.diagnostics(
            a["box_u"], problem.f(grid.points)[..., 0], problem.yd(grid.points)[..., 0], problem
        )
        if d["projected_residual"] > 2e-9:
            raise ValueError("box reference")
        close(grid.solve(a["box_u"] + problem.f(grid.points)[..., 0]), a["box_y"])
        close(grid.solve(a["box_y"] - problem.yd(grid.points)[..., 0]), a["box_p"])
        y, u, mu = a["cnp_y"], a["cnp_u"], a["cnp_mu"]
        close(grid.apply(y), u)
        residual = y - a["cnp_yd"] + problem.alpha * grid.apply(u) + mu
        if (
            np.max(np.abs(residual)) > 8e-8
            or y.max() > problem.state_cap + 1e-11
            or mu.min() < -8e-8
            or np.max(np.abs(mu * (y - problem.state_cap))) > 1e-8
        ):
            raise ValueError("state-constrained reference KKT")
        j = float(0.5 * grid.norm(y - a["cnp_yd"]) ** 2 + 0.5 * problem.alpha * grid.norm(u) ** 2)
        close(j, row["state_constraint"]["objective"])
        records.append(dict(grid=n, box_optimality=True, state_constraint_kkt=True))
    return records


def run(root):
    root = Path(root)
    if not (root / "INDEPENDENT_ASSESSMENT_COMPLETE").is_file():
        raise RuntimeError("assessment required")
    frozen = verify_freeze(root)
    test = load_holdout(root)
    tr, _ = load_data(root / "data", "train")
    va, _ = load_data(root / "data", "val")
    check_disjoint(tr, va)
    check_disjoint(tr, test)
    check_disjoint(va, test)
    torch.set_num_threads(2)
    problem = Problem()
    checks = []
    rows = json.loads((root / "assessment/summary.json").read_text())["rows"]
    if {r["key"] for r in rows} != {e["key"] for e in frozen["entries"]} or len(rows) != 21:
        raise ValueError("assessment inventory")
    for e in frozen["entries"]:
        folder = root / "assessment" / e["key"]
        row = json.loads((folder / "metrics.json").read_text())
        if row["checkpoint_sha256"] != e["sha256"] or row != next(
            r for r in rows if r["key"] == e["key"]
        ):
            raise ValueError("assessment identity")
        model, c, split, b = load(e["checkpoint"])
        instance = c.method in ("osnn", "admm", "cnp")
        with np.load(folder / "fixed_fields.npz", allow_pickle=False) as a:
            fixed = dict(a)
        if instance:
            with np.load(folder / "gauss_fields.npz", allow_pickle=False) as a:
                gauss = dict(a)
            x, w = quadrature(96)
            close(x, gauss["points"])
            close(w[:, 0], gauss["weights"])
            actual = instance_fields(model, c, split, x)
            for k, v in actual.items():
                close(v, gauss[k])
            if c.method == "cnp":
                j = float(
                    0.5
                    * np.sum(
                        w[:, 0]
                        * (
                            (gauss["y"] - problem.sine(x)[:, 0]) ** 2
                            + problem.alpha * gauss["u"] ** 2
                        )
                    )
                )
                close(j, row["fixed"]["quadrature"]["objective"])
            else:
                rel = float(
                    np.sqrt(
                        np.sum(w[:, 0] * (gauss["u"] - problem.u(x)[:, 0]) ** 2)
                        / np.sum(w[:, 0] * problem.u(x)[:, 0] ** 2)
                    )
                )
                close(rel, row["fixed"]["quadrature"]["control_relative_l2"])
        else:
            normalization(model, tr, c.method)
            grid = Grid(c.grid)
            coarse, _ = predict(
                model,
                c,
                problem.f(grid.points)[None, ..., 0],
                problem.yd(grid.points)[None, ..., 0],
            )
        for n in (32, 64, 128):
            grid = Grid(n)
            stored = {
                k.removeprefix(f"grid{n}_"): v
                for k, v in fixed.items()
                if k.startswith(f"grid{n}_")
            }
            if instance:
                actual = instance_fields(
                    model, c, split, grid.points.reshape(-1, 2), derivatives=False
                )
                for k, v in actual.items():
                    close(v.reshape(n - 1, n - 1), stored[k])
            else:
                expected = coarse["u"][0] if n == c.grid else prolong(coarse["u"][0], n)
                close(expected, stored["u"], rtol=3e-4, atol=2e-6)
            with np.load(root / "classical" / f"grid{n}.npz", allow_pickle=False) as a:
                base = dict(a)
            if c.method != "cnp":
                values = control(
                    grid,
                    stored["u"],
                    problem.f(grid.points)[..., 0],
                    problem.yd(grid.points)[..., 0],
                    {k: base["box_" + k] for k in ("y", "u", "p")},
                    stored.get("y"),
                    stored.get("p"),
                )
                for k, v in values.items():
                    close(v, row["fixed"]["grid" + str(n)][k])
            else:
                y, u = stored["y"], stored["u"]
                yd = problem.sine(grid.points)[..., 0]
                recovered = grid.solve(u)
                close(
                    float(0.5 * grid.norm(y - yd) ** 2 + 0.5 * problem.alpha * grid.norm(u) ** 2),
                    row["fixed"]["grid" + str(n)]["raw_objective"],
                )
                close(
                    float(grid.norm(u - base["cnp_u"]) / grid.norm(base["cnp_u"])),
                    row["fixed"]["grid" + str(n)]["control_relative_l2"],
                )
                close(
                    float(np.maximum(recovered - problem.state_cap, 0).max()),
                    row["fixed"]["grid" + str(n)]["recovered_state_violation"],
                )
        if not instance:
            pred, _ = predict(model, c, test["f"], test["yd"])
            with np.load(folder / "heldout_predictions.npz", allow_pickle=False) as a:
                saved = dict(a)
            if not np.array_equal(saved["sample_id"], test["sample_id"]):
                raise ValueError("heldout IDs")
            for k, v in pred.items():
                close(v, saved[k], rtol=3e-4, atol=2e-6)
            values = control(
                Grid(32), saved["u"], test["f"], test["yd"], test, saved.get("y"), saved.get("p")
            )
            for k, v in values.items():
                close(v, saved["metric_" + k])
                close(aggregate({k: v})[k]["mean"], row["heldout"]["metrics"][k]["mean"])
            for n in (64, 128):
                fine = Grid(n)
                fields = inputs({k: v[:8] for k, v in test.items() if k.startswith("coeff_")}, n)
                ref = {k: saved[f"grid{n}_reference_{k}"] for k in ("u", "y", "p")}
                close(ref["y"], fine.solve(ref["u"] + fields["f"]))
                close(ref["p"], fine.solve(ref["y"] - fields["yd"]))
                if (
                    np.max(
                        fine.diagnostics(ref["u"], fields["f"], fields["yd"], problem)[
                            "projected_residual"
                        ]
                    )
                    > 2e-9
                ):
                    raise ValueError("refined reference")
                close(prolong(saved["u"][:8], n), saved[f"grid{n}_u"])
                values = control(fine, saved[f"grid{n}_u"], fields["f"], fields["yd"], ref)
                for k, v in values.items():
                    close(v, saved[f"grid{n}_metric_{k}"])
            if c.method != "mimo":
                _, acts = action_diagnostics(model, c, test)
                with np.load(folder / "action_diagnostics.npz", allow_pickle=False) as a:
                    for k, v in acts.items():
                        if k.endswith(("_state", "_adjoint", "_jvp")):
                            close(v, a[k], rtol=8e-4, atol=3e-6)
        checks.append(
            dict(
                key=e["key"],
                checkpoint_reloaded=True,
                predictions_reproduced=True,
                physical_metrics_recomputed=True,
            )
        )
        print("audit", e["key"], flush=True)
    result = dict(
        status="passed",
        methods=7,
        seeds=[0, 1, 2],
        training_configurations=21,
        checkpoint_variants=21,
        family_variants=12,
        heldout_count=128,
        checks=checks,
        reference_checks=references(root),
        old_source_protection=old_sources(),
        selections_unchanged=True,
        train_only_normalization=True,
        source_frozen_before_test=True,
    )
    verify_freeze(root)
    write(root / "DELIVERY_AUDIT.json", result)
    (root / "NUMERICAL_AUDIT_COMPLETE").write_text(
        "Checkpoint prediction, physical metrics and reference optimality audit passed.\n"
    )
    return result
