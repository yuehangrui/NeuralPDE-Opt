"""Per-problem feasibility, optimality, objective, active-set and state diagnostics.

Book/code notation:
Book 5.5: projected optimality uses physical p. Feasible z, inner u, predicted state and true recovered state must not be silently interchanged in ADMM diagnostics.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from chapters.ch05.reference import Grid
from chapters.ch05.problem import Problem


def aggregate(values):
    return {
        k: dict(
            mean=float(np.mean(v)),
            std=float(np.std(v)),
            max=float(np.max(v)),
            p95=float(np.quantile(v, 0.95)),
        )
        for k, v in values.items()
    }


def control(g, u, f, yd, reference, y_prediction=None, p_prediction=None):
    problem = Problem()
    u = np.atleast_3d(u) if np.asarray(u).ndim < 2 else np.asarray(u, dtype=float)
    y, adj, grad = g.recover(u, f, yd, problem.alpha)
    d = g.diagnostics(u, f, yd, problem)
    refj = (
        0.5 * g.norm(reference["y"] - yd) ** 2 + 0.5 * problem.alpha * g.norm(reference["u"]) ** 2
    )
    relative = lambda v, r: g.norm(v - r) / np.maximum(g.norm(r), 1e-12)
    result = dict(
        control_relative_l2=relative(u, reference["u"]),
        recovered_state_relative_l2=relative(y, reference["y"]),
        true_objective=d["objective"],
        reference_objective=refj,
        objective_gap=d["objective"] - refj,
        objective_gap_relative=(d["objective"] - refj) / np.maximum(abs(refj), 1e-14),
        projected_residual_l2=d["projected_residual"],
        projected_residual_relative=d["projected_residual"] / np.maximum(g.norm(u), 1e-12),
        bound_violation=d["bound_violation"],
        recovered_state_residual=d["state_residual"],
        recovered_adjoint_residual=d["adjoint_residual"],
    )
    for tol in (1e-4, 1e-3, 1e-2):
        classify = lambda v: np.where(
            v <= problem.lower + tol, -1, np.where(v >= problem.upper - tol, 1, 0)
        )
        labels = classify(u)
        truth = classify(reference["u"])
        result["active_mismatch_" + str(tol)] = np.mean(labels != truth, axis=(-2, -1))
        result["active_lower_fraction_" + str(tol)] = np.mean(labels == -1, axis=(-2, -1))
        result["active_upper_fraction_" + str(tol)] = np.mean(labels == 1, axis=(-2, -1))
    if y_prediction is not None:
        yp = np.asarray(y_prediction)
        result.update(
            raw_state_relative_l2=relative(yp, reference["y"]),
            raw_state_residual=g.norm(g.apply(yp) - u - f),
            raw_objective=0.5 * g.norm(yp - yd) ** 2 + 0.5 * problem.alpha * g.norm(u) ** 2,
        )
        if p_prediction is not None:
            pp = np.asarray(p_prediction)
            result.update(
                raw_adjoint_relative_l2=relative(pp, reference["p"]),
                raw_adjoint_residual=g.norm(g.apply(pp) - yp + yd),
                raw_projection_residual=g.norm(
                    u - np.clip(-pp / problem.alpha, problem.lower, problem.upper)
                ),
            )
    return result


def prolong(u, n):
    u = np.asarray(u)
    single = u.ndim == 2
    if single:
        u = u[None]
    m = u.shape[-1] + 1
    t = np.arange(m + 1) / m
    target = Grid(n).points.reshape(-1, 2)
    # Edge extension affects interpolation near the boundary; no zero-control BC is imposed.
    pad = np.pad(u, [(0, 0), (1, 1), (1, 1)], mode="edge")
    out = np.stack(
        [
            RegularGridInterpolator((t, t), a, bounds_error=True)(target).reshape(n - 1, n - 1)
            for a in pad
        ]
    )
    return out[0] if single else out
