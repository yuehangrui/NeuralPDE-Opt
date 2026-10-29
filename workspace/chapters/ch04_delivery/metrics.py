"""Common physical metrics; these functions never select a trained model.

Book/code notation:
Book 4.6: distinguish predicted y from the state recovered by solving the true PDE for u. control_gradient=alpha*u+p uses that recovered state and true adjoint.
"""

import numpy as np
from chapters.ch04_adaptation.operator_data import ALPHA, EIGENVALUES, kkt


def control_metrics(f, yd, u, y_prediction=None, p_prediction=None):
    f, yd, u = [np.atleast_2d(np.asarray(v, dtype=float)) for v in (f, yd, u)]
    if (
        f.shape != yd.shape
        or f.shape != u.shape
        or f.shape[1] != 17
        or not all(np.isfinite(v).all() for v in (f, yd, u))
    ):
        raise ValueError("invalid coefficient fields")
    y = (f + u) / EIGENVALUES
    ystar, ustar, _ = kkt(f, yd)
    norm = lambda v: np.linalg.norm(v, axis=1)
    rel = lambda e, ref: norm(e) / np.maximum(norm(ref), 1e-14)
    control_gradient = ALPHA * u + (y - yd) / EIGENVALUES
    g0 = (f / EIGENVALUES - yd) / EIGENVALUES
    j = 0.125 * (np.sum((y - yd) ** 2, 1) + ALPHA * np.sum(u * u, 1))
    jstar = 0.125 * (np.sum((ystar - yd) ** 2, 1) + ALPHA * np.sum(ustar * ustar, 1))
    result = dict(
        control_relative_l2=rel(u - ustar, ustar),
        recovered_state_relative_l2=rel(y - ystar, ystar),
        true_objective=j,
        reference_objective=jstar,
        objective_gap=j - jstar,
        objective_gap_relative=(j - jstar) / np.maximum(abs(jstar), 1e-14),
        true_gradient_l2=0.5 * norm(control_gradient),
        true_gradient_relative=rel(control_gradient, g0),
    )
    if y_prediction is not None:
        yp = np.atleast_2d(y_prediction)
        result.update(
            raw_state_relative_l2=rel(yp - ystar, ystar),
            state_residual_relative=rel(EIGENVALUES * yp - u - f, f),
            raw_objective=0.125 * (np.sum((yp - yd) ** 2, 1) + ALPHA * np.sum(u * u, 1)),
        )
        if p_prediction is not None:
            pp = np.atleast_2d(p_prediction)
            result.update(
                adjoint_residual_relative=rel(EIGENVALUES * pp - yp + yd, yd),
                optimality_residual_l2=0.5 * norm(ALPHA * u + pp),
            )
    return result


def state_metrics(rhs, y):
    rhs, y = np.asarray(rhs), np.asarray(y)
    truth = rhs / EIGENVALUES
    return dict(
        state_relative_l2=np.linalg.norm(y - truth, axis=1) / np.linalg.norm(truth, axis=1),
        state_residual_relative=np.linalg.norm(EIGENVALUES * y - rhs, axis=1)
        / np.linalg.norm(rhs, axis=1),
    )


def aggregate(values):
    return {
        key: dict(mean=float(np.mean(a)), std=float(np.std(a)), max=float(np.max(a)))
        for key, a in values.items()
    }
