"""Fixed HINTS blending and book Algorithm 2 with a residual safeguard.

Book/code notation:
Book (3.43) and Algorithm 2: r_h=rhs-matrix@y. The weighted norm is sqrt(r_h^T*D^{-1}*r_h). eta is neural-step acceptance, omega is Jacobi damping, and rho certifies classical contraction.
"""

import time
import numpy as np
from scipy import sparse


def validate_system(matrix, rhs):
    """The D^-1 residual contraction proof requires a symmetric matrix."""
    if not sparse.issparse(matrix):
        raise ValueError("Expected a scipy sparse matrix")
    rhs = np.asarray(rhs, dtype=np.float64)
    if rhs.ndim != 1 or not len(rhs) or matrix.shape != (len(rhs), len(rhs)):
        raise ValueError("Expected a nonempty 1D rhs and matching square matrix")
    matrix = matrix.astype(np.float64).tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if (
        not np.isfinite(matrix.data).all()
        or not np.isfinite(rhs).all()
        or np.any(matrix.diagonal() <= 0)
    ):
        raise ValueError("Expected finite data and positive diagonal")
    skew = matrix - matrix.T
    # P1 assembly is exactly symmetric; allow only floating-point roundoff.
    scale = np.max(abs(matrix.data))
    if skew.nnz and np.max(abs(skew.data)) > 32 * np.finfo(float).eps * scale:
        raise ValueError("Symmetry required for the weighted contraction certificate")
    return matrix, rhs


def contraction_bound(matrix, omega):
    diagonal = matrix.diagonal()
    radius = np.max((np.asarray(abs(matrix).sum(1)).ravel() - diagonal) / diagonal)
    low, high = 1 - radius, 1 + radius
    if low <= 0 or high * omega >= 2:
        raise ValueError("Classical contraction is not certified by this diagonal-dominance bound")
    return float(max(abs(1 - omega * low), abs(1 - omega * high))), float(low), float(high)


def solve(
    matrix,
    rhs,
    corrector=None,
    mode="safeguarded",
    period=10,
    eta=0.9,
    omega=2 / 3,
    rtol=1e-8,
    atol=1e-12,
    maxiter=10000,
    trace=False,
):
    matrix, rhs = validate_system(matrix, rhs)
    if mode not in ["jacobi", "fixed", "safeguarded"] or (mode != "jacobi" and corrector is None):
        raise ValueError("Invalid hybrid mode")
    if (
        type(period) is not int
        or period < 1
        or not 0 < eta < 1
        or not 0 < omega < 2
        or type(maxiter) is not int
        or maxiter < 0
        or not np.isfinite([rtol, atol]).all()
        or min(rtol, atol) < 0
    ):
        raise ValueError("Invalid solver controls")
    diagonal = matrix.diagonal()
    if (
        matrix.shape != (len(rhs), len(rhs))
        or not np.isfinite(rhs).all()
        or not np.isfinite(matrix.data).all()
        or np.any(diagonal <= 0)
    ):
        raise ValueError("Invalid system")
    rho, low, high = contraction_bound(matrix, omega)
    norm = lambda r: float(np.sqrt(np.sum(r * r / diagonal)))
    rhs_norm = norm(rhs)
    tolerance = atol + rtol * rhs_norm
    y = np.zeros_like(rhs)
    r = rhs.copy()
    residual_norm = rhs_norm
    count = {
        "iterations": 0,
        "neural_calls": 0,
        "accepted_neural": 0,
        "rejected_neural": 0,
        "classical_steps": 0,
        "matrix_vector_products": 0,
    }
    history = [residual_norm] if trace else []
    events = []
    max_classical_ratio = 0.0
    max_accepted_ratio = 0.0
    failure = None
    start = time.perf_counter()
    for k in range(maxiter):
        if residual_norm <= tolerance:
            break
        used_neural = False
        before = residual_norm
        if mode != "jacobi" and k % period == 0:
            count["neural_calls"] += 1
            correction = np.asarray(corrector(r.copy()), dtype=np.float64)
            finite = correction.shape == rhs.shape and np.isfinite(correction).all()
            if finite:
                trial_y = y + correction
                trial_residual = rhs - matrix @ trial_y
                count["matrix_vector_products"] += 1
                trial_norm = norm(trial_residual)
                finite = np.isfinite(trial_norm)
            else:
                trial_norm = float("inf")
            accepted = finite and (mode == "fixed" or trial_norm <= eta * residual_norm)
            if accepted:
                y, r, residual_norm = trial_y, trial_residual, trial_norm
                used_neural = True
                count["accepted_neural"] += 1
                max_accepted_ratio = max(max_accepted_ratio, trial_norm / before)
            else:
                count["rejected_neural"] += 1
                if mode == "fixed":
                    failure = "nonfinite neural proposal"
                    break
            if trace:
                events.append(
                    {
                        "iteration": k,
                        "accepted": bool(accepted),
                        "ratio": trial_norm / before if np.isfinite(trial_norm) else None,
                    }
                )
        if not used_neural:
            y = y + omega * r / diagonal
            r = rhs - matrix @ y
            count["matrix_vector_products"] += 1
            residual_norm = norm(r)
            count["classical_steps"] += 1
            max_classical_ratio = max(max_classical_ratio, residual_norm / before)
        count["iterations"] = k + 1
        if trace:
            history.append(residual_norm)
        if not np.isfinite(residual_norm) or residual_norm > max(rhs_norm, 1e-30) * 1e12:
            failure = "divergence limit"
            break
    # Includes the last permitted update; no off-by-one nonconvergence label.
    r = rhs - matrix @ y
    count["matrix_vector_products"] += 1
    residual_norm = norm(r)
    converged = bool(np.isfinite(residual_norm) and residual_norm <= tolerance)
    record = {
        **count,
        "converged": converged,
        "failure": failure if failure else (None if converged else "iteration limit"),
        "final_residual": residual_norm,
        "rhs_norm": rhs_norm,
        "tolerance": tolerance,
        "relative_residual": residual_norm / max(rhs_norm, 1e-30),
        "classical_contraction_bound": rho,
        "scaled_eigenvalue_bounds": [low, high],
        "max_classical_observed_ratio": max_classical_ratio,
        "max_accepted_neural_ratio": max_accepted_ratio,
        "residual_history": history,
        "neural_events": events,
        "elapsed_seconds": time.perf_counter() - start,
    }
    return y, record


def residual_coordinates(basis_interior, residual):
    """Phi_I^T r = Phi^T M f when f_I=M_II^{-1}r and f_boundary=0."""
    return np.asarray(basis_interior).T @ np.asarray(residual)
