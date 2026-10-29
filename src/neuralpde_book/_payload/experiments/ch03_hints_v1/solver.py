"""Fixed HINTS blending and book Algorithm 2 with a residual safeguard.

Book/code notation:
Book 3.5: a neural correction solves a residual equation. The retained public safeguarded solver is chapters/ch03/hybrid.py; this file preserves the historical mathematical kernel.
"""

import time
import numpy as np


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
    rhs = np.asarray(rhs, dtype=np.float64)
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
    bnorm = norm(rhs)
    tol = atol + rtol * bnorm
    y = np.zeros_like(rhs)
    r = rhs.copy()
    rn = bnorm
    count = {
        "iterations": 0,
        "neural_calls": 0,
        "accepted_neural": 0,
        "rejected_neural": 0,
        "classical_steps": 0,
        "matrix_vector_products": 0,
    }
    history = [rn] if trace else []
    events = []
    max_classical_ratio = 0.0
    max_accepted_ratio = 0.0
    failure = None
    start = time.perf_counter()
    for k in range(maxiter):
        if rn <= tol:
            break
        used_neural = False
        before = rn
        if mode != "jacobi" and k % period == 0:
            count["neural_calls"] += 1
            delta = np.asarray(corrector(r), dtype=np.float64)
            finite = delta.shape == rhs.shape and np.isfinite(delta).all()
            if finite:
                trial = y + delta
                rt = rhs - matrix @ trial
                count["matrix_vector_products"] += 1
                nt = norm(rt)
                finite = np.isfinite(nt)
            else:
                nt = float("inf")
            accepted = finite and (mode == "fixed" or nt <= eta * rn)
            if accepted:
                y, r, rn = trial, rt, nt
                used_neural = True
                count["accepted_neural"] += 1
                max_accepted_ratio = max(max_accepted_ratio, nt / before)
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
                        "ratio": nt / before if np.isfinite(nt) else None,
                    }
                )
        if not used_neural:
            y = y + omega * r / diagonal
            r = rhs - matrix @ y
            count["matrix_vector_products"] += 1
            rn = norm(r)
            count["classical_steps"] += 1
            max_classical_ratio = max(max_classical_ratio, rn / before)
        count["iterations"] = k + 1
        if trace:
            history.append(rn)
        if not np.isfinite(rn) or rn > max(bnorm, 1e-30) * 1e12:
            failure = "divergence limit"
            break
    # Includes the last permitted update; no off-by-one nonconvergence label.
    r = rhs - matrix @ y
    count["matrix_vector_products"] += 1
    rn = norm(r)
    converged = bool(np.isfinite(rn) and rn <= tol)
    record = {
        **count,
        "converged": converged,
        "failure": failure if failure else (None if converged else "iteration limit"),
        "final_residual": rn,
        "rhs_norm": bnorm,
        "tolerance": tol,
        "relative_residual": rn / max(bnorm, 1e-30),
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
