"""Correct consistent-mass semilinear derivatives for future nonlinear ports.

Residual R=K y+M(y^3)-M u-f_load. A=K+3 M diag(y^2) is
generally nonsymmetric; sparse LU and A.T are required. Each sample uses
its OWN linearization. Loads and nodal controls are distinct arguments.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def jacobian(K, M, y):
    return sparse.csc_matrix(K) + 3 * sparse.csc_matrix(M) @ sparse.diags(np.asarray(y) ** 2)


def state(K, M, u, f_load, tol=1e-11, maxiter=40):
    K, M = sparse.csc_matrix(K), sparse.csc_matrix(M)
    y = splu(K).solve(M @ u + f_load)
    scale = max(1.0, np.linalg.norm(M @ u + f_load))
    residual = lambda z: K @ z + M @ (z**3) - M @ u - f_load
    for _ in range(maxiter):
        r = residual(y)
        if np.linalg.norm(r) <= tol * scale:
            return y
        dy = splu(jacobian(K, M, y)).solve(-r)
        step = 1.0
        while step >= 2**-20:
            if np.linalg.norm(residual(y + step * dy)) < np.linalg.norm(r):
                break
            step *= 0.5
        if step < 2**-20:
            raise RuntimeError("Newton line search failed")
        y = y + step * dy
    raise RuntimeError("nonconverged state label")


def jvp(K, M, y, directions):
    """y (batch,n), directions (batch,n,r); no batch averaging."""
    y = np.asarray(y)
    directions = np.asarray(directions)
    if y.ndim != 2 or directions.ndim != 3 or directions.shape[:2] != y.shape:
        raise ValueError("expected one state and set of directions per sample")
    return np.stack([splu(jacobian(K, M, yi)).solve(M @ vi) for yi, vi in zip(y, directions)])


def reduced_gradient(K, M, y, u, yd, alpha):
    """Euclidean gradient of .5(y-yd)^T M(y-yd)+alpha/2 u^T M u."""
    p = splu(jacobian(K, M, y)).solve(M @ (y - yd), trans="T")
    return alpha * (M @ u) + M.T @ p
