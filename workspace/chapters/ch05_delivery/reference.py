"""Independent constrained-state QP and reference refinement, no neural ansatz.

Book/code notation:
Book 5.2.2: state_qp is an independent convex reference for an upper state constraint, not an obstacle/VI state equation and not the box-control manufactured solution.
"""

import time
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize
from chapters.ch05.reference import Grid
from chapters.ch05.problem import Problem


def state_qp(n, cap=0.2, alpha=0.001, max_iter=150):
    grid = Grid(n)
    m = n - 1
    t = diags([-np.ones(m - 1), 2 * np.ones(m), -np.ones(m - 1)], [-1, 0, 1], format="csr") * n * n
    a = kron(t, eye(m)) + kron(eye(m), t)
    h = (eye(m * m) + alpha * (a @ a)).tocsr()
    yd = Problem().sine(grid.points)[..., 0].ravel()
    c = h.diagonal()
    y = spsolve(h, yd)
    mu = np.zeros_like(y)
    history = []
    seen = set()
    started = time.time()
    for k in range(max_iter):
        active = mu + c * (y - cap) > 0
        key = active.tobytes()
        if key in seen:
            raise RuntimeError("state-QP active set cycle before convergence")
        seen.add(key)
        free = ~active
        y = np.full_like(yd, cap)
        y[free] = spsolve(h[free][:, free], yd[free] - h[free][:, active] @ y[active])
        grad = h @ y - yd
        mu = np.where(active, -grad, 0.0)
        violation = float(np.maximum(y - cap, 0).max())
        dual = float(np.maximum(-mu, 0).max())
        stationarity = float(np.max(np.abs(grad + mu)))
        history.append(
            dict(
                iteration=k + 1,
                active=int(active.sum()),
                violation=violation,
                dual_violation=dual,
                stationarity_inf=stationarity,
            )
        )
        if violation < 1e-11 and dual < 5e-8 and stationarity < 5e-8:
            break
    else:
        raise RuntimeError("state reference did not converge")
    u = (a @ y).reshape(m, m)
    y = y.reshape(m, m)
    mu = mu.reshape(m, m)
    result = dict(
        n=n,
        iterations=k + 1,
        seconds=time.time() - started,
        objective=float(
            0.5 * grid.norm(y - yd.reshape(m, m)) ** 2 + 0.5 * alpha * grid.norm(u) ** 2
        ),
        feasibility=violation,
        dual_violation=dual,
        stationarity_inf=stationarity,
        complementarity_inf=float(np.max(np.abs(mu * (y - cap)))),
        history=history,
        definition="FD state-space convex QP, H=I+alpha*A^T*A, y<=cap; not a continuous optimality certificate",
    )
    return dict(y=y, u=u, mu=mu, yd=yd.reshape(m, m)), result


def independent_box(n=8):
    grid = Grid(n)
    problem = Problem()
    f = problem.f(grid.points)[..., 0]
    yd = problem.yd(grid.points)[..., 0]

    def fg(v):
        y, _, grad = grid.recover(v.reshape(n - 1, n - 1), f, yd, problem.alpha)
        return (
            float(0.5 * np.sum((y - yd) ** 2) + 0.5 * problem.alpha * np.sum(v * v)),
            grad.ravel(),
        )

    r = minimize(
        fg,
        np.full((n - 1) ** 2, 0.5),
        jac=True,
        bounds=[(0.3, 1.0)] * ((n - 1) ** 2),
        method="L-BFGS-B",
        options=dict(gtol=1e-12, ftol=1e-15, maxiter=5000),
    )
    if not r.success:
        raise RuntimeError(r.message)
    return r.x.reshape(n - 1, n - 1)
