"""Section 4.3: reduced-space optimization with exact discrete adjoints."""

import numpy as np
from scipy.optimize import minimize
from chapters.ch04.fem import Reference
from chapters.ch04.problem import Problem
from .common import ALPHA, DIM, SpectralProblem


def spectral(maxiter=300):
    problem = SpectralProblem()
    history = []

    def record(u):
        history.append(problem.diagnostics(u))

    record(np.zeros(DIM))
    opt = minimize(
        problem.fun,
        np.zeros(DIM),
        jac=True,
        method="BFGS",
        callback=record,
        options=dict(gtol=1e-12, maxiter=maxiter),
    )
    record(opt.x)
    return dict(
        control=opt.x,
        history=history,
        diagnostics=history[-1],
        optimizer_success=bool(opt.success),
        message=str(opt.message),
        counts=problem.counts,
        iterations=int(opt.nit),
    )


def fem(mesh=32, maxiter=500):
    ref = Reference(Problem(), mesh, 4)
    ix = ref.interior
    counts = dict(state=0, adjoint=0)

    def fun(u):
        y = ref.lu.solve(ref.fl[ix] + ref.Mi @ u)
        p = ref.lu.solve(ref.Mi @ y - ref.ydl[ix], trans="T")
        counts["state"] += 1
        counts["adjoint"] += 1
        # SciPy takes the Euclidean derivative, NOT the L2 Riesz gradient.
        control_gradient = ref.Mi @ (ALPHA * u + p)
        value = ref.objective(ref.at_quad(ref.expand(y)), ref.at_quad(ref.expand(u)))
        return value, control_gradient

    history = []

    def record(u):
        value, g = fun(u)
        # Recover the physical L2 gradient rather than using ||g||_2 as a proxy.
        y = ref.lu.solve(ref.fl[ix] + ref.Mi @ u)
        p = ref.lu.solve(ref.Mi @ y - ref.ydl[ix], trans="T")
        counts["state"] += 1
        counts["adjoint"] += 1
        g_l2 = ALPHA * u + p
        history.append(
            dict(
                objective=value,
                gradient_l2=float(np.sqrt(g_l2 @ ref.Mi @ g_l2)),
                euclidean_derivative_norm=float(np.linalg.norm(g)),
            )
        )

    record(np.zeros(len(ix)))
    opt = minimize(
        fun,
        np.zeros(len(ix)),
        jac=True,
        method="L-BFGS-B",
        callback=record,
        options=dict(gtol=1e-14, ftol=0.0, maxiter=maxiter, maxls=40),
    )
    record(opt.x)
    delta = opt.x - ref.u[ix]
    discrete_error = np.sqrt(delta @ ref.Mi @ delta) / np.sqrt(ref.u[ix] @ ref.Mi @ ref.u[ix])
    return dict(
        mesh=mesh,
        dofs=len(ix),
        optimizer_success=bool(opt.success),
        message=str(opt.message),
        iterations=int(opt.nit),
        counts=counts,
        control=opt.x,
        reference=ref.report(),
        history=history,
        control_error_to_discrete_kkt=float(discrete_error),
        objective_gap_to_discrete_kkt=history[-1]["objective"] - ref.reference_objective,
    )
