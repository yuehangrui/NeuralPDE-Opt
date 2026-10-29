"""Section 4.4.1: neural lower-level PDE training and implicit hypergradients.

q is the state-network head for y=B q/lambda, u is the control-network head.
The lower loss is 1/2 ||B(q-u-f)||^2; H^{-1} is applied by a linear solve.
No unrolling, detached state surrogate, or direct KKT solve replaces the bilevel loop.

Book/code notation:
Book 4.4.1: the inner state solve depends on the outer control. The implicit derivative includes the lower Hessian solve; lower stationarity and outer stationarity are distinct.
"""

import numpy as np
import torch
from scipy.optimize import minimize
from .common import ALPHA, DIM, LAM, SpectralProblem, SineHead, tensor, train_quadratic


class Bilevel:
    def __init__(self):
        self.problem = SpectralProblem()
        self.state_net = SineHead()
        self.G = tensor(self.problem.gram)
        self.calls = 0
        self.inner_calls = 0
        self.last = None

    def lower(self, q, u):
        r = q - u - tensor(self.problem.f)
        return 0.5 * r @ self.G @ r

    def solve_inner(self, u):
        u = tensor(u)
        info = train_quadratic(
            self.state_net.parameters(), lambda: self.lower(self.state_net.weight, u)
        )
        self.inner_calls += info["closure_calls"]
        if info["gradient_max"] > 1e-8:
            raise RuntimeError("lower-level neural solve failed its residual tolerance")
        return self.state_net.weight.detach().clone(), info

    def fun(self, u_np):
        self.calls += 1
        q0, inner = self.solve_inner(u_np)
        q = q0.requires_grad_(True)
        u = tensor(u_np).requires_grad_(True)
        y = q / tensor(LAM)
        e = y - tensor(self.problem.yd)
        outer = 0.5 * e @ self.G @ e + 0.5 * ALPHA * u @ self.G @ u
        jq, ju = torch.autograd.grad(outer, (q, u), retain_graph=True)
        hessian = torch.autograd.functional.hessian(lambda qq: self.lower(qq, u), q)
        cross = torch.autograd.functional.jacobian(
            lambda uu: torch.autograd.grad(self.lower(q, uu), q, create_graph=True)[0], u
        )
        v = torch.linalg.solve(hessian, jq)
        hyper = ju - cross.T @ v
        self.last = dict(
            q=q0.detach().numpy(),
            control=np.asarray(u_np).copy(),
            inner=inner,
            hypergradient=hyper.detach().numpy(),
            hessian_solve_residual=float(torch.linalg.vector_norm(hessian @ v - jq)),
        )
        return float(outer.detach()), hyper.detach().numpy()


def run(maxiter=300):
    solver = Bilevel()
    history = []
    best = None

    def record(u):
        nonlocal best
        solver.fun(u)
        y = solver.last["q"] / LAM
        diag = solver.problem.diagnostics(u, y)
        row = dict(
            **diag,
            inner=solver.last["inner"],
            hessian_solve_residual=solver.last["hessian_solve_residual"],
        )
        history.append(row)
        if best is None or diag["validation_score"] < best["score"]:
            best = dict(
                score=diag["validation_score"],
                control=u.copy(),
                q=solver.last["q"].copy(),
                index=len(history) - 1,
            )

    record(np.zeros(DIM))
    opt = minimize(
        solver.fun,
        np.zeros(DIM),
        jac=True,
        method="BFGS",
        callback=record,
        options=dict(gtol=1e-12, maxiter=maxiter),
    )
    record(opt.x)
    return dict(
        control=best["control"],
        state_head=best["q"],
        best_index=best["index"],
        history=history,
        diagnostics=history[best["index"]],
        optimizer_success=bool(opt.success),
        message=str(opt.message),
        iterations=int(opt.nit),
        counts=dict(outer_evaluations=solver.calls, inner_closures=solver.inner_calls),
    )
