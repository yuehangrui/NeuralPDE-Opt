"""Section 4.5.1, equations 4.21-4.24, translated to the known nonzero source.

Use ||K y-f||^2 in J_mod and r=K y-u-f. The multiplier z is NOT the
physical adjoint p: at a saddle z=alpha*u/2=-p/2 for this modified functional.
Fixed sine neural features make the inner head training convex in this version.

Book/code notation:
Book (4.22)-(4.24), adapted to nonzero f: z is the modified-functional multiplier. On a feasible optimum z=alpha*u/2=-p/2; q=K*y-f in this kernel is not a scaled adjoint.
"""

import numpy as np
import torch
from .common import ALPHA, DIM, LAM, SpectralProblem, SineHead, tensor


def modified_loss(q, u, z, problem):
    G = tensor(problem.gram)
    source = q - tensor(problem.f)
    e = q / tensor(LAM) - tensor(problem.yd)
    return 0.5 * e @ G @ e + 0.25 * ALPHA * (u @ G @ u + source @ G @ source) + z @ G @ (source - u)


def run(outer_steps=80, rho=None):
    rho = 0.2 * ALPHA if rho is None else rho
    if not np.isfinite(rho) or not 0 < rho < ALPHA / 2 or outer_steps < 1:
        raise ValueError("require 0 < rho < alpha/2 and positive iteration budget")
    problem = SpectralProblem()
    state, control = SineHead(), SineHead()
    z = torch.zeros(DIM, dtype=torch.float64)
    history, best, inner_calls = [], None, 0
    for k in range(outer_steps):
        # Diagonally preconditioned gradient training of the quadratic heads.
        # This avoids subtractive cancellation in a near-converged line search.
        # The quadrature Gram is nearly diagonal, but full autograd derivatives
        # still include every off-diagonal entry; no optimal-control labels enter.
        diagonal = tensor(np.diag(problem.gram))
        hq = diagonal * (1 / tensor(LAM) ** 2 + ALPHA / 2) / ALPHA
        hu = diagonal / 2
        for inner_step in range(20):
            loss = modified_loss(state.weight, control.weight, z, problem) / ALPHA
            gq, gu = torch.autograd.grad(loss, (state.weight, control.weight))
            gradmax = max(float(gq.abs().max()), float(gu.abs().max()))
            if gradmax < 1e-10:
                break
            with torch.no_grad():
                state.weight.sub_(gq / hq)
                control.weight.sub_(gu / hu)
        info = dict(
            closure_calls=inner_step + 1,
            gradient_max=gradmax,
            loss=float(loss.detach()),
            optimizer="diagonally preconditioned gradient descent",
        )
        inner_calls += info["closure_calls"]
        if info["gradient_max"] > 1e-7:
            raise RuntimeError("Uzawa inner training did not meet stationarity tolerance")
        q, u = state.weight.detach().numpy().copy(), control.weight.detach().numpy().copy()
        diag = problem.diagnostics(u, q / LAM)
        history.append(
            dict(
                iteration=k,
                **diag,
                inner=info,
                multiplier_residual_l2=float(
                    np.sqrt((q - u - problem.f) @ problem.gram @ (q - u - problem.f))
                ),
            )
        )
        if best is None or diag["validation_score"] < best["score"]:
            best = dict(score=diag["validation_score"], control=u, q=q, z=z.clone(), index=k)
        if diag["validation_score"] < 1e-8:
            break
        # L2 multiplier ascent; do not multiply by the Gram matrix a second time.
        z = z + rho * tensor(q - u - problem.f)
    return dict(
        control=best["control"],
        state_head=best["q"],
        multiplier=best["z"].numpy(),
        best_index=best["index"],
        history=history,
        diagnostics=history[best["index"]],
        iterations=len(history),
        rho=rho,
        counts=dict(inner_closures=inner_calls),
    )
