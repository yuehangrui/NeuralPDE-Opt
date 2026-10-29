"""Independent symmetric five-point FD reference; interior values, mass h^2 I.

This is a new reference adapter, not the legacy P1 FEM discretization.
Controls and quadrature are interior-grid quantities. No artificial control BC.

Book/code notation:
Book (5.2)-(5.3): Grid.recover returns physical p and alpha*u+p. In PDOL, p_dual is an observation-space dual; applying the adjoint operator yields the PDE adjoint. The implemented primal step retains +u in its proximal numerator.
"""

import numpy as np
from scipy.fft import dstn, idstn
from .problem import clip


class Grid:
    def __init__(self, n=32):
        if type(n) is not int or n < 4:
            raise ValueError("n must be an integer >=4 (number of intervals)")
        self.n, self.h = n, 1 / n
        t = np.arange(1, n) / n
        self.points = np.stack(np.meshgrid(t, t, indexing="ij"), -1)
        l = 4 * n * n * np.sin(np.pi * np.arange(1, n) / (2 * n)) ** 2
        self.eigenvalues = l[:, None] + l[None, :]
        self.s_norm = 1 / self.eigenvalues[0, 0]

    def check(self, v):
        v = np.asarray(v, dtype=np.float64)
        if v.shape[-2:] != (self.n - 1,) * 2 or not np.isfinite(v).all():
            raise ValueError("invalid interior grid field")
        return v

    def solve(self, v):
        return idstn(
            dstn(self.check(v), type=1, norm="ortho", axes=(-2, -1)) / self.eigenvalues,
            type=1,
            norm="ortho",
            axes=(-2, -1),
        )

    def apply(self, v):
        v = self.check(v)
        padded_field = np.pad(v, [(0, 0)] * (v.ndim - 2) + [(1, 1), (1, 1)])
        return self.n**2 * (
            4 * v
            - padded_field[..., 2:, 1:-1]
            - padded_field[..., :-2, 1:-1]
            - padded_field[..., 1:-1, 2:]
            - padded_field[..., 1:-1, :-2]
        )

    def norm(self, v):
        return np.sqrt(np.sum(np.asarray(v) ** 2, axis=(-2, -1))) * self.h

    def recover(self, u, f, yd, alpha):
        y = self.solve(u + f)
        p = self.solve(y - yd)  # exact transpose in the h^2 I metric
        return y, p, alpha * u + p

    def diagnostics(self, u, f, yd, problem):
        u, f, yd = self.check(u), self.check(f), self.check(yd)
        y, p, control_gradient = self.recover(u, f, yd, problem.alpha)
        residual = u - clip(-p / problem.alpha, problem.lower, problem.upper)
        return dict(
            objective=0.5 * self.norm(y - yd) ** 2 + 0.5 * problem.alpha * self.norm(u) ** 2,
            projected_residual=self.norm(residual),
            bound_violation=np.maximum(np.maximum(problem.lower - u, u - problem.upper), 0).max(
                axis=(-2, -1)
            ),
            state_residual=self.norm(self.apply(y) - u - f),
            adjoint_residual=self.norm(self.apply(p) - y + yd),
        )

    def optimal_control(self, f, yd, problem, tol=1e-9, max_iter=4000):
        """Projected gradient with known Lipschitz bound; fail closed on bad labels."""
        if tol <= 0 or max_iter < 1:
            raise ValueError("invalid reference stopping settings")
        f, yd = self.check(f), self.check(yd)
        if f.shape != yd.shape:
            raise ValueError("source and target shapes differ")
        step = 1 / (problem.alpha + self.s_norm**2)
        u = np.full_like(f, np.clip(0, problem.lower, problem.upper))
        for k in range(1, max_iter + 1):
            y, p, control_gradient = self.recover(u, f, yd, problem.alpha)
            residual = self.norm(u - clip(-p / problem.alpha, problem.lower, problem.upper))
            if np.max(residual) <= tol:
                return dict(u=u, y=y, p=p, iterations=k, projected_residual=residual)
            u = clip(u - step * control_gradient, problem.lower, problem.upper)
        raise RuntimeError(f"reference unconverged: projected residual={np.max(residual):.3e}")


def primal_dual(state, adjoint, f, yd, problem, s_norm, max_iter=500, tol=1e-7):
    """PDOL update ported from pdol/solver.py, classical step condition.

    Reports final (not historical first-hit) surrogate fixed-point residual.
    The exact PDE optimality residual must be assessed separately by Grid.
    No relaxation >1: that can destroy feasibility of the reported control.
    """
    import torch

    if not np.isfinite(s_norm) or s_norm <= 0 or max_iter < 1 or tol <= 0:
        raise ValueError("invalid PD settings")
    tau = 1 / problem.alpha
    sigma = 0.9 / (tau * s_norm**2)
    u = torch.full_like(f, float(np.clip(0, problem.lower, problem.upper)))
    p_dual = torch.zeros_like(f)  # dual in observation space, NOT PDE adjoint p
    with torch.no_grad():
        for k in range(1, max_iter + 1):
            u_next = torch.clamp(
                (u - tau * adjoint(p_dual)) / (1 + problem.alpha * tau),
                problem.lower,
                problem.upper,
            )
            p_dual_next = (p_dual + sigma * (state(2 * u_next - u + f) - yd)) / (1 + sigma)
            u, p_dual = u_next, p_dual_next
            if k % 5 == 0 or k == max_iter:
                y = state(u + f)
                r_u = (
                    (
                        u
                        - torch.clamp(
                            -adjoint(p_dual) / problem.alpha, problem.lower, problem.upper
                        )
                    )
                    .flatten(1)
                    .square()
                    .mean(1)
                    .sqrt()
                )
                r_d = (p_dual - y + yd).flatten(1).square().mean(1).sqrt()
                r = torch.maximum(r_u, r_d)
                if not torch.isfinite(r).all():
                    raise FloatingPointError("PDOL diverged")
                if bool((r <= tol).all()):
                    break
    return u, dict(
        iterations=k,
        surrogate_fixed_point_rms=r.cpu().tolist(),
        surrogate_converged=(r <= tol).cpu().tolist(),
        tau=tau,
        sigma=sigma,
        convergence_claim="surrogate residual only; exact-pair step bound is not a surrogate theorem",
    )
