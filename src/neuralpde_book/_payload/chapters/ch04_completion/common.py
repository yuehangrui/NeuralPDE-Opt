"""A declared finite neural trial space for the running Poisson control example.

The hidden features are fixed sine products; only the linear heads are trained.
This is a reproducible method demonstrator, not a deep-network paper replication.
No manufactured optimum is supplied to an optimizer or checkpoint selector.

Book/code notation:
Book 4.3-4.5: the legacy hybrid demonstrations use a fixed 17-mode sine space. LAM are Laplacian eigenvalues, not Lagrange multipliers; ALPHA is the control cost coefficient.
"""

from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch
from torch import nn
from chapters.ch04.problem import Problem

DTYPE = torch.float64
MODES = np.array([(i, j) for i in range(1, 5) for j in range(1, 5)] + [(8, 8)])
LAM = np.pi**2 * (MODES**2).sum(1)
DIM = len(LAM)
ALPHA = 1e-3


def tensor(x):
    if isinstance(x, np.ndarray) and not x.flags.writeable:
        x = x.copy()
    return torch.as_tensor(x, dtype=DTYPE)


def basis(x):
    x = np.asarray(x)
    return np.sin(np.pi * x[..., 0, None] * MODES[:, 0]) * np.sin(
        np.pi * x[..., 1, None] * MODES[:, 1]
    )


def quadrature(n):
    if type(n) is not int or n < 24:
        raise ValueError("quadrature must resolve all 17 declared modes (n >= 24)")
    z, w = np.polynomial.legendre.leggauss(n)
    x, y = np.meshgrid((z + 1) / 2, (z + 1) / 2, indexing="ij")
    return np.stack([x.ravel(), y.ravel()], 1), np.outer(w / 2, w / 2).ravel()


class SpectralProblem:
    """Exact PDE on the declared subspace; f and yd projected from public data."""

    def __init__(self, quadrature_order=32):
        self.problem = Problem()
        self.x, self.w = quadrature(quadrature_order)
        self.b = basis(self.x)
        self.gram = self.b.T @ (self.w[:, None] * self.b)
        self.f = np.linalg.solve(self.gram, self.b.T @ (self.w * self.problem.f(self.x)[:, 0]))
        self.yd = np.linalg.solve(self.gram, self.b.T @ (self.w * self.problem.yd(self.x)[:, 0]))
        self.counts = dict(state=0, adjoint=0)

    def state(self, u):
        u = np.asarray(u, dtype=float)
        if u.shape[-1] != DIM or not np.isfinite(u).all():
            raise ValueError("invalid control")
        self.counts["state"] += 1 if u.ndim == 1 else len(u)
        return (self.f + u) / LAM

    def objective(self, y, u):
        e = y - self.yd
        return float(0.5 * e @ self.gram @ e + 0.5 * ALPHA * u @ self.gram @ u)

    def fun(self, u):
        y = self.state(u)
        self.counts["adjoint"] += 1
        control_gradient = (self.gram @ (y - self.yd)) / LAM + ALPHA * self.gram @ u
        return self.objective(y, u), control_gradient

    def diagnostics(self, u, y=None):
        """Validation on a different quadrature rule; no optimum for selection."""
        x, w = quadrature(56)
        b = basis(x)
        recovered = (self.f + u) / LAM
        y = recovered if y is None else np.asarray(y)
        r = b @ (LAM * y - u) - self.problem.f(x)[:, 0]
        control_gradient = b @ (ALPHA * u + (recovered - self.yd) / LAM)
        g0 = b @ ((self.f / LAM - self.yd) / LAM)
        norm = lambda v: float(np.sqrt(w @ (v * v)))
        feasibility = norm(r) / norm(self.problem.f(x)[:, 0])
        grad = norm(control_gradient) / norm(g0)
        # Analytic errors are reporting only, never used to select models.
        exact_u = self.problem.u(x)[:, 0]
        exact_y = self.problem.y(x)[:, 0]
        return dict(
            state_residual_relative=feasibility,
            true_gradient_relative=grad,
            validation_score=math.hypot(feasibility, grad),
            control_relative_l2=norm(b @ u - exact_u) / norm(exact_u),
            recovered_state_relative_l2=norm(b @ recovered - exact_y) / norm(exact_y),
            true_objective=self.objective(recovered, u),
            raw_objective=self.objective(y, u),
        )


class SineHead(nn.Module):
    """One fixed sine-feature hidden layer and a trainable linear output head."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(DIM, dtype=DTYPE))

    def forward(self, features):
        return features @ self.weight


def train_quadratic(parameters, closure_loss, maxiter=60, tol=1e-11):
    """Train a convex neural-head subproblem, with explicit closure accounting."""
    params = list(parameters)
    opt = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=maxiter,
        tolerance_grad=tol,
        tolerance_change=1e-24,
        line_search_fn="strong_wolfe",
    )
    calls = 0

    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True)
        loss = closure_loss()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite training objective")
        loss.backward()
        calls += 1
        return loss

    opt.step(closure)
    loss = closure_loss()
    grads = torch.autograd.grad(loss, params)
    gradmax = max(float(g.detach().abs().max()) for g in grads)
    return dict(closure_calls=calls, gradient_max=gradmax, loss=float(loss.detach()))


def identity():
    root = Path(__file__).resolve().parents[2]
    files = list(Path(__file__).parent.glob("*.py"))
    files += [root / "chapters/ch04" / f for f in ("__init__.py", "problem.py", "fem.py")]
    return {
        str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def save_checkpoint(path, method, state, config):
    torch.save(
        dict(
            schema="ch04-completion-v1",
            method=method,
            source_identity=identity(),
            config=config,
            state=state,
        ),
        path,
    )


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("schema") != "ch04-completion-v1" or ck["source_identity"] != identity():
        raise ValueError("checkpoint source identity mismatch")

    def finite(value):
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite checkpoint")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        if isinstance(value, (list, tuple)):
            for item in value:
                finite(item)

    finite(ck["state"])
    return ck
