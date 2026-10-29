"""Numerical building blocks; no optimal-control labels in training utilities.

Book/code notation:
Book 4.5.1/4.5.3: FDGrid uses h^2-weighted norms; Mesh uses consistent P1 mass. CoordinateNet is a trainable nonlinear field. Its role determines whether its output is state y_theta or source f_theta.
"""

from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch
from torch import nn
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.fft import dstn, idstn
from scipy.interpolate import RegularGridInterpolator

ALPHA = 1e-3
DTYPE = torch.float64


def tensor(a, device="cpu"):
    return torch.as_tensor(a, dtype=DTYPE, device=device)


def sp_tensor(a, device="cpu"):
    a = a.tocoo()
    return torch.sparse_coo_tensor(
        np.array([a.row, a.col]), a.data, a.shape, dtype=DTYPE, device=device
    ).coalesce()


def smv(a, x):
    return torch.sparse.mm(a, x[:, None])[:, 0]


class CoordinateNet(nn.Module):
    """Train every hidden layer; Fourier inputs are not a finite output basis."""

    def __init__(self, width=32, depth=2, boundary=False):
        super().__init__()
        if type(width) is not int or width < 2 or type(depth) is not int or depth < 1:
            raise ValueError("positive depth and width >= 2 required")
        self.width, self.depth, self.boundary = width, depth, boundary
        self.register_buffer("frequencies", tensor([1.0, 2.0, 4.0, 8.0]))
        layers = [nn.Linear(18, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers).double()
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        phase = math.pi * x[..., :, None] * self.frequencies
        features = torch.cat([2 * x - 1, phase.sin().flatten(-2), phase.cos().flatten(-2)], -1)
        y = self.net(features)[..., 0]
        if self.boundary:
            y = y * 16 * x[..., 0] * (1 - x[..., 0]) * x[..., 1] * (1 - x[..., 1])
        return y


def public_fields(x):
    """Problem data f and yd only, not optimal y/u/p labels."""
    xp = torch if isinstance(x, torch.Tensor) else np
    s1 = xp.sin(math.pi * x[..., 0]) * xp.sin(math.pi * x[..., 1])
    s8 = xp.sin(8 * math.pi * x[..., 0]) * xp.sin(8 * math.pi * x[..., 1])
    f = 2 * math.pi**2 * s1 + s8 / (128 * math.pi**2 * ALPHA)
    return f, s1 - s8


class FDGrid:
    def __init__(self, n):
        if type(n) is not int or n < 16:
            raise ValueError("FD grid must resolve frequency 8: n >= 16")
        self.n = n
        t = np.arange(1, n) / n
        xx, yy = np.meshgrid(t, t, indexing="ij")
        self.x = np.stack([xx, yy], -1)
        k = np.arange(1, n)
        d = 4 * n * n * np.sin(math.pi * k / (2 * n)) ** 2
        self.eigenvalues = d[:, None] + d[None, :]
        self.f, self.yd = public_fields(self.x)
        self.gradient_scale = self.norm(self.solve(self.solve(self.f) - self.yd))

    def norm(self, a):
        return float(np.linalg.norm(a) / self.n)

    def solve(self, f):
        return idstn(dstn(f, type=1, norm="ortho") / self.eigenvalues, type=1, norm="ortho")

    def apply(self, y):
        if isinstance(y, torch.Tensor):
            padded_y = torch.nn.functional.pad(y, (1, 1, 1, 1))
        else:
            padded_y = np.pad(y, 1)
        return self.n**2 * (
            4 * y
            - padded_y[:-2, 1:-1]
            - padded_y[2:, 1:-1]
            - padded_y[1:-1, :-2]
            - padded_y[1:-1, 2:]
        )

    def diagnostics(self, raw_y, u):
        y_recovered = self.solve(self.f + u)
        p = self.solve(y_recovered - self.yd)
        control_gradient = ALPHA * u + p
        state_residual = self.apply(raw_y) - u - self.f
        relative_feasibility = self.norm(state_residual) / self.norm(self.f)
        stationarity = self.norm(control_gradient) / self.gradient_scale
        return dict(
            validation_score=float(np.hypot(relative_feasibility, stationarity)),
            pde_relative=relative_feasibility,
            true_gradient_relative=stationarity,
            true_objective=0.5 * self.norm(y_recovered - self.yd) ** 2
            + 0.5 * ALPHA * self.norm(u) ** 2,
            raw_objective=0.5 * self.norm(raw_y - self.yd) ** 2 + 0.5 * ALPHA * self.norm(u) ** 2,
        )


def transfer_control(u, n, points):
    """Linear interpolation/extrapolation of interior controls, no imposed BC."""
    t = np.arange(1, n) / n
    return RegularGridInterpolator((t, t), u, bounds_error=False, fill_value=None)(points)


class Mesh:
    """Structured P1 mesh, consistent mass and exact sparse transpose solve.

    Assembly follows the frozen chapters/ch04/fem.py; this version constructs
    geometry only and never computes analytic optimal states during training.
    The network source is projected by nodal P1 interpolation, load M f_theta.
    """

    def __init__(self, n):
        if type(n) is not int or n < 8:
            raise ValueError("mesh n >= 8 required")
        self.n = n
        t = np.arange(n + 1) / n
        xx, yy = np.meshgrid(t, t, indexing="ij")
        self.nodes = np.stack([xx.ravel(), yy.ravel()], 1)
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        a = (ii * (n + 1) + jj).ravel()
        b, c, d = a + n + 1, a + n + 2, a + 1
        elements = np.stack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)], 1).reshape(-1, 3)
        jac = np.stack(
            [
                self.nodes[elements[:, 1]] - self.nodes[elements[:, 0]],
                self.nodes[elements[:, 2]] - self.nodes[elements[:, 0]],
            ],
            -1,
        )
        det = np.linalg.det(jac)
        grads = np.einsum(
            "ad,edk->eak", np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]), np.linalg.inv(jac)
        )
        lk = np.einsum("ead,ebd,e->eab", grads, grads, det / 2)
        lm = det[:, None, None] * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])[None] / 24
        rows = np.broadcast_to(elements[:, :, None], lk.shape).ravel()
        cols = np.broadcast_to(elements[:, None, :], lk.shape).ravel()
        shape = (len(self.nodes),) * 2
        self.K = sparse.coo_matrix((lk.ravel(), (rows, cols)), shape=shape).tocsr()
        self.M = sparse.coo_matrix((lm.ravel(), (rows, cols)), shape=shape).tocsr()
        self.ix = np.flatnonzero(np.all((self.nodes > 0) & (self.nodes < 1), axis=1))
        self.Ki = self.K[self.ix][:, self.ix].tocsc()
        self.Mi = self.M[self.ix][:, self.ix].tocsc()
        self.L = self.M[self.ix]
        self.lu = splu(self.Ki)
        self.counts = dict(forward=0, transpose=0)

    def observe(self, points):
        points = np.asarray(points)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.isfinite(points).all()
            or np.any((points < 0) | (points > 1))
        ):
            raise ValueError("observations require finite points in [0,1]^2")
        ij = np.minimum((points * self.n).astype(int), self.n - 1)
        r, s = (points * self.n - ij).T
        a = ij[:, 0] * (self.n + 1) + ij[:, 1]
        b, c, d = a + self.n + 1, a + self.n + 2, a + 1
        low = s <= r
        indices = np.stack([a, np.where(low, b, c), np.where(low, c, d)], 1)
        weights = np.where(
            low[:, None], np.stack([1 - r, r - s, s], 1), np.stack([1 - s, r, s - r], 1)
        )
        H = sparse.coo_matrix(
            (weights.ravel(), (np.repeat(np.arange(len(points)), 3), indices.ravel())),
            shape=(len(points), len(self.nodes)),
        ).tocsr()
        return H[:, self.ix]


class SparseSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, load, mesh):
        ctx.mesh = mesh
        mesh.counts["forward"] += 1
        return tensor(mesh.lu.solve(load.detach().cpu().numpy()), load.device)

    @staticmethod
    def backward(ctx, derivative):
        ctx.mesh.counts["transpose"] += 1
        return (
            tensor(
                ctx.mesh.lu.solve(derivative.detach().cpu().numpy(), trans="T"), derivative.device
            ),
            None,
        )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    root = Path(__file__).resolve().parents[2]
    files = sorted(Path(__file__).parent.glob("*.py"))
    files += [root / "chapters/ch04/problem.py", root / "chapters/ch04/fem.py"]
    return {str(p.relative_to(root)).replace("\\", "/"): digest(p) for p in files}


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["source_identity"] != identity():
        raise ValueError("checkpoint source identity mismatch")
    net = CoordinateNet(**checkpoint["network"])
    net.load_state_dict(checkpoint["model"])
    return net, checkpoint
