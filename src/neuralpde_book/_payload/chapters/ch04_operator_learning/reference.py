"""Self-adjoint Dirichlet sine-Galerkin reference on interior grid values.

The nodal mass is h^2 I. DST-I with orthonormal normalization represents
all tensor sine modes 1..n-1. This is spectral, not a five-point FD Laplacian.
Boundary values are not control degrees of freedom.

Book/code notation:
Book (4.5): recover(f,yd,u) solves for y and physical p and returns the reduced control gradient alpha*u+p. No control boundary condition is imposed.
"""

import numpy as np
from scipy.fft import dstn, idstn


class Poisson:
    def __init__(self, n=32, alpha=1e-3):
        if type(n) is not int or n < 4 or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("invalid reference")
        self.n, self.alpha = n, alpha
        k = np.arange(1, n)
        self.eigenvalues = np.pi**2 * (k[:, None] ** 2 + k[None, :] ** 2)
        t = np.linspace(0, 1, n + 1)
        x, y = np.meshgrid(t, t, indexing="ij")
        self.points = np.stack([x, y], -1)
        self.mask = np.zeros((n + 1, n + 1))
        self.mask[1:-1, 1:-1] = 1

    def transform(self, x, factor):
        x = np.asarray(x)
        if x.shape[-2:] != (self.n + 1, self.n + 1) or not np.isfinite(x).all():
            raise ValueError("invalid field")
        out = np.zeros_like(x, dtype=np.float64)
        out[..., 1:-1, 1:-1] = idstn(
            dstn(x[..., 1:-1, 1:-1], type=1, norm="ortho", axes=(-2, -1)) * factor,
            type=1,
            norm="ortho",
            axes=(-2, -1),
        )
        return out

    def solve(self, rhs):
        return self.transform(rhs, 1 / self.eigenvalues)

    def apply(self, y):
        return self.transform(y, self.eigenvalues)

    def norm(self, x):
        return np.sqrt(np.sum(np.asarray(x)[..., 1:-1, 1:-1] ** 2, axis=(-2, -1))) / self.n

    def inner(self, a, b):
        return np.sum(a[..., 1:-1, 1:-1] * b[..., 1:-1, 1:-1], axis=(-2, -1)) / self.n**2

    def optimal(self, f, yd):
        y = self.transform(
            f, self.alpha * self.eigenvalues / (1 + self.alpha * self.eigenvalues**2)
        ) + self.transform(yd, 1 / (1 + self.alpha * self.eigenvalues**2))
        u = self.transform(
            yd, self.eigenvalues / (1 + self.alpha * self.eigenvalues**2)
        ) - self.transform(f, 1 / (1 + self.alpha * self.eigenvalues**2))
        return y, u, -self.alpha * u

    def recover(self, f, yd, u):
        y = self.solve(f + u)
        p = self.solve(y - yd)
        control_gradient = self.alpha * u + p
        j = 0.5 * self.inner(y - yd, y - yd) + 0.5 * self.alpha * self.inner(u, u)
        return y, p, control_gradient, j
