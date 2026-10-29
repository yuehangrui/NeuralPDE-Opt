"""Book (2.10): -Delta y = f in Omega, y = g on the boundary.

The 1D benchmark has g(0)=0 and g(1)=1; the 2D benchmark has g=0.
Exact y and grad(y) are for validation. Training uses f, the boundary lift G,
and the boundary factor D; there are no solution labels in the training loss.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass(frozen=True)
class Poisson:
    """Analytic Poisson data on Omega=(0,1)^dim, for dim=1 or 2."""

    dim: int

    def __post_init__(self):
        if self.dim not in (1, 2):
            raise ValueError("Only dimensions 1 and 2 are implemented")

    @property
    def name(self):
        return f"poisson{self.dim}d"

    def exact(self, x):
        """Reference solution y(x), retaining a final scalar axis (..., 1)."""
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        if self.dim == 1:
            return x[..., :1] + sin(math.pi * x[..., :1])
        return sin(math.pi * x[..., :1]) * sin(math.pi * x[..., 1:2])

    def gradient(self, x):
        """Reference spatial gradient grad(y), with shape (..., dim)."""
        is_torch = isinstance(x, torch.Tensor)
        sin, cos = (torch.sin, torch.cos) if is_torch else (np.sin, np.cos)
        if self.dim == 1:
            return 1 + math.pi * cos(math.pi * x)
        dy_dx1 = math.pi * cos(math.pi * x[..., :1]) * sin(math.pi * x[..., 1:2])
        dy_dx2 = math.pi * sin(math.pi * x[..., :1]) * cos(math.pi * x[..., 1:2])
        return torch.cat((dy_dx1, dy_dx2), dim=-1) if is_torch else np.concatenate((dy_dx1, dy_dx2), axis=-1)

    def source(self, x):
        """PDE source f(x) = -Delta y(x), with shape (..., 1)."""
        if self.dim == 1:
            sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
            return math.pi**2 * sin(math.pi * x[..., :1])
        return 2 * math.pi**2 * self.exact(x)

    def lifting(self, x):
        """Boundary lift G(x): G=g on the boundary, book 2.3.2."""
        return x[..., :1] if self.dim == 1 else x[..., :1] * 0

    def boundary_factor(self, x):
        """D(x)=product_i 4*x_i*(1-x_i); used with torch neural inputs."""
        coordinate_factors = 4 * x * (1 - x)
        return coordinate_factors.prod(dim=-1, keepdim=True)

    def boundary_points(self, n=101):
        """Independent boundary check points; these are not training labels."""
        if self.dim == 1:
            return np.array([[0.0], [1.0]])
        t = np.linspace(0, 1, n)
        return np.concatenate([np.stack((t, t*0), -1), np.stack((t, t*0+1), -1),
                               np.stack((t*0, t), -1), np.stack((t*0+1, t), -1)])
