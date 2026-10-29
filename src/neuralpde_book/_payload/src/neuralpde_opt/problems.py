"""Two Poisson problems; all reference data are analytic, not learned labels."""
from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass(frozen=True)
class Poisson:
    dim: int

    def __post_init__(self):
        if self.dim not in (1, 2):
            raise ValueError("Only dimensions 1 and 2 are implemented")

    @property
    def name(self):
        return f"poisson{self.dim}d"

    def exact(self, x):
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        if self.dim == 1:
            return x[..., :1] + sin(math.pi * x[..., :1])
        return sin(math.pi * x[..., :1]) * sin(math.pi * x[..., 1:2])

    def gradient(self, x):
        is_torch = isinstance(x, torch.Tensor)
        sin, cos = (torch.sin, torch.cos) if is_torch else (np.sin, np.cos)
        if self.dim == 1:
            return 1 + math.pi * cos(math.pi * x)
        a = math.pi * cos(math.pi * x[..., :1]) * sin(math.pi * x[..., 1:2])
        b = math.pi * sin(math.pi * x[..., :1]) * cos(math.pi * x[..., 1:2])
        return torch.cat((a, b), dim=-1) if is_torch else np.concatenate((a, b), axis=-1)

    def source(self, x):
        if self.dim == 1:
            sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
            return math.pi**2 * sin(math.pi * x[..., :1])
        return 2 * math.pi**2 * self.exact(x)

    def lifting(self, x):
        return x[..., :1] if self.dim == 1 else x[..., :1] * 0

    def boundary_factor(self, x):
        z = 4 * x * (1 - x)
        return z.prod(dim=-1, keepdim=True)

    def boundary_points(self, n=101):
        if self.dim == 1:
            return np.array([[0.0], [1.0]])
        t = np.linspace(0, 1, n)
        return np.concatenate([np.stack((t, t*0), -1), np.stack((t, t*0+1), -1),
                               np.stack((t*0, t), -1), np.stack((t*0+1, t), -1)])
