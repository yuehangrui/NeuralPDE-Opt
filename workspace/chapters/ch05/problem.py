"""Elliptic box example from ocpde_dl/pinn/elliptic_cc.py; physical p=alpha*q.

Book/code notation:
Book (5.1)-(5.8): y is state, u is box-constrained control, p is physical adjoint, q=p/alpha, and u=projection(-q). state_cap is psi in the separate CNP problem.
"""

from dataclasses import dataclass
import math
import numpy as np
import torch


def clip(x, lo, hi):
    return torch.clamp(x, lo, hi) if isinstance(x, torch.Tensor) else np.clip(x, lo, hi)


@dataclass(frozen=True)
class Problem:
    alpha: float = 1e-3
    lower: float = 0.3
    upper: float = 1.0
    state_cap: float = 0.2

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.alpha, self.lower, self.upper, self.state_cap)):
            raise ValueError("nonfinite problem parameter")
        if self.alpha <= 0 or self.lower >= self.upper or self.state_cap <= 0:
            raise ValueError("require alpha>0, lower<upper, state_cap>0")

    def sine(self, x):
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        return sin(math.pi * x[..., 0:1]) * sin(math.pi * x[..., 1:2])

    def y(self, x):
        return math.pi**2 * self.alpha * self.sine(x)

    def q(self, x):
        return -2 * self.sine(x)

    def p(self, x):
        return self.alpha * self.q(x)

    def u(self, x):
        return clip(-self.q(x), self.lower, self.upper)

    def f(self, x):
        return 2 * math.pi**4 * self.alpha * self.sine(x) - self.u(x)

    def yd(self, x):
        return 5 * math.pi**2 * self.alpha * self.sine(x)


def prox_l1_box(v, threshold, lower, upper):
    """Exact prox for threshold*|u| + indicator_[lower,upper]."""
    if threshold < 0 or not math.isfinite(threshold) or lower > upper:
        raise ValueError("invalid proximal parameters")
    if isinstance(v, torch.Tensor):
        soft = v.sign() * torch.relu(v.abs() - threshold)
    else:
        soft = np.sign(v) * np.maximum(np.abs(v) - threshold, 0)
    return clip(soft, lower, upper)
