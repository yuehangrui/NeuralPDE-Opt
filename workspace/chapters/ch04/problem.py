"""Manufactured control problem adapted from ocpde_dl/pinn/elliptic_uc.py.

Book/code notation:
Book (4.2), (4.5): -Delta y=u+f, -Delta p=y-yd, alpha*u+p=0. y/u/p methods below are manufactured references, not training labels. yd denotes y_d.
"""

from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass(frozen=True)
class Problem:
    alpha: float = 1e-3
    control_frequency: int = 8

    def __post_init__(self):
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if type(self.control_frequency) is not int or self.control_frequency < 1:
            raise ValueError("control_frequency must be a positive integer")

    def sine(self, x, frequency):
        sin = torch.sin if isinstance(x, torch.Tensor) else np.sin
        return sin(math.pi * frequency * x[..., 0:1]) * sin(math.pi * frequency * x[..., 1:2])

    def y(self, x):
        return self.sine(x, 1)

    def p(self, x):
        k = self.control_frequency
        return self.sine(x, k) / (2 * (math.pi * k) ** 2)

    def u(self, x):
        return -self.p(x) / self.alpha

    def f(self, x):
        return 2 * math.pi**2 * self.y(x) - self.u(x)

    def yd(self, x):
        return self.y(x) - self.sine(x, self.control_frequency)
