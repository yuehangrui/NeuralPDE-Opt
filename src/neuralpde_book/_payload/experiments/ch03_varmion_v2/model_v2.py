"""Fixed-basis VarMiON: dense control and compact SPD inverse matrices.

Book/code notation:
Book 3.4.3: matrix(sensors) is an inverse-like reduced map K_theta(c), not the PDE stiffness. SPD mode returns L*L^T; operator supervision compares its action with the independently assembled reduced operator.
"""

import numpy as np
import torch
from torch import nn


class MatrixMap(nn.Module):
    def __init__(self, kind, mean, std, k0, width=None):
        super().__init__()
        k0 = np.asarray(k0, dtype=np.float64)
        if kind not in {"dense", "spd"} or k0.ndim != 2 or k0.shape[0] != k0.shape[1]:
            raise ValueError("Invalid matrix model")
        if not np.isfinite(k0).all() or not np.isfinite(std) or std <= 0:
            raise ValueError("Invalid normalization or initial inverse")
        np.testing.assert_allclose(k0, k0.T, atol=1e-12)
        l0 = np.linalg.cholesky(k0)
        self.kind, self.rank = kind, len(k0)
        self.register_buffer("mean", torch.tensor(float(mean), dtype=torch.float64))
        self.register_buffer("std", torch.tensor(float(std), dtype=torch.float64))
        self.register_buffer("k0", torch.from_numpy(k0.copy()))
        self.register_buffer("l0", torch.from_numpy(l0))
        self.register_buffer("inverse_l0", torch.from_numpy(np.linalg.inv(l0)))
        self.register_buffer("eye", torch.eye(self.rank, dtype=torch.float64))
        if kind == "dense":
            width = 128 if width is None else width
            self.branch = nn.Sequential(
                nn.Linear(289, width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, self.rank * self.rank),
            )
        else:
            width = 32 if width is None else width
            rows, cols = torch.tril_indices(self.rank, self.rank)
            self.register_buffer("rows", rows)
            self.register_buffer("cols", cols)
            self.branch = nn.Sequential(
                nn.Linear(289, width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, len(rows)),
            )
            nn.init.zeros_(self.branch[-1].weight)
            nn.init.zeros_(self.branch[-1].bias)

    def matrix(self, sensors):
        if sensors.ndim != 2 or sensors.shape[1] != 289:
            raise ValueError("Expected 17x17 sensors")
        raw = self.branch((sensors - self.mean) / self.std)
        if self.kind == "dense":
            return self.k0 + 0.1 * raw.reshape(-1, self.rank, self.rank)
        packed = raw.new_zeros((len(raw), self.rank, self.rank))
        packed[:, self.rows, self.cols] = 0.1 * raw
        # Positive diagonal plus unconstrained strict lower triangle. The clamp
        # guards exponent overflow, while preserving positive factors.
        diag = torch.exp(torch.diagonal(packed, dim1=-2, dim2=-1).clamp(-8, 8))
        factor = torch.tril(packed, diagonal=-1) + torch.diag_embed(diag)
        factor = self.l0 @ factor
        return factor @ factor.transpose(-1, -2)

    def forward(self, sensors, loads):
        return torch.einsum("bij,bj->bi", self.matrix(sensors), loads)


def operator_loss(k, whitened_operator, inverse_l0):
    """||B Q-I||_F^2/r with B=L0^T Ar L0, Q=L0^-1 K L0^-T."""
    q = inverse_l0 @ k @ inverse_l0.T
    defect = whitened_operator @ q - torch.eye(k.shape[-1], device=k.device, dtype=k.dtype)
    return defect.square().sum((-2, -1)).mean() / k.shape[-1]


def supervised_loss(alpha, target, denominator):
    return ((alpha.double() - target).square().sum(1) / denominator).mean()
