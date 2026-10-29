"""Book Eq. (3.33)-(3.35): a nonlinear c branch acts linearly on the load.

Book/code notation:
Book (3.33)-(3.35): z_f=Phi^T*M*f, beta=K_theta(c)*z_f, y_theta=Phi*beta. Stored z/alpha arrays retain historical names; alpha here is a reduced output coefficient, not the control penalty of Chapters 4-5.
"""

import numpy as np
import torch
from torch import nn


class VariationalMap(nn.Module):
    def __init__(self, mean_c, std_c, k0, width=128):
        super().__init__()
        rank = len(k0)
        self.rank = rank
        self.register_buffer("mean_c", torch.tensor(float(mean_c)))
        self.register_buffer("std_c", torch.tensor(float(std_c)))
        self.register_buffer("k0", torch.tensor(k0, dtype=torch.float32))
        self.branch = nn.Sequential(
            nn.Linear(289, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, rank * rank),
        )

    def matrix(self, c):
        x = ((c[:, ::4, ::4] - self.mean_c) / self.std_c).flatten(1)
        return self.k0 + 0.1 * self.branch(x).reshape(-1, self.rank, self.rank)

    def forward(self, c, z):
        return torch.einsum("bij,bj->bi", self.matrix(c), z)


class UnstructuredMap(nn.Module):
    def __init__(self, mean_c, std_c, k0, width=384):
        super().__init__()
        rank = len(k0)
        self.register_buffer("mean_c", torch.tensor(float(mean_c)))
        self.register_buffer("std_c", torch.tensor(float(std_c)))
        self.register_buffer("k0", torch.tensor(k0, dtype=torch.float32))
        layers = []
        size = 289 + rank
        for _ in range(4):
            layers.extend([nn.Linear(size, width), nn.GELU()])
            size = width
        layers.append(nn.Linear(size, rank))
        self.network = nn.Sequential(*layers)

    def forward(self, c, z):
        x = ((c[:, ::4, ::4] - self.mean_c) / self.std_c).flatten(1)
        return z @ self.k0.T + 0.1 * self.network(torch.cat([x, z], dim=1))


def model_for(name, norm, k0):
    if name == "varmion":
        return VariationalMap(norm["c_mean"], norm["c_std"], k0)
    if name == "unstructured":
        return UnstructuredMap(norm["c_mean"], norm["c_std"], k0)
    raise ValueError("Unknown structured model")


def fit_uncentered_pod(y, mass, rank):
    flat = np.asarray(y, dtype=np.float64).reshape(len(y), -1)
    gram = flat @ (mass @ flat.T) / len(flat)
    values, vectors = np.linalg.eigh((gram + gram.T) / 2)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if not 1 <= rank <= len(y) or values[rank - 1] <= values[0] * 1e-12:
        raise ValueError("Insufficient numerical rank")
    basis = flat.T @ vectors[:, :rank] / np.sqrt(len(y) * values[:rank])
    for j in range(rank):
        if basis[np.argmax(abs(basis[:, j])), j] < 0:
            basis[:, j] *= -1
    if np.max(abs(basis.T @ (mass @ basis) - np.eye(rank))) > 1e-9:
        raise ValueError("Mass orthogonality")
    return basis, values


def structure_checks(model, c, z1, z2):
    model.eval()
    with torch.no_grad():
        a, b = 0.7, -1.3
        predicted_coefficients = model(c, a * z1 + b * z2)
        expected = a * model(c, z1) + b * model(c, z2)
        scale = torch.linalg.vector_norm(expected, dim=1).clamp_min(1e-12)
        return {
            "zero_load_max": float(model(c, torch.zeros_like(z1)).abs().max()),
            "relative_homogeneity_max": float(
                (
                    torch.linalg.vector_norm(model(c, 2.5 * z1) - 2.5 * model(c, z1), dim=1)
                    / torch.linalg.vector_norm(2.5 * model(c, z1), dim=1).clamp_min(1e-12)
                ).max()
            ),
            "relative_superposition_max": float(
                (torch.linalg.vector_norm(predicted_coefficients - expected, dim=1) / scale).max()
            ),
        }
