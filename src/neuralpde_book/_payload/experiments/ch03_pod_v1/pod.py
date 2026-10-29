"""Train-only centered POD in the consistent finite-element mass inner product.

Book/code notation:
Book 3.4.2: POD basis is orthonormal in the P1 mass inner product. Snapshot coefficients are reduced coordinates, not PDE control variables or regularization parameters.
"""

import numpy as np
import torch
from torch import nn


def fit_pod(train, geometry, rank):
    ids = list(map(str, train["sample_id"]))
    y = np.asarray(train["y"], dtype=np.float64)
    if not ids or not all(s.startswith("reaction-v1:train:") for s in ids):
        raise ValueError("POD fit requires training IDs")
    if len(set(ids)) != len(ids) or len(ids) != len(y):
        raise ValueError("Invalid training IDs")
    if y.shape[1:] != geometry.grid_shape or not np.isfinite(y).all():
        raise ValueError("Invalid training fields")
    if type(rank) is not int or not 1 <= rank < len(y):
        raise ValueError("Centered rank must be below sample count")
    flat = y.reshape(len(y), -1)
    if np.max(abs(flat[:, geometry.boundary])) > 1e-14:
        raise ValueError("POD fields must satisfy zero Dirichlet boundary")
    mean = flat.mean(0)
    centered = flat - mean
    gram = centered @ (geometry.mass @ centered.T) / len(y)
    eigenvalues, vectors = np.linalg.eigh((gram + gram.T) / 2)
    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[indices], vectors[:, indices]
    if eigenvalues[0] <= 0 or eigenvalues[rank - 1] <= eigenvalues[0] * 1e-12:
        raise ValueError("Requested rank exceeds numerically resolved snapshot rank")
    if eigenvalues.min() < -1e-10 * eigenvalues[0]:
        raise ValueError("Snapshot Gram matrix is not positive semidefinite")
    modes = centered.T @ vectors[:, :rank] / np.sqrt(len(y) * eigenvalues[:rank])
    for j in range(rank):
        # Resolve sign ambiguity for readable, reproducible stored modes.
        if modes[np.argmax(abs(modes[:, j])), j] < 0:
            modes[:, j] *= -1
    gram_modes = modes.T @ (geometry.mass @ modes)
    defect = float(np.max(abs(gram_modes - np.eye(rank))))
    if defect > 1e-9:
        raise ValueError("Mass orthonormality check failed")
    return {
        "mean": mean,
        "modes": modes,
        "eigenvalues": np.maximum(eigenvalues, 0),
        "fit_sample_ids": np.asarray(ids),
        "grid_shape": np.asarray(geometry.grid_shape),
        "orthogonality_defect": np.asarray(defect),
    }


def coordinates(y, basis, geometry, rank):
    centered = np.asarray(y, dtype=np.float64).reshape(len(y), -1) - basis["mean"]
    return centered @ (geometry.mass @ basis["modes"][:, :rank])


def reconstruct(alpha, basis, rank):
    values = basis["mean"] + np.asarray(alpha, dtype=np.float64) @ basis["modes"][:, :rank].T
    return values.reshape((-1, *map(int, basis["grid_shape"])))


def encode_inputs(c, normalization, stride):
    c = np.asarray(c, dtype=np.float64)
    if (
        type(stride) is not int
        or stride < 1
        or c.ndim != 3
        or any((n - 1) % stride for n in c.shape[1:])
    ):
        raise ValueError("Stride must preserve both grid endpoints")
    if not np.isfinite(c).all() or np.min(c) <= 0:
        raise ValueError("Expected positive finite physical reaction coefficients")
    if (
        not np.isfinite(normalization["c_mean"])
        or not np.isfinite(normalization["c_std"])
        or normalization["c_std"] <= 0
    ):
        raise ValueError("Invalid training normalization")
    return ((c[:, ::stride, ::stride] - normalization["c_mean"]) / normalization["c_std"]).reshape(
        len(c), -1
    )


def prepare_arrays(data, basis, geometry, normalization, rank, stride):
    arrays = {}
    scales = np.sqrt(basis["eigenvalues"][:rank])
    for split, part in data.items():
        alpha = coordinates(part["y"], basis, geometry, rank)
        projection = reconstruct(alpha, basis, rank)
        denominator = geometry.l2_squared(part["y"])
        if np.any(denominator <= 0):
            raise ValueError("Relative error requires nonzero targets")
        arrays[split] = {
            "x": torch.tensor(encode_inputs(part["c"], normalization, stride), dtype=torch.float32),
            "z": torch.tensor(alpha / scales, dtype=torch.float32),
            "alpha": alpha,
            "denominator": torch.tensor(denominator, dtype=torch.float64),
            "projection_squared": torch.tensor(
                geometry.l2_squared(projection - part["y"]), dtype=torch.float64
            ),
        }
    return arrays


class CoefficientNet(nn.Module):
    def __init__(self, input_size, rank, width=128, depth=3):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(input_size, width), nn.GELU()])
            input_size = width
        layers.append(nn.Linear(width, rank))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def coefficient_loss(z_pred, alpha_true, scales, denominator):
    # Float64 metric preserves small POD coefficients while network parameters are float32.
    error = z_pred.to(torch.float64) * scales - alpha_true
    return (error.square().sum(1) / denominator).mean()


def evaluate_coefficients(model, arrays, scales):
    model.eval()
    with torch.no_grad():
        alpha = model(arrays["x"]).to(torch.float64) * scales
        error = alpha - torch.as_tensor(arrays["alpha"], dtype=torch.float64)
        squared = error.square().sum(1) + arrays["projection_squared"]
        score = torch.sqrt(squared / arrays["denominator"]).mean().item()
    return alpha.numpy(), score
