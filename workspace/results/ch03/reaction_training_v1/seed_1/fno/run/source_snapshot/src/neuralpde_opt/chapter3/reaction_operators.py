"""Explicit c -> y interface for the fixed-epsilon reaction problem.

Two-channel backbones receive [normalized c, zeros]. The second slot represents
the known fixed forcing, not an estimated random field; no constant-field std
is fitted. Backbones are reusable neural maps and contain no diffusion PDE.

Book/code notation:
Book 3.3: c -> y_theta. Arrays use (batch, nx, ny); the fixed f=1 channel is encoded as zero after normalization. Backbone latent channels are not PDE fields.
"""

import numpy as np
import torch
from torch import nn
from .operator_training import make_model


def fit_reaction_normalization(train, geometry):
    if set(train) != {"c", "y", "sample_id"} or any(
        not str(s).startswith("reaction-v1:train:") for s in train["sample_id"]
    ):
        raise ValueError("Normalization requires explicit reaction training data")
    c, y = train["c"], train["y"]
    if (
        c.shape != y.shape
        or c.shape[1:] != geometry.grid_shape
        or not np.isfinite(c).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Invalid normalization arrays")
    mean = float(np.mean(c.reshape(len(c), -1) @ geometry.weights))
    std = float(np.sqrt(np.mean(geometry.l2_squared(c - mean))))
    rms = float(np.sqrt(np.mean(geometry.l2_squared(y))))
    if std <= 1e-12 or rms <= 1e-12:
        raise ValueError("Degenerate training scale")
    return {
        "schema_version": "ch03-reaction-normalization-v1",
        "fit_split": "train",
        "fit_sample_ids": list(map(str, train["sample_id"])),
        "c_mean": mean,
        "c_std": std,
        "y_rms": rms,
        "output_centered": False,
        "second_channel": "fixed f=1 encoded as zeros",
        "integration": "consistent P1 mass",
    }


class ReactionOperator(nn.Module):
    def __init__(self, backbone, normalization, epsilon=0.01):
        super().__init__()
        if normalization.get("schema_version") != "ch03-reaction-normalization-v1":
            raise ValueError("Cannot load diffusion normalization as reaction")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Invalid fixed diffusion")
        self.backbone = backbone
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32))
        for name in ["c_mean", "c_std", "y_rms"]:
            v = normalization[name]
            if not np.isfinite(v) or (name != "c_mean" and v <= 0):
                raise ValueError("Invalid normalization: " + name)
            self.register_buffer(name, torch.tensor(v, dtype=torch.float32))

    def forward(self, c, **query_options):
        normalized = (c - self.c_mean) / self.c_std
        return self.backbone(normalized, torch.zeros_like(normalized), **query_options) * self.y_rms


def make_reaction_model(spec, normalization, epsilon=0.01):
    return ReactionOperator(make_model(spec), normalization, epsilon)
