"""Retain frozen legacy backbones; adapt representation to the declared family.

The 17-mode range is a declared finite-family restriction, not a universal
operator. No PDE inverse or exact KKT formula occurs in model inference.
The optional final head is learned from training input/solution pairs only.

Book/code notation:
Book 4.4.2: physical outputs are reconstructed from standardized coefficients. Pure neural and train-only head-polished stages retain separate checkpoint identities.
"""

import numpy as np
import torch
from torch import nn
from chapters.ch04_operator_learning.models import FNO, DeepONet
from chapters.ch04_operator_learning.reference import Poisson
from chapters.ch04_adaptation.operator_data import basis, MODES


class AdaptedOperator(nn.Module):
    def __init__(self, architecture="fno", role="state", n=32, small=False):
        super().__init__()
        if (
            architecture not in ("fno", "deeponet", "fedonet")
            or role not in ("state", "kkt")
            or n != 32
        ):
            raise ValueError("expected supported architecture/role on n=32")
        self.architecture, self.role, self.n = architecture, role, n
        self.outputs = 1 if role == "state" else 2
        self.dimension = 17 if role == "state" else 34
        self.ref = Poisson(n)
        b = basis(self.ref.points.reshape(-1, 2)) * self.ref.mask.reshape(-1, 1)
        for name, value in dict(
            B=b,
            mask=self.ref.mask,
            eigenvalues=np.pi**2 * (MODES**2).sum(1),
            input_mean=np.zeros(self.dimension),
            input_scale=np.ones(self.dimension),
            output_mean=np.zeros((self.outputs, 17)),
            output_scale=np.ones((self.outputs, 17)),
        ).items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.float64))
        if architecture == "fno":
            self.net = FNO(
                self.outputs,
                modes=3 if small else 12,
                width=4 if small else 32,
                layers=2 if small else 4,
            ).double()
            self.sine_head = None
        else:
            latent = 6 if small else 56
            self.net = DeepONet(
                n,
                self.outputs,
                width=12 if small else 300,
                depth=2 if small else 3,
                latent=latent,
                fourier=architecture == "fedonet",
                n_freq=8 if small else 64,
            ).double()
            self.sine_head = nn.Linear(latent, self.outputs * 17, bias=False, dtype=torch.float64)
            nn.init.normal_(self.sine_head.weight, std=0.01)
        self.register_buffer(
            "head",
            torch.zeros(self.dimension + self.outputs * 17, self.outputs * 17, dtype=torch.float64),
        )
        self.register_buffer("head_active", torch.tensor(False))
        self.small = small

    def encode_fields(self, x):
        return 4 * x.flatten(-2) @ self.B / self.n**2

    def inputs(self, x, check=True):
        if x.ndim != 4 or x.shape[1:] != (2, 33, 33):
            raise ValueError("expected (batch,2,33,33)")
        if check:
            if not bool(torch.isfinite(x).all()):
                raise ValueError("nonfinite input")
            c = self.encode_fields(x * self.mask)
            back = (c @ self.B.T).reshape_as(x)
            rel = (back - x * self.mask).flatten(1).norm(dim=1) / x.flatten(1).norm(
                dim=1
            ).clamp_min(1e-12)
            if bool((rel > 1e-8).any()):
                raise ValueError("input outside the declared 17-mode family")
        else:
            c = self.encode_fields(x * self.mask)
        return c.sum(1) if self.role == "state" else c.flatten(1)

    def set_normalization(self, x, y):
        with torch.no_grad():
            c = self.inputs(x)
            t = self.encode_fields(y)
            for name, value in (
                ("input_mean", c.mean(0)),
                ("input_scale", c.std(0, unbiased=False)),
                ("output_mean", t.mean(0)),
                ("output_scale", t.std(0, unbiased=False)),
            ):
                if name.endswith("scale"):
                    value = value.clamp_min(1e-10)
                getattr(self, name).copy_(value)

    def standardized(self, z):
        # State depends on f+u; use one normalized RHS duplicated into the
        # two-channel inherited interface. The split cannot create fake gradients.
        c = z[:, None, :].expand(-1, 2, -1) if self.role == "state" else z.reshape(-1, 2, 17)
        field = (c @ self.B.T).reshape(-1, 2, 33, 33) / np.sqrt(17)
        if self.architecture == "fno":
            raw = self.net(field)
        else:
            branch = self.net.branch1(field[:, 0].flatten(1)) + self.net.branch2(
                field[:, 1].flatten(1)
            )
            xy = self.net.points
            if self.net.fourier:
                phase = 2 * torch.pi * (xy @ self.net.B.T)
                xy = torch.cat([phase.sin(), phase.cos()], 1)
            trunk = self.net.trunk(xy).reshape(-1, self.outputs, self.net.latent)
            raw = torch.einsum("bp,nop->bon", branch, trunk).reshape(-1, self.outputs, 33, 33)
        coeff = self.encode_fields(raw * self.mask)
        if self.sine_head is not None:
            coeff = coeff + self.sine_head(branch).reshape(-1, self.outputs, 17)
        return coeff

    def neural_coefficients(self, c):
        return self.output_mean + self.output_scale * self.standardized(
            (c - self.input_mean) / self.input_scale
        )

    def head_features(self, c):
        q = self.standardized((c - self.input_mean) / self.input_scale).flatten(1)
        q0 = self.standardized((-self.input_mean / self.input_scale)[None]).flatten(1)
        return torch.cat([c / self.input_scale, q - q0], 1)

    def coefficients(self, c):
        if bool(self.head_active):
            out = self.head_features(c) @ self.head
            return out.reshape(-1, self.outputs, 17) * self.output_scale
        return self.neural_coefficients(c)

    def forward(self, x, check=True):
        c = self.inputs(x, check)
        out = self.coefficients(c)
        return (out @ self.B.T).reshape(-1, self.outputs, 33, 33)

    def fields(self, x):
        if self.role != "kkt":
            raise ValueError("KKT role required")
        y, u = self(x).unbind(1)
        return y, u, -0.001 * u

    def fit_head(self, x, y):
        """Freeze neural features and solve the TRAIN-only global linear head.

        Known linearity is encoded by including raw input coefficients. SVD fits
        all weights jointly; no reference eigenvalues/solution formula is used.
        Thus gains may be dominated by the linear component and must be reported
        alongside a standalone linear baseline and the pre-polish neural model.
        """
        with torch.no_grad():
            c = self.inputs(x)
            design = self.head_features(c).cpu().numpy()
            target = (self.encode_fields(y) / self.output_scale).flatten(1).cpu().numpy()
            scale = np.linalg.norm(design, axis=0).clip(1e-12)
            # Penalize only the nonlinear-feature weights, removing rank-deficient
            # ambiguities without biasing the known homogeneous linear subspace.
            ridge = np.zeros((design.shape[1] - self.dimension, design.shape[1]))
            ridge[:, self.dimension :] = 1e-3 * np.eye(design.shape[1] - self.dimension)
            augmented = np.concatenate([design / scale, ridge])
            rhs = np.concatenate([target, np.zeros((len(ridge), target.shape[1]))])
            solution, _, rank, s = np.linalg.lstsq(augmented, rhs, rcond=1e-12)
            solution = solution / scale[:, None]
            if not np.isfinite(solution).all():
                raise FloatingPointError("nonfinite fitted head")
            self.head.copy_(torch.as_tensor(solution, device=self.head.device))
            self.head_active.fill_(True)
            error = np.linalg.norm(design @ solution - target) / np.linalg.norm(target)
        return dict(
            rank=int(rank),
            features=design.shape[1],
            condition=float(s[0] / s[-1]),
            nonlinear_penalty=1e-6,
            training_relative_residual=float(error),
            fit_split="train",
            uses_pde_inverse=False,
            neural_head_weight_norm=float(np.linalg.norm(solution[self.dimension :])),
            linear_head_weight_norm=float(np.linalg.norm(solution[: self.dimension])),
        )
