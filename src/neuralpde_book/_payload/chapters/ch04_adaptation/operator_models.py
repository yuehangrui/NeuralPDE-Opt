"""Adapted ocpde_dl MIMOONet and RB_DINO building blocks.

See provenance.json for exact source hashes and changes. No imports from legacy.
MIMO uses four multiplicatively fused branches, with a shared enriched trunk.
RB-DINO uses fixed train-only PCA encoders/decoders and a latent GELU network.

Book/code notation:
Book 4.4.2: state and KKT output roles differ. Network parameters and coordinate encodings are kept separate from physical state/control coefficients.
"""

import torch
from torch import nn
from chapters.ch04.models import boundary_factor, laplace


def mlp(n, hidden, out, activation=nn.Tanh):
    layers = []
    for width in hidden:
        layers.extend([nn.Linear(n, width), activation()])
        n = width
    return nn.Sequential(*layers, nn.Linear(n, out))


class MIMO(nn.Module):
    def __init__(self, sensors=289, width=64, latent=48, modes=12, alpha=1e-3):
        super().__init__()
        self.alpha = alpha
        self.branches = nn.ModuleList([mlp(sensors, (width,) * 3, latent) for _ in range(4)])
        self.trunk = mlp(2, (width,) * 2, latent)
        k = (
            torch.cartesian_prod(torch.arange(1, modes + 1), torch.arange(1, modes + 1)).double()
            * torch.pi
        )
        self.register_buffer("frequencies", k)
        self.spectral_trunk = nn.Linear(modes * modes, latent, bias=False)
        self.register_buffer("input_mean", torch.zeros(2, sensors, dtype=torch.float64))
        self.register_buffer("input_scale", torch.ones(2, dtype=torch.float64))
        self.register_buffer("output_scale", torch.ones(2, dtype=torch.float64))

    def basis(self, x):
        s = torch.sin(x[:, 0:1] * self.frequencies[:, 0]) * torch.sin(
            x[:, 1:2] * self.frequencies[:, 1]
        )
        return boundary_factor(x) * self.trunk(x) + self.spectral_trunk(s)

    def forward(self, f, yd, x):
        f = (f - self.input_mean[0]) / self.input_scale[0]
        yd = (yd - self.input_mean[1]) / self.input_scale[1]
        by = self.branches[0](f) * self.branches[1](yd)
        bu = self.branches[2](f) * self.branches[3](yd)
        t = self.basis(x)
        y = (by @ t.T) * self.output_scale[0]
        u = (bu @ t.T) * self.output_scale[1]
        return y, u, -self.alpha * u  # expose physical p; alpha*u+p identically zero

    def residuals(self, f_s, yd_s, x, f_values, yd_values, nonlinear=False):
        # One coordinate graph per instance, no differentiation through a later
        # reshaped/repeated tensor, no no_grad around nonlinear reaction terms.
        states = []
        adjoints = []
        for i in range(len(f_s)):
            xi = x.detach().clone().requires_grad_(True)
            y, u, p = self(f_s[i : i + 1], yd_s[i : i + 1], xi)
            y, u, p = y.T, u.T, p.T
            rs = -laplace(y, xi) - u - f_values[i, :, None]
            ra = -laplace(p, xi) - y + yd_values[i, :, None]
            if nonlinear:
                rs = rs + y**3
                ra = ra + 3 * y * y * p
            states.append(rs[:, 0])
            adjoints.append(ra[:, 0])
        return torch.stack(states), torch.stack(adjoints)


class RBDINO(nn.Module):
    def __init__(self, dimension=17, rank=17, width=64):
        super().__init__()
        self.latent_net = mlp(rank, (width,) * 3, rank, nn.GELU)
        for name, value in (
            ("Psi", torch.eye(dimension, rank)),
            ("Phi", torch.eye(dimension, rank)),
            ("input_mean", torch.zeros(dimension)),
            ("output_mean", torch.zeros(dimension)),
            ("input_scale", torch.ones(rank)),
            ("output_scale", torch.ones(rank)),
        ):
            self.register_buffer(name, value.double())

    def encode(self, x):
        return ((x - self.input_mean) @ self.Psi) / self.input_scale

    def decode(self, z):
        return (z * self.output_scale) @ self.Phi.T + self.output_mean

    def forward(self, x):
        return self.decode(self.latent_net(self.encode(x)))

    def jacobian(self, x):
        return torch.func.vmap(torch.func.jacfwd(self.latent_net))(self.encode(x))

    def physical_jacobian(self, x):
        j = self.jacobian(x) * self.output_scale[None, :, None] / self.input_scale[None, None, :]
        return torch.einsum("or,bru,iu->boi", self.Phi, j, self.Psi)
