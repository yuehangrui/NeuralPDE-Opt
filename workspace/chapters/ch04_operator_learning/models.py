"""Adapted from legacy neural_operator/fno.py and deeponet.py.

Retain FNO Fourier+pointwise GELU layers and DeepONet additive branches.
FEDONet retains the random Fourier trunk embedding. Fix FFT shapes/dtypes,
make output channels explicit, apply BCs without in-place autograd mutations.

Book/code notation:
Book 4.4.2: these reusable backbones use latent channels internally. Method role and normalization are defined by the surrounding adapter, not inferred from a generic forward variable name.
"""

import math
import torch
from torch import nn


def mlp(n, width, depth, out):
    layers = []
    for _ in range(depth):
        layers.extend([nn.Linear(n, width), nn.Tanh()])
        n = width
    return nn.Sequential(*layers, nn.Linear(n, out))


class SpectralConv(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        # Real-pair storage follows module.float()/double() consistently.
        scale = 1 / (width * width * math.sqrt(2))
        self.w1 = nn.Parameter(scale * torch.randn(width, width, modes, modes, 2))
        self.w2 = nn.Parameter(scale * torch.randn(width, width, modes, modes, 2))

    def forward(self, x):
        b, c, h, w = x.shape
        m1 = min(self.modes, h // 2)
        m2 = min(self.modes, w // 2 + 1)
        ft = torch.fft.rfft2(x)
        top = torch.einsum(
            "bcxy,coxy->boxy",
            ft[:, :, :m1, :m2],
            torch.view_as_complex(self.w1[:, :, :m1, :m2].contiguous()),
        )
        bottom = torch.einsum(
            "bcxy,coxy->boxy",
            ft[:, :, -m1:, :m2],
            torch.view_as_complex(self.w2[:, :, :m1, :m2].contiguous()),
        )
        middle = ft.new_zeros(b, c, h - 2 * m1, m2)
        left = torch.cat([top, middle, bottom], 2)
        output = torch.cat([left, ft.new_zeros(b, c, h, w // 2 + 1 - m2)], 3)
        return torch.fft.irfft2(output, s=(h, w))


class FNO(nn.Module):
    def __init__(self, outputs=2, modes=12, width=32, layers=4):
        super().__init__()
        self.lift = nn.Linear(2, width)
        self.spectral = nn.ModuleList([SpectralConv(width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, outputs)

    def forward(self, x):
        z = self.lift(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for a, b in zip(self.spectral, self.local):
            z = nn.functional.gelu(a(z) + b(z))
        z = nn.functional.gelu(self.fc1(z.permute(0, 2, 3, 1)))
        return self.fc2(z).permute(0, 3, 1, 2)


class DeepONet(nn.Module):
    def __init__(
        self, n=32, outputs=2, width=300, depth=3, latent=56, fourier=False, n_freq=64, sigma=1.0
    ):
        super().__init__()
        self.outputs, self.latent = outputs, latent
        self.branch1 = mlp((n + 1) ** 2, width, depth, latent)
        self.branch2 = mlp((n + 1) ** 2, width, depth, latent)
        self.fourier = fourier
        if fourier:
            self.register_buffer("B", torch.randn(n_freq, 2) * sigma)
        self.trunk = mlp(2 * n_freq if fourier else 2, width, depth, outputs * latent)
        t = torch.linspace(0, 1, n + 1)
        x, y = torch.meshgrid(t, t, indexing="ij")
        self.register_buffer("points", torch.stack([x, y], -1).reshape(-1, 2))

    def forward(self, x):
        b = self.branch1(x[:, 0].flatten(1)) + self.branch2(x[:, 1].flatten(1))
        coords = self.points
        if self.fourier:
            z = 2 * torch.pi * (coords @ self.B.T)
            coords = torch.cat([torch.sin(z), torch.cos(z)], 1)
        t = self.trunk(coords).reshape(-1, self.outputs, self.latent)
        out = torch.einsum("bp,nop->bon", b, t)
        return out.reshape(len(x), self.outputs, *x.shape[-2:])


class Operator(nn.Module):
    def __init__(self, architecture="fno", role="kkt", n=32, alpha=1e-3, small=False):
        super().__init__()
        if architecture not in ("fno", "deeponet", "fedonet") or role not in ("state", "kkt"):
            raise ValueError("invalid architecture/role")
        self.architecture, self.role, self.n, self.alpha = architecture, role, n, alpha
        outputs = 1 if role == "state" else 2
        if architecture == "fno":
            self.net = FNO(
                outputs,
                modes=3 if small else 12,
                width=4 if small else 32,
                layers=2 if small else 4,
            )
        else:
            self.net = DeepONet(
                n,
                outputs,
                width=12 if small else 300,
                depth=2 if small else 3,
                latent=6 if small else 56,
                fourier=architecture == "fedonet",
                n_freq=8 if small else 64,
            )
        self.register_buffer("input_scale", torch.ones(1, 2, 1, 1))
        self.register_buffer("output_scale", torch.ones(1, outputs, 1, 1))
        t = torch.linspace(0, 1, n + 1)
        x, y = torch.meshgrid(t, t, indexing="ij")
        self.register_buffer("boundary", 16 * x * (1 - x) * y * (1 - y))
        mask = torch.zeros(n + 1, n + 1)
        mask[1:-1, 1:-1] = 1
        self.register_buffer("mask", mask)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (2, self.n + 1, self.n + 1):
            raise ValueError("fixed-grid (batch,2,n+1,n+1) input required")
        z = x * self.mask / self.input_scale
        # Enforce the homogeneous PDE's zero response; subtract in normalized units.
        raw = self.net(z) - self.net(torch.zeros_like(z[:1]))
        return raw * self.boundary * self.output_scale

    def fields(self, x):
        if self.role != "kkt":
            raise ValueError("state surrogate does not directly predict optimal controls")
        out = self(x)
        y, u = out[:, 0], out[:, 1]
        return y, u, -self.alpha * u
