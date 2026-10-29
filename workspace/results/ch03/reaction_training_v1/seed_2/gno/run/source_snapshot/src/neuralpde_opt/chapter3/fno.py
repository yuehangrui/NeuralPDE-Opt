"""A small Fourier neural operator for the Chapter 3 common field data.

Arrays use ``indexing='ij'``: axis -2 is x and axis -1 is y.  The same
parameters can be evaluated on another regular grid, but accuracy there must
be measured separately.  This implementation uses zero padding to soften the
periodic wraparound of Fourier layers and optionally enforces the known
homogeneous Dirichlet condition by an output factor.

Book/code notation:
Book 3.3.2: Fourier layers act on latent features. Spectral mode indices, physical coordinates, and solution values have distinct roles; public weight names remain checkpoint-compatible.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class SpectralConv2d(nn.Module):
    """Truncated real-FFT multiplier with disjoint first-axis frequency blocks.

    The first block contains zero and positive x frequencies; the second
    contains negative x frequencies (including x Nyquist for even lengths).
    Block sizes are clipped separately so small grids cannot overlap.  Real
    and imaginary parameters are stored as pairs of real numbers: ordinary
    ``model.float()`` and ``model.double()`` therefore also change the complex
    multiplier precision correctly.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = _positive_integer(in_channels, "in_channels")
        self.out_channels = _positive_integer(out_channels, "out_channels")
        self.modes = _positive_integer(modes, "modes")
        shape = (in_channels, out_channels, modes, modes, 2)
        scale = 1.0 / math.sqrt(2.0 * in_channels)
        self.weight_positive = nn.Parameter(scale * torch.randn(shape))
        self.weight_negative = nn.Parameter(scale * torch.randn(shape))

    @staticmethod
    def _multiply(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # A dtype cast preserves the FFT's complex64/complex128 precision even
        # when this layer is used by itself with an input of another precision.
        real_dtype = values.real.dtype
        multiplier = torch.view_as_complex(weight.to(dtype=real_dtype).contiguous())
        return torch.einsum("bixy,ioxy->boxy", values, multiplier)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != self.in_channels:
            raise ValueError("spectral input must have shape [B, in_channels, H, W]")
        if values.dtype not in (torch.float32, torch.float64):
            raise TypeError("spectral input must use float32 or float64")
        height, width = values.shape[-2:]
        if height < 2 or width < 2:
            raise ValueError("both grid axes must contain at least two points")
        transformed = torch.fft.rfft2(values, norm="ortho")
        output = transformed.new_zeros((values.shape[0], self.out_channels, height, width // 2 + 1))
        positive = min(self.modes, (height + 1) // 2)
        negative = min(self.modes, height // 2)
        y_modes = min(self.modes, width // 2 + 1)
        output[:, :, :positive, :y_modes] = self._multiply(
            transformed[:, :, :positive, :y_modes],
            self.weight_positive[:, :, :positive, :y_modes],
        )
        output[:, :, -negative:, :y_modes] = self._multiply(
            transformed[:, :, -negative:, :y_modes],
            # Stored rows label [-modes, ..., -1]. Keep the tail when clipped
            # so, for example, frequency -1 uses the same parameter on every grid.
            self.weight_negative[:, :, -negative:, :y_modes],
        )
        # Explicit size is essential: irfft2 otherwise assumes an even last axis.
        return torch.fft.irfft2(output, s=(height, width), norm="ortho")


class FNO2d(nn.Module):
    """Joint ``(a, f) -> y`` FNO on closed regular grids of the unit square.

    ``a`` and ``f`` have shape ``[B, H, W]`` and must have the same dtype/device
    as the model.  Use ``model.double()`` for float64.  Inputs may already be
    normalized by a training-set normalizer; physical coordinates are added
    here and are never normalized.  The output has shape ``[B, H, W]``.
    """

    supports_queries = False

    def __init__(self, modes=8, width=16, layers=3, hard_boundary=True):
        super().__init__()
        modes = _positive_integer(modes, "modes")
        width = _positive_integer(width, "width")
        layers = _positive_integer(layers, "layers")
        if not isinstance(hard_boundary, bool):
            raise ValueError("hard_boundary must be a bool")
        self.hard_boundary = hard_boundary
        self.lift = nn.Linear(4, width)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes) for _ in range(layers)]
        )
        self.local_layers = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(layers)]
        )
        self.project = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, 1)
        )

    def representation_metadata(self):
        """Describe finite inputs separately from cross-grid accuracy claims."""
        return {
            "architecture": "fno",
            "input_representation": "all supplied closed regular-grid values of a and f",
            "input_layout": "[batch, x-node, y-node]; unit square; indexing=ij",
            "coordinate_features": ["x", "y"],
            "modes_per_axis": self.spectral_layers[0].modes,
            "width": self.lift.out_features,
            "layers": len(self.spectral_layers),
            "fft_normalization": "ortho",
            "latent_padding_fraction": 0.125,
            "output_representation": "values on the supplied regular grid",
            "supports_queries": False,
            "hard_boundary": self.hard_boundary,
            "transfer_limit": "Compatible grids permit execution; accuracy requires independent references.",
        }

    @staticmethod
    def _coordinates(values: torch.Tensor):
        height, width = values.shape[-2:]
        return torch.meshgrid(
            torch.linspace(0, 1, height, dtype=values.dtype, device=values.device),
            torch.linspace(0, 1, width, dtype=values.dtype, device=values.device),
            indexing="ij",
        )

    def forward(self, a: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        if not isinstance(a, torch.Tensor) or not isinstance(f, torch.Tensor):
            raise TypeError("a and f must be torch tensors")
        if a.ndim != 3 or a.shape[0] == 0 or f.shape != a.shape:
            raise ValueError("a and f must have matching shapes [B, H, W]")
        if a.dtype != f.dtype or a.device != f.device:
            raise ValueError("a and f must have the same dtype and device")
        if a.dtype not in (torch.float32, torch.float64):
            raise TypeError("a and f must use float32 or float64")
        if a.dtype != self.lift.weight.dtype or a.device != self.lift.weight.device:
            raise ValueError("input dtype/device must match the model; use model.to(...)")
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(f).all()):
            raise ValueError("a and f must contain finite values")
        height, width = a.shape[-2:]
        if height < 2 or width < 2:
            raise ValueError("both grid axes must contain at least two points")
        x, y = self._coordinates(a)
        x_batch = x.expand(a.shape[0], -1, -1)
        y_batch = y.expand(a.shape[0], -1, -1)
        values = self.lift(torch.stack((a, f, x_batch, y_batch), dim=-1))
        values = values.permute(0, 3, 1, 2)
        # Pad the latent field by 12.5% on the two upper ends. Coordinates and
        # the boundary factor refer to the original closed domain throughout.
        values = F.pad(values, (0, math.ceil(width / 8), 0, math.ceil(height / 8)))
        for spectral, local in zip(self.spectral_layers, self.local_layers):
            values = F.gelu(spectral(values) + local(values))
        values = values[:, :, :height, :width].permute(0, 2, 3, 1)
        result = self.project(values).squeeze(-1)
        if self.hard_boundary:
            result = result * (16.0 * x * (1.0 - x) * y * (1.0 - y))
        return result
