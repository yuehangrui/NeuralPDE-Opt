"""Isolated upstream operators for Chapter 3 train/val adaptation.

Core authorship/licenses/pins are in third_party/ch03_upstream. Existing four
teaching models are unchanged. These adapters are fixed-grid (65 by 65), and
accept already normalized a/f; PhysicalOperator supplies train-only scaling.

Book/code notation:
Adapted author backbones: Fourier-enhanced DeepONet and Galerkin Transformer. Vendored kernels are unchanged; adapters define the PDE inputs, output scaling, and boundary mask.
"""

import math
import torch
from torch import nn
from .upstream_vendor.deepxde_core import DeepONetCartesianProd
from .upstream_vendor.galerkin_model import FourierTransformer2D


def _grid(size):
    axis = torch.arange(size, dtype=torch.float64) / (size - 1)
    return torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1)


def _validate(a, f, size, parameter):
    if not isinstance(a, torch.Tensor) or not isinstance(f, torch.Tensor):
        raise TypeError("a and f must be tensors")
    if a.ndim != 3 or a.shape[0] < 1 or a.shape[1:] != (size, size) or f.shape != a.shape:
        raise ValueError(f"Expected matching nonempty [B,{size},{size}] fields")
    if a.dtype not in (torch.float32, torch.float64) or f.dtype != a.dtype:
        raise TypeError("Expected matching float32/float64 fields")
    if a.device != f.device or a.device != parameter.device or a.dtype != parameter.dtype:
        raise ValueError("Model and fields must share device and dtype")
    if not torch.isfinite(a).all() or not torch.isfinite(f).all():
        raise ValueError("Fields must be finite")


class DeepXDEOperator(nn.Module):
    """Author Cartesian-product core; joint a/f branch, hard Dirichlet adapter.

    The port retains the author's activated trunk output and unscaled inner
    product. The original teaching implementation used a linear trunk output
    and divided the product by sqrt(rank). Fourier coordinates are explicitly
    an adapter ablation, not part of the author's standard core.
    """

    supports_queries = False

    def __init__(
        self,
        grid_size=65,
        sensor_stride=4,
        latent_dim=56,
        width=300,
        depth=3,
        coordinate_modes=0,
        branch_activation="relu",
        trunk_activation="tanh",
    ):
        super().__init__()
        for key, val, lower in [
            ("grid_size", grid_size, 3),
            ("sensor_stride", sensor_stride, 1),
            ("latent_dim", latent_dim, 1),
            ("width", width, 1),
            ("depth", depth, 1),
            ("coordinate_modes", coordinate_modes, 0),
        ]:
            if type(val) is not int or val < lower:
                raise ValueError(key)
        if (grid_size - 1) % sensor_stride:
            raise ValueError("Stride must divide grid_size-1")
        self.grid_size = grid_size
        self.sensor_stride = sensor_stride
        self.coordinate_modes = coordinate_modes
        self.register_buffer("grid", _grid(grid_size))
        n = ((grid_size - 1) // sensor_stride + 1) ** 2
        self.core = DeepONetCartesianProd(
            [2 * n] + [width] * depth + [latent_dim],
            [2 + 4 * coordinate_modes] + [width] * depth + [latent_dim],
            {"branch": branch_activation, "trunk": trunk_activation},
            "Glorot normal",
        )

    def core_inputs(self, a, f):
        s = self.sensor_stride
        branch = torch.cat([a[:, ::s, ::s].flatten(1), f[:, ::s, ::s].flatten(1)], dim=-1)
        p = self.grid.to(a).reshape(-1, 2)
        coord = 2 * p - 1
        if self.coordinate_modes:
            k = torch.arange(1, self.coordinate_modes + 1, device=a.device, dtype=a.dtype)
            phase = math.pi * p.unsqueeze(-1) * k
            coord = torch.cat([coord, phase.sin().flatten(1), phase.cos().flatten(1)], dim=-1)
        boundary = 16 * p[:, 0] * (1 - p[:, 0]) * p[:, 1] * (1 - p[:, 1])
        return branch, coord, boundary

    def forward(self, a, f):
        _validate(a, f, self.grid_size, next(self.parameters()))
        branch, coord, boundary = self.core_inputs(a, f)
        return (self.core((branch, coord)) * boundary).reshape(-1, self.grid_size, self.grid_size)

    def representation_metadata(self):
        return {
            "model": "DeepXDE DeepONetCartesianProd port",
            "supports_queries": False,
            "grid_size": self.grid_size,
            "sensor_stride": self.sensor_stride,
            "coordinate_modes": self.coordinate_modes,
            "boundary": "16*x*(1-x)*y*(1-y)",
            "source": "lululxvi/deepxde@99b6620386d18cefb1549dddb7b7fe468cfad607",
            "license": "LGPL-2.1 core; separate source and notices included",
        }


class GalerkinOperator(nn.Module):
    """Author Darcy architecture: CNN scales, six Galerkin blocks, FFT decoder.

    Adaptations: two input fields, 65->33->17 attention grid, 17->33->65
    decoder grid, fixed zero Dirichlet boundary. Author dropout bug is patched
    in the vendored layers; no dependency on original data/training loaders.
    """

    supports_queries = False

    def __init__(self, grid_size=65, width=128, layers=6, heads=4, modes=12, dropout=0.05):
        super().__init__()
        if grid_size != 65:
            raise ValueError("This port is configured for public 65x65 grid")
        if (
            type(width) is not int
            or width < 6
            or type(heads) is not int
            or heads < 1
            or width % heads
        ):
            raise ValueError("width must be >=6 and divisible by heads")
        if type(layers) is not int or layers < 1 or type(modes) is not int or not 1 <= modes <= 32:
            raise ValueError("Invalid layers or Fourier modes")
        if not 0 <= dropout < 1:
            raise ValueError("Invalid dropout")
        self.grid_size = grid_size
        self.register_buffer("grid", _grid(grid_size))
        self.register_buffer("attention_grid", _grid(17).reshape(-1, 2))
        self.core_config = dict(
            node_feats=2,
            pos_dim=2,
            n_targets=1,
            n_hidden=width,
            num_feat_layers=0,
            num_encoder_layers=layers,
            n_head=heads,
            dim_feedforward=2 * width,
            feat_extract_type=None,
            attention_type="galerkin",
            xavier_init=0.01,
            diagonal_weight=0.01,
            symmetric_init=False,
            layer_norm=False,
            attn_norm=True,
            norm_eps=1e-7,
            batch_norm=False,
            return_attn_weight=False,
            return_latent=False,
            decoder_type="ifft2",
            spacial_dim=2,
            spacial_fc=True,
            upsample_mode="interp",
            downsample_mode="interp",
            freq_dim=32,
            boundary_condition="dirichlet",
            num_regressor_layers=2,
            fourier_modes=modes,
            regressor_activation="silu",
            downscaler_activation="relu",
            upscaler_activation="silu",
            last_activation=True,
            dropout=0.0,
            downscaler_dropout=dropout,
            upscaler_dropout=0.0,
            ffn_dropout=dropout,
            encoder_dropout=dropout,
            decoder_dropout=0.0,
            debug=False,
            downscaler_size=[33, 17],
            upscaler_size=[33, 65],
        )
        self.core = FourierTransformer2D(**self.core_config)

    def core_inputs(self, a, f):
        return dict(
            node=torch.stack([a, f], dim=-1),
            edge=None,
            pos=self.attention_grid.to(a).unsqueeze(0).expand(a.shape[0], -1, -1),
            grid=self.grid.to(a).unsqueeze(0).expand(a.shape[0], -1, -1, -1),
        )

    def forward(self, a, f):
        _validate(a, f, self.grid_size, next(self.parameters()))
        return self.core(**self.core_inputs(a, f))["preds"].squeeze(-1)

    def representation_metadata(self):
        return {
            "model": "Galerkin Transformer author Darcy port",
            "supports_queries": False,
            "source": "scaomath/galerkin-transformer@f9e7d6abc24e958b1165f48b22625c5bab688043",
            "core_config": self.core_config,
            "axis_order": "ij: x then y",
            "boundary": "Author zero boundary nodes; no polynomial output factor",
            "license": "MIT; source and notices included",
        }


FACTORIES = {"deepxde": DeepXDEOperator, "galerkin": GalerkinOperator}


def make_model(spec):
    if set(spec) != {"name", "kwargs"} or spec["name"] not in FACTORIES:
        raise ValueError("Unknown upstream model specification")
    return FACTORIES[spec["name"]](**spec["kwargs"])
