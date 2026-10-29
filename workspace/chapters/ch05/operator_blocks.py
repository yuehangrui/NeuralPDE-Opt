"""Adapted ocpde_dl iUzawa/FNO blocks; source hashes in provenance.json.

Chapter-5 variant: fixed training resolution, exact box prox, zero-preserving
state/adjoint surrogates, corrected matrix-Gram SAP kernel. Weight sharing
between nonlinear FNOs is NOT a proof of a Hilbert adjoint. No mesh-transfer
or convergence guarantee is claimed. Public training uses sa_type='sa' for
the self-adjoint Poisson target and reports its empirical pairing defect.

Book/code notation:
Book (5.12)-(5.13) overloads p: the unrolled p_dual is an observation-space dual (equals y-yd at an exact fixed point). Physical PDE adjoint is S^*p_dual. Preserve this distinction from q=p_physical/alpha used by OSNN.
"""

from __future__ import annotations

import itertools
import math
from typing import Literal, Sequence, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

einsum_symbols = "bioacdeghjklmnqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

SATypes = Literal["sa", "sa_time", "none"]


class UzawaModuleParams(TypedDict):
    """Widths of the learnable modules inside one Uzawa layer."""

    qa_width: int  # retained source configuration key; exact prox has no weights
    qb_width: int  # channel width of the SAP Fourier layer in Q_B
    qb_width_pre_lift: int  # pre-lifting width of Q_B
    fno_layers: int  # number of Fourier layers in S (and S*)
    fno_width: int  # channel width of S
    fno_width_pre_proj: int  # pre-projection width of S


class LinearFourier(nn.Module):
    """Spectral convolution layer, optionally self-adjoint positive (SAP).

    Input and output live on the same grid: (B, C_in, d1, ..., dn) ->
    (B, C_out, d1, ..., dn). This port accepts only ``train_res``. The original
    cross-grid subsampling branch did not preserve the adjoint pairing.

    With ``is_sap=True`` the spectral kernel is parameterized as Φ·Φ* and the
    pointwise matrix as V·Vᵀ, so the layer is self-adjoint positive
    semidefinite — the structural property required of the preconditioner Q_B.
    """

    def __init__(
        self,
        domain_dim: int,
        input_channels: int,
        output_channels: int,
        modes: Sequence[int],
        train_res: Sequence[int],
        is_sap: bool = False,
    ) -> None:
        super().__init__()

        if is_sap:
            assert input_channels == output_channels, (
                "Input and output channels must be equal for self-adjoint " "positive layers."
            )
        self.input_domain_dim = domain_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.modes = modes
        assert len(self.modes) == self.input_domain_dim
        if any(type(m) is not int or m < 1 for m in modes):
            raise ValueError("positive integer modes required")
        if any(2 * m + 1 > r for m, r in zip(modes, train_res)):
            raise ValueError("Fourier bands overlap or exceed grid resolution")
        self.is_self_adjoint = is_sap
        self.train_res = train_res
        assert (
            len(self.train_res) == self.input_domain_dim
        ), "Fourier resolution must match the input domain dimension."

        phi_size = (
            [self.input_channels, self.output_channels]
            + [2 * m + 1 for m in self.modes[:-1]]
            + [self.modes[-1] + 1]
        )
        init_normal_std = 1.0 / (self.input_channels + self.output_channels)
        self.phi = nn.Parameter(
            torch.normal(0.0, init_normal_std, size=phi_size, dtype=torch.complex64)
        )
        self.local_mat = nn.Parameter(
            torch.normal(
                0.0,
                init_normal_std,
                size=(self.input_channels, self.output_channels),
            )
        )

        domain_str = einsum_symbols[-self.input_domain_dim :]
        self.einsum_str_fourier = "bi" + domain_str + ",io" + domain_str + "->bo" + domain_str
        self.einsum_str_local = "bi" + domain_str + ",io->bo" + domain_str

    def _complex_mul_fourier(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum(self.einsum_str_fourier, x, weights)

    def _complex_mul_local(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum(self.einsum_str_local, x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.ndim == len(self.train_res) + 2
        ), "Input tensor must have shape (batch_size, input_channels, d1, ..., dn)."
        assert tuple(x.shape[2:]) == tuple(
            self.train_res
        ), "This audited port only supports the training resolution."

        input_resample_factor = [size // res for res, size in zip(self.train_res, x.shape[2:])]
        input_resample_index = [slice(None, None, rf) for rf in input_resample_factor]

        x_ft = x[(..., *input_resample_index)]
        x_ft = torch.fft.rfftn(
            x_ft,
            dim=tuple(range(-self.input_domain_dim, 0)),
            norm="ortho",
        )
        output_ft_shape = (
            x.size()[0:1] + (self.output_channels,) + x.size()[2:-1] + (x.size(-1) // 2 + 1,)
        )
        output_ft = torch.zeros(output_ft_shape, dtype=x_ft.dtype, device=x_ft.device)

        mode_indices = [((None, m + 1), (-m, None)) for m in self.modes[:-1]] + [
            ((None, self.modes[-1] + 1),)
        ]
        pre_index = (slice(None, None, None), slice(None, None, None))

        if self.is_self_adjoint:
            kernel = torch.einsum("ik...,jk...->ij...", self.phi, self.phi.conj())
            mat = torch.matmul(self.local_mat, self.local_mat.transpose(-2, -1))
        else:
            kernel = self.phi
            mat = self.local_mat

        for boundaries in itertools.product(*mode_indices):
            mode_index = tuple(slice(*b) for b in boundaries)
            full_index = pre_index + mode_index
            output_ft[full_index] = self._complex_mul_fourier(x_ft[full_index], kernel[full_index])

        output_ft = torch.fft.irfftn(
            output_ft,
            s=x.shape[2:],
            dim=tuple(range(-self.input_domain_dim, 0)),
            norm="ortho",
        )
        output_ft = (
            torch.sqrt(
                torch.prod(
                    torch.tensor(
                        input_resample_factor, dtype=torch.float32, device=output_ft.device
                    )
                )
            )
            * output_ft
        )

        output_local = self._complex_mul_local(x, mat)
        return output_ft + output_local


class SAPFourier(nn.Module):
    """Learned self-adjoint positive semidefinite operator Q_B.

    The symmetric sandwich  Wᵀ · (SAP Fourier layer) · W  with learnable
    lifting W keeps the whole map self-adjoint positive semidefinite.
    Channels-last: (B, d1, ..., dn, C).
    """

    def __init__(
        self,
        domain_dim: int,
        channels: int,
        modes: Sequence[int],
        width: int,
        width_pre_lifting: int,
        train_res: Sequence[int],
        padding: Sequence[int] | None = None,
    ):
        super().__init__()

        self.domain_dim = domain_dim
        self.channels = channels
        self.modes = modes
        self.width = width
        self.width_pre_lifting = width_pre_lifting
        self.train_res = train_res
        self.padding = padding

        fourier_layers_train_res = (
            [tr + pd for tr, pd in zip(self.train_res, self.padding)]
            if self.padding is not None
            else self.train_res
        )
        self.pre_lifting_mat = nn.Parameter(
            torch.empty(self.channels, self.width_pre_lifting, dtype=torch.float32)
        )
        self.lifting = nn.Parameter(
            torch.empty(self.width_pre_lifting, self.width, dtype=torch.float32)
        )
        self.fourier_layer_sap = LinearFourier(
            domain_dim=self.domain_dim,
            input_channels=self.width,
            output_channels=self.width,
            modes=self.modes,
            train_res=fourier_layers_train_res,
            is_sap=True,
        )
        nn.init.kaiming_uniform_(self.pre_lifting_mat, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lifting, a=math.sqrt(5))

        domain_str = einsum_symbols[-self.domain_dim :]
        self.einsum_str = f"b{domain_str}i,io->b{domain_str}o"

    def _mat_mul(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum(self.einsum_str, x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._mat_mul(x, self.pre_lifting_mat)
        x = self._mat_mul(x, self.lifting)
        x = x.permute(0, -1, *list(range(1, x.ndim - 1)))

        if self.padding is not None:
            # Scale the padding with resolution so non-periodic inputs keep
            # working at any grid of the 2^k + 1 family.
            resample_factor = tuple(
                (size - 1) // (res - 1) for res, size in zip(self.train_res, x.shape[2:])
            )
            padding_size = tuple(rf * (pd + 1) - 1 for rf, pd in zip(resample_factor, self.padding))
            pad_info = tuple(dim_info for ps in reversed(padding_size) for dim_info in (0, ps))
            unpad_info = tuple(slice(None, -ps) if ps > 0 else slice(None) for ps in padding_size)
            x = F.pad(x, pad=pad_info, value=0.0)

        x = self.fourier_layer_sap(x)

        if self.padding is not None:
            x = x[(..., *unpad_info)]

        x = x.permute(0, *list(range(2, x.ndim)), 1)
        x = self._mat_mul(x, self.lifting.transpose(-2, -1))
        x = self._mat_mul(x, self.pre_lifting_mat.transpose(-2, -1))
        return x


class FNO(nn.Module):
    """Standard Fourier neural operator, used for the PDE solution operator S.

    Channels-last: (B, d1, ..., dn, C_in) -> (B, d1, ..., dn, C_out).
    Normalized coordinate channels are appended before the lifting.
    """

    def __init__(
        self,
        domain_dim: int,
        input_channels: int,
        output_channels: int,
        layers: int,
        modes: Sequence[int],
        width: int,
        width_pre_proj: int,
        train_res: Sequence[int],
        padding: Sequence[int] | None = None,
    ) -> None:
        super().__init__()

        assert len(modes) == domain_dim, "Number of modes must match the domain dimension."
        assert (
            padding is None or len(padding) == domain_dim
        ), "Padding must be None or match the domain dimension."
        if padding is not None:
            assert all(p >= 0 for p in padding), "Padding values must be non-negative."

        self.domain_dim = domain_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.modes = modes
        self.width = width
        self.width_pre_proj = width_pre_proj
        self.layers = layers
        self.padding = padding
        self.train_res = train_res

        fourier_layers_train_res = (
            [tr + pd for tr, pd in zip(self.train_res, self.padding)]
            if self.padding is not None
            else self.train_res
        )
        self.lifting = nn.Linear(self.domain_dim + self.input_channels, self.width)
        self.fourier_layers = nn.ModuleList(
            [
                LinearFourier(
                    domain_dim=domain_dim,
                    input_channels=self.width,
                    output_channels=self.width,
                    modes=modes,
                    train_res=fourier_layers_train_res,
                    is_sap=False,
                )
                for _ in range(layers)
            ]
        )
        self.pre_projection = nn.Linear(self.width, self.width_pre_proj)
        self.projection = nn.Linear(self.width_pre_proj, self.output_channels)
        self.activation = nn.GELU()

    def _get_grid(self, shape, device):
        batch_size, sizes = shape[0], shape[1:-1]
        grids = torch.meshgrid(
            *[torch.linspace(0, 1, size, dtype=torch.float, device=device) for size in sizes],
            indexing="ij",
        )
        grids = [
            g.reshape(1, *sizes, 1).repeat(batch_size, *(1,) * self.domain_dim, 1) for g in grids
        ]
        return torch.cat(grids, dim=-1).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self._get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.lifting(x)
        x = x.permute(0, -1, *list(range(1, x.ndim - 1)))

        if self.padding is not None:
            resample_factor = tuple(
                (size - 1) // (res - 1) for res, size in zip(self.train_res, x.shape[2:])
            )
            padding_size = tuple(rf * (pd + 1) - 1 for rf, pd in zip(resample_factor, self.padding))
            pad_info = tuple(dim_info for ps in reversed(padding_size) for dim_info in (0, ps))
            unpad_info = tuple(slice(None, -ps) if ps > 0 else slice(None) for ps in padding_size)
            x = F.pad(x, pad=pad_info, value=0.0)

        for layer in self.fourier_layers:
            x = layer(x)
            x = self.activation(x)

        if self.padding is not None:
            x = x[(..., *unpad_info)]
        x = x.permute(0, *list(range(2, x.ndim)), 1)
        x = self.pre_projection(x)
        x = self.activation(x)
        x = self.projection(x)
        return x


class UzawaLayer(nn.Module):
    """One unrolled inexact-Uzawa iteration (one layer of the iUzawa-Net).

    Parameters
    ----------
    sa_type : how the learned adjoint surrogate is represented:
        'sa'      — tied state/adjoint weights, not an algebraic adjoint guarantee,
        'sa_time' — S* = flip_t ∘ S ∘ flip_t (parabolic; time is the last
                    spatial axis, i.e. axis -2 of the channels-last tensor),
        'none'    — an independent FNO.
    Explicit lower/upper fields are required by this port's exact box prox.
    """

    def __init__(
        self,
        domain_dim: int,
        channels: int,
        modes: Sequence[int],
        train_res: Sequence[int],
        alpha: float,
        module_params: UzawaModuleParams,
        padding: Sequence[int] | None = None,
        sa_type: SATypes | None = None,
        with_bounds: bool = True,
    ):
        super().__init__()

        self.domain_dim = domain_dim
        self.channels = channels
        self.modes = modes
        self.alpha = alpha
        self.with_bounds = with_bounds

        self.qb_sap_fourier_layers = SAPFourier(
            domain_dim=domain_dim,
            channels=channels,
            modes=modes,
            width=module_params["qb_width"],
            width_pre_lifting=module_params["qb_width_pre_lift"],
            train_res=train_res,
            padding=padding,
        )
        self.qa_delta = 10.0  # Q_A = alpha*(1+delta) I; tau=alpha*delta

        self.S_oper = FNO(
            domain_dim=domain_dim,
            input_channels=channels,
            output_channels=channels,
            layers=module_params["fno_layers"],
            modes=modes,
            width=module_params["fno_width"],
            width_pre_proj=module_params["fno_width_pre_proj"],
            train_res=train_res,
            padding=padding,
        )
        self.sa_type: SATypes = "none" if sa_type is None else sa_type
        if self.sa_type == "none":
            self.S_adj_fno = FNO(
                domain_dim=domain_dim,
                input_channels=channels,
                output_channels=channels,
                layers=module_params["fno_layers"],
                modes=modes,
                width=module_params["fno_width"],
                width_pre_proj=module_params["fno_width_pre_proj"],
                train_res=train_res,
                padding=padding,
            )
        elif self.sa_type not in ("sa", "sa_time"):
            raise ValueError(f"Invalid self-adjoint type: {sa_type}")

    def state_action(self, x):
        # Respect S(0)=0; small physical scale avoids huge initial prox inputs.
        return (self.S_oper(x) - self.S_oper(torch.zeros_like(x))) / (2 * math.pi**2)

    def S_adj_oper(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the (learned) adjoint S* according to ``sa_type``."""
        if self.sa_type == "sa":
            return self.state_action(x)
        if self.sa_type == "sa_time":
            return self.state_action(x.flip([-2])).flip([-2])
        return (self.S_adj_fno(x) - self.S_adj_fno(torch.zeros_like(x))) / (2 * math.pi**2)

    def _prox_oper(
        self,
        input: torch.Tensor,
        u_a: torch.Tensor | None,
        u_b: torch.Tensor | None,
    ) -> torch.Tensor:
        if u_a is None or u_b is None:
            raise ValueError("explicit box bounds required in this port")
        if (
            not torch.isfinite(u_a).all()
            or not torch.isfinite(u_b).all()
            or bool((u_a > u_b).any())
        ):
            raise ValueError("invalid bounds")
        return torch.maximum(torch.minimum(input, u_b), u_a)

    def _u_step(
        self,
        u: torch.Tensor,
        p_dual: torch.Tensor,
        u_a: torch.Tensor | None,
        u_b: torch.Tensor | None,
    ) -> torch.Tensor:
        u_new = self.qa_delta / (1.0 + self.qa_delta) * u - (
            1.0 / ((1.0 + self.qa_delta) * self.alpha)
        ) * self.S_adj_oper(p_dual)
        return self._prox_oper(u_new, u_a, u_b)

    def _p_step(
        self,
        u: torch.Tensor,
        p_dual: torch.Tensor,
        f: torch.Tensor,
        yd: torch.Tensor,
    ) -> torch.Tensor:
        dual_increment = self.qb_sap_fourier_layers(self.state_action(u + f) - p_dual - yd)
        return p_dual + dual_increment

    def forward(
        self,
        u: torch.Tensor,
        p_dual: torch.Tensor,
        yd: torch.Tensor,
        f: torch.Tensor,
        u_a: torch.Tensor | None = None,
        u_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u = self._u_step(u, p_dual, u_a, u_b)
        p_dual = self._p_step(u, p_dual, f, yd)
        return u, p_dual


class _IUzawaNetBase(nn.Module):
    """Shared constructor arguments and forward loop of the two variants."""

    def __init__(
        self,
        domain_dim: int,
        channels: int,
        modes: Sequence[int],
        train_res: Sequence[int],
        layers: int,
        alpha: float,
        layer_module_params: UzawaModuleParams,
        padding: Sequence[int] | None = None,
        sa_type: SATypes | None = None,
        with_bounds: bool = True,
    ) -> None:
        super().__init__()
        self.domain_dim = domain_dim
        self.channels = channels
        self.modes = modes
        self.layers = layers
        self.alpha = alpha
        self.with_bounds = with_bounds
        self._layer_kwargs = dict(
            domain_dim=domain_dim,
            channels=channels,
            modes=modes,
            alpha=alpha,
            train_res=train_res,
            module_params=layer_module_params,
            padding=padding,
            sa_type=sa_type,
            with_bounds=with_bounds,
        )

    def _iter_layers(self):
        raise NotImplementedError

    def forward(
        self,
        yd: torch.Tensor,
        f: torch.Tensor,
        u_a: torch.Tensor | None = None,
        u_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.with_bounds and (u_a is None or u_b is None):
            raise ValueError(
                "This model was built with with_bounds=True; " "pass u_a and u_b tensors."
            )
        if u_a is None or u_b is None:
            raise ValueError("explicit finite box bounds required")
        # An interior start avoids the all-active zero-gradient initialization
        # caused by replacing the old learned prox with an exact box projection.
        u = (u_a + u_b) / 2
        p_dual = torch.zeros_like(yd)
        for layer in self._iter_layers():
            u, p_dual = layer(u, p_dual, yd, f, u_a, u_b)
        return u


class IUzawaNetS(_IUzawaNetBase):
    """iUzawa-Net-S: one Uzawa layer, weight-tied across all iterations.

    Depth-transferable: ``self.layers`` may be changed after loading a
    checkpoint to unroll more (or fewer) iterations at inference time.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.uzawa_layer = UzawaLayer(**self._layer_kwargs)

    def _iter_layers(self):
        return (self.uzawa_layer for _ in range(self.layers))


class IUzawaNetF(_IUzawaNetBase):
    """iUzawa-Net-F: independent weights for each unrolled layer."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.uzawa_layers = nn.ModuleList(
            [UzawaLayer(**self._layer_kwargs) for _ in range(self.layers)]
        )

    def _iter_layers(self):
        return iter(self.uzawa_layers)
