"""FNO projected-KKT map, paired PDOL surrogates, corrected iUzawa F/S ports.

Book/code notation:
Book 5.2.1: historical method ID mimo selects projected KKT-FNO, not Chapter 4 multiplicative MIMOONet. Its output is decoded to physical (u,y,p); PDOL/iUzawa retain their own iterative roles.
"""

import torch
from torch import nn
from .operator_blocks import FNO, IUzawaNetF, IUzawaNetS
from .reference import Grid, primal_dual


class Operators(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.problem = config.problem()
        c = config
        resolution = (c.grid - 1,) * 2
        kw = dict(domain_dim=2, modes=(c.modes,) * 2, train_res=resolution, padding=(1, 1))

        def fno(inp, out):
            return FNO(
                input_channels=inp,
                output_channels=out,
                layers=c.fno_depth,
                width=c.fno_width,
                width_pre_proj=c.projection_width,
                **kw,
            )

        if c.method == "mimo":
            self.map = fno(2, 2)
        elif c.method == "pdol":
            self.state = fno(1, 1)
            self.adjoint = fno(1, 1)
            for k in ("state_in", "state_out", "adjoint_in", "adjoint_out"):
                self.register_buffer(k, torch.ones(()))
        else:
            cls = IUzawaNetF if c.method == "iuzawa_f" else IUzawaNetS
            params = dict(
                qa_width=c.projection_width,
                qb_width=c.qb_width,
                qb_width_pre_lift=c.projection_width,
                fno_layers=c.fno_depth,
                fno_width=c.fno_width,
                fno_width_pre_proj=c.projection_width,
            )
            self.unrolled = cls(
                channels=1,
                layers=c.uzawa_layers,
                alpha=c.alpha,
                layer_module_params=params,
                sa_type="sa",
                with_bounds=True,
                **kw,
            )
        self.register_buffer("input_scale", torch.ones(2))
        self.register_buffer("output_scale", torch.ones(2))

    def set_scales(self, data):
        def rms(a):
            return max(float((a * a).mean() ** 0.5), 1e-8)

        with torch.no_grad():
            self.input_scale.copy_(
                torch.tensor([rms(data["f"]), rms(data["yd"])], device=self.input_scale.device)
            )
            self.output_scale.copy_(
                torch.tensor(
                    [rms(data["y"]), rms(data["p"] / self.problem.alpha)],
                    device=self.output_scale.device,
                )
            )
            if self.config.method == "pdol":
                for key, arr in (
                    ("state_in", "v"),
                    ("state_out", "sv"),
                    ("adjoint_in", "w"),
                    ("adjoint_out", "sw"),
                ):
                    getattr(self, key).fill_(rms(data[arr]))

    def action(self, v, role):
        scale_in = getattr(self, role + "_in")
        scale_out = getattr(self, role + "_out")
        net = getattr(self, role)
        # S(0)=0 explicitly; centering statistics never use validation/test.
        return (net(v / scale_in) - net(torch.zeros_like(v))) * scale_out

    def forward(self, f, yd):
        if self.config.method == "mimo":
            state_scaled_adjoint = (
                self.map(torch.cat((f, yd), -1) / self.input_scale) * self.output_scale
            )
            y, q_scaled = state_scaled_adjoint[..., 0:1], state_scaled_adjoint[..., 1:2]
            return (
                torch.clamp(-q_scaled, self.problem.lower, self.problem.upper),
                y,
                self.problem.alpha * q_scaled,
            )
        if self.config.method == "pdol":
            u, info = primal_dual(
                lambda v: self.action(v, "state"),
                lambda w: self.action(w, "adjoint"),
                f,
                yd,
                self.problem,
                Grid(self.config.grid).s_norm,
                max_iter=self.config.pd_iterations,
            )
            return u, info
        return self.unrolled(
            yd, f, torch.full_like(f, self.problem.lower), torch.full_like(f, self.problem.upper)
        )
