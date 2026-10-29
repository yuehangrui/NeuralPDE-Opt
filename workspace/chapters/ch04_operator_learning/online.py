"""Book 4.4.2: fixed learned state operator + task-specific neural control.

This is distinct from RB-DINO's classical coefficient optimization. Every
reported online iterate is re-solved with the independent spectral PDE.
"""

from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn
from .training import load, write
from .data import load as load_data, arrays
from .reference import Poisson
from .models import mlp


class ControlNet(nn.Module):
    def __init__(self, ref, modes=12):
        super().__init__()
        x = torch.tensor(ref.points.reshape(-1, 2), dtype=torch.float64)
        k = (
            torch.cartesian_prod(torch.arange(1, modes + 1), torch.arange(1, modes + 1)).double()
            * torch.pi
        )
        self.register_buffer("points", x)
        self.register_buffer(
            "basis", torch.sin(x[:, 0:1] * k[:, 0]) * torch.sin(x[:, 1:2] * k[:, 1])
        )
        self.register_buffer("mask", torch.tensor(ref.mask))
        self.n = ref.n
        self.net = mlp(2, 64, 3, 1).double()
        self.coefficients = nn.Parameter(torch.zeros(modes * modes, dtype=torch.float64))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self):
        return (self.net(self.points)[:, 0] + self.basis @ self.coefficients).reshape(
            self.n + 1, self.n + 1
        ) * self.mask


def derivative_probe(model, inputs, ref):
    model = model.eval()
    param = next(model.parameters())
    rng = np.random.default_rng(944)
    rows = []
    for i in range(min(4, len(inputs))):
        x = torch.tensor(
            inputs[i : i + 1], dtype=param.dtype, device=param.device, requires_grad=True
        )
        # Smooth observable and perturbation, defined without solution labels.
        noise = rng.normal(size=(ref.n + 1, ref.n + 1)) * ref.mask
        w = ref.solve(noise)
        w = w / max(float(ref.norm(w)), 1e-14)
        v = ref.solve(rng.normal(size=noise.shape))
        v = v / max(float(ref.norm(v)), 1e-14)
        wt = torch.tensor(w, dtype=x.dtype, device=x.device)
        value = (model(x)[0, 0] * wt).sum() / ref.n**2
        control_gradient = torch.autograd.grad(value, x)[0][0, 1].detach().cpu().numpy()
        truth = ref.solve(w) / ref.n**2
        h = 1e-4
        direction = torch.zeros_like(x)
        direction[0, 1] = torch.tensor(v, device=x.device, dtype=x.dtype)
        with torch.no_grad():
            fd = float(
                ((model(x + h * direction)[0, 0] - model(x - h * direction)[0, 0]) * wt).sum()
                / (2 * h * ref.n**2)
            )
        ad = float(np.sum(control_gradient * v))
        if not np.isclose(fd, ad, rtol=2e-4, atol=1e-8):
            raise AssertionError("surrogate AD/FD inconsistency")
        rows.append(
            dict(
                sample=i,
                adjoint_action_relative=float(ref.norm(control_gradient - truth) / ref.norm(truth)),
                derivative_ad=ad,
                derivative_fd=fd,
                ad_fd_absolute_error=abs(ad - fd),
            )
        )
    return dict(
        role="validation",
        mean_adjoint_action_relative=float(np.mean([r["adjoint_action_relative"] for r in rows])),
        rows=rows,
    )


def run(checkpoint, data_path, output, device="cuda"):
    torch.set_num_threads(2)
    model, cfg, ck = load(checkpoint, device, torch.float64)
    if cfg.role != "state":
        raise ValueError("online optimization requires a state operator")
    _, va, manifest = load_data(data_path)
    if manifest["hashes"] != ck["data_hashes"]:
        raise ValueError("online data identity mismatch")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    ref = Poisson(cfg.n, cfg.alpha)
    model.eval().requires_grad_(False)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    probe = derivative_probe(model, arrays(va, "state")[0], ref)
    write(root / "derivatives.json", probe)
    rows = []
    predictions = {}
    for i in range(2):
        torch.manual_seed(444 + i)
        control = ControlNet(ref).to(device)
        f = torch.tensor(va["f"][i], device=device, dtype=torch.float64)
        yd = torch.tensor(va["yd"][i], device=device, dtype=torch.float64)
        trace = []
        _, _, g0, _ = ref.recover(va["f"][i], va["yd"][i], np.zeros_like(va["f"][i]))

        def objective():
            u = control()
            y = model(torch.stack([f, u])[None])[0, 0]
            return 0.5 * ((y - yd) ** 2 + cfg.alpha * u * u).sum() / ref.n**2

        def record(stage):
            with torch.no_grad():
                u = control().cpu().numpy()
                yp = model(torch.stack([f, control()])[None])[0, 0].cpu().numpy()
            yr, _, g, j = ref.recover(va["f"][i], va["yd"][i], u)
            trace.append(
                dict(
                    stage=stage,
                    J_true=float(j),
                    J_surrogate=float(
                        0.5 * ref.inner(yp - va["yd"][i], yp - va["yd"][i])
                        + 0.5 * cfg.alpha * ref.inner(u, u)
                    ),
                    control_relative_l2=float(ref.norm(u - va["u"][i]) / ref.norm(va["u"][i])),
                    true_gradient_relative=float(ref.norm(g) / ref.norm(g0)),
                    state_surrogate_relative=float(ref.norm(yp - yr) / ref.norm(yr)),
                )
            )

        record("initial")
        opt = torch.optim.Adam(control.parameters(), lr=0.01)
        for step in range(1, 101):
            opt.zero_grad(set_to_none=True)
            loss = objective()
            loss.backward()
            if not bool(torch.isfinite(loss)) or any(
                p.grad is not None and not bool(torch.isfinite(p.grad).all())
                for p in control.parameters()
            ):
                raise FloatingPointError("nonfinite control optimization")
            opt.step()
            if step % 20 == 0:
                record("adam:" + str(step))
        opt = torch.optim.LBFGS(
            control.parameters(),
            lr=1.0,
            max_iter=100,
            tolerance_grad=1e-8,
            tolerance_change=1e-12,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            opt.zero_grad(set_to_none=True)
            value = objective()
            value.backward()
            if not bool(torch.isfinite(value)) or any(
                p.grad is not None and not bool(torch.isfinite(p.grad).all())
                for p in control.parameters()
            ):
                raise FloatingPointError("nonfinite LBFGS control update")
            return value

        opt.step(closure)
        record("lbfgs:last")
        rows.append(
            dict(
                id=str(va["ids"][i]),
                closure_calls=calls,
                lbfgs_iterations=int(opt.state[next(control.parameters())]["n_iter"]),
                trajectory=trace,
            )
        )
        predictions[str(i)] = control().detach().cpu().numpy()
        torch.save(control.cpu().state_dict(), root / ("control_" + str(i) + ".pt"))
    if any(
        not torch.equal(v, before[k])
        for k, v in {k: v.detach().cpu() for k, v in model.state_dict().items()}.items()
    ):
        raise AssertionError("frozen surrogate changed")
    np.savez_compressed(root / "controls.npz", **predictions)
    result = dict(
        role="validation",
        selection_use=False,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        fixed_surrogate_verified=True,
        online_method="neural control, Adam 100 + LBFGS up to 100",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows,
        test_evaluations=0,
    )
    write(root / "online.json", result)
    print(cfg.architecture, "online validation finished", flush=True)
