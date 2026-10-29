"""Align neural online control with the declared finite training family.

The original wider-space online pilot is retained in online/. The true optimal
control for every task in this family lies in this 17-mode space. Projection
does not use optimal-control labels or alter the offline checkpoint selection.
"""

from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn
from chapters.ch04_adaptation.operator_data import basis
from .online import ControlNet
from .data import load as load_data, arrays
from .training import load, write
from .reference import Poisson


class SupportedControl(nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.raw = ControlNet(ref)
        self.n = ref.n
        b = basis(ref.points.reshape(-1, 2)) * ref.mask.reshape(-1, 1)
        self.register_buffer("B", torch.tensor(b, dtype=torch.float64))

    def project(self, u):
        return (self.B @ (4 * self.B.T @ u.reshape(-1) / self.n**2)).reshape(self.n + 1, self.n + 1)

    def forward(self):
        return self.project(self.raw())


def run():
    torch.set_num_threads(2)
    root = Path("results/ch04/operator_learning_v1")
    out = root / "online_supported"
    out.mkdir(exist_ok=False)
    _, va, manifest = load_data("data/ch04/operator_learning_v1")
    ref = Poisson()
    summaries = {}
    for architecture in ("fno", "deeponet", "fedonet"):
        ckpath = root / (architecture + "_state") / "best.pt"
        model, cfg, ck = load(ckpath, "cuda", torch.float64)
        if ck["data_hashes"] != manifest["hashes"]:
            raise ValueError("data identity mismatch")
        model.eval().requires_grad_(False)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        folder = out / architecture
        folder.mkdir()
        rows = []
        controls = {}
        b = basis(ref.points.reshape(-1, 2))
        project = lambda u: (b @ (4 * b.T @ u.reshape(-1) / ref.n**2)).reshape(ref.n + 1, ref.n + 1)
        derivative = []
        rng = np.random.default_rng(944)
        for i in range(4):
            inp = arrays(va, "state")[0][i : i + 1]
            x = torch.tensor(inp, device="cuda", dtype=torch.float64, requires_grad=True)
            w = project(ref.solve(rng.normal(size=ref.mask.shape)))
            w /= ref.norm(w)
            wt = torch.tensor(w, device="cuda")
            value = (model(x)[0, 0] * wt).sum() / ref.n**2
            control_gradient = torch.autograd.grad(value, x)[0][0, 1].detach().cpu().numpy()
            truth = ref.solve(w) / ref.n**2
            derivative.append(float(ref.norm(project(control_gradient) - truth) / ref.norm(truth)))
        for i in range(2):
            torch.manual_seed(444 + i)
            control = SupportedControl(ref).cuda()
            f = torch.tensor(va["f"][i], device="cuda")
            yd = torch.tensor(va["yd"][i], device="cuda")
            trace = []
            _, _, g0, _ = ref.recover(va["f"][i], va["yd"][i], np.zeros_like(va["f"][i]))

            def objective():
                u = control()
                y = model(torch.stack([f, u])[None])[0, 0]
                return 0.5 * ((y - yd) ** 2 + ref.alpha * u * u).sum() / ref.n**2

            def record(stage):
                with torch.no_grad():
                    u = control().cpu().numpy()
                    yp = model(torch.stack([f, control()])[None])[0, 0].cpu().numpy()
                yr, p, g, j = ref.recover(va["f"][i], va["yd"][i], u)
                trace.append(
                    dict(
                        stage=stage,
                        J_true=float(j),
                        J_surrogate=float(
                            0.5 * ref.inner(yp - va["yd"][i], yp - va["yd"][i])
                            + 0.5 * ref.alpha * ref.inner(u, u)
                        ),
                        control_relative_l2=float(ref.norm(u - va["u"][i]) / ref.norm(va["u"][i])),
                        true_gradient_relative=float(ref.norm(g) / ref.norm(g0)),
                        support_projection_error=float(ref.norm(u - project(u))),
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
                    raise FloatingPointError("nonfinite supported control")
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

            def closure():
                opt.zero_grad(set_to_none=True)
                loss = objective()
                loss.backward()
                if not bool(torch.isfinite(loss)) or any(
                    p.grad is not None and not bool(torch.isfinite(p.grad).all())
                    for p in control.parameters()
                ):
                    raise FloatingPointError("nonfinite supported LBFGS")
                return loss

            opt.step(closure)
            record("lbfgs:last")
            controls[str(i)] = control().detach().cpu().numpy()
            torch.save(control.cpu().state_dict(), folder / ("control_" + str(i) + ".pt"))
            rows.append(
                dict(
                    id=str(va["ids"][i]),
                    trajectory=trace,
                    lbfgs_iterations=int(opt.state[next(control.parameters())]["n_iter"]),
                )
            )
        if any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()):
            raise AssertionError("state surrogate changed")
        np.savez_compressed(folder / "controls.npz", **controls)
        result = dict(
            role="validation",
            selection_use=False,
            checkpoint_sha256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            fixed_surrogate_verified=True,
            projection="L2-orthogonal projection onto the 17-mode training family; exact family optimum retained",
            supported_adjoint_action_relative=derivative,
            rows=rows,
            test_evaluations=0,
        )
        write(folder / "online.json", result)
        summaries[architecture] = float(
            np.mean([r["trajectory"][-1]["control_relative_l2"] for r in rows])
        )
        print(architecture, summaries[architecture], flush=True)
    write(out / "summary.json", summaries)
    (out / "VALIDATION_COMPLETE").write_text(
        "Six supported-space online pilots complete; errors reported without a convergence claim.\n"
    )


if __name__ == "__main__":
    run()
