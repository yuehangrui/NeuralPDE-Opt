"""Scale the surrogate objective and use a relative BFGS stopping criterion.

No retraining or model reselection. True PDE calls are diagnostics only.
The initial reduced-gradient infinity norm defines the scale independently
for each task; normalized gtol=1e-6 replaces an overly strict absolute 1e-10.
"""

from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from scipy.optimize import minimize
from .train_operators import load
from .operator_data import EIGENVALUES, ALPHA
from chapters.ch04.training import write_json


def run():
    torch.set_num_threads(2)
    base = Path("results/ch04/adaptation_v1")
    root = base / "online_adaptation"
    root.mkdir(exist_ok=False)
    for method in ("rb_state", "rb_dino"):
        folder = base / "operators" / method
        model, ck = load(folder / "best.pt")
        model.eval().requires_grad_(False)
        data = dict(np.load(folder / "validation.npz"))
        rows = []
        for i in range(8):
            f, yd = data["f"][i], data["yd"][i]
            trace = []

            def raw(u):
                ut = torch.tensor(u, dtype=torch.float64, requires_grad=True)
                y = model((ut + torch.tensor(f))[None])[0]
                value = 0.125 * ((y - torch.tensor(yd)).square().sum() + ALPHA * ut.square().sum())
                control_gradient = torch.autograd.grad(value, ut)[0].numpy()
                return float(value.detach()), control_gradient

            j0, g0 = raw(np.zeros_like(f))
            scale = max(float(np.linalg.norm(g0, ord=np.inf)), 1e-12)

            def fun(u):
                j, g = raw(u)
                return (j - j0) / scale, g / scale

            true_g0 = 0.25 * ((f / EIGENVALUES - yd) / EIGENVALUES)

            def record(u):
                y = (f + u) / EIGENVALUES
                control_gradient = 0.25 * ((y - yd) / EIGENVALUES + ALPHA * u)
                j, gp = raw(u)
                trace.append(
                    dict(
                        J_true=float(0.125 * (np.sum((y - yd) ** 2) + ALPHA * np.sum(u * u))),
                        J_surrogate=j,
                        true_gradient_relative=float(
                            np.linalg.norm(control_gradient) / np.linalg.norm(true_g0)
                        ),
                        surrogate_gradient_relative_inf=float(
                            np.linalg.norm(gp, ord=np.inf) / scale
                        ),
                    )
                )

            record(np.zeros_like(f))
            opt = minimize(
                fun,
                np.zeros_like(f),
                method="BFGS",
                jac=True,
                callback=record,
                options=dict(gtol=1e-6, maxiter=1000),
            )
            record(opt.x)
            # Independently check the achieved criterion, retain SciPy exit status.
            converged = trace[-1]["surrogate_gradient_relative_inf"] <= 1e-6
            if not converged:
                raise RuntimeError("scaled online optimization did not meet its declared criterion")
            rows.append(
                dict(
                    id=str(data["ids"][i]),
                    optimizer_success=bool(opt.success),
                    message=str(opt.message),
                    nit=int(opt.nit),
                    surrogate_stationarity_verified=converged,
                    scale=scale,
                    control_relative_l2=float(
                        np.linalg.norm(opt.x - data["u"][i]) / np.linalg.norm(data["u"][i])
                    ),
                    trajectory=trace,
                )
            )
        result = dict(
            role="validation",
            selection_use=False,
            method=method,
            checkpoint_sha256=hashlib.sha256((folder / "best.pt").read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            gtol_relative_inf=1e-6,
            mean_control_relative_l2=float(np.mean([r["control_relative_l2"] for r in rows])),
            rows=rows,
            test_evaluations=0,
        )
        write_json(root / (method + ".json"), result)
        print(
            method,
            result["mean_control_relative_l2"],
            sum(r["optimizer_success"] for r in rows),
            flush=True,
        )
    (root / "VALIDATION_COMPLETE").write_text(
        "16 validation optimizations met the declared surrogate stationarity criterion.\n"
    )


if __name__ == "__main__":
    run()
