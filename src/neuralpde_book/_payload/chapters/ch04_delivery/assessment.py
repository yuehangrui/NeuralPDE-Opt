"""Uniform, post-freeze independent assessment of all delivered methods.

Problem-family models get a new held-out set; instance-wise solvers get common
independent quadrature and refined FEM checks. No outcome updates a checkpoint.
"""

from pathlib import Path
import argparse
import gc
import json
import time
import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize
from chapters.ch04.problem import Problem
from chapters.ch04.fem import Reference
from chapters.ch04.models import residuals
from chapters.ch04.evaluation import predict
from chapters.ch04_adaptation.operator_data import MODES, EIGENVALUES, ALPHA, basis
from chapters.ch04_completion.common import SpectralProblem, quadrature
from chapters.ch04_completion.ismo import Surrogate
from . import operators, hybrids
from .frozen_jobs import write
from .protocol import verify_freeze, load_holdout, sha
from .metrics import control_metrics, state_metrics, aggregate


class CoefficientField(nn.Module):
    def __init__(self, y, u, p=None):
        super().__init__()
        self.placeholder = nn.Parameter(torch.zeros(1, dtype=torch.float64), requires_grad=False)
        self.register_buffer("modes", torch.tensor(MODES, dtype=torch.float64))
        self.register_buffer("u", torch.tensor(u, dtype=torch.float64))
        self.register_buffer("y", None if y is None else torch.tensor(y, dtype=torch.float64))
        self.register_buffer("p", None if p is None else torch.tensor(p, dtype=torch.float64))

    def forward(self, x):
        b = torch.sin(torch.pi * x[:, 0:1] * self.modes[:, 0]) * torch.sin(
            torch.pi * x[:, 1:2] * self.modes[:, 1]
        )
        return tuple(None if a is None else (b @ a)[:, None] for a in (self.y, self.u, self.p))


def load_model(entry, device):
    kind = entry["kind"]
    path = entry["checkpoint"]
    if kind == "pinn":
        from chapters.ch04.training import load_checkpoint

        model, _, _ = load_checkpoint(path, device)
    elif kind == "adapted_pinn":
        from chapters.ch04_adaptation.instance import load

        model, _, _ = load(path, device)
    elif kind == "grid_operator":
        from chapters.ch04_operator_adaptation.training import load

        model, _, _ = load(path, device)
    elif kind == "modal_operator":
        model, _ = operators.load(path, device)
    else:
        return hybrids.load(path)
    return model.eval().requires_grad_(False)


def coefficient_predict(model, entry, inputs):
    """Differentiable physical-coefficient interface, without grid reprojection."""
    if entry["kind"] == "grid_operator":
        c = inputs if entry["method"].endswith("_state") else inputs.flatten(1)
        value = model.coefficients(c)
        return value[:, 0] if entry["method"].endswith("_state") else value
    return model(inputs)


def online(model, entry, f, yd):
    """Optimize all supplied independent controls together; model stays frozen."""
    device = next(model.parameters()).device
    ft = torch.tensor(f, dtype=torch.float64, device=device)
    dt = torch.tensor(yd, dtype=torch.float64, device=device)
    n = len(f)
    calls = 0
    history = []
    start = time.perf_counter()

    def raw(z_np, scales=None):
        u = torch.tensor(
            z_np.reshape(n, 17), dtype=torch.float64, device=device, requires_grad=True
        )
        y = coefficient_predict(model, entry, ft + u)
        values = 0.125 * ((y - dt).square().sum(1) + ALPHA * u.square().sum(1))
        loss = values.sum() if scales is None else (values / scales).sum()
        (gradient,) = torch.autograd.grad(loss, u)
        return (
            float(loss.detach()),
            gradient.detach().cpu().numpy().ravel(),
            y.detach().cpu().numpy(),
        )

    initial = np.zeros(n * 17)
    _, g0, _ = raw(initial)
    scales = torch.tensor(
        np.linalg.norm(g0.reshape(n, 17), axis=1).clip(1e-12), dtype=torch.float64, device=device
    )

    def fun(z):
        nonlocal calls
        calls += 1
        j, g, _ = raw(z, scales)
        return j, g

    def record(z):
        m = control_metrics(f, yd, z.reshape(n, 17))
        history.append(
            dict(
                iteration=len(history),
                true_objective=m["true_objective"].tolist(),
                true_gradient_relative=m["true_gradient_relative"].tolist(),
            )
        )

    record(initial)
    opt = minimize(
        fun,
        initial,
        jac=True,
        method="L-BFGS-B",
        callback=record,
        options=dict(gtol=1e-10, ftol=1e-15, maxiter=500, maxls=40, maxcor=50),
    )
    record(opt.x)
    _, g, y = raw(opt.x, scales)
    return (
        opt.x.reshape(n, 17),
        y,
        dict(
            success=bool(opt.success),
            message=str(opt.message),
            iterations=int(opt.nit),
            evaluations=calls,
            seconds=time.perf_counter() - start,
            scaled_surrogate_gradient_max=float(abs(g).max()),
            optimization_variables="control coefficients only; frozen learned state map",
            history=history,
        ),
    )


def assess_family(model, entry, data, out):
    device = next(model.parameters()).device
    state_role = entry["method"].endswith("_state") or entry["method"] in ("rb_state", "rb_dino")
    result = {}
    arrays = {}
    start = time.perf_counter()
    with torch.no_grad():
        inputs = data["rhs"] if state_role else np.stack([data["f"], data["yd"]], 1)
        prediction = (
            coefficient_predict(
                model, entry, torch.tensor(inputs, dtype=torch.float64, device=device)
            )
            .cpu()
            .numpy()
        )
    result["inference_seconds"] = time.perf_counter() - start
    arrays.update(ids=data["ids"], prediction=prediction)
    if state_role:
        values = state_metrics(data["rhs"], prediction)
        result["state"] = aggregate(values)
        for k, v in values.items():
            arrays["state_" + k] = v
        # Fixed first eight holdout probes, never used to change the model.
        x = torch.tensor(data["rhs"][:8], dtype=torch.float64, device=device)
        d = torch.tensor(
            np.roll(data["rhs"], 1, axis=0)[:8] - data["rhs"][:8],
            dtype=torch.float64,
            device=device,
        )
        _, j = torch.autograd.functional.jvp(lambda v: coefficient_predict(model, entry, v), x, d)
        target = d / torch.tensor(EIGENVALUES, device=device)
        derivative = (j - target).norm(dim=1) / target.norm(dim=1)
        result["state_jvp_relative"] = aggregate({"error": derivative.cpu().numpy()})["error"]
        arrays["jvp_prediction"] = j.detach().cpu().numpy()
        arrays["jvp_reference"] = target.cpu().numpy()
        u, y, info = online(model, entry, data["f"][:8], data["yd"][:8])
        values = control_metrics(data["f"][:8], data["yd"][:8], u, y)
        result["online"] = dict(**aggregate(values), optimizer=info)
        result["matched_control_8"] = aggregate(values)
        arrays.update(online_controls=u, online_states=y, online_ids=data["ids"][:8])
        for k, v in values.items():
            arrays["online_" + k] = v
    else:
        y, u = prediction[:, 0], prediction[:, 1]
        values = control_metrics(data["f"], data["yd"], u, y, -ALPHA * u)
        result["direct_control"] = aggregate(values)
        result["matched_control_8"] = aggregate({k: v[:8] for k, v in values.items()})
        for k, v in values.items():
            arrays["direct_" + k] = v
    np.savez_compressed(out / "heldout_predictions.npz", **arrays)
    return result


def fixed_field(model, entry):
    problem = SpectralProblem()
    f, yd = problem.f[None], problem.yd[None]
    meta = {}
    if entry["kind"] in ("pinn", "adapted_pinn"):
        return model, meta
    if entry["kind"] == "hybrid":
        state = model["state"]
        method = entry["method"]
        u = state["control"].numpy()
        y = None
        p = None
        if "state_head" in state:
            y = state["state_head"].numpy() / EIGENVALUES
        if method == "deep_uzawa":
            p = -2 * state["multiplier"].numpy()
        if method == "ismo":
            metrics = json.loads((Path(entry["directory"]) / "metrics.json").read_text())
            index = metrics["best_round"]
            if index >= 0:
                surrogate = Surrogate()
                surrogate.load_state_dict(state["round_models"][index])
                with torch.no_grad():
                    y = surrogate(torch.tensor(u, dtype=torch.float64)).numpy()
            meta["surrogate_round_associated_with_control"] = index
        if method == "fem_nn":
            meta["learned_source_coefficients"] = (
                EIGENVALUES * state["source_head"].numpy()
            ).tolist()
        return CoefficientField(y, u, p), meta
    state_role = entry["method"].endswith("_state") or entry["method"] in ("rb_state", "rb_dino")
    if state_role:
        u, y, info = online(model, entry, f, yd)
        meta["online_optimizer"] = info
        return CoefficientField(y[0], u[0]), meta
    device = next(model.parameters()).device
    with torch.no_grad():
        v = (
            coefficient_predict(
                model, entry, torch.tensor(np.stack([f, yd], 1), device=device, dtype=torch.float64)
            )
            .cpu()
            .numpy()[0]
        )
    return CoefficientField(v[0], v[1], -ALPHA * v[1]), meta


def assess_fixed(field, meta, references, out):
    problem = Problem()
    x, w = quadrature(96)
    b = SpectralProblem()
    y, u, p = predict(field, x)
    norm = lambda a: float(np.sqrt(w @ (a * a)))
    truth_u = problem.u(x)[:, 0]
    truth_y = problem.y(x)[:, 0]
    result = dict(quadrature_order=96, control_relative_l2=norm(u - truth_u) / norm(truth_u))
    arrays = dict(points=x, weights=w, u=u, u_reference=truth_u, y_reference=truth_y)
    if y is not None:
        result.update(
            raw_state_relative_l2=norm(y - truth_y) / norm(truth_y),
            raw_objective=float(0.5 * (w @ ((y - problem.yd(x)[:, 0]) ** 2 + ALPHA * u * u))),
        )
        arrays["y"] = y
        if p is not None:
            arrays["p"] = p
        device = next(field.parameters()).device
        rs = []
        ra = []
        for i in range(0, len(x), 512):
            xi = torch.tensor(
                x[i : i + 512], device=device, dtype=torch.float64, requires_grad=True
            )
            r, a = residuals(field, problem, xi)
            rs.append(r.detach().cpu().numpy()[:, 0])
            if a is not None:
                ra.append(a.detach().cpu().numpy()[:, 0])
        rs = np.concatenate(rs)
        arrays["state_residual"] = rs
        result["state_residual_relative"] = norm(rs) / norm(problem.f(x)[:, 0])
        if ra:
            ra = np.concatenate(ra)
            arrays["adjoint_residual"] = ra
            result["adjoint_residual_relative"] = norm(ra) / norm(problem.yd(x)[:, 0])
    else:
        result["raw_state_evaluation"] = "not a neural state output; true PDE recovery below"
    fem_rows = []
    for ref in references:
        _, uq, _ = predict(field, ref.points.reshape(-1, 2), chunk=8192)
        uq = uq.reshape(ref.weights.shape)
        yr, pr, gq = ref.recover(uq)
        j = ref.objective(ref.at_quad(yr), uq)
        row = dict(
            mesh=ref.n,
            control_relative_l2=ref.norm(uq - problem.u(ref.points)[..., 0])
            / ref.norm(problem.u(ref.points)[..., 0]),
            recovered_objective=j,
            reference_objective_h=ref.reference_objective,
            objective_gap_h=j - ref.reference_objective,
            reduced_gradient_l2=ref.norm(gq),
            reduced_gradient_relative=ref.norm(gq) / ref.gradient_scale,
            recovered_state_relative_l2=ref.norm(ref.at_quad(yr) - problem.y(ref.points)[..., 0])
            / ref.norm(problem.y(ref.points)[..., 0]),
        )
        if "learned_source_coefficients" in meta:
            coeff = np.asarray(meta["learned_source_coefficients"])
            # Avoid a quadrature-points by 17 dense basis allocation at n=256.
            source = np.zeros_like(ref.weights)
            for c, (kx, ky) in zip(coeff, MODES):
                source += (
                    c
                    * np.sin(np.pi * kx * ref.points[..., 0])
                    * np.sin(np.pi * ky * ref.points[..., 1])
                )
            observed = np.zeros(len(ref.interior))
            for c, (kx, ky), lam in zip(b.f, MODES, EIGENVALUES):
                pts = ref.nodes[ref.interior]
                observed += (
                    (c / lam) * np.sin(np.pi * kx * pts[:, 0]) * np.sin(np.pi * ky * pts[:, 1])
                )
            learned = ref.lu.solve(ref.load(source)[ref.interior])
            delta = learned - observed
            row["independent_source_observation_relative_l2"] = float(
                np.sqrt(delta @ ref.Mi @ delta) / np.sqrt(observed @ ref.Mi @ observed)
            )
        fem_rows.append(row)
    t = np.linspace(0, 1, 129)
    boundary = np.concatenate(
        [
            np.stack([t, 0 * t], 1),
            np.stack([t, 0 * t + 1], 1),
            np.stack([0 * t, t], 1),
            np.stack([0 * t + 1, t], 1),
        ]
    )
    by, _, bp = predict(field, boundary)
    result["boundary_y_max"] = None if by is None else float(abs(by).max())
    result["boundary_p_max"] = None if bp is None else float(abs(bp).max())
    result["fem"] = fem_rows
    result["metadata"] = meta
    result["fem_gap_note"] = (
        "Sine/continuous-network controls and P1 reference controls are different discrete spaces; signed gaps retained."
    )
    np.savez_compressed(out / "fixed_fields.npz", **arrays)
    return result


def run(root, device="cuda", resume=False):
    root = Path(root)
    frozen = verify_freeze(root)
    data = load_holdout(root)
    assessment = root / "assessment"
    assessment.mkdir(exist_ok=resume)
    torch.set_num_threads(2)
    started = time.perf_counter()
    references = [Reference(Problem(), n, 4) for n in (128, 256)]
    write(assessment / "reference.json", {str(r.n): r.report() for r in references})
    rows = []
    entries = frozen["entries"]
    for index, entry in enumerate(entries):
        write(
            assessment / "status.json",
            dict(
                stage="evaluating",
                index=index,
                total=len(entries),
                key=entry["key"],
                seconds=time.perf_counter() - started,
            ),
        )
        out = assessment / entry["key"]
        if resume and (out / "metrics.json").is_file():
            row = json.loads((out / "metrics.json").read_text())
            if row["sha256"] != entry["sha256"] or row["key"] != entry["key"]:
                raise ValueError("resume identity mismatch")
            rows.append(row)
            continue
        out.mkdir(exist_ok=resume)
        start = time.perf_counter()
        model = load_model(entry, device)
        row = dict(
            **entry,
            test_role=(
                "fresh held-out problems"
                if entry["track"] == "problem_family"
                else "independent spatial/mesh assessment of fixed instance"
            ),
        )
        if entry["track"] == "problem_family":
            row["heldout"] = assess_family(model, entry, data, out)
        field, meta = fixed_field(model, entry)
        row["fixed_instance"] = assess_fixed(field, meta, references, out)
        row["assessment_seconds"] = time.perf_counter() - start
        if sha(entry["checkpoint"]) != entry["sha256"]:
            raise ValueError("model changed during assessment")
        write(out / "metrics.json", row)
        rows.append(row)
        print(
            entry["key"], "fixed control", row["fixed_instance"]["control_relative_l2"], flush=True
        )
        del model, field
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    verify_freeze(root)
    write(
        assessment / "summary.json",
        dict(
            rows=rows,
            seconds=time.perf_counter() - started,
            selected_checkpoints=len(rows),
            model_selection_unchanged=True,
        ),
    )
    write(
        assessment / "status.json",
        dict(stage="complete", total=len(rows), seconds=time.perf_counter() - started),
    )
    (root / "INDEPENDENT_ASSESSMENT_COMPLETE").write_text(
        "All selected variants evaluated without model updates.\n"
    )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    run(a.root, a.device, a.resume)
