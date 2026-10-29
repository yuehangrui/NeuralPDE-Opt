"""Independent residual points plus quadrature/FEM recovery for every method."""

import numpy as np
import torch
from .models import residuals


def sample_points(seed, role, count):
    if count < 1 or role not in (1, 2):
        raise ValueError("invalid sample role or count")
    return np.random.default_rng(np.random.SeedSequence([seed, role, 20261023])).uniform(
        0, 1, (count, 2)
    )


def predict(model, points, chunk=4096):
    parameter = next(model.parameters())
    outputs = [[], [], []]
    with torch.no_grad():
        for start in range(0, len(points), chunk):
            x = torch.as_tensor(
                points[start : start + chunk], dtype=parameter.dtype, device=parameter.device
            )
            for values, parts in zip(model(x), outputs):
                if values is not None:
                    parts.append(values[:, 0].cpu().numpy())
    return tuple(np.concatenate(parts) if parts else None for parts in outputs)


def evaluate(model, problem, ref, validation_points, state_scale):
    model.eval()
    parameter = next(model.parameters())
    rms_s, rms_a, count = 0.0, 0.0, 0
    for start in range(0, len(validation_points), 512):
        x = torch.tensor(
            validation_points[start : start + 512],
            dtype=parameter.dtype,
            device=parameter.device,
            requires_grad=True,
        )
        state, adjoint = residuals(model, problem, x)
        rms_s += float(state.detach().square().sum())
        count += len(x)
        if adjoint is not None:
            rms_a += float(adjoint.detach().square().sum())
    yq, uq, pq = predict(model, ref.points.reshape(-1, 2))
    yq, uq = yq.reshape(ref.weights.shape), uq.reshape(ref.weights.shape)
    recovered_y, recovered_p, gq = ref.recover(uq)
    j = ref.objective(ref.at_quad(recovered_y), uq)
    grad = ref.norm(gq)
    state_rms = (rms_s / count) ** 0.5
    score = ((state_rms / state_scale) ** 2 + (grad / ref.gradient_scale) ** 2) ** 0.5
    report = dict(
        validation_score=float(score),
        pde_rms=state_rms,
        pde_relative=state_rms / state_scale,
        reduced_gradient_l2=grad,
        reduced_gradient_relative=grad / ref.gradient_scale,
        J_raw=ref.objective(yq, uq),
        J_recovered=j,
        J_reference_h=ref.reference_objective,
        objective_gap_h=j - ref.reference_objective,
        mesh_n=ref.n,
        quadrature_order=ref.order,
    )
    for name, values in [
        ("y_raw", yq),
        ("u_raw", uq),
        ("y_recovered", ref.at_quad(recovered_y)),
        ("p_recovered", ref.at_quad(recovered_p)),
    ]:
        truth = getattr(problem, name[0])(ref.points)[..., 0]
        report["relative_l2_" + name] = ref.norm(values - truth) / ref.norm(truth)
    if pq is not None:
        truth = problem.p(ref.points)[..., 0]
        report["relative_l2_p_raw"] = ref.norm(pq.reshape(ref.weights.shape) - truth) / ref.norm(
            truth
        )
        report["adjoint_rms"] = (rms_a / count) ** 0.5
    t = np.linspace(0, 1, 101)
    boundary = np.concatenate(
        [
            np.stack([t, 0 * t], 1),
            np.stack([t, 0 * t + 1], 1),
            np.stack([0 * t, t], 1),
            np.stack([0 * t + 1, t], 1),
        ]
    )
    by, _, bp = predict(model, boundary)
    report["boundary_y_max"] = float(abs(by).max())
    if bp is not None:
        report["boundary_p_max"] = float(abs(bp).max())
    if not all(np.isfinite(v) for v in report.values()):
        raise FloatingPointError("nonfinite independent diagnostics")
    return report
