"""Independent quadrature errors and formulation-specific diagnostics."""
import math
import numpy as np
import torch
from .mesh import make_mesh
from .models import gradient, laplacian
from .objectives import Objective


def error_metrics(problem, x, weights, y_approx, grad_y_approx):
    """Integrate solution and gradient errors against the exact solution y.

    y_approx denotes either y_theta or y_h. A final scalar axis (..., 1)
    is retained for field values and weights; gradients end in spatial dim.
    h1_semi_* measures the gradient seminorm, not the full H1 norm.
    The energy-gap identity requires matching essential boundary data.
    """
    y_exact = problem.exact(x)
    grad_y_exact = problem.gradient(x)
    l2_error_squared = float(np.sum(weights*(y_approx-y_exact)**2))
    h1_error_squared = float(np.sum(weights*np.sum((grad_y_approx-grad_y_exact)**2, axis=-1, keepdims=True)))
    l2_norm_squared = float(np.sum(weights*y_exact**2))
    h1_norm_squared = float(np.sum(weights*np.sum(grad_y_exact**2, axis=-1, keepdims=True)))
    energy = float(np.sum(weights*(0.5*np.sum(grad_y_approx**2, axis=-1, keepdims=True)-problem.source(x)*y_approx)))
    exact_energy = float(np.sum(weights*(0.5*np.sum(grad_y_exact**2, axis=-1, keepdims=True)-problem.source(x)*y_exact)))
    return dict(l2_abs=math.sqrt(l2_error_squared), l2_rel=math.sqrt(l2_error_squared/l2_norm_squared),
                h1_semi_abs=math.sqrt(h1_error_squared), h1_semi_rel=math.sqrt(h1_error_squared/h1_norm_squared),
                energy=energy, exact_energy=exact_energy, energy_gap=energy-exact_energy,
                energy_identity_defect=abs((energy-exact_energy)-0.5*h1_error_squared))


def evaluate(model, problem, n, order=5, weak_n=None, strong=True, batch=4096):
    """Evaluate y_theta on an independent mesh, optionally enriching test space.

    This reports diagnostics; it does not select a model or update parameters.
    ``n`` and ``weak_n`` are interval counts per coordinate, not sample counts.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    mesh = make_mesh(problem.dim, n)
    quadrature_points, weights, _ = mesh.quadrature(order)
    points = quadrature_points.reshape(-1, problem.dim)
    y_batches, grad_y_batches, residual_batches = [], [], []
    for start in range(0, len(points), batch):
        x = torch.as_tensor(points[start:start+batch], device=device, dtype=dtype).requires_grad_(True)
        y_theta = model(x)
        grad_y_theta = gradient(y_theta, x)
        y_batches.append(y_theta.detach().cpu().numpy())
        grad_y_batches.append(grad_y_theta.detach().cpu().numpy())
        if strong:
            strong_residual = -laplacian(y_theta, x)-problem.source(x)
            residual_batches.append(strong_residual.detach().cpu().numpy())
    result = error_metrics(
        problem, points, weights.reshape(-1, 1),
        np.concatenate(y_batches), np.concatenate(grad_y_batches),
    )
    x_boundary = torch.as_tensor(problem.boundary_points(), device=device, dtype=dtype)
    with torch.no_grad():
        result["boundary_max"] = float((model(x_boundary)-problem.exact(x_boundary)).abs().max().cpu())
    if strong:
        strong_residual = np.concatenate(residual_batches)
        result["strong_residual_l2"] = float(np.sqrt(np.sum(weights.reshape(-1, 1)*strong_residual*strong_residual)))
        result["strong_residual_max"] = float(np.max(np.abs(strong_residual)))
    if weak_n:
        weak_objective = Objective(problem, make_mesh(problem.dim, weak_n), order, "vpinn", device, dtype)
        result["weak_residual_enriched"] = math.sqrt(max(0., float(weak_objective(model).detach().cpu())))
        result["weak_validation_n"] = weak_n
    result["validation_n"] = n
    result["validation_order"] = order
    return result
