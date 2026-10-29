"""Independent quadrature errors and formulation-specific diagnostics."""
import math
import numpy as np
import torch
from .mesh import make_mesh
from .models import gradient, laplacian
from .objectives import Objective


def error_metrics(problem, x, weights, values, gradients):
    exact = problem.exact(x)
    exact_grad = problem.gradient(x)
    diff2 = float(np.sum(weights*(values-exact)**2))
    grad2 = float(np.sum(weights*np.sum((gradients-exact_grad)**2, axis=-1, keepdims=True)))
    norm2 = float(np.sum(weights*exact**2))
    grad_norm2 = float(np.sum(weights*np.sum(exact_grad**2, axis=-1, keepdims=True)))
    energy = float(np.sum(weights*(0.5*np.sum(gradients**2, axis=-1, keepdims=True)-problem.source(x)*values)))
    exact_energy = float(np.sum(weights*(0.5*np.sum(exact_grad**2, axis=-1, keepdims=True)-problem.source(x)*exact)))
    return dict(l2_abs=math.sqrt(diff2), l2_rel=math.sqrt(diff2/norm2),
                h1_semi_abs=math.sqrt(grad2), h1_semi_rel=math.sqrt(grad2/grad_norm2),
                energy=energy, exact_energy=exact_energy, energy_gap=energy-exact_energy,
                energy_identity_defect=abs((energy-exact_energy)-0.5*grad2))


def evaluate(model, problem, n, order=5, weak_n=None, strong=True, batch=4096):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    mesh = make_mesh(problem.dim, n)
    pts, weights, _ = mesh.quadrature(order)
    points = pts.reshape(-1, problem.dim)
    ys, gs, residuals = [], [], []
    for start in range(0, len(points), batch):
        x = torch.as_tensor(points[start:start+batch], device=device, dtype=dtype).requires_grad_(True)
        y = model(x)
        g = gradient(y, x)
        ys.append(y.detach().cpu().numpy())
        gs.append(g.detach().cpu().numpy())
        if strong:
            r = -laplacian(y, x)-problem.source(x)
            residuals.append(r.detach().cpu().numpy())
    result = error_metrics(problem, points, weights.reshape(-1, 1), np.concatenate(ys), np.concatenate(gs))
    xb = torch.as_tensor(problem.boundary_points(), device=device, dtype=dtype)
    with torch.no_grad():
        result["boundary_max"] = float((model(xb)-problem.exact(xb)).abs().max().cpu())
    if strong:
        r = np.concatenate(residuals)
        result["strong_residual_l2"] = float(np.sqrt(np.sum(weights.reshape(-1, 1)*r*r)))
        result["strong_residual_max"] = float(np.max(np.abs(r)))
    if weak_n:
        obj = Objective(problem, make_mesh(problem.dim, weak_n), order, "vpinn", device, dtype)
        result["weak_residual_enriched"] = math.sqrt(max(0., float(obj(model).detach().cpu())))
        result["weak_validation_n"] = weak_n
    result["validation_n"] = n
    result["validation_order"] = order
    return result
