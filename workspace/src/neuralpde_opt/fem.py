"""Small P1 FEM reference implementation with independent error integration."""
import time
import numpy as np
from scipy.sparse.linalg import spsolve
from .mesh import make_mesh
from .metrics import error_metrics


def solve(problem, n, order=6):
    start = time.perf_counter()
    mesh = make_mesh(problem.dim, n)
    matrix = mesh.stiffness()
    load = mesh.load(problem, order)
    values = problem.lifting(mesh.nodes).reshape(-1).copy()
    ii = mesh.interior
    # Solve for the homogeneous correction, then add the nonzero lifting.
    values[ii] += spsolve(matrix[ii][:, ii], (load-matrix@values)[ii])
    solve_seconds = time.perf_counter()-start
    # Higher-order quadrature on every element; no nodal-error shortcut.
    x, w, phi = mesh.quadrature(order+2)
    local_values = values[mesh.elements]
    yh = np.einsum("qa,ea->eq", phi, local_values)[..., None]
    gh = np.einsum("ea,ead->ed", local_values, mesh.gradients)
    gh = np.broadcast_to(gh[:, None, :], x.shape)
    result = error_metrics(problem, x, w[..., None], yh, gh)
    residual = matrix@values-load
    boundary = np.setdiff1d(np.arange(len(mesh.nodes)), ii)
    result.update(problem=problem.name, dim=problem.dim, method="fem", n=n,
                  dofs=len(ii), solve_seconds=solve_seconds,
                  algebraic_residual_l2=float(np.linalg.norm(residual[ii])),
                  boundary_max=float(np.max(np.abs(values[boundary]-problem.exact(mesh.nodes[boundary])[:, 0]))))
    return mesh, values, result
