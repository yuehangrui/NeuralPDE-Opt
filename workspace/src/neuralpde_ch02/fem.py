"""Small P1 FEM reference implementation with independent error integration."""
import time
import numpy as np
from scipy.sparse.linalg import spsolve
from .mesh import make_mesh
from .metrics import error_metrics


def solve(problem, n, order=6):
    """Solve K_h y_h = F_h after boundary lifting; return nodal y_h and errors.

    K_h is the P1 stiffness matrix and F_h is the source load. The nodal lift
    is initialized first; only the homogeneous interior correction is solved.
    """
    start = time.perf_counter()
    mesh = make_mesh(problem.dim, n)
    stiffness_matrix = mesh.stiffness()
    load_vector = mesh.load(problem, order)
    y_h_nodes = problem.lifting(mesh.nodes).reshape(-1).copy()
    interior = mesh.interior
    # Solve for the homogeneous correction, then add the nonzero lifting.
    y_h_nodes[interior] += spsolve(
        stiffness_matrix[interior][:, interior],
        (load_vector-stiffness_matrix@y_h_nodes)[interior],
    )
    solve_seconds = time.perf_counter()-start
    # Higher-order quadrature on every element; no nodal-error shortcut.
    x, weights, phi = mesh.quadrature(order+2)
    element_values = y_h_nodes[mesh.elements]
    y_h = np.einsum("qa,ea->eq", phi, element_values)[..., None]
    grad_y_h = np.einsum("ea,ead->ed", element_values, mesh.gradients)
    grad_y_h = np.broadcast_to(grad_y_h[:, None, :], x.shape)
    result = error_metrics(problem, x, weights[..., None], y_h, grad_y_h)
    algebraic_residual = stiffness_matrix@y_h_nodes-load_vector
    boundary = np.setdiff1d(np.arange(len(mesh.nodes)), interior)
    result.update(problem=problem.name, dim=problem.dim, method="fem", n=n,
                  dofs=len(interior), solve_seconds=solve_seconds,
                  algebraic_residual_l2=float(np.linalg.norm(algebraic_residual[interior])),
                  boundary_max=float(np.max(np.abs(y_h_nodes[boundary]-problem.exact(mesh.nodes[boundary])[:, 0]))))
    return mesh, y_h_nodes, result
