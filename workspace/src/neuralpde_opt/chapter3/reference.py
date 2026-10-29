"""Variable-coefficient P1 FEM, separate from the frozen Chapter 2 solver.

Analytic fields are evaluated at quadrature points, rather than interpolated
from the public observation grid. The shared Chapter 2 mesh is reused unchanged.
"""

from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from ..mesh import make_mesh


@dataclass
class Solution:
    mesh: object
    values: np.ndarray
    diagnostics: dict


def solve_fields(fields, n, order=4, mesh=None):
    start = time.perf_counter()
    mesh = make_mesh(2, n) if mesh is None else mesh
    points, weights, phi = mesh.quadrature(order)
    a = fields.coefficient(points)
    f = fields.source(points)
    if not np.all(np.isfinite(a)) or np.min(a) <= 0 or not np.all(np.isfinite(f)):
        raise ValueError("Non-finite fields or non-positive coefficient")
    local = np.einsum("ead,ebd,e->eab", mesh.gradients, mesh.gradients, np.sum(weights * a, axis=1))
    rows = np.broadcast_to(mesh.elements[:, :, None], local.shape)
    cols = np.broadcast_to(mesh.elements[:, None, :], local.shape)
    matrix = sparse.coo_matrix(
        (local.ravel(), (rows.ravel(), cols.ravel())), shape=(len(mesh.nodes), len(mesh.nodes))
    ).tocsr()
    local_load = np.einsum("eq,eq,qa->ea", weights, f, phi)
    load = np.zeros(len(mesh.nodes))
    np.add.at(load, mesh.elements.ravel(), local_load.ravel())
    ii = mesh.interior
    reduced = matrix[ii][:, ii]
    values = np.zeros(len(mesh.nodes))
    values[ii] = spsolve(reduced, load[ii])
    residual = reduced @ values[ii] - load[ii]
    tiny = np.finfo(float).tiny
    boundary = np.ones(len(values), dtype=bool)
    boundary[ii] = False
    work = float(values @ load)
    energy = float(values @ (matrix @ values))
    return Solution(
        mesh,
        values,
        {
            "n": n,
            "dofs": len(ii),
            "quadrature_order": order,
            "relative_algebraic_residual": float(
                np.linalg.norm(residual) / max(np.linalg.norm(load[ii]), tiny)
            ),
            "relative_energy_identity_defect": abs(energy - work) / max(abs(work), tiny),
            "boundary_max": float(np.max(np.abs(values[boundary]))),
            "a_quadrature_min": float(a.min()),
            "a_quadrature_max": float(a.max()),
            "seconds": time.perf_counter() - start,
        },
    )


def evaluate_p1(values, n, points):
    """Evaluate values and gradients on the shared SW--NE triangulation."""
    shape = points.shape[:-1]
    p = np.asarray(points).reshape(-1, 2)
    scaled = p * n
    cell = np.minimum(np.floor(scaled).astype(int), n - 1)
    uv = scaled - cell
    u, v = uv.T
    a = cell[:, 0] * (n + 1) + cell[:, 1]
    b, c, d = a + n + 1, a + n + 2, a + 1
    z = values
    lower = v <= u
    out = np.where(
        lower,
        (1 - u) * z[a] + (u - v) * z[b] + v * z[c],
        (1 - v) * z[a] + u * z[c] + (v - u) * z[d],
    )
    gx = np.where(lower, z[b] - z[a], z[c] - z[d]) * n
    gy = np.where(lower, z[c] - z[b], z[d] - z[a]) * n
    return out.reshape(shape), np.stack((gx, gy), -1).reshape(shape + (2,))


def compare_p1(values, n, reference, order=3):
    """Integrated L2 and gradient difference on the finer triangulation."""
    p, w, phi = reference.mesh.quadrature(order)
    v, g = evaluate_p1(values, n, p)
    local = reference.values[reference.mesh.elements]
    y = np.einsum("qa,ea->eq", phi, local)
    grad = np.einsum("ea,ead->ed", local, reference.mesh.gradients)[:, None, :]
    return {
        "relative_l2": float(np.sqrt(np.sum(w * (v - y) ** 2) / np.sum(w * y**2))),
        "relative_h1_seminorm": float(
            np.sqrt(np.sum(w[..., None] * (g - grad) ** 2) / np.sum(w[..., None] * grad**2))
        ),
    }


def analytic_errors(solution, exact, gradient, order=5):
    p, w, phi = solution.mesh.quadrature(order)
    local = solution.values[solution.mesh.elements]
    y = np.einsum("qa,ea->eq", phi, local)
    grad = np.einsum("ea,ead->ed", local, solution.mesh.gradients)[:, None, :]
    true, gt = exact(p), gradient(p)
    return {
        "relative_l2": float(np.sqrt(np.sum(w * (y - true) ** 2) / np.sum(w * true**2))),
        "relative_h1_seminorm": float(
            np.sqrt(np.sum(w[..., None] * (grad - gt) ** 2) / np.sum(w[..., None] * gt**2))
        ),
    }


class Manufactured:
    """a=1 or 1+0.25*x+0.125*y, y=sin(pi*x)sin(pi*y)."""

    def __init__(self, variable=True):
        self.variable = variable

    def coefficient(self, p):
        return 1 + 0.25 * p[..., 0] + 0.125 * p[..., 1] if self.variable else np.ones(p.shape[:-1])

    @staticmethod
    def exact(p):
        return np.sin(np.pi * p[..., 0]) * np.sin(np.pi * p[..., 1])

    @staticmethod
    def gradient(p):
        x, y = p[..., 0], p[..., 1]
        return np.pi * np.stack(
            (np.cos(np.pi * x) * np.sin(np.pi * y), np.sin(np.pi * x) * np.cos(np.pi * y)), -1
        )

    def source(self, p):
        f = 2 * np.pi**2 * self.coefficient(p) * self.exact(p)
        if self.variable:
            g = self.gradient(p)
            f = f - 0.25 * g[..., 0] - 0.125 * g[..., 1]
        return f


def save_geometry(path, n):
    """Store topology, positive quadrature and both consistent/lumped mass."""
    mesh = make_mesh(2, n)
    _, w, phi = mesh.quadrature(4)
    local = np.einsum("eq,qa,qb->eab", w, phi, phi)
    rows = np.broadcast_to(mesh.elements[:, :, None], local.shape)
    cols = np.broadcast_to(mesh.elements[:, None, :], local.shape)
    mass = sparse.coo_matrix(
        (local.ravel(), (rows.ravel(), cols.ravel())), shape=(len(mesh.nodes), len(mesh.nodes))
    ).tocsr()
    np.savez_compressed(
        path,
        coordinates=mesh.nodes.reshape(n + 1, n + 1, 2),
        elements=mesh.elements,
        interior=mesh.interior,
        gradients=mesh.gradients,
        determinants=mesh.determinants,
        quadrature_barycentric=phi,
        quadrature_reference_weights=w[0] / mesh.determinants[0],
        lumped_mass_weights=np.asarray(mass.sum(axis=1)).reshape(n + 1, n + 1),
        mass_data=mass.data,
        mass_indices=mass.indices,
        mass_indptr=mass.indptr,
        mass_shape=np.array(mass.shape),
    )
