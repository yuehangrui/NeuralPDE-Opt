"""Interval/triangular P1 mesh and positive Gauss/Duffy quadrature.

The same geometry supplies FEM baselines and finite-element test functions for
the neural weak residual. A neural trial function remains mesh-independent.
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse


@dataclass
class Mesh:
    """P1 geometry; n counts intervals per coordinate and h=1/n.

    Assembly indices: e=element, q=quadrature point, a/b=local basis,
    d=spatial direction. ``gradients[e,a,d]`` is a physical basis gradient.
    """

    dim: int
    n: int
    nodes: np.ndarray
    elements: np.ndarray
    interior: np.ndarray
    gradients: np.ndarray
    determinants: np.ndarray

    def quadrature(self, order):
        """Return points (E,Q,dim), physical weights (E,Q), and phi (Q,A)."""
        gauss_points, gauss_weights = leggauss(order)
        gauss_points, gauss_weights = (gauss_points + 1)/2, gauss_weights/2
        if self.dim == 1:
            phi = np.stack((1-gauss_points, gauss_points), -1)
            weights = gauss_weights
        else:
            # Duffy map from a reference square to the reference triangle.
            square_xi, square_eta = np.meshgrid(gauss_points, gauss_points, indexing="ij")
            weights_xi, weights_eta = np.meshgrid(gauss_weights, gauss_weights, indexing="ij")
            xi, eta = square_xi.ravel(), ((1-square_xi)*square_eta).ravel()
            phi = np.stack((1-xi-eta, xi, eta), -1)
            weights = (weights_xi*weights_eta*(1-square_xi)).ravel()
        points = np.einsum("qa,ead->eqd", phi, self.nodes[self.elements])
        return points, self.determinants[:, None]*weights[None, :], phi

    def stiffness(self):
        """Assemble K_h[i,j] = integral grad(phi_i) . grad(phi_j)."""
        volume = self.determinants / (2 if self.dim == 2 else 1)
        local = np.einsum("ead,ebd,e->eab", self.gradients, self.gradients, volume)
        rows = np.broadcast_to(self.elements[:, :, None], local.shape)
        cols = np.broadcast_to(self.elements[:, None, :], local.shape)
        return sparse.coo_matrix((local.ravel(), (rows.ravel(), cols.ravel())),
                                 shape=(len(self.nodes), len(self.nodes))).tocsr()

    def load(self, problem, order=6):
        """Assemble F_h[j] = integral f * phi_j, including boundary nodes."""
        x, weights, phi = self.quadrature(order)
        local = np.einsum("eq,eq,qa->ea", weights, problem.source(x)[..., 0], phi)
        result = np.zeros(len(self.nodes))
        np.add.at(result, self.elements.ravel(), local.ravel())
        return result


def make_mesh(dim, n):
    """Uniform interval or positively oriented triangular mesh of (0,1)^dim."""
    if n < 2:
        raise ValueError("At least two intervals per axis are required")
    t = np.linspace(0, 1, n+1)
    if dim == 1:
        nodes = t[:, None]
        elements = np.stack((np.arange(n), np.arange(1, n+1)), -1)
        det = np.diff(t)
        gradients = np.broadcast_to(np.array([[-1.0], [1.0]]), (n, 2, 1)) / det[:, None, None]
    elif dim == 2:
        x1_grid, x2_grid = np.meshgrid(t, t, indexing="ij")
        nodes = np.stack((x1_grid.ravel(), x2_grid.ravel()), -1)
        elements = []
        for i in range(n):
            for j in range(n):
                node_0 = i*(n+1)+j
                node_1, node_2, node_3 = node_0+n+1, node_0+n+2, node_0+1
                elements.extend(((node_0, node_1, node_2), (node_0, node_2, node_3)))
        elements = np.asarray(elements, dtype=int)
        vertices = nodes[elements]
        jac = np.stack((vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]), -1)
        det = np.linalg.det(jac)
        gradients = np.einsum("ad,edk->eak", np.array([[-1., -1.], [1., 0.], [0., 1.]]), np.linalg.inv(jac))
    else:
        raise ValueError(dim)
    interior = np.flatnonzero(np.all((nodes > 0) & (nodes < 1), axis=1))
    return Mesh(dim, n, nodes, elements, interior, gradients, det)
