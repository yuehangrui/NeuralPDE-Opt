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
    dim: int
    n: int
    nodes: np.ndarray
    elements: np.ndarray
    interior: np.ndarray
    gradients: np.ndarray
    determinants: np.ndarray

    def quadrature(self, order):
        z, w = leggauss(order)
        z, w = (z + 1)/2, w/2
        if self.dim == 1:
            phi = np.stack((1-z, z), -1)
            weights = w
        else:
            u, v = np.meshgrid(z, z, indexing="ij")
            wu, wv = np.meshgrid(w, w, indexing="ij")
            r, s = u.ravel(), ((1-u)*v).ravel()
            phi = np.stack((1-r-s, r, s), -1)
            weights = (wu*wv*(1-u)).ravel()
        points = np.einsum("qa,ead->eqd", phi, self.nodes[self.elements])
        return points, self.determinants[:, None]*weights[None, :], phi

    def stiffness(self):
        volume = self.determinants / (2 if self.dim == 2 else 1)
        local = np.einsum("ead,ebd,e->eab", self.gradients, self.gradients, volume)
        a = self.elements.shape[1]
        rows = np.broadcast_to(self.elements[:, :, None], local.shape)
        cols = np.broadcast_to(self.elements[:, None, :], local.shape)
        return sparse.coo_matrix((local.ravel(), (rows.ravel(), cols.ravel())),
                                 shape=(len(self.nodes), len(self.nodes))).tocsr()

    def load(self, problem, order=6):
        x, w, phi = self.quadrature(order)
        local = np.einsum("eq,eq,qa->ea", w, problem.source(x)[..., 0], phi)
        result = np.zeros(len(self.nodes))
        np.add.at(result, self.elements.ravel(), local.ravel())
        return result


def make_mesh(dim, n):
    if n < 2:
        raise ValueError("At least two intervals per axis are required")
    t = np.linspace(0, 1, n+1)
    if dim == 1:
        nodes = t[:, None]
        elements = np.stack((np.arange(n), np.arange(1, n+1)), -1)
        det = np.diff(t)
        gradients = np.broadcast_to(np.array([[-1.0], [1.0]]), (n, 2, 1)) / det[:, None, None]
    elif dim == 2:
        xx, yy = np.meshgrid(t, t, indexing="ij")
        nodes = np.stack((xx.ravel(), yy.ravel()), -1)
        elements = []
        for i in range(n):
            for j in range(n):
                a = i*(n+1)+j
                b, c, d = a+n+1, a+n+2, a+1
                elements.extend(((a, b, c), (a, c, d)))
        elements = np.asarray(elements, dtype=int)
        vertices = nodes[elements]
        jac = np.stack((vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]), -1)
        det = np.linalg.det(jac)
        gradients = np.einsum("ad,edk->eak", np.array([[-1., -1.], [1., 0.], [0., 1.]]), np.linalg.inv(jac))
    else:
        raise ValueError(dim)
    interior = np.flatnonzero(np.all((nodes > 0) & (nodes < 1), axis=1))
    return Mesh(dim, n, nodes, elements, interior, gradients, det)
