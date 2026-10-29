"""Independent P1 reference for the linear-quadratic control problem.

Geometry and positive Duffy quadrature follow the existing project mesh.py.
Consistent mass, quadrature loads, sparse KKT LU and exact discrete adjoints.

Book/code notation:
Book 4.3: stiffness K and consistent mass M define the P1 state/adjoint systems. A Euclidean coefficient gradient includes the mass matrix; it is not interchangeable with the L2 gradient alpha*u+p.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve


class Reference:
    def __init__(self, problem, n=64, order=4):
        if type(n) is not int or n < 2 * problem.control_frequency or order < 2:
            raise ValueError(
                "mesh must resolve the manufactured control: n >= 2*frequency; order >= 2"
            )
        self.problem, self.n, self.order = problem, n, order
        t = np.linspace(0, 1, n + 1)
        xx, yy = np.meshgrid(t, t, indexing="ij")
        self.nodes = np.stack([xx.ravel(), yy.ravel()], axis=1)
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        a = (ii * (n + 1) + jj).ravel()
        b, c, d = a + n + 1, a + n + 2, a + 1
        self.elements = np.stack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)], 1).reshape(-1, 3)
        vertices = self.nodes[self.elements]
        jac = np.stack([vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]], -1)
        det = np.linalg.det(jac)
        grads = np.einsum(
            "ad,edk->eak", np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]), np.linalg.inv(jac)
        )
        local_k = np.einsum("ead,ebd,e->eab", grads, grads, det / 2)
        local_m = det[:, None, None] * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])[None, :, :] / 24
        rows = np.broadcast_to(self.elements[:, :, None], local_k.shape).ravel()
        cols = np.broadcast_to(self.elements[:, None, :], local_k.shape).ravel()
        shape = (len(self.nodes),) * 2
        self.K = sparse.coo_matrix((local_k.ravel(), (rows, cols)), shape=shape).tocsr()
        self.M = sparse.coo_matrix((local_m.ravel(), (rows, cols)), shape=shape).tocsr()
        self.interior = np.flatnonzero(np.all((self.nodes > 0) & (self.nodes < 1), axis=1))
        z, w = leggauss(order)
        z, w = (z + 1) / 2, w / 2
        r, s = np.meshgrid(z, z, indexing="ij")
        wr, ws = np.meshgrid(w, w, indexing="ij")
        q1, q2 = r.ravel(), ((1 - r) * s).ravel()
        self.phi = np.stack([1 - q1 - q2, q1, q2], axis=1)
        self.points = np.einsum("qa,ead->eqd", self.phi, vertices)
        self.weights = det[:, None] * (wr * ws * (1 - r)).ravel()[None, :]
        self.fq = problem.f(self.points)[..., 0]
        self.ydq = problem.yd(self.points)[..., 0]
        self.fl = self.load(self.fq)
        self.ydl = self.load(self.ydq)
        ix = self.interior
        self.Ki = self.K[ix][:, ix].tocsc()
        self.Mi = self.M[ix][:, ix].tocsc()
        self.lu = splu(self.Ki)
        system = sparse.bmat(
            [[self.Ki, self.Mi / problem.alpha], [-self.Mi, self.Ki]], format="csc"
        )
        yp = spsolve(system, np.concatenate([self.fl[ix], -self.ydl[ix]]))
        self.y, self.p = self.expand(yp[: len(ix)]), self.expand(yp[len(ix) :])
        self.u = -self.p / problem.alpha
        self.reference_objective = self.objective(self.at_quad(self.y), self.at_quad(self.u))
        _, p0, _ = self.recover(np.zeros_like(self.fq))
        self.gradient_scale = max(self.norm(self.at_quad(p0)), 1e-12)
        self.algebraic_residual = float(
            np.linalg.norm(system @ yp - np.concatenate([self.fl[ix], -self.ydl[ix]]))
        )

    def expand(self, values):
        out = np.zeros(len(self.nodes))
        out[self.interior] = values
        return out

    def at_quad(self, nodal):
        return np.einsum("qa,ea->eq", self.phi, nodal[self.elements])

    def load(self, values):
        if values.shape != self.weights.shape or not np.isfinite(values).all():
            raise ValueError("invalid quadrature field")
        local = np.einsum("eq,eq,qa->ea", self.weights, values, self.phi)
        out = np.zeros(len(self.nodes))
        np.add.at(out, self.elements.ravel(), local.ravel())
        return out

    def norm(self, values):
        return float(np.sqrt(np.sum(self.weights * values**2)))

    def objective(self, yq, uq):
        return float(
            0.5 * np.sum(self.weights * (yq - self.ydq) ** 2)
            + 0.5 * self.problem.alpha * np.sum(self.weights * uq**2)
        )

    def recover(self, uq):
        """Re-solve the PDE for the supplied control, then its discrete adjoint."""
        y = self.expand(self.lu.solve((self.fl + self.load(uq))[self.interior]))
        p = self.expand(self.lu.solve((self.M @ y - self.ydl)[self.interior], trans="T"))
        gq = self.problem.alpha * uq + self.at_quad(p)
        return y, p, gq

    def report(self):
        errors = {}
        for name in ("y", "u", "p"):
            exact = getattr(self.problem, name)(self.points)[..., 0]
            errors["relative_l2_" + name] = self.norm(
                self.at_quad(getattr(self, name)) - exact
            ) / self.norm(exact)
        _, _, gradient = self.recover(self.at_quad(self.u))
        return dict(
            mesh_n=self.n,
            dofs=len(self.interior),
            quadrature_order=self.order,
            objective=self.reference_objective,
            algebraic_residual=self.algebraic_residual,
            reduced_gradient_l2=self.norm(gradient),
            **errors,
        )
