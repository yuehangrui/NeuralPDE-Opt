"""Book 2.3--2.4: strong residual, Ritz energy, and finite-test weak residual.

All methods evaluate the same neural field y_theta. Only the scalar objective
changes. Quadrature weights include element Jacobians; no extra mean is taken.
"""
import torch
from .models import gradient, laplacian


class Objective:
    """Quadrature realization of J_PINN, E, or r^T G_W^{-1} r.

    Flattened coordinates have shape (n_points, dim); scalar fields and weights
    have shape (n_points, 1). For assembly, ``shape`` is (n_elements, n_quad).
    ``phi`` and ``grad_phi`` are local P1 basis values and gradients, which
    assemble into the global test functions phi_j that vanish on the boundary.
    """

    def __init__(self, problem, mesh, order, method, device="cpu", dtype=torch.float64):
        self.problem, self.mesh, self.method = problem, mesh, method
        x, weights, phi = mesh.quadrature(order)
        self.shape = x.shape[:2]
        self.x = torch.as_tensor(x.reshape(-1, problem.dim), dtype=dtype, device=device)
        self.weights = torch.as_tensor(weights.reshape(-1, 1), dtype=dtype, device=device)
        self.f = problem.source(self.x).detach()
        if method == "vpinn":
            self.phi = torch.as_tensor(phi, dtype=dtype, device=device)
            self.grad_phi = torch.as_tensor(mesh.gradients, dtype=dtype, device=device)
            self.indices = torch.as_tensor(mesh.elements.ravel(), dtype=torch.long, device=device)
            self.interior = torch.as_tensor(mesh.interior, dtype=torch.long, device=device)

            # G_W[i,j] = integral grad(phi_i) . grad(phi_j), book (2.40).
            gram_matrix = mesh.stiffness()[mesh.interior][:, mesh.interior].toarray()
            self.gram_cholesky = torch.linalg.cholesky(
                torch.as_tensor(gram_matrix, dtype=dtype, device=device)
            )
            # Element load: integral f * phi_a. Indices: element e, point q,
            # local basis a; spatial direction d is used in weak_residual.
            self.element_load = torch.einsum(
                "eq,eq,qa->ea",
                self.weights.reshape(self.shape), self.f.reshape(self.shape), self.phi,
            )
        elif method not in ("pinn", "ritz"):
            raise ValueError(method)

    def weak_residual(self, y_theta, x):
        """Return r_j = a(y_theta, phi_j) - ell(phi_j), book (2.36), (2.40)."""
        grad_y_theta = gradient(y_theta, x).reshape(*self.shape, self.problem.dim)
        element_residual = torch.einsum(
            "eq,eqd,ead->ea", self.weights.reshape(self.shape), grad_y_theta, self.grad_phi,
        ) - self.element_load
        weak_residual = torch.zeros(len(self.mesh.nodes), dtype=x.dtype, device=x.device)
        weak_residual = weak_residual.index_add(0, self.indices, element_residual.reshape(-1))
        return weak_residual[self.interior]

    def __call__(self, model):
        x = self.x.detach().requires_grad_(True)
        y_theta = model(x)
        if self.method == "pinn":
            # J_PINN, book (2.11): the boundary term vanishes by construction.
            strong_residual = -laplacian(y_theta, x) - self.f
            return (self.weights * strong_residual.square()).sum()
        if self.method == "ritz":
            # E(y_theta), book (2.29)--(2.32): minimize energy, not its square.
            grad_y_theta = gradient(y_theta, x)
            return (
                self.weights * (0.5 * grad_y_theta.square().sum(-1, keepdim=True) - self.f * y_theta)
            ).sum()

        # Solve G_W * riesz_coefficients = r without explicitly inverting G_W.
        weak_residual = self.weak_residual(y_theta, x)
        riesz_coefficients = torch.cholesky_solve(
            weak_residual[:, None], self.gram_cholesky,
        )[:, 0]
        return torch.dot(weak_residual, riesz_coefficients)
