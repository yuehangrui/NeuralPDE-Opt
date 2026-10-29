"""Three distinct continuum formulations, realized by explicit quadrature."""
import torch
from .models import gradient, laplacian


class Objective:
    def __init__(self, problem, mesh, order, method, device="cpu", dtype=torch.float64):
        self.problem, self.mesh, self.method = problem, mesh, method
        x, w, phi = mesh.quadrature(order)
        self.shape = x.shape[:2]
        self.x = torch.as_tensor(x.reshape(-1, problem.dim), dtype=dtype, device=device)
        self.weights = torch.as_tensor(w.reshape(-1, 1), dtype=dtype, device=device)
        self.f = problem.source(self.x).detach()
        if method == "vpinn":
            self.phi = torch.as_tensor(phi, dtype=dtype, device=device)
            self.grad_phi = torch.as_tensor(mesh.gradients, dtype=dtype, device=device)
            self.indices = torch.as_tensor(mesh.elements.ravel(), dtype=torch.long, device=device)
            self.interior = torch.as_tensor(mesh.interior, dtype=torch.long, device=device)
            gram = mesh.stiffness()[mesh.interior][:, mesh.interior].toarray()
            self.chol = torch.linalg.cholesky(torch.as_tensor(gram, dtype=dtype, device=device))
            self.load = torch.einsum("eq,eq,qa->ea", self.weights.reshape(self.shape),
                                     self.f.reshape(self.shape), self.phi)
        elif method not in ("pinn", "ritz"):
            raise ValueError(method)

    def weak_residual(self, y, x):
        grad_y = gradient(y, x).reshape(*self.shape, self.problem.dim)
        local = torch.einsum("eq,eqd,ead->ea", self.weights.reshape(self.shape), grad_y, self.grad_phi) - self.load
        residual = torch.zeros(len(self.mesh.nodes), dtype=x.dtype, device=x.device)
        residual = residual.index_add(0, self.indices, local.reshape(-1))
        return residual[self.interior]

    def __call__(self, model):
        x = self.x.detach().requires_grad_(True)
        y = model(x)
        if self.method == "pinn":
            r = -laplacian(y, x)-self.f
            return (self.weights*r.square()).sum()
        if self.method == "ritz":
            g = gradient(y, x)
            return (self.weights*(0.5*g.square().sum(-1, keepdim=True)-self.f*y)).sum()
        r = self.weak_residual(y, x)
        z = torch.cholesky_solve(r[:, None], self.chol)[:, 0]
        return torch.dot(r, z)
