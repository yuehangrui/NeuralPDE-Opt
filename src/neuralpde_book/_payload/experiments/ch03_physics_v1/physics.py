"""Exact P1 energy for nodal c/f; algebraic residuals are interior load vectors.

Book/code notation:
Book 3.4.1: E_h(y)=0.5*y^T*A_h(c)*y-b_h^T*y. A_h combines epsilon*K and reaction mass. Physics residual uses A_h*y-b_h; HINTS uses the opposite sign b_h-A_h*y for corrections.
"""

from math import factorial
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import torch
from neuralpde_opt.mesh import make_mesh


class P1System:
    def __init__(self, n=64, epsilon=0.01):
        if type(n) is not int or n < 2 or not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Invalid grid/diffusion")
        self.n, self.epsilon = n, float(epsilon)
        self.mesh = make_mesh(2, n)
        self.size = len(self.mesh.nodes)
        self.elements = self.mesh.elements
        self.interior = self.mesh.interior
        self.boundary = np.ones(self.size, dtype=bool)
        self.boundary[self.interior] = False
        self.area = self.mesh.determinants / 2
        self.local_mass = self.area[:, None, None] * (np.ones((3, 3)) + np.eye(3)) / 12
        self.local_stiffness = np.einsum(
            "ead,ebd,e->eab", self.mesh.gradients, self.mesh.gradients, self.area
        )
        third = np.empty((3, 3, 3))
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    counts = np.bincount([a, b, c], minlength=3)
                    third[a, b, c] = 2 * np.prod([factorial(int(k)) for k in counts]) / factorial(5)
        self.third = third
        self.rows = np.broadcast_to(self.elements[:, :, None], self.local_mass.shape).ravel()
        self.cols = np.broadcast_to(self.elements[:, None, :], self.local_mass.shape).ravel()
        self.mass = self.assemble_local(self.local_mass)
        self.lumped = np.asarray(self.mass.sum(1)).ravel()

    def assemble_local(self, local):
        return sparse.coo_matrix(
            (local.ravel(), (self.rows, self.cols)), shape=(self.size, self.size)
        ).tocsr()

    def local_matrices(self, c):
        c = np.asarray(c, dtype=np.float64)
        if (
            c.ndim != 3
            or c.shape[1:] != (self.n + 1, self.n + 1)
            or not np.isfinite(c).all()
            or c.min() <= 0
        ):
            raise ValueError("Positive finite nodal c required")
        values = c.reshape(len(c), -1)[:, self.elements]
        return self.epsilon * self.local_stiffness[None] + np.einsum(
            "nek,abk,e->neab", values, self.third, self.area
        )

    def assemble(self, c, f=None):
        c = np.asarray(c)
        matrix = self.assemble_local(self.local_matrices(c[None])[0])
        f = np.ones(self.size) if f is None else np.asarray(f, dtype=np.float64).reshape(-1)
        if f.shape != (self.size,) or not np.isfinite(f).all():
            raise ValueError("Invalid nodal source")
        return matrix, self.mass @ f

    def solve(self, c, f=None):
        system_matrix, load_vector = self.assemble(c, f)
        interior = self.interior
        y = np.zeros(self.size)
        y[interior] = spsolve(system_matrix[interior][:, interior], load_vector[interior])
        load_residual = (load_vector - system_matrix @ y)[interior]
        if np.linalg.norm(load_residual) > 1e-10 * max(
            np.linalg.norm(load_vector[interior]), 1e-15
        ):
            raise RuntimeError("Direct solve residual")
        return y.reshape(self.n + 1, self.n + 1)

    def diagnostics(self, c, prediction, f=None):
        c = np.asarray(c)
        prediction = np.asarray(prediction)
        if prediction.shape != c.shape:
            raise ValueError("Prediction shape mismatch")
        records = []
        for j in range(len(c)):
            system_matrix, load_vector = self.assemble(c[j], None if f is None else f[j])
            y = prediction[j].ravel()
            load_residual = (system_matrix @ y - load_vector)[self.interior]
            residual_scale_squared = np.sum(
                load_vector[self.interior] ** 2 / self.lumped[self.interior]
            )
            residual = np.sqrt(
                np.sum(load_residual * load_residual / self.lumped[self.interior])
                / max(residual_scale_squared, 1e-30)
            )
            records.append(
                {
                    "relative_lumped_dual_residual": float(residual),
                    "energy": float(0.5 * y @ (system_matrix @ y) - load_vector @ y),
                    "boundary_max": float(np.max(abs(y[self.boundary]))),
                }
            )
        return records


class TorchP1:
    def __init__(self, system, device="cpu", dtype=torch.float32):
        self.system = system
        self.device = device
        self.dtype = dtype
        self.elements = torch.tensor(system.elements, device=device, dtype=torch.long)
        self.ii = torch.tensor(system.interior, device=device, dtype=torch.long)
        self.mass = torch.tensor(system.local_mass, device=device, dtype=dtype)
        self.load = torch.tensor(system.lumped, device=device, dtype=dtype)
        self.size = system.size

    def local(self, c):
        return torch.tensor(self.system.local_matrices(c), device=self.device, dtype=self.dtype)

    def norm2(self, y):
        local = y.flatten(1)[:, self.elements]
        return torch.einsum("nea,eab,neb->n", local, self.mass, local)

    def energy(self, y, local_matrix):
        local = y.flatten(1)[:, self.elements]
        return 0.5 * torch.einsum("nea,neab,neb->n", local, local_matrix, local) - (
            y.flatten(1) * self.load
        ).sum(1)

    def residual(self, y, local_matrix):
        local = y.flatten(1)[:, self.elements]
        value = torch.einsum("neab,neb->nea", local_matrix, local)
        result = torch.zeros((len(y), self.size), device=y.device, dtype=y.dtype)
        result.index_add_(1, self.elements.flatten(), value.flatten(1))
        return (result - self.load)[:, self.ii]

    def relative_residual(self, y, local_matrix):
        residual = self.residual(y, local_matrix)
        weights = self.load[self.ii]
        return torch.sqrt((residual.square() / weights).sum(1) / weights.sum())


def objective(
    prediction, target, label_mask, local_matrix, kernel, physics_weight, label_count, total_count
):
    # Unbiased per-batch estimator of mean loss over the prescribed labeled subset.
    data = prediction.sum() * 0
    if label_count:
        selected = label_mask.bool()
        if selected.any():
            truth = target[selected]
            data = (
                (kernel.norm2(prediction[selected] - truth) / kernel.norm2(truth)).sum()
                * total_count
                / (label_count * len(prediction))
            )
    energy = (
        2 * kernel.energy(prediction, local_matrix).mean()
        if physics_weight
        else prediction.sum() * 0
    )
    return data + physics_weight * energy, data, energy
