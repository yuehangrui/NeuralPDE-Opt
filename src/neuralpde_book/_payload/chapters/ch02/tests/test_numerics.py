"""Meaningful checks against exact identities and expected P1 convergence."""
import unittest
import numpy as np
import torch
from neuralpde_ch02.problems import Poisson
from neuralpde_ch02.mesh import make_mesh
from neuralpde_ch02.models import CoordinateMLP, gradient, laplacian
from neuralpde_ch02.objectives import Objective
from neuralpde_ch02.metrics import evaluate
from neuralpde_ch02.fem import solve


class NumericalTests(unittest.TestCase):
    def test_quadrature_and_basis(self):
        for dim in (1, 2):
            mesh = make_mesh(dim, 5)
            x, weights, phi = mesh.quadrature(4)
            self.assertAlmostEqual(float(weights.sum()), 1., places=12)
            self.assertAlmostEqual(float((weights[..., None]*x**2).sum()), dim/3, places=12)
            self.assertTrue(np.allclose(phi.sum(-1), 1))
            self.assertTrue(np.allclose(mesh.gradients.sum(1), 0))
            # Linear precision detects transposed triangle mappings.
            grad_x = np.einsum("eak,ead->ekd", mesh.nodes[mesh.elements], mesh.gradients)
            self.assertTrue(np.allclose(grad_x, np.eye(dim)))

    def test_exact_residual_and_autograd(self):
        for dim in (1, 2):
            problem = Poisson(dim)
            x = torch.rand(31, dim, dtype=torch.float64, requires_grad=True)
            y = problem.exact(x)
            self.assertLess(float((-laplacian(y, x)-problem.source(x)).abs().max().detach()), 1e-11)
            self.assertLess(float((gradient(y, x)-problem.gradient(x)).abs().max().detach()), 1e-12)
            objective = Objective(problem, make_mesh(dim, 8), 6, "vpinn")
            self.assertLess(float(objective(problem.exact).detach()), 1e-19)

    def test_hard_boundary_and_energy_identity(self):
        for dim in (1, 2):
            torch.manual_seed(9)
            problem = Poisson(dim)
            model = CoordinateMLP(problem, 8, 2).double()
            metrics = evaluate(model, problem, 12, order=7, strong=False)
            self.assertLess(metrics["boundary_max"], 1e-14)
            self.assertLess(metrics["energy_identity_defect"], 1e-9)

    def test_fem_rates(self):
        for dim in (1, 2):
            coarse = solve(Poisson(dim), 8)[2]
            fine = solve(Poisson(dim), 16)[2]
            self.assertGreater(coarse["l2_abs"]/fine["l2_abs"], 3.7)
            self.assertGreater(coarse["h1_semi_abs"]/fine["h1_semi_abs"], 1.9)
            self.assertLess(fine["algebraic_residual_l2"], 1e-10)

    def test_all_objectives_differentiate(self):
        for dim in (1, 2):
            for method in ("pinn", "ritz", "vpinn"):
                torch.manual_seed(1)
                problem = Poisson(dim)
                model = CoordinateMLP(problem, 8, 2).double()
                objective = Objective(problem, make_mesh(dim, 4), 3, method)
                loss = objective(model)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()))

    def test_gram_basis_scaling_invariance(self):
        mesh = make_mesh(2, 5)
        gram_matrix = mesh.stiffness()[mesh.interior][:, mesh.interior].toarray()
        rng = np.random.default_rng(0)
        weak_residual = rng.normal(size=len(gram_matrix))
        basis_scaling = np.geomspace(.1, 10., len(gram_matrix))
        self.assertAlmostEqual(float(weak_residual@np.linalg.solve(gram_matrix, weak_residual)),
                               float((basis_scaling*weak_residual)@np.linalg.solve(basis_scaling[:, None]*gram_matrix*basis_scaling[None, :], basis_scaling*weak_residual)), places=10)


if __name__ == "__main__":
    unittest.main()
