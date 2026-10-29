"""Numerical contract tests, independently checking derivatives and semantics."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from . import classical, bilevel, deep_uzawa, fem_nn, ismo
from .common import (ALPHA, DIM, LAM, SpectralProblem, tensor, basis, quadrature,
                     save_checkpoint, load_checkpoint)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_01_quadrature_and_projection(self):
        p = SpectralProblem()
        np.testing.assert_allclose(p.gram, .25*np.eye(DIM), atol=1e-14)
        x, _ = quadrature(56)
        np.testing.assert_allclose(basis(x)@p.f, p.problem.f(x)[:, 0], atol=1e-12)
        np.testing.assert_allclose(basis(x)@p.yd, p.problem.yd(x)[:, 0], atol=1e-13)

    def test_02_exact_adjoint_derivative(self):
        p = SpectralProblem(); rng = np.random.default_rng(8)
        u, v = rng.normal(size=(2, DIM)); eps = 1e-5
        fd = (p.fun(u+eps*v)[0]-p.fun(u-eps*v)[0])/(2*eps)
        self.assertAlmostEqual(fd, p.fun(u)[1]@v, delta=1e-10)

    def test_03_spectral_baseline(self):
        out = classical.spectral()
        self.assertLess(out['diagnostics']['true_gradient_relative'], 1e-6)

    def test_04_fem_baseline_discrete_kkt(self):
        out = classical.fem(16)
        self.assertLess(out['control_error_to_discrete_kkt'], 1e-5)
        self.assertLess(abs(out['objective_gap_to_discrete_kkt']), 1e-10)

    def test_05_bilevel_hypergradient(self):
        solver = bilevel.Bilevel(); rng = np.random.default_rng(18)
        u, v = rng.normal(size=(2, DIM)); eps = 1e-4
        _, g = solver.fun(u)
        fd = (solver.fun(u+eps*v)[0]-solver.fun(u-eps*v)[0])/(2*eps)
        np.testing.assert_allclose(g, solver.problem.fun(u)[1], rtol=1e-8, atol=1e-10)
        self.assertAlmostEqual(fd, g@v, delta=1e-8)
        self.assertLess(solver.last['hessian_solve_residual'], 1e-12)

    def test_06_modified_objective_agrees_on_feasible_set(self):
        p = SpectralProblem(); u = np.random.default_rng(5).normal(size=DIM)
        value = deep_uzawa.modified_loss(tensor(p.f+u), tensor(u), tensor(u*2), p)
        self.assertAlmostEqual(float(value), p.objective(p.state(u), u), delta=1e-12)

    def test_07_uzawa_saddle_sign_and_stationarity(self):
        p = SpectralProblem()
        u = (LAM*p.yd-p.f)/(1+ALPHA*LAM**2)
        q = tensor(p.f+u).requires_grad_(True); ut = tensor(u).requires_grad_(True)
        grad = torch.autograd.grad(deep_uzawa.modified_loss(q, ut, tensor(ALPHA*u/2), p), (q, ut))
        self.assertLess(max(float(g.abs().max()) for g in grad), 1e-12)

    def test_08_uzawa_step_range(self):
        for rho in (0., ALPHA/2, float('nan')):
            with self.assertRaises(ValueError):
                deep_uzawa.run(1, rho)

    def test_09_uzawa_training(self):
        out = deep_uzawa.run(60)
        self.assertLess(out['diagnostics']['validation_score'], 1e-5)

    def test_10_sparse_solver_transpose(self):
        # An explicitly nonsymmetric matrix makes accidental forward solves fail.
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import splu
        from types import SimpleNamespace
        A = np.array([[2., .7], [-.2, 1.4]])
        fem = SimpleNamespace(ref=SimpleNamespace(lu=splu(csc_matrix(A))), counts=dict(state=0, adjoint=0))
        b = tensor([.3, .4]).requires_grad_(True)
        y = fem_nn.SparseSolve.apply(b, fem)
        g, = torch.autograd.grad(tensor([.8, -.3])@y, b)
        np.testing.assert_allclose(g.numpy(), np.linalg.solve(A.T, [.8, -.3]), atol=1e-14)

    def test_11_fem_nn_parameter_gradient(self):
        problem = fem_nn.SourceIdentification(16)
        rng = np.random.default_rng(22)
        theta = tensor(rng.normal(scale=.01, size=DIM)).requires_grad_(True)
        d = tensor(rng.normal(size=DIM)); eps = 1e-5
        g, = torch.autograd.grad(problem.loss(theta), theta)
        fd = (problem.loss(theta.detach()+eps*d)-problem.loss(theta.detach()-eps*d))/(2*eps)
        self.assertAlmostEqual(float(fd), float(g@d), delta=1e-8)

    def test_12_ismo_enriches_and_retrains(self):
        out = ismo.run(seed=41, rounds=2, epochs=10, initial_samples=20, enrichment=2)
        self.assertEqual(len(out['history']), 2)
        self.assertEqual(out['history'][0]['training_samples'], 20)
        self.assertEqual(out['history'][1]['training_samples'], 24)
        self.assertEqual(len(out['data']['train_controls']), 28)
        for candidate in out['data']['candidates'].reshape(-1, DIM):
            self.assertTrue(np.any(np.all(out['data']['train_controls'] == candidate, axis=1)))
        self.assertLessEqual(out['history'][1]['best_true_objective'], out['history'][0]['best_true_objective'])

    def test_13_checkpoint_identity_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'ck.pt'
            save_checkpoint(path, 'bilevel', {'control':tensor(np.zeros(DIM))}, {})
            self.assertEqual(load_checkpoint(path)['method'], 'bilevel')
            ck = torch.load(path, weights_only=True); ck['source_identity'] = {}; torch.save(ck, path)
            with self.assertRaises(ValueError): load_checkpoint(path)
            save_checkpoint(path, 'bilevel', {'control':tensor([float('nan')])}, {})
            with self.assertRaises(ValueError): load_checkpoint(path)

    def test_14_validation_mesh_must_be_independent(self):
        with self.assertRaises(ValueError): fem_nn.run(16, 16)
        with self.assertRaises(ValueError): quadrature(16)


if __name__ == '__main__':
    unittest.main()
