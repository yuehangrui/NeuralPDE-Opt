import unittest
import numpy as np
import torch
from .common import (ALPHA, CoordinateNet, FDGrid, Mesh, SparseSolve, tensor,
                     sp_tensor, smv, transfer_control, public_fields)
from .training import observations, InverseProblem, defaults, validate_config, train_fem


class NumericalContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.rng = np.random.default_rng(8)

    def test_fd_inverse_and_self_adjoint(self):
        g = FDGrid(24)
        a, b = self.rng.normal(size=(2,23,23))
        np.testing.assert_allclose(g.apply(g.solve(a)), a, atol=2e-14)
        self.assertAlmostEqual(float(np.sum(a*g.solve(b))), float(np.sum(b*g.solve(a))), places=12)

    def test_fd_numpy_torch_agree_and_gradient(self):
        g = FDGrid(16); a = self.rng.normal(size=(15,15))
        t = tensor(a).requires_grad_()
        np.testing.assert_allclose(g.apply(t).detach(), g.apply(a), atol=1e-12)
        v = tensor(self.rng.normal(size=a.shape))
        grad, = torch.autograd.grad((g.apply(t)*v).sum(), t)
        np.testing.assert_allclose(grad, g.apply(v), atol=1e-12)

    def test_modified_objective_on_feasible_set(self):
        g = FDGrid(16); u = self.rng.normal(size=(15,15)); y = g.solve(g.f+u)
        original = .5*g.norm(y-g.yd)**2+.5*ALPHA*g.norm(u)**2
        modified = .5*g.norm(y-g.yd)**2+.25*ALPHA*(g.norm(u)**2+g.norm(g.apply(y)-g.f)**2)
        self.assertAlmostEqual(original, modified, places=12)

    def test_exact_control_block_and_multiplier_sign(self):
        z = tensor(self.rng.normal(size=(3,3)))
        u = (2*z/ALPHA).clone().requires_grad_()
        grad, = torch.autograd.grad((.25*ALPHA*u.square()-z*u).sum(), u)
        np.testing.assert_allclose(grad, 0, atol=1e-15)

    def test_control_interpolation_does_not_impose_zero_boundary(self):
        g = FDGrid(16)
        u = 1+g.x[...,0]+2*g.x[...,1]
        points = np.array([[0,0],[1,1],[.325,.735]])
        np.testing.assert_allclose(transfer_control(u,16,points),1+points[:,0]+2*points[:,1],atol=1e-13)

    def test_network_boundary_and_trainable_hidden_layers(self):
        net = CoordinateNet(12,2,True)
        x = tensor([[0,.3],[1,.3],[.3,0],[.3,1]])
        np.testing.assert_array_equal(net(x).detach(), np.zeros(4))
        x = tensor(self.rng.random((20,2)))
        net(x).square().sum().backward()
        for p in net.parameters():
            self.assertGreater(float(p.grad.norm()), 0)

    def test_network_not_restricted_to_17_modes(self):
        net = CoordinateNet(12,2,True)
        g = FDGrid(24); x = g.x.reshape(-1,2)
        modes = [(i,j) for i in range(1,5) for j in range(1,5)]+[(8,8)]
        B = np.stack([np.sin(np.pi*i*x[:,0])*np.sin(np.pi*j*x[:,1]) for i,j in modes],1)
        y = net(tensor(x)).detach().numpy()
        residual = y-B@np.linalg.lstsq(B,y,rcond=None)[0]
        self.assertGreater(np.linalg.norm(residual)/np.linalg.norm(y), 1e-3)

    def test_mesh_assembly_and_affine_observation(self):
        m = Mesh(8)
        self.assertAlmostEqual(float(m.M.sum()),1.,places=13)
        np.testing.assert_allclose(m.K@np.ones(len(m.nodes)),0,atol=2e-14)
        xy = self.rng.uniform(.2,.8,(20,2))
        values = 1+2*m.nodes[m.ix,0]-3*m.nodes[m.ix,1]
        np.testing.assert_allclose(m.observe(xy)@values,1+2*xy[:,0]-3*xy[:,1],atol=1e-14)

    def test_mesh_matches_independent_frozen_assembly(self):
        from chapters.ch04.fem import Reference
        from chapters.ch04.problem import Problem
        m, r = Mesh(16), Reference(Problem(),16,2)
        self.assertLess(np.linalg.norm((m.K-r.K).data),1e-12)
        self.assertLess(np.linalg.norm((m.M-r.M).data),1e-12)

    def test_sparse_solve_transpose_derivative(self):
        m = Mesh(8)
        load = tensor(self.rng.normal(size=len(m.ix))).requires_grad_()
        d = tensor(self.rng.normal(size=len(m.ix)))
        target = (SparseSolve.apply(load,m).sin()).sum()
        grad, = torch.autograd.grad(target,load)
        eps = 1e-6
        plus = np.sin(m.lu.solve((load+eps*d).detach().numpy())).sum()
        minus = np.sin(m.lu.solve((load-eps*d).detach().numpy())).sum()
        self.assertAlmostEqual(float(grad@d),(plus-minus)/(2*eps),delta=2e-8)

    def test_full_inverse_network_directional_derivative(self):
        data = observations(16,8,.01)
        problem = InverseProblem(8,data['train_x'],data['train_observed'],float(data['observation_scale']))
        net = CoordinateNet(6,1)
        params = list(net.parameters())
        direction = [torch.randn_like(p) for p in params]
        grad = torch.autograd.grad(problem.loss(net,1e-4),params)
        expected = float(sum((g*d).sum() for g,d in zip(grad,direction)))
        original = [p.detach().clone() for p in params]
        vals = []
        for sign in (1,-1):
            with torch.no_grad():
                for p,o,d in zip(params,original,direction): p.copy_(o+sign*1e-6*d)
            vals.append(float(problem.loss(net,1e-4).detach()))
        self.assertAlmostEqual(expected,(vals[0]-vals[1])/2e-6,delta=1e-7)

    def test_observation_split_noise_and_scaling(self):
        noisy, clean = observations(), observations(noise=0.)
        self.assertEqual(len(set(map(tuple,np.concatenate([noisy['train_x'],noisy['val_x']])))),384)
        np.testing.assert_array_equal(noisy['train_x'],clean['train_x'])
        self.assertGreater(np.linalg.norm(noisy['train_observed']-clean['train_observed']),0)
        self.assertAlmostEqual(float(noisy['observation_scale']),float(np.sqrt(np.mean(noisy['train_observed']**2))))

    def test_observation_validation(self):
        with self.assertRaises(ValueError): Mesh(8).observe(np.array([[np.nan,0]]))
        with self.assertRaises(ValueError): observations(noise=-1.)
        with self.assertRaises(ValueError): FDGrid(8)

    def test_public_fields_match_frozen_problem(self):
        from chapters.ch04.problem import Problem
        x = self.rng.random((20,2)); f,yd = public_fields(x)
        np.testing.assert_allclose(f,Problem().f(x)[:,0],atol=1e-14)
        np.testing.assert_allclose(yd,Problem().yd(x)[:,0],atol=1e-14)

    def test_exact_inner_uzawa_matches_independent_optimum(self):
        from .assessment import exact_uzawa
        result = exact_uzawa(24)
        self.assertLess(result['control_error_to_discrete_optimum'],1e-11)
        self.assertLess(result['history'][-1]['true_gradient_relative'],1e-11)

    def test_invalid_config_fails(self):
        c = defaults('uzawa'); c['validation_mesh']=c['mesh']
        with self.assertRaises(ValueError): validate_config(c)
        c = defaults('fem_noisy'); c['noise']=float('nan')
        with self.assertRaises(ValueError): validate_config(c)
        c = defaults('fem_noisy'); c['unknown_option']=2
        with self.assertRaises(ValueError): validate_config(c)

    def test_tiny_training_selection_reload_and_corruption(self):
        import tempfile
        import json
        from pathlib import Path
        from .assessment import audit_training
        from .common import write_json
        torch.set_num_threads(2)
        c = defaults('fem_clean')
        c.update(mesh=16,validation_mesh=24,adam_steps=3,lbfgs_steps=2,record_every=1,
                 width=6,depth=1,train_observations=16,val_observations=8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'training/fem_clean_seed0'
            write_json(root/'protocol.json',dict(jobs=[c]))
            train_fem(c,p)
            self.assertEqual(len(audit_training(root)),1)
            ck=torch.load(p/'best.pt',weights_only=False)
            ck['selected_index'] += 1
            torch.save(ck,p/'best.pt')
            with self.assertRaises(ValueError): audit_training(root)


if __name__ == '__main__':
    unittest.main()
