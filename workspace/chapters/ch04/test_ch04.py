"""Numerical and workflow checks for the migrated control methods."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .problem import Problem
from .models import Fields, laplace, loss_terms, multiplier_update
from .fem import Reference
from .evaluation import sample_points, evaluate
from .training import Config, build, train, load_checkpoint, tolerances_met


class ControlChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.problem = Problem()
        cls.ref = Reference(cls.problem,16,4)

    def test_manufactured_kkt(self):
        x = torch.rand(51,2,dtype=torch.float64,requires_grad=True)
        p = self.problem
        state = -laplace(p.y(x),x)-p.u(x)-p.f(x)
        adj = -laplace(p.p(x),x)-p.y(x)+p.yd(x)
        self.assertLess(float(state.abs().max()),1e-12)
        self.assertLess(float(adj.abs().max()),1e-12)
        self.assertLess(float((p.alpha*p.u(x)+p.p(x)).abs().max()),1e-16)

    def test_invalid_problem(self):
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): Problem(alpha=value)
        with self.assertRaises(ValueError): Problem(control_frequency=1.5)

    def test_sampling_is_independent_and_reproducible(self):
        a,b = sample_points(0,1,100),sample_points(0,2,100)
        self.assertTrue(np.array_equal(a,sample_points(0,1,100)))
        self.assertEqual(len(set(map(tuple,a)) & set(map(tuple,b))),0)
        self.assertTrue(np.all((a>0)&(a<1)))

    def test_hard_state_and_adjoint_boundaries(self):
        model = Fields('kkt',width=4,depth=1).double()
        x = torch.tensor([[0.,.2],[1.,.7],[.3,0.],[.8,1.]],dtype=torch.float64)
        y,u,p = model(x)
        self.assertEqual(float(y.abs().max()),0)
        self.assertEqual(float(p.abs().max()),0)
        self.assertEqual(float((model.alpha*u+p).abs().max()),0)

    def test_direct_control_is_not_boundary_constrained(self):
        model = Fields('direct',width=4,depth=1).double()
        with torch.no_grad():
            for p in model.second_net.parameters(): p.zero_()
            model.second_net[-1].bias.fill_(1.)
        y,u,p = model(torch.tensor([[0.,.4]],dtype=torch.float64))
        self.assertEqual(float(u),1.); self.assertIsNone(p)

    def test_direct_loss_weights(self):
        model = Fields('direct',width=4,depth=1).double()
        x = torch.rand(12,2,dtype=torch.float64,requires_grad=True)
        loss,obj,r = loss_terms(model,self.problem,x,penalty=2.5)
        self.assertAlmostEqual(float(loss),float(obj+2.5*r.square().mean()),places=12)
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_kkt_exact_loss(self):
        problem = self.problem
        class Exact(torch.nn.Module):
            method = 'kkt'
            def forward(self,x): return problem.y(x),problem.u(x),problem.p(x)
        x = torch.rand(25,2,dtype=torch.float64,requires_grad=True)
        loss,_,_ = loss_terms(Exact(),problem,x)
        self.assertLess(float(loss),1e-24)

    def test_alm_multiplier_uses_old_penalty_and_detaches(self):
        r = torch.tensor([[2.],[-3.]],requires_grad=True)
        update = multiplier_update(torch.ones_like(r),r,.2)
        self.assertTrue(torch.allclose(update,torch.tensor([[1.8],[-.2]])))
        self.assertFalse(update.requires_grad)

    def test_feasibility_alone_cannot_stop(self):
        self.assertFalse(tolerances_met({'pde_relative':0.,'reduced_gradient_relative':1.},Config()))

    def test_fem_mass_and_stiffness(self):
        r = self.ref
        self.assertLess(np.max(abs((r.K-r.K.T).data),initial=0),1e-14)
        self.assertLess(np.max(abs((r.M-r.M.T).data),initial=0),1e-14)
        self.assertAlmostEqual(float(np.ones(len(r.nodes))@r.M@np.ones(len(r.nodes))),1.,places=13)
        self.assertGreater(np.linalg.eigvalsh(r.Ki.toarray()).min(),0)

    def test_fem_kkt_and_recovered_stationarity(self):
        report = self.ref.report()
        self.assertLess(report['algebraic_residual'],1e-11)
        self.assertLess(report['reduced_gradient_l2'],1e-12)

    def test_discrete_adjoint_directional_derivative(self):
        r = self.ref
        u = .2*np.sin(3*np.pi*r.points[...,0])*np.sin(2*np.pi*r.points[...,1])
        v = np.cos(2*np.pi*r.points[...,0])
        y,p,g = r.recover(u)
        grad = np.sum(r.weights*g*v)
        def objective(e):
            yy,_,_ = r.recover(u+e*v)
            return r.objective(r.at_quad(yy),u+e*v)
        h = 1e-4
        fd = (objective(h)-objective(-h))/(2*h)
        self.assertLess(abs(fd-grad),1e-9)

    def test_feasible_nonoptimal_control(self):
        y,p,g = self.ref.recover(-self.ref.fq)
        self.assertLess(np.linalg.norm(y),1e-12)
        self.assertGreater(self.ref.norm(g),1e-3)

    def test_refinement_reduces_error(self):
        coarse = self.ref.report()
        fine = Reference(self.problem,32,4).report()
        for name in ('y','u','p'):
            self.assertLess(fine['relative_l2_'+name],coarse['relative_l2_'+name])

    def test_underresolved_mesh_rejected(self):
        with self.assertRaises(ValueError): Reference(self.problem,8)

    def test_invalid_configs(self):
        for kwargs in ({'train_points':0},{'mu_growth':.5},{'seed':-1},{'adam_steps':0,'lbfgs_steps':0},{'penalty':float('nan')}):
            with self.assertRaises(ValueError): Config(**kwargs).validate()

    def test_nonfinite_control_rejected(self):
        with self.assertRaises(ValueError): self.ref.recover(np.full_like(self.ref.fq,np.nan))

    def test_checkpoint_selection_and_output_protection(self):
        cfg = Config(method='direct',width=4,depth=1,train_points=24,validation_points=12,
            validation_mesh=16,quadrature_order=2,adam_steps=2,lbfgs_steps=1,evaluate_every=1,smoke=True)
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder)/'run'
            result = train(cfg,dest)
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['run_kind'],'smoke')
            rows = [json.loads(s) for s in (dest/'history.jsonl').read_text().splitlines()]
            best = min(rows,key=lambda r:r['validation_score'])
            self.assertEqual(best['progress'],result['best_progress'])
            model,loaded,checkpoint = load_checkpoint(dest/'best.pt')
            self.assertEqual(loaded.method,cfg.method)
            self.assertEqual(checkpoint['progress'],result['best_progress'])
            self.assertEqual(checkpoint['sample_hashes'],result['sample_hashes'])
            with self.assertRaises(FileExistsError): train(cfg,dest)

    def test_changed_source_identity_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'bad.pt'
            torch.save({'schema':'ch04-instance-v1','source_hashes':{}},path)
            with self.assertRaises(ValueError): load_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
