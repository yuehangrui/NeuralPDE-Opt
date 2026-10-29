"""Numerical and lifecycle regressions for the adapted algorithms."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from .problem import Problem,prox_l1_box
from .reference import Grid,primal_dual
from .models import Fields,SplitState,laplace,loss_terms
from .operator_blocks import LinearFourier,SAPFourier,FNO
from .operators import Operators
from .common import Config,smoke_config,sha,write_json
from .data import generate,load_split
from .training import save,load,train


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)
    def setUp(self): torch.manual_seed(91); self.p=Problem(); self.g=Grid(8)

    def test_manufactured_projected_kkt(self):
        x=torch.rand(30,2,dtype=torch.float64,requires_grad=True)
        p=self.p
        self.assertLess(float((-laplace(p.y(x),x)-p.u(x)-p.f(x)).abs().max().detach()),1e-13)
        self.assertLess(float((-laplace(p.p(x),x)-p.y(x)+p.yd(x)).abs().max().detach()),1e-13)
        torch.testing.assert_close(p.u(x),torch.clamp(-p.p(x)/p.alpha,p.lower,p.upper))

    def test_prox_l1_box_scalar_minimization(self):
        for v in (-3.,-.2,.1,1.,3.):
            actual=prox_l1_box(np.array(v),.4,-.7,1.1)
            mesh=np.linspace(-.7,1.1,30001); cost=.5*(mesh-v)**2+.4*np.abs(mesh)
            self.assertLess(abs(actual-mesh[np.argmin(cost)]),7e-5)

    def test_invalid_problem(self):
        for kw in ({'alpha':0},{'lower':2},{'upper':float('nan')},{'state_cap':-1}):
            with self.assertRaises(ValueError): Problem(**kw)

    def test_fd_inverse(self):
        v=np.random.default_rng(0).normal(size=(2,7,7))
        np.testing.assert_allclose(self.g.apply(self.g.solve(v)),v,atol=2e-14)

    def test_discrete_adjoint(self):
        rng=np.random.default_rng(2); a=rng.normal(size=(7,7)); b=rng.normal(size=(7,7))
        self.assertLess(abs(np.sum(a*self.g.solve(b))-np.sum(b*self.g.solve(a))),1e-14)

    def test_reduced_gradient_finite_difference(self):
        rng=np.random.default_rng(3); u=rng.normal(size=(7,7)); v=rng.normal(size=u.shape)
        f=self.p.f(self.g.points)[...,0]; yd=self.p.yd(self.g.points)[...,0]
        def j(w): return .5*self.g.norm(self.g.solve(w+f)-yd)**2+.5*self.p.alpha*self.g.norm(w)**2
        g=self.g.recover(u,f,yd,self.p.alpha)[2]
        self.assertLess(abs((j(u+1e-5*v)-j(u-1e-5*v))/2e-5-np.sum(g*v)/64),1e-12)

    def test_reference_against_independent_optimizer(self):
        f=self.p.f(self.g.points)[...,0]; yd=self.p.yd(self.g.points)[...,0]
        result=self.g.optimal_control(f,yd,self.p)
        def fg(v):
            y,p,g=self.g.recover(v.reshape(7,7),f,yd,self.p.alpha)
            return .5*np.sum((y-yd)**2)+.5*self.p.alpha*np.sum(v*v),g.ravel()
        other=minimize(fg,np.ones(49)*.5,jac=True,bounds=[(.3,1.)]*49,method='L-BFGS-B',options=dict(gtol=1e-12,ftol=1e-15,maxiter=1000))
        self.assertTrue(other.success)
        self.assertLess(np.max(np.abs(result['u']-other.x.reshape(7,7))),1e-5)

    def test_unconverged_labels_rejected(self):
        with self.assertRaises(RuntimeError): self.g.optimal_control(np.zeros((7,7)),np.ones((7,7)),self.p,max_iter=1)

    def test_reference_refinement(self):
        errors=[]
        for n in (8,16,32):
            grid=Grid(n); p=self.p; ref=grid.optimal_control(p.f(grid.points)[...,0],p.yd(grid.points)[...,0],p)
            errors.append(float(grid.norm(ref['u']-p.u(grid.points)[...,0])))
        self.assertLess(errors[2],errors[1]); self.assertLess(errors[1],errors[0])

    def test_exact_paired_primal_dual(self):
        g=self.g; p=self.p
        f=torch.tensor(p.f(g.points)[...,0][None,...,None]); yd=torch.tensor(p.yd(g.points)[...,0][None,...,None])
        action=lambda v:torch.from_numpy(g.solve(v.numpy()[...,0]))[...,None]
        u,info=primal_dual(action,action,f,yd,p,g.s_norm,max_iter=1500,tol=1e-9)
        ref=g.optimal_control(f.numpy()[...,0],yd.numpy()[...,0],p)
        self.assertTrue(all(info['surrogate_converged']))
        np.testing.assert_allclose(u.numpy()[...,0],ref['u'],atol=5e-9)

    def test_hard_boundaries(self):
        x=torch.tensor([[0.,.2],[1.,.3],[.4,0],[.5,1]],dtype=torch.float64,requires_grad=True)
        for method in ('osnn','admm','cnp'):
            model=Fields(self.p,method,8,2).double()
            self.assertLess(float(model.y(x).abs().max().detach()),1e-15)

    def test_cnp_feasibility_and_true_derivatives(self):
        model=Fields(self.p,'cnp',8,2).double()
        x=torch.rand(12,2,dtype=torch.float64,requires_grad=True)
        self.assertLessEqual(float(model.y(x).max().detach()),self.p.state_cap)
        h=2e-4; approx=torch.zeros_like(model.y(x))
        for axis in (0,1):
            delta=torch.zeros_like(x); delta[:,axis]=h
            approx+=(model.y(x+delta)-2*model.y(x)+model.y(x-delta))/h**2
        torch.testing.assert_close(laplace(model.y(x),x),approx,atol=4e-6,rtol=4e-5)

    def test_loss_backward(self):
        for method in ('osnn','admm','cnp'):
            model=Fields(self.p,method,8,2).double(); x=torch.rand(13,2,dtype=torch.float64,requires_grad=True)
            sum(loss_terms(model,x).values()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_admm_sign_and_stored_iterates(self):
        model=Fields(self.p,'admm',8,2).double(); split=SplitState(self.p,.002)
        x=torch.rand(9,2,dtype=torch.float64); previous=split.at(x)
        split.append(model); actual=split.at(x)
        u=(.002*previous['z']-previous['lam']-self.p.alpha*model.q(x))/(self.p.alpha+.002)
        torch.testing.assert_close(actual['u'],u)
        torch.testing.assert_close(actual['lam'],previous['lam']+.002*(u-actual['z']))
        self.assertFalse(torch.equal(actual['u'],model.u(x)))
        split.append(model)
        rebuilt=SplitState(self.p,.002); rebuilt.load(split.state_dict(),model)
        newx=torch.rand(17,2,dtype=torch.float64)
        for k,v in split.at(newx).items(): torch.testing.assert_close(v,rebuilt.at(newx)[k])

    def test_fourier_odd_shapes_and_backward(self):
        for n in (7,8):
            layer=LinearFourier(2,2,3,(2,2),(n,n))
            x=torch.rand(2,2,n,n,requires_grad=True); y=layer(x)
            self.assertEqual(y.shape,(2,3,n,n)); y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_fourier_rejects_transfer_and_overlap(self):
        with self.assertRaises(ValueError): LinearFourier(2,1,1,(4,4),(8,8))
        layer=LinearFourier(2,1,1,(2,2),(8,8))
        with self.assertRaises(AssertionError): layer(torch.ones(1,1,16,16))

    def test_sap_symmetry_positive(self):
        for n in (7,8):
            layer=SAPFourier(2,1,(2,2),3,4,(n,n),padding=(1,1))
            a=torch.randn(2,n,n,1); b=torch.randn_like(a)
            la,lb=layer(a),layer(b)
            self.assertLess(float(((la*b).sum()-(a*lb).sum()).abs().detach()),2e-6)
            self.assertGreaterEqual(float((a*la).sum().detach()),-1e-7)

    def test_unrolled_feasibility_and_backward(self):
        for method in ('iuzawa_f','iuzawa_s'):
            c=smoke_config(method); model=Operators(c)
            f=torch.randn(3,7,7,1); yd=.05*torch.randn_like(f)
            u=model(f,yd)
            self.assertGreaterEqual(float(u.min().detach()),c.lower-1e-7)
            self.assertLessEqual(float(u.max().detach()),c.upper+1e-7)
            u.square().mean().backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads)); self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)

    def test_pdol_zero_mapping(self):
        model=Operators(smoke_config('pdol')); z=torch.zeros(2,7,7,1)
        for role in ('state','adjoint'): torch.testing.assert_close(model.action(z,role),z)

    def test_reject_invalid_configs(self):
        for kw in ({'method':'bad'},{'beta':0},{'width':2.5},{'grid':3},{'learning_rate':float('nan')}):
            with self.assertRaises(ValueError): Config(**kw).validate()


class LifecycleTests(unittest.TestCase):
    def test_data_validation_and_no_test_read(self):
        with tempfile.TemporaryDirectory(prefix='ch05-tests-') as temp:
            root=Path(temp)/'data'; generate(root,8,(3,2,2))
            tr,meta=load_split(root,'train'); va,_=load_split(root,'val')
            self.assertEqual(np.intersect1d(tr['sample_id'],va['sample_id']).size,0)
            # Missing held-out test must not prevent train/val use.
            (root/'test.npz').unlink(); load_split(root,'train'); load_split(root,'val')
            with self.assertRaises(FileExistsError): generate(root,8,(3,2,2))
            with (root/'train.npz').open('ab') as stream: stream.write(b'corrupt')
            with self.assertRaises(ValueError): load_split(root,'train')

    def test_semantic_label_validation(self):
        with tempfile.TemporaryDirectory(prefix='ch05-tests-') as temp:
            root=Path(temp)/'data'; generate(root,8,(2,2,2))
            data,meta=load_split(root,'train'); data['p']+=.1
            np.savez_compressed(root/'train.npz',**data)
            meta['splits']['train']['sha256']=sha(root/'train.npz'); write_json(root/'manifest.json',meta)
            with self.assertRaises(ValueError): load_split(root,'train')

    def test_checkpoint_split_reload(self):
        with tempfile.TemporaryDirectory(prefix='ch05-tests-') as temp:
            c=smoke_config('admm'); model=Fields(c.problem(),c.method,c.width,c.depth).double()
            split=SplitState(c.problem(),c.beta); split.append(model); split.append(model)
            path=Path(temp)/'model.pt'; save(path,model,c,2,1.,split)
            restored,_,rs,_=load(path); x=torch.rand(16,2,dtype=torch.float64)
            torch.testing.assert_close(model.y(x),restored.y(x))
            for k,v in split.at(x).items(): torch.testing.assert_close(v,rs.at(x)[k])

    def test_source_identity_enforced(self):
        with tempfile.TemporaryDirectory(prefix='ch05-tests-') as temp:
            c=smoke_config('osnn'); model=Fields(c.problem(),c.method,c.width,c.depth).double(); path=Path(temp)/'model.pt'
            save(path,model,c,0,1.); blob=torch.load(path,weights_only=True)
            blob['source_hashes']['models.py']='incorrect'; torch.save(blob,path)
            with self.assertRaises(ValueError): load(path)

    def test_training_earliest_best_reload_and_fresh_directory(self):
        with tempfile.TemporaryDirectory(prefix='ch05-tests-') as temp:
            root=Path(temp)/'run'; c=smoke_config('osnn'); train(c,root)
            history=json.loads((root/'history.json').read_text()); best=min(history,key=lambda item:item['selection_score'])
            _,_,_,blob=load(root/'best.pt')
            self.assertEqual(blob['selection_score'],best['selection_score'])
            self.assertEqual(blob['optimizer_updates'],best['optimizer_updates'])
            with self.assertRaises(FileExistsError): train(c,root)


if __name__=='__main__': unittest.main()
