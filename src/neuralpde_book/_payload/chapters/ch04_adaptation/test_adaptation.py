"""Numerical regression tests for the actual port failure modes."""
import unittest
import numpy as np
import torch
from scipy import sparse
from .instance import quadrature,EnrichedFields,weighted_loss,InstanceConfig
from .operator_data import basis,generate,validate,EIGENVALUES
from .operator_models import MIMO,RBDINO
from .discrete_derivatives import state,jacobian,jvp,reduced_gradient
from chapters.ch04.models import laplace
from chapters.ch04.problem import Problem

class AdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_quadrature_mass(self):
        x,w=quadrature(48);b=basis(x)
        np.testing.assert_allclose(b.T@(w*b),np.eye(b.shape[1])/4,atol=2e-14)
        self.assertAlmostEqual(w.sum(),1.)

    def test_enrichment_derivatives(self):
        torch.manual_seed(4);m=EnrichedFields('direct',width=8,depth=2,modes=4).double()
        with torch.no_grad():m.state_head.normal_();m.control_head.normal_()
        x=torch.rand(11,2,dtype=torch.float64,requires_grad=True);p=Problem()
        y,u,_=m(x);a=-laplace(y,x)-u-p.f(x);b=m.state_residual(x,p)
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)

    def test_weighted_loss_gradient(self):
        torch.manual_seed(3);m=EnrichedFields('direct',width=8,depth=1,modes=3).double();p=Problem()
        x,w=quadrature(16);x=torch.tensor(x,requires_grad=True);w=torch.tensor(w)
        fn=lambda:weighted_loss(m,p,x,w,.01,torch.zeros_like(w))[0]
        loss=fn();g=torch.autograd.grad(loss,m.control_head)[0][2].item();h=1e-5
        with torch.no_grad():m.control_head[2]+=h
        a=fn().item()
        with torch.no_grad():m.control_head[2]-=2*h
        b=fn().item()
        self.assertAlmostEqual(g,(a-b)/(2*h),places=8)

    def test_multiplier_sign(self):
        p=Problem();x=torch.rand(15,2,dtype=torch.float64);lam=-p.p(x)
        torch.testing.assert_close(p.alpha*p.u(x)-lam,torch.zeros_like(lam))

    def test_exact_linear_head_solve(self):
        from .head_polish import fit_heads
        torch.manual_seed(4);m=EnrichedFields('alm',width=8,depth=2,modes=3).double()
        xn,wn=quadrature(20);x=torch.tensor(xn,requires_grad=True);p=Problem(control_frequency=2)
        multiplier=np.random.default_rng(4).normal(size=wn.shape)*.001
        report=fit_heads(m,p,x,wn,.01,multiplier)
        loss,_,_=weighted_loss(m,p,x,torch.tensor(wn),.01,torch.tensor(multiplier))
        grads=torch.autograd.grad(loss,(m.state_head,m.control_head))
        self.assertLess(float(torch.cat(grads).norm()),1e-11)
        self.assertEqual(report['rank'],18)

    def test_config_validation(self):
        for kwargs in [dict(quadrature=8),dict(mu=float('nan')),dict(adam_steps=0,lbfgs_steps=0),dict(sampling='test')]:
            with self.assertRaises(ValueError):InstanceConfig(**kwargs).validate()

    def test_data_roles_and_labels(self):
        a,b=generate('train',16),generate('validation',16);validate(a,b)
        np.testing.assert_allclose(a['state']*EIGENVALUES,a['rhs'],atol=1e-12)
        with self.assertRaises(ValueError):generate('test',16)

    def test_overlap_and_bad_labels_rejected(self):
        a,b=generate('train',8),generate('validation',8)
        with self.assertRaises(ValueError):validate(a,a)
        b['u'][0,0]+=1
        with self.assertRaises(ValueError):validate(a,b)

    def test_mimo_boundary_and_scaling(self):
        m=MIMO(sensors=5,width=8,latent=4,modes=3).double()
        x=torch.tensor([[0.,.3],[1.,.6],[.5,0.],[.2,1.]],dtype=torch.float64)
        y,u,p=m(torch.rand(2,5,dtype=torch.float64),torch.rand(2,5,dtype=torch.float64),x)
        self.assertLess(float(y.abs().max()),1e-14);self.assertLess(float(p.abs().max()),1e-14)
        torch.testing.assert_close(.001*u+p,torch.zeros_like(p))

    def test_mimo_residual_backprop(self):
        torch.manual_seed(9);m=MIMO(sensors=5,width=8,latent=4,modes=3).double()
        f,d=torch.randn(2,5,dtype=torch.float64),torch.randn(2,5,dtype=torch.float64)
        x=torch.rand(6,2,dtype=torch.float64);values=torch.randn(2,6,dtype=torch.float64)
        a,b=m.residuals(f,d,x,values,values,nonlinear=True)
        loss=a.square().mean()+b.square().mean();loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters()))
        separate=m.residuals(f[1:2],d[1:2],x,values[1:2],values[1:2],nonlinear=True)
        torch.testing.assert_close(a[1],separate[0][0])

    def test_dino_chain_rule(self):
        torch.manual_seed(3);m=RBDINO(dimension=4,rank=4,width=8).double()
        with torch.no_grad():
            m.Psi.copy_(torch.linalg.qr(torch.randn(4,4,dtype=torch.float64))[0])
            m.Phi.copy_(torch.linalg.qr(torch.randn(4,4,dtype=torch.float64))[0])
            m.input_scale.copy_(torch.tensor([1.,2.,3.,4.],dtype=torch.float64))
            m.output_scale.copy_(torch.tensor([.1,.2,.3,.4],dtype=torch.float64))
        x=torch.randn(2,4,dtype=torch.float64);v=torch.randn_like(x);h=1e-5
        fd=(m(x+h*v)-m(x-h*v))/(2*h)
        exact=torch.einsum('bij,bj->bi',m.physical_jacobian(x),v)
        torch.testing.assert_close(fd,exact,atol=1e-10,rtol=1e-7)

    def matrices(self):
        k=sparse.diags([-np.ones(3),3*np.ones(4),-np.ones(3)],[-1,0,1]).tocsc()
        m=sparse.diags([.2*np.ones(3),np.ones(4),.2*np.ones(3)],[-1,0,1]).tocsc()
        return k,m

    def test_samplewise_nonlinear_jvp(self):
        k,m=self.matrices();u=np.array([[1.,2.,3.,4.],[5.,-1.,2.,1.]]);f=np.ones(4)
        y=np.stack([state(k,m,ui,f) for ui in u]);v=np.random.default_rng(3).normal(size=(2,4,2))
        d=jvp(k,m,y,v);h=1e-5
        for b in range(2):
            for j in range(2):
                fd=(state(k,m,u[b]+h*v[b,:,j],f)-state(k,m,u[b]-h*v[b,:,j],f))/(2*h)
                np.testing.assert_allclose(d[b,:,j],fd,atol=1e-9,rtol=1e-7)
        self.assertGreater(np.linalg.norm(jacobian(k,m,y[0]).toarray()-jacobian(k,m,y[0]).toarray().T),.01)

    def test_transpose_adjoint_gradient(self):
        k,m=self.matrices();u=np.array([1.,2.,3.,4.]);f=np.ones(4);yd=np.array([.2,.4,.7,-.3]);alpha=.001
        y=state(k,m,u,f);g=reduced_gradient(k,m,y,u,yd,alpha);v=np.array([.4,-.7,1.,.2])
        def objective(c):
            yy=state(k,m,c,f)-yd
            return .5*yy@(m@yy)+alpha/2*c@(m@c)
        h=1e-5;self.assertAlmostEqual(g@v,(objective(u+h*v)-objective(u-h*v))/(2*h),places=8)

    def test_fail_closed_nonlinear_solve(self):
        k,m=self.matrices()
        with self.assertRaises(RuntimeError):state(k,m,np.ones(4)*100,np.zeros(4),maxiter=0)
        with self.assertRaises(ValueError):jvp(k,m,np.ones((2,4)),np.ones((1,4,3)))

if __name__=='__main__':unittest.main()
