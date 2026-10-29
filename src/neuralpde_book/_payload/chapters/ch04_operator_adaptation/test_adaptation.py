import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from .models import AdaptedOperator
from .training import Config,physics_loss,tangent_loss
from chapters.ch04_operator_learning.data import make_split,arrays
from chapters.ch04_operator_learning.reference import Poisson

class AdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.tr=make_split('train',128);cls.va=make_split('validation',8)
    def model(self,arch='deeponet',role='state'):
        torch.manual_seed(0);m=AdaptedOperator(arch,role,small=True)
        x,y=(torch.tensor(a) for a in arrays(self.tr,role));m.set_normalization(x,y);return m,x,y
    def test_mass_projection(self):
        m,x,y=self.model();torch.testing.assert_close(4*m.B.T@m.B/1024,torch.eye(17,dtype=torch.float64),atol=1e-14,rtol=1e-14)
        torch.testing.assert_close((m.encode_fields(y)@m.B.T).reshape_as(y),y,atol=1e-13,rtol=1e-13)
    def test_unsupported_mode_rejected(self):
        m,x,y=self.model();p=m.ref.points
        bad=x[:1].clone();bad[:,0]+=torch.tensor(np.sin(6*np.pi*p[...,0])*np.sin(7*np.pi*p[...,1]))
        with self.assertRaises(ValueError):m(bad)
        bad=x[:1].clone();bad[0,0,2,2]=float('nan')
        with self.assertRaises(ValueError):m(bad)
    def test_state_split_invariance(self):
        m,x,y=self.model();delta=x[:2,0]*.123;other=x[:2].clone();other[:,0]+=delta;other[:,1]-=delta
        torch.testing.assert_close(m(x[:2]),m(other),atol=1e-12,rtol=1e-12)
    def test_six_interfaces_and_tangent_backward(self):
        for arch in ('fno','deeponet','fedonet'):
            for role in ('state','kkt'):
                m,x,y=self.model(arch,role);c=m.inputs(x[:4]);z=(c-m.input_mean)/m.input_scale
                t=(m.encode_fields(y[:4])-m.output_mean)/m.output_scale
                loss=tangent_loss(m,z,z.flip(0)-z,t.flip(0)-t)+physics_loss(m,c,m.standardized(z))
                loss.backward();self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters()))
                self.assertEqual(m(x[:2]).shape,y[:2].shape)
    def test_tangent_finite_difference(self):
        for arch in ('fno','deeponet','fedonet'):
            m,x,y=self.model(arch);c=m.inputs(x[:2]);z=(c-m.input_mean)/m.input_scale;v=torch.randn_like(z)
            _,j=torch.autograd.functional.jvp(m.standardized,z,v)
            h=1e-5;fd=(m.standardized(z+h*v)-m.standardized(z-h*v))/(2*h)
            torch.testing.assert_close(j,fd,atol=2e-8,rtol=2e-5)
    def test_reference_derivative_label(self):
        ref=Poisson()
        for role in ('state','kkt'):
            x,y=arrays(self.tr,role);d=x[1:]-x[:-1]
            exact=ref.solve(d[:,0]+d[:,1])[:,None] if role=='state' else np.stack(ref.optimal(d[:,0],d[:,1])[:2],1)
            np.testing.assert_allclose(exact,y[1:]-y[:-1],atol=3e-13,rtol=1e-10)
    def test_physics_zero_on_labels(self):
        for role in ('state','kkt'):
            m,x,y=self.model(role=role);c=m.inputs(x);target=(m.encode_fields(y)-m.output_mean)/m.output_scale
            self.assertLess(float(physics_loss(m,c,target)),1e-22)
    def test_train_only_head_and_derivative(self):
        for arch in ('fno','deeponet','fedonet'):
            for role in ('state','kkt'):
                m,x,y=self.model(arch,role);weights={k:v.detach().clone() for k,v in m.named_parameters()};fit=m.fit_head(x,y)
                self.assertTrue(all(torch.equal(v,weights[k]) for k,v in m.named_parameters()))
                vx,vy=(torch.tensor(a) for a in arrays(self.va,role))
                torch.testing.assert_close(m(vx),vy,atol=1e-8,rtol=1e-8)
                self.assertLess(float(m(torch.zeros_like(vx[:1])).abs().max()),1e-12)
                c=m.inputs(vx[:2]);v=torch.randn_like(c);h=1e-5
                _,j=torch.autograd.functional.jvp(m.coefficients,c,v)
                fd=(m.coefficients(c+h*v)-m.coefficients(c-h*v))/(2*h)
                torch.testing.assert_close(j,fd,atol=1e-8,rtol=1e-7)
    def test_config_rejects_invalid(self):
        for kw in (dict(epochs=0),dict(seed=-1),dict(tangent_samples=0),dict(tangent_weight=float('nan'))):
            with self.assertRaises(ValueError):Config(**kw).validate()

if __name__=='__main__':unittest.main()
