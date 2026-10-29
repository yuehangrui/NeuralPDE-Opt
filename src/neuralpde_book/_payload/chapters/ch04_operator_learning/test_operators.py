import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .models import Operator,SpectralConv
from .reference import Poisson
from .data import create,load,check_split
from .online import ControlNet,derivative_probe


class OperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_spectral_eigenmode(self):
        ref=Poisson(32);x=ref.points
        y=np.sin(8*np.pi*x[...,0])*np.sin(8*np.pi*x[...,1]);rhs=128*np.pi**2*y
        np.testing.assert_allclose(ref.solve(rhs),y,atol=1e-14)
    def test_reference_adjoint(self):
        ref=Poisson(16);r=np.random.default_rng(4);u=r.normal(size=(17,17))*ref.mask;v=r.normal(size=u.shape)*ref.mask
        self.assertAlmostEqual(ref.inner(ref.solve(u),v),ref.inner(u,ref.solve(v)),places=14)
    def test_kkt_and_gradient(self):
        ref=Poisson(16);r=np.random.default_rng(7);f=r.normal(size=(17,17))*ref.mask;d=r.normal(size=f.shape)*ref.mask
        y,u,p=ref.optimal(f,d)
        np.testing.assert_allclose(ref.apply(y)-u-f,0,atol=1e-12)
        np.testing.assert_allclose(ref.apply(p)-y+d,0,atol=1e-12)
        _,_,g,j=ref.recover(f,d,u);self.assertLess(ref.norm(g),1e-14)
        c=r.normal(size=f.shape)*ref.mask;v=r.normal(size=f.shape)*ref.mask;h=1e-5
        _,_,g,_=ref.recover(f,d,c)
        fd=(ref.recover(f,d,c+h*v)[3]-ref.recover(f,d,c-h*v)[3])/(2*h)
        self.assertAlmostEqual(fd,ref.inner(g,v),places=10)
    def test_data_archive_and_tampering(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'data';create(path,train_count=8,validation_count=4);tr,va,m=load(path)
            self.assertFalse(set(tr['ids'])&set(va['ids']))
            tr['state_y'][0,3,4]=np.nan
            with self.assertRaises(ValueError):check_split(tr,32)
            tr,_,_=load(path);tr['u'][0,3,4]+=.1;np.savez(path/'train.npz',**tr)
            with self.assertRaises(ValueError):load(path)
    def test_fft_shapes_and_dtypes(self):
        for h,w in ((9,11),(10,12),(7,6)):
            for dtype in (torch.float32,torch.float64):
                m=SpectralConv(2,12).to(dtype=dtype);x=torch.randn(2,2,h,w,dtype=dtype,requires_grad=True);y=m(x)
                self.assertEqual(y.shape,x.shape);self.assertEqual(y.dtype,dtype)
                y.square().sum().backward();self.assertTrue(bool(torch.isfinite(x.grad).all()))
    def test_six_interfaces_and_backward(self):
        for arch in ('fno','deeponet','fedonet'):
            for role in ('state','kkt'):
                m=Operator(arch,role,n=8,small=True).double();x=torch.randn(2,2,9,9,dtype=torch.float64,requires_grad=True)
                out=m(x);self.assertEqual(out.shape,(2,1 if role=='state' else 2,9,9))
                self.assertLess(float((out*(1-m.mask)).abs().max()),1e-14)
                out.square().sum().backward();self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters()))
                if role=='kkt':
                    y,u,p=m.fields(x);torch.testing.assert_close(.001*u+p,torch.zeros_like(p))
    def test_homogeneous_and_fixed_grid(self):
        for arch in ('fno','deeponet','fedonet'):
            m=Operator(arch,'state',n=8,small=True).double()
            self.assertLess(float(m(torch.zeros(2,2,9,9,dtype=torch.float64)).abs().max()),1e-14)
            with self.assertRaises(ValueError):m(torch.zeros(1,2,10,10,dtype=torch.float64))
    def test_operator_input_derivatives(self):
        ref=Poisson(8);x=np.random.default_rng(4).normal(size=(2,2,9,9))*ref.mask
        for arch in ('fno','deeponet','fedonet'):
            m=Operator(arch,'state',n=8,small=True).double();result=derivative_probe(m,x,ref)
            self.assertEqual(len(result['rows']),2)
    def test_control_network_frozen_surrogate(self):
        ref=Poisson(8);m=Operator('fedonet','state',n=8,small=True).double().requires_grad_(False)
        c=ControlNet(ref,modes=3);u=c();self.assertEqual(float(u.abs().max()),0.)
        x=torch.stack([torch.ones_like(u),u])[None];loss=m(x).square().sum()+.001*u.square().sum();loss.backward()
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().max())>0 for p in c.parameters()))

    def test_control_family_projection(self):
        from .online_supported import SupportedControl
        from .data import make_split
        ref=Poisson(32);net=SupportedControl(ref);data=make_split('validation',2)
        u=torch.tensor(data['u'][0]);torch.testing.assert_close(net.project(u),u,atol=1e-12,rtol=1e-12)
        x=ref.points;outside=torch.tensor(np.sin(6*np.pi*x[...,0])*np.sin(7*np.pi*x[...,1]))
        self.assertLess(float(net.project(outside).norm()),1e-12)
        a=torch.randn_like(u);b=torch.randn_like(u)
        torch.testing.assert_close(net.project(net.project(a)),net.project(a),atol=1e-12,rtol=1e-12)
        torch.testing.assert_close((net.project(a)*b).sum(),(a*net.project(b)).sum(),atol=1e-12,rtol=1e-12)

if __name__=='__main__':unittest.main()
