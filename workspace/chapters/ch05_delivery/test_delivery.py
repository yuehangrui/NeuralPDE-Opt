import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from chapters.ch05.models import Fields,SplitState,laplace
from chapters.ch05.problem import Problem
from chapters.ch05.reference import Grid
from . import data
from .common import Config,quadrature
from .reference import state_qp,independent_box
from .training import instance_loss,save,load,validation
from .metrics import control,prolong
from .protocol import check_disjoint,pretest,freeze,parameter_accounting,check_split_identity
from .queue import validate_config_data
from chapters.ch05.operators import Operators
from chapters.ch05.common import smoke_config

class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_quadrature_and_separate_validation(self):
        x,w=quadrature(8);self.assertAlmostEqual(float(w.sum()),1.)
        self.assertAlmostEqual(float((x[:,0:1]**4*w).sum()),.2)
        c=Config(method='osnn',width=8,depth=2,grid=8,val_points=20)
        m=Fields(c.problem(),c.method,c.width,c.depth).double();s=SplitState(c.problem(),c.beta)
        a=validation(m,c,s,None);c.seed=7;b=validation(m,c,s,None)
        self.assertEqual(a,b)
    def test_reconstructible_fields(self):
        a,_=data.make('train',3,8);data.validate(a,8)
        fine=data.inputs(a,16)
        np.testing.assert_allclose(fine['f'][:,1::2,1::2],a['f'],atol=1e-13)
    def test_semantic_and_duplicate_checks(self):
        a,_=data.make('train',2,8);b,_=data.make('val',2,8)
        self.assertFalse(data.fingerprints(a)&data.fingerprints(b))
        a['p'][0,0,0]+=.1
        with self.assertRaises(AssertionError):data.validate(a,8)
    def test_no_heldout_during_data_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)/'data';data.create_train_val(r,(2,2),8)
            self.assertFalse((r/'test.npz').exists());data.load(r,'train');data.load(r,'val')
    def test_weighted_instance_parameter_gradient(self):
        torch.manual_seed(2);c=Config(method='cnp',width=4,depth=1)
        m=Fields(c.problem(),c.method,c.width,c.depth).double();xx,ww=quadrature(5)
        x=torch.tensor(xx,requires_grad=True);w=torch.tensor(ww)
        p=next(m.parameters());loss=instance_loss(m,x,w);g=torch.autograd.grad(loss,p)[0][0,0].item()
        original=p[0,0].item();values=[]
        for delta in (-1e-6,1e-6):
            with torch.no_grad():p[0,0]=original+delta
            values.append(float(instance_loss(m,x,w).detach()))
        with torch.no_grad():p[0,0]=original
        self.assertAlmostEqual((values[1]-values[0])/2e-6,g,places=6)
    def test_state_reference_qp(self):
        n=8;g=Grid(n);a,report=state_qp(n);yd=a['yd']
        def fg(v):
            y=v.reshape(n-1,n-1);u=g.apply(y)
            return .5*np.sum((y-yd)**2)+.0005*np.sum(u**2),(y-yd+.001*g.apply(u)).ravel()
        r=minimize(fg,np.zeros((n-1)**2),jac=True,bounds=[(None,.2)]*((n-1)**2),method='L-BFGS-B',options=dict(ftol=1e-15,gtol=1e-10,maxiter=5000))
        self.assertTrue(r.success);np.testing.assert_allclose(r.x.reshape(n-1,n-1),a['y'],atol=1e-7)
        self.assertLess(report['feasibility'],1e-12);self.assertLess(report['dual_violation'],1e-8)
    def test_box_reference_independent(self):
        g=Grid(8);p=Problem();ref=g.optimal_control(p.f(g.points)[...,0],p.yd(g.points)[...,0],p)
        np.testing.assert_allclose(independent_box(8),ref['u'],atol=1e-5)
    def test_checkpoint_identity_and_split(self):
        c=Config(method='admm',width=8,depth=2);m=Fields(c.problem(),c.method,c.width,c.depth).double()
        s=SplitState(c.problem(),c.beta);s.append(m)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'best.pt';save(p,m,c,3,1.,s,None,{})
            restored,_,rs,b=load(p);x=torch.rand(10,2,dtype=torch.float64)
            for k,v in s.at(x).items():torch.testing.assert_close(v,rs.at(x)[k])
            b['source_hashes']['bad']='bad';torch.save(b,p)
            with self.assertRaises(ValueError):load(p)
    def test_physical_metrics_at_optimum(self):
        g=Grid(8);p=Problem();f=p.f(g.points)[...,0];yd=p.yd(g.points)[...,0];ref=g.optimal_control(f,yd,p)
        v=control(g,ref['u'],f,yd,ref,ref['y'],ref['p'])
        self.assertLess(float(v['projected_residual_l2']),1e-9)
        self.assertEqual(float(v['control_relative_l2']),0.)
        self.assertEqual(float(v['objective_gap']),0.)
    def test_prolongation_bounds_and_constant(self):
        u=np.full((7,7),.7);np.testing.assert_allclose(prolong(u,16),.7,atol=1e-15)
        u=np.random.default_rng(0).uniform(.3,1,(3,7,7));v=prolong(u,16)
        self.assertGreaterEqual(v.min(),.3);self.assertLessEqual(v.max(),1.)
    def test_partial_input_overlap_rejected(self):
        a,_=data.make('train',2,8);b,_=data.make('val',2,8)
        b['coeff_f'][0]=a['coeff_f'][0];b['coeff_yd'][0]=a['coeff_yd'][0]
        with self.assertRaises(ValueError):check_disjoint(a,b)
    def test_no_freeze_before_complete_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):pretest(tmp)
            with self.assertRaises(FileNotFoundError):freeze(tmp)
    def test_declared_unused_final_dual_block(self):
        c=smoke_config('iuzawa_f');m=Operators(c);f=torch.randn(2,7,7,1);yd=.02*torch.randn_like(f)
        with torch.no_grad():
            before=m(f,yd)
            for p in m.unrolled.uzawa_layers[-1].qb_sap_fourier_layers.parameters():p.add_(.1)
            after=m(f,yd)
        torch.testing.assert_close(before,after,atol=0,rtol=0)
        accounting=parameter_accounting(m,c.method)
        self.assertGreater(accounting['structurally_inactive_real_parameters'],0)
    def test_problem_and_job_data_must_match(self):
        c=Config(method='pdol');metadata=dict(problem=vars(c.problem()),grid=c.grid)
        validate_config_data(c,metadata);c.alpha=.002
        with self.assertRaises(ValueError):validate_config_data(c,metadata)
        c=Config(method='cnp',grid=16)
        with self.assertRaises(ValueError):validate_config_data(c,metadata)
    def test_split_identity_is_not_only_unique(self):
        a,_=data.make('train',2,8);check_split_identity(a,'train')
        with self.assertRaises(ValueError):check_split_identity(a,'val')
        a['sample_id']=np.arange(2)
        with self.assertRaises(ValueError):check_split_identity(a,'train')

if __name__=='__main__':unittest.main()
