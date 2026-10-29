"""Contracts for common data/metrics, derivative interfaces and selection gates."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from chapters.ch04_adaptation.operator_data import generate,kkt,EIGENVALUES,ALPHA
from chapters.ch04_completion.common import SpectralProblem
from chapters.ch04_operator_adaptation.training import build,Config
from .metrics import control_metrics,state_metrics
from .operators import ModalMIMO
from .assessment import CoefficientField,coefficient_predict,online,assess_fixed
from .protocol import freeze,generate_holdout,verify_freeze,sha
from .hybrids import identity,bilevel_run,uzawa_run


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_01_true_kkt_metrics(self):
        d=generate('validation',5,0)
        m=control_metrics(d['f'],d['yd'],d['u'],d['y'],d['p'])
        for key in ('control_relative_l2','true_gradient_relative','state_residual_relative','adjoint_residual_relative','optimality_residual_l2'):
            self.assertLess(float(max(abs(m[key]))),1e-12)
        self.assertLess(float(max(abs(m['objective_gap']))),1e-14)

    def test_02_objective_gradient_consistency(self):
        d=generate('validation',1,4);u=np.random.default_rng(3).normal(size=(1,17));v=np.ones_like(u);eps=1e-4
        val=lambda z:control_metrics(d['f'],d['yd'],z)['true_objective'][0]
        fd=(val(u+eps*v)-val(u-eps*v))/(2*eps)
        y=(d['f']+u)/EIGENVALUES;g=.25*(ALPHA*u+(y-d['yd'])/EIGENVALUES)
        self.assertAlmostEqual(fd,float(np.sum(g*v)),delta=1e-10)

    def test_03_modal_mimo_jvp_matches_difference(self):
        torch.manual_seed(9);model=ModalMIMO();x=torch.randn(2,2,17,dtype=torch.float64);d=torch.randn_like(x)
        _,j=torch.autograd.functional.jvp(model,x,d)
        eps=1e-5;fd=(model(x+eps*d)-model(x-eps*d))/(2*eps)
        torch.testing.assert_close(j,fd,rtol=1e-5,atol=1e-8)
        loss=model(x).square().sum();loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().max()>0 for p in model.net.branches.parameters()))

    def test_04_grid_coefficient_interface(self):
        for arch in ('fno','deeponet','fedonet'):
            for role in ('state','kkt'):
                model=build(Config(architecture=arch,role=role,device='cpu',small=True))
                c=torch.randn(2,2,17,dtype=torch.float64)
                x=(c@model.B.T).reshape(2,2,33,33)
                inputs=c.sum(1) if role=='state' else c
                entry=dict(kind='grid_operator',method=arch+'_'+role)
                physical=coefficient_predict(model,entry,inputs)
                field=model.encode_fields(model(x))
                torch.testing.assert_close(physical,field[:,0] if role=='state' else field,rtol=1e-10,atol=1e-11)

    def test_05_online_exact_model(self):
        class Exact(torch.nn.Module):
            def __init__(self):
                super().__init__();self.placeholder=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64),requires_grad=False)
            def forward(self,x):return x/torch.tensor(EIGENVALUES)
        d=generate('validation',4,7)
        u,y,info=online(Exact(),dict(kind='modal_operator',method='rb_dino'),d['f'],d['yd'])
        self.assertLess(float(control_metrics(d['f'],d['yd'],u)['true_gradient_relative'].max()),1e-5)

    def test_06_no_holdout_before_all_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):freeze(tmp)
            with self.assertRaises(FileNotFoundError):generate_holdout(tmp)
            self.assertFalse((Path(tmp)/'heldout').exists())

    def test_07_freeze_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.py';source.write_text('original')
            manifest=dict(source_hashes={str(source):sha(source)},config_hashes={},training_data_hashes={},entries=[])
            (root/'SELECTION_FROZEN.json').write_text(json.dumps(manifest))
            verify_freeze(root);source.write_text('changed')
            with self.assertRaises(ValueError):verify_freeze(root)

    def test_08_portable_hybrid_identity(self):
        self.assertTrue(all(not Path(p).is_absolute() for p in identity()))

    def test_09_actual_initialization_study(self):
        a,b=bilevel_run(0),bilevel_run(1)
        self.assertFalse(np.array_equal(a['initial_state_head'],b['initial_state_head']))
        self.assertLess(a['diagnostics']['true_gradient_relative'],1e-6)
        self.assertLess(b['diagnostics']['true_gradient_relative'],1e-6)

    def test_10_fixed_field_metrics(self):
        from chapters.ch04.fem import Reference
        from chapters.ch04.problem import Problem
        p=SpectralProblem();y,u,adj=kkt(p.f,p.yd)
        with tempfile.TemporaryDirectory() as tmp:
            result=assess_fixed(CoefficientField(y,u,adj),{},[Reference(Problem(),16,4)],Path(tmp))
        self.assertLess(result['control_relative_l2'],1e-12)
        self.assertLess(result['state_residual_relative'],1e-12)

    def test_11_metrics_reject_nonfinite(self):
        z=np.zeros((2,17));z[0,0]=np.nan
        with self.assertRaises(ValueError):control_metrics(z,z,z)


if __name__=='__main__':unittest.main()
