import tempfile
from pathlib import Path
import unittest
import numpy as np
from scipy import sparse
import bootstrap
from integrity import certify, verify, inside, require
from validation import validate_fields, disjoint
from hybrid import solve


class ReleaseTests(unittest.TestCase):
    def test_unexpected_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'a').write_text('valid'); certify(p); verify(p)
            (p/'extra').write_text('unexpected')
            with self.assertRaises(ValueError): verify(p)

    def test_changed_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'a').write_text('valid'); certify(p); (p/'a').write_text('changed')
            with self.assertRaises(ValueError): verify(p)

    def test_unsafe_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for path in ['../outside','/etc/passwd','']:
                with self.assertRaises(ValueError): inside(tmp,path)

    def test_validation_is_active_under_optimization(self):
        with self.assertRaises(ValueError): require(False,'gate')

    def test_split_overlap_rejected(self):
        with self.assertRaises(ValueError): disjoint({'a','b'},{'b'})

    def test_group_and_sample_correspondence(self):
        c=np.ones((2,65,65)); y=c.copy(); y[:,[0,-1],:]=0; y[:,:,[0,-1]]=0
        d=dict(c=c,y=y,f=c.copy(),sample_id=np.array(['g:f0','g:f1']),coefficient_group=np.array(['g','g']))
        d['f'][1]*=2; validate_fields(d,1,2)
        d['coefficient_group']=np.array(['g','wrong'])
        with self.assertRaises(ValueError): validate_fields(d)

    def test_same_group_different_field_rejected(self):
        c=np.ones((2,65,65)); c[1]*=1.1; y=c.copy(); y[:,[0,-1],:]=0; y[:,:,[0,-1]]=0
        d=dict(c=c,y=y,sample_id=np.array(['g:f0','g:f1']),coefficient_group=np.array(['g','g']))
        with self.assertRaises(ValueError): validate_fields(d,1,2)

    def test_nonsymmetric_contraction_claim_rejected(self):
        a=sparse.csr_matrix([[2.,.3],[.1,2.]])
        with self.assertRaises(ValueError): solve(a,np.ones(2),mode='jacobi')

    def test_rhs_column_rejected(self):
        with self.assertRaises(ValueError): solve(sparse.eye(2),np.ones((2,1)),mode='jacobi')

    def test_corrector_cannot_mutate_residual(self):
        a=sparse.eye(2); b=np.array([1.,2.])
        def bad(r): r[:]=0.; return np.full_like(r,np.nan)
        x,r=solve(a,b,bad,mode='safeguarded',omega=1.,maxiter=1)
        np.testing.assert_array_equal(x,b); self.assertTrue(r['converged']); self.assertEqual(r['rejected_neural'],1)

    def test_last_step_and_zero_rhs(self):
        a=sparse.eye(2)
        _,r=solve(a,np.ones(2),mode='jacobi',omega=1.,maxiter=1)
        self.assertTrue(r['converged']); self.assertEqual(r['iterations'],1)
        _,r=solve(a,np.zeros(2),mode='jacobi',maxiter=0)
        self.assertTrue(r['converged']); self.assertEqual(r['iterations'],0)

    def test_fixed_divergence_is_reported(self):
        _,r=solve(sparse.eye(2),np.ones(2),lambda r:np.full_like(r,np.nan),mode='fixed')
        self.assertFalse(r['converged']); self.assertEqual(r['failure'],'nonfinite neural proposal')

    def test_valid_solver_matches_historical_numerics(self):
        old=bootstrap.module('old_hints_solver','experiments/ch03_hints_v1/solver.py')
        a=sparse.diags([-.1*np.ones(3),np.ones(4),-.1*np.ones(3)],[-1,0,1]).tocsr(); b=np.arange(1.,5.)
        for mode in ['jacobi','fixed','safeguarded']:
            kw=dict(mode=mode,corrector=lambda r:r*.2,period=3,trace=True)
            x,r=solve(a,b,**kw); y,s=old.solve(a,b,**kw)
            np.testing.assert_array_equal(x,y); self.assertEqual(r['residual_history'],s['residual_history'])


if __name__=='__main__': unittest.main()
