import unittest
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from solver import solve,residual_coordinates,contraction_bound

class HINTSTests(unittest.TestCase):
    def setUp(self):
        self.a=sparse.diags([-np.ones(11),2.5*np.ones(12),-np.ones(11)],[-1,0,1],format='csr');self.b=np.arange(12,dtype=float)+1
    def test_exact_neural_correction_and_last_step_check(self):
        y,r=solve(self.a,self.b,lambda v:spsolve(self.a,v),maxiter=1,trace=True)
        self.assertTrue(r['converged']);self.assertEqual(r['iterations'],1);self.assertEqual(r['accepted_neural'],1);self.assertEqual(r['classical_steps'],0)
        np.testing.assert_allclose(self.a@y,self.b,atol=1e-12)
    def test_bad_neural_step_rejected_and_fallback_converges(self):
        _,r=solve(self.a,self.b,lambda v:-10*v,period=3,maxiter=1000,trace=True)
        self.assertTrue(r['converged']);self.assertGreater(r['rejected_neural'],0);self.assertEqual(r['accepted_neural'],0)
        self.assertLessEqual(r['max_classical_observed_ratio'],r['classical_contraction_bound']+1e-6)
        self.assertTrue(all(not e['accepted'] for e in r['neural_events']))
    def test_nonfinite_proposal_falls_back_only_when_safeguarded(self):
        bad=lambda v:np.full_like(v,np.nan)
        _,r=solve(self.a,self.b,bad,maxiter=1000);self.assertTrue(r['converged']);self.assertGreater(r['rejected_neural'],0)
        _,r=solve(self.a,self.b,bad,mode='fixed',maxiter=10);self.assertFalse(r['converged']);self.assertEqual(r['failure'],'nonfinite neural proposal')
    def test_zero_rhs_and_budget_failure(self):
        y,r=solve(self.a,np.zeros(12),mode='jacobi',maxiter=0);self.assertTrue(r['converged']);self.assertEqual(r['iterations'],0)
        _,r=solve(self.a,self.b,mode='jacobi',maxiter=0);self.assertFalse(r['converged'])
        for name in ['rtol','atol']:
            for value in [np.nan,np.inf,-np.inf,-1.]:
                with self.subTest(control=name,value=value):
                    with self.assertRaises(ValueError):solve(self.a,self.b,mode='jacobi',**{name:value})
    def test_residual_load_conversion_not_raw_nodal_values(self):
        m=sparse.diags([.1*np.ones(11),np.linspace(1,2,12),.1*np.ones(11)],[-1,0,1],format='csr');phi=np.random.default_rng(7).normal(size=(12,4));r=np.arange(12)+1.
        f=spsolve(m,r);np.testing.assert_allclose(residual_coordinates(phi,r),phi.T@(m@f),atol=1e-13)
        self.assertGreater(np.linalg.norm(residual_coordinates(phi,r)-phi.T@(m@r)),1.)
    def test_classical_bound_matches_scaled_spectrum(self):
        rho,low,high=contraction_bound(self.a,2/3);d=self.a.diagonal();a=self.a.toarray()/np.sqrt(d[:,None]*d[None,:]);ev=np.linalg.eigvalsh(a)
        self.assertGreaterEqual(ev.min(),low-1e-12);self.assertLessEqual(ev.max(),high+1e-12);self.assertLess(max(abs(1-2/3*ev)),rho)
if __name__=='__main__':unittest.main()
