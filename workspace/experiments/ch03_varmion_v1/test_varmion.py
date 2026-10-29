import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from model import VariationalMap,UnstructuredMap,structure_checks,fit_uncentered_pod
from data import CFField,parameters
from train_coeff import fit
from neuralpde_opt.chapter3.reference import save_geometry
from neuralpde_opt.chapter3.operator_training import P1Geometry

class VarMiONTests(unittest.TestCase):
    def test_exact_source_linearity_and_unstructured_control(self):
        torch.manual_seed(1);c=torch.rand(5,65,65,dtype=torch.float64)+.5;z1=torch.randn(5,4,dtype=torch.float64);z2=torch.randn_like(z1)
        model=VariationalMap(1,.2,np.eye(4),width=8).double();r=structure_checks(model,c,z1,z2)
        self.assertEqual(r['zero_load_max'],0);self.assertLess(r['relative_superposition_max'],1e-13);self.assertLess(r['relative_homogeneity_max'],1e-13)
        other=UnstructuredMap(1,.2,np.eye(4),width=8).double();r=structure_checks(other,c,z1,z2);self.assertGreater(r['zero_load_max'],1e-8)
    def test_uncentered_mass_basis_and_zero_source_reconstruction(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'g.npz';save_geometry(path,4);g=P1Geometry(path)
            rng=np.random.default_rng(5);y=rng.normal(size=(12,5,5));y.reshape(12,-1)[:,g.boundary]=0
            phi,ev=fit_uncentered_pod(y,g.mass,8)
            np.testing.assert_allclose(phi.T@(g.mass@phi),np.eye(8),atol=1e-12)
            np.testing.assert_array_equal(phi[g.boundary],0);np.testing.assert_array_equal(phi@np.zeros(8),0)
            dense=np.linalg.svd(y.reshape(12,-1)@np.linalg.cholesky(g.mass.toarray()),compute_uv=False)
            np.testing.assert_allclose(ev[:8],dense[:8]**2/12,atol=1e-13)
    def test_source_law_is_linear_and_split_streams_are_disjoint(self):
        config={'seed':123,'source_modes':4,'source_length_scale':.25,'source_exponent':2.}
        base={'seed':1,'reaction':{'modes':4,'length_scale':.2,'exponent':2.,'include_constant':True,'lower':.5,'upper':1.5,'latent_scale':1.}}
        cc,f1=parameters(config,base,'train',1,0);same,f2=parameters(config,base,'train',1,1);cv,_=parameters(config,base,'val',1,0)
        np.testing.assert_array_equal(cc,same);self.assertFalse(np.array_equal(cc,cv));p=np.random.default_rng(2).random((13,2))
        a=CFField(cc,f1,base['reaction']);b=CFField(cc,f2,base['reaction']);mixed=CFField(cc,.7*f1-1.3*f2,base['reaction'])
        np.testing.assert_allclose(mixed.source(p),.7*a.source(p)-1.3*b.source(p),atol=1e-14)
        with self.assertRaises(ValueError):parameters(config,base,'test',0,0)
    def assert_nested(self,a,b):
        if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.assert_nested(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_nested(x,y)
        else:self.assertEqual(a,b)
    def resume(self,device):
        rng=np.random.default_rng(5);arrays={}
        for split,n in [('train',12),('val',5)]:
            z=rng.normal(size=(n,3));alpha=z*.8
            arrays[split]={'c':rng.random((n,65,65))+.5,'z':z,'alpha':alpha,'denominator':(alpha*alpha).sum(1)+.01,'projection_squared':np.full(n,.01)}
        config={'epochs':4,'batch_size':5,'learning_rate':.001,'weight_decay':1e-5,'min_lr':1e-6,'checkpoint_every':1,'grad_clip':1.}
        factory=lambda:VariationalMap(1.,.2,np.eye(3),width=8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=fit(root/'a',config,2,factory,arrays,{'synthetic':True},device)
            fit(root/'b',config,2,factory,arrays,{'synthetic':True},device,stop_after=2)
            b=fit(root/'b',config,2,factory,arrays,{'synthetic':True},device,resume=True)
            for key in a:
                if key!='elapsed_seconds':self.assert_nested(a[key],b[key])
            self.assertEqual(a['steps'],12);self.assertEqual(a['visits'],48)
    def test_resume_cpu(self):self.resume('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_resume_cuda(self):self.resume('cuda')
if __name__=='__main__':unittest.main()
