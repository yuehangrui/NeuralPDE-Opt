import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from bootstrap import ROOT
from model_v2 import MatrixMap, operator_loss
from trainer_v2 import fit
from model import VariationalMap


class V2Tests(unittest.TestCase):
    def test_dense_control_matches_existing_architecture(self):
        rng=np.random.default_rng(1);q=rng.normal(size=(4,4));k0=q@q.T+np.eye(4)
        c=torch.as_tensor(rng.uniform(.5,1.5,size=(3,65,65)),dtype=torch.float32)
        z=torch.as_tensor(rng.normal(size=(3,4)),dtype=torch.float32)
        torch.manual_seed(7);old=VariationalMap(1.,.2,k0).float()
        torch.manual_seed(7);new=MatrixMap('dense',1.,.2,k0).float()
        self.assertEqual(sum(p.numel() for p in old.parameters()),sum(p.numel() for p in new.parameters()))
        torch.testing.assert_close(old(c,z),new(c[:,::4,::4].flatten(1),z),rtol=1e-6,atol=1e-6)

    def test_spd_initialization_and_linearity_after_nontrivial_change(self):
        rng=np.random.default_rng(2);q=rng.normal(size=(5,5));k0=q@q.T+np.eye(5)
        model=MatrixMap('spd',1.,.2,k0,width=8).double()
        x=torch.as_tensor(rng.uniform(.5,1.5,size=(6,289)))
        np.testing.assert_allclose(model.matrix(x).detach().numpy(),np.broadcast_to(k0,(6,5,5)),atol=1e-13)
        with torch.no_grad():
            model.branch[-1].weight.normal_(0,.3);model.branch[-1].bias.normal_(0,.3)
        k=model.matrix(x)
        self.assertGreater(float(torch.linalg.eigvalsh(k).min()),0)
        z=torch.as_tensor(rng.normal(size=(6,5)));w=torch.as_tensor(rng.normal(size=(6,5)))
        torch.testing.assert_close(model(x,.7*z-1.3*w),.7*model(x,z)-1.3*model(x,w),rtol=1e-13,atol=1e-13)
        self.assertEqual(float(model(x,torch.zeros_like(z)).abs().max()),0)

    def test_operator_constraint_has_exact_inverse_zero_and_detects_unseen_direction(self):
        rng=np.random.default_rng(3);q=rng.normal(size=(5,5));a=q@q.T+np.eye(5)
        q0=rng.normal(size=(5,5));k0=q0@q0.T+np.eye(5);l=np.linalg.cholesky(k0);li=np.linalg.inv(l)
        b=torch.as_tensor((l.T@a@l)[None]);k=torch.as_tensor(np.linalg.inv(a)[None])
        self.assertLess(float(operator_loss(k,b,torch.as_tensor(li))),1e-25)
        wrong=k.clone();wrong[0,4,4]+=.1
        probe=torch.zeros(5,dtype=torch.float64);probe[0]=1
        torch.testing.assert_close(wrong@probe,k@probe)
        self.assertGreater(float(operator_loss(wrong,b,torch.as_tensor(li))),1e-4)

    def test_operator_gradient_can_fit_known_spd_action(self):
        model=MatrixMap('spd',1.,.2,np.eye(3),width=8).double()
        x=torch.ones(4,289,dtype=torch.float64)
        b=torch.diag(torch.tensor([.8,1.1,1.3],dtype=torch.float64)).expand(4,-1,-1)
        opt=torch.optim.Adam(model.parameters(),lr=.03)
        initial=float(operator_loss(model.matrix(x),b,model.inverse_l0))
        for _ in range(30):
            opt.zero_grad();loss=operator_loss(model.matrix(x),b,model.inverse_l0);loss.backward();opt.step()
        self.assertLess(float(operator_loss(model.matrix(x),b,model.inverse_l0)),initial*.1)

    def assert_nested(self,a,b):
        if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for key in a:self.assert_nested(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_nested(x,y)
        else:self.assertEqual(a,b)

    def resume_check(self,device):
        rng=np.random.default_rng(5);arrays={}
        for split,n in [('train',12),('val',5)]:
            z=rng.normal(size=(n,3));alpha=.8*z
            arrays[split]=dict(sensors=rng.uniform(.5,1.5,size=(n,289)),z=z,alpha=alpha,
                denominator=(alpha*alpha).sum(1)+.01,projection_squared=np.full(n,.01))
        pool=dict(sensors=rng.uniform(.5,1.5,size=(7,289)),B=np.broadcast_to(1.25*np.eye(3),(7,3,3)).copy())
        cfg=dict(epochs=4,batch_size=5,operator_batch_size=4,learning_rate=.001,weight_decay=1e-5,
                 min_lr=1e-6,grad_clip=1.,checkpoint_every=1,operator_weight=1.)
        factory=lambda:MatrixMap('spd',1.,.2,np.eye(3),width=8)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            a=fit(root/'a',cfg,2,factory,arrays,pool,{'synthetic':True},device=device)
            fit(root/'b',cfg,2,factory,arrays,pool,{'synthetic':True},device=device,stop_after=2)
            b=fit(root/'b',cfg,2,factory,arrays,pool,{'synthetic':True},device=device,resume=True)
            for key in a:
                if key!='elapsed_seconds':self.assert_nested(a[key],b[key])
            self.assertEqual(a['steps'],12);self.assertEqual(a['visits'],48);self.assertEqual(a['operator_visits'],48)
            changed=copy.deepcopy(cfg);changed['operator_weight']=2.
            with self.assertRaises(ValueError):
                fit(root/'b',changed,2,factory,arrays,pool,{'synthetic':True},device=device,resume=True)

    def test_exact_resume_cpu(self):self.resume_check('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_exact_resume_cuda(self):self.resume_check('cuda')

if __name__=='__main__':unittest.main()
