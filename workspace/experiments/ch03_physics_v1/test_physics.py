import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from neuralpde_opt.chapter3.reference import save_geometry
from neuralpde_opt.chapter3.operator_training import P1Geometry
from neuralpde_opt.chapter3.reaction import ReactionFEM
from physics import P1System,TorchP1,objective
from run import normalize_inputs,label_mask,train_one,finish_one,verify,validate_config


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.system=P1System(4,.01);save_geometry(self.root/'geom.npz',4);self.geometry=P1Geometry(self.root/'geom.npz')
        rng=np.random.default_rng(999);self.data={}
        for split,count in [('train',12),('val',5)]:
            c=1+.2*rng.random((count,5,5));y=np.stack([self.system.solve(v) for v in c])
            self.data[split]={'c':c,'y':y,'sample_id':np.asarray([f'reaction-v1:{split}:{i:04d}' for i in range(count)])}
        self.kernel=TorchP1(self.system,dtype=torch.float64)
    def tearDown(self):self.tmp.cleanup()

    def test_exact_cubic_mass_against_quadrature(self):
        system=self.system;c=self.data['train']['c'];local=system.local_matrices(c)
        _,weights,phi=system.mesh.quadrature(4)
        cq=np.einsum('nek,qk->neq',c.reshape(12,-1)[:,system.elements],phi)
        independent=system.epsilon*system.local_stiffness[None]+np.einsum('eq,neq,qa,qb->neab',weights,cq,phi,phi)
        np.testing.assert_allclose(local,independent,rtol=1e-13,atol=1e-15)

    def test_independent_continuous_linear_coefficient_assembly(self):
        class Field:
            @staticmethod
            def reaction(p):return 1+.3*p[...,0]+.2*p[...,1]
            @staticmethod
            def source(p):return 1+.1*p[...,0]-.1*p[...,1]
        field=Field();p=self.system.mesh.nodes
        matrix,load=self.system.assemble(field.reaction(p).reshape(5,5),field.source(p))
        other,rhs,_=ReactionFEM(4,.01,4).assemble(field)
        np.testing.assert_allclose(matrix.toarray(),other.toarray(),atol=1e-14)
        np.testing.assert_allclose(load,rhs,atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(matrix[self.system.interior][:,self.system.interior].toarray()).min(),0)

    def check_gradient(self,device):
        torch.use_deterministic_algorithms(True)
        k=TorchP1(self.system,device,dtype=torch.float64);c=self.data['train']['c'][:2]
        rng=np.random.default_rng(42);values=rng.normal(size=(2,5,5))
        values.reshape(2,-1)[:,self.system.boundary]=0
        y=torch.tensor(values,device=device,requires_grad=True);local=k.local(c)
        energy=k.energy(y,local);gradient=torch.autograd.grad(energy.sum(),y)[0].detach().cpu().numpy()
        residual=k.residual(y,local).detach().cpu().numpy()
        for j in range(2):
            a,b=self.system.assemble(c[j]);expected=a@values[j].ravel()-b
            np.testing.assert_allclose(gradient[j].ravel(),expected,rtol=1e-12,atol=1e-13)
            np.testing.assert_allclose(residual[j],expected[self.system.interior],rtol=1e-12,atol=1e-13)
            self.assertAlmostEqual(energy[j].item(),.5*values[j].ravel()@(a@values[j].ravel())-b@values[j].ravel(),places=12)
    def test_energy_gradient_and_residual_cpu(self):self.check_gradient('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_energy_gradient_and_residual_cuda(self):self.check_gradient('cuda')

    def test_direct_solution_stationarity_and_energy_identity(self):
        c=self.data['train']['c'][:2];u=torch.tensor(self.data['train']['y'][:2]);local=self.kernel.local(c)
        self.assertLess(float(self.kernel.relative_residual(u,local).max()),1e-12)
        y=u+torch.tensor(.1*np.random.default_rng(1).normal(size=u.shape));y.flatten(1)[:,self.system.boundary]=0
        difference=self.kernel.energy(y,local)-self.kernel.energy(u,local)
        e=y-u;el=e.flatten(1)[:,self.kernel.elements]
        exact=.5*torch.einsum('nea,neab,neb->n',el,local,el)
        np.testing.assert_allclose(difference.numpy(),exact.numpy(),atol=1e-13)

    def test_zero_labels_do_not_depend_on_training_targets(self):
        c=self.data['train']['c'];pred=torch.tensor(self.data['train']['y']*.8,requires_grad=True)
        local=self.kernel.local(c);mask=torch.zeros(12,dtype=torch.bool)
        first=objective(pred,None,mask,local,self.kernel,1.,0,12)[0]
        second=objective(pred,torch.full_like(pred,float('nan')),mask,local,self.kernel,1.,0,12)[0]
        self.assertTrue(torch.equal(first,second));self.assertTrue(torch.isfinite(second))
        n=normalize_inputs(self.data,self.geometry);changed=copy.deepcopy(self.data);changed['train']['y']*=99
        self.assertEqual(n,normalize_inputs(changed,self.geometry));self.assertEqual(n['y_rms'],1.)

    def test_subset_mask_and_minibatch_unbiased_loss(self):
        mask=label_mask(12,3,8);self.assertEqual(mask.sum(),3)
        self.assertTrue(np.all(mask<=label_mask(12,8,8)))
        y=torch.tensor(self.data['train']['y']);pred=y*.8;local=self.kernel.local(self.data['train']['c']);mask=torch.tensor(mask)
        full=objective(pred,y,mask,local,self.kernel,1.,3,12)[0]
        pieces=[]
        for start in range(0,12,5):
            sl=slice(start,min(start+5,12));count=len(pred[sl])
            pieces.append(objective(pred[sl],y[sl],mask[sl],local[sl],self.kernel,1.,3,12)[0]*count/12)
        self.assertAlmostEqual(full.item(),sum(pieces).item(),places=13)

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
        config=json.loads((Path(__file__).parent/'config.json').read_text())
        config.update(epochs=4,batch_size=5,eval_batch_size=5,checkpoint_every=1,train_eval_every=1,cpu_threads=1)
        config['model']['kwargs'].update(modes=2,width=4,layers=1)
        case={'name':'test_mixed','labels':4,'physics_weight':1.};norm=normalize_inputs(self.data,self.geometry)
        args=(config,case,2,self.data,self.geometry,self.system,norm,{'synthetic':True},device)
        first=train_one(self.root/'continuous',*args)
        train_one(self.root/'resume',*args,stop_after=2);second=train_one(self.root/'resume',*args,resume=True)
        for key in first:
            if key!='elapsed_seconds':self.assert_nested(first[key],second[key])
        result=finish_one(self.root/'resume',second,config,case,self.data,self.geometry,self.system,device)
        self.assertEqual(result['optimizer_steps'],12);self.assertEqual(result['supervised_label_visits'],16);verify(self.root/'resume')
        (self.root/'resume/history.json').write_text('[]')
        with self.assertRaises(ValueError):verify(self.root/'resume')
    def test_exact_resume_selection_and_certificate_cpu(self):self.resume_check('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_exact_resume_selection_and_certificate_cuda(self):self.resume_check('cuda')

    def test_config_rejects_empty_physics_and_test(self):
        config=json.loads((Path(__file__).parent/'config.json').read_text());validate_config(config)
        with self.assertRaises(ValueError):validate_config({**config,'test_evaluations':1})
        config['cases']=[{'name':'empty','labels':0,'physics_weight':0}]
        with self.assertRaises(ValueError):validate_config(config)

if __name__=='__main__':unittest.main()
