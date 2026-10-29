"""Independent numerical identities and resume/selection integration checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from neuralpde_opt.chapter3.reference import save_geometry
from neuralpde_opt.chapter3.operator_training import P1Geometry
from pod import fit_pod, coordinates, reconstruct, encode_inputs, prepare_arrays, coefficient_loss
from train import train_one, finish_one, validate_completed
from run import validate_config


class PODTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        save_geometry(self.root/'geometry.npz', 4)
        self.g = P1Geometry(self.root/'geometry.npz')
        rng = np.random.default_rng(817)
        self.data = {}
        for split, count in [('train',12),('val',5)]:
            y = rng.normal(size=(count,5,5)) + 2
            y.reshape(count,-1)[:,self.g.boundary] = 0
            self.data[split] = {'y':y, 'c':1+.2*np.tanh(rng.normal(size=(count,5,5))),
                                'sample_id':np.asarray([f'reaction-v1:{split}:{i}' for i in range(count)])}
        self.b = fit_pod(self.data['train'],self.g,4)
        self.norm = {'c_mean':1., 'c_std':.2}
        self.config = json.loads((Path(__file__).parent/'config.json').read_text())
        self.config.update(epochs=4,batch_size=5,checkpoint_every=1,width=8,depth=2,sensor_stride=2)

    def tearDown(self): self.temp.cleanup()

    def test_mass_basis_and_independent_dense_svd(self):
        x = self.data['train']['y'].reshape(12,-1)
        np.testing.assert_array_equal(self.b['mean'],x.mean(0))
        m = self.g.mass.toarray()
        s = np.linalg.svd((x-x.mean(0))@np.linalg.cholesky(m),compute_uv=False)
        np.testing.assert_allclose(self.b['eigenvalues'][:9],s[:9]**2/12,rtol=1e-12,atol=1e-15)
        phi = self.b['modes']
        np.testing.assert_allclose(phi.T@m@phi,np.eye(4),atol=1e-12)
        np.testing.assert_array_equal(phi[self.g.boundary],0)
        self.assertTrue(all(phi[np.argmax(abs(phi[:,j])),j] > 0 for j in range(4)))

    def test_projection_monotonicity_and_pythagoras(self):
        y = self.data['val']['y']
        previous = np.full(len(y),np.inf)
        for r in range(1,5):
            alpha = coordinates(y,self.b,self.g,r)
            projected = reconstruct(alpha,self.b,r)
            e = self.g.l2_squared(projected-y)
            self.assertTrue(np.all(e<=previous+1e-14));previous=e
            shifted = alpha + .123
            total = self.g.l2_squared(reconstruct(shifted,self.b,r)-y)
            np.testing.assert_allclose(total,e+np.sum((shifted-alpha)**2,axis=1),atol=1e-12)
            residual = (y-projected).reshape(len(y),-1)
            np.testing.assert_allclose(residual@(self.g.mass@self.b['modes'][:,:r]),0,atol=1e-12)

    def test_fit_rejects_validation_and_degenerate_snapshots(self):
        with self.assertRaises(ValueError): fit_pod(self.data['val'],self.g,2)
        d = copy.deepcopy(self.data['train']);d['sample_id'][1] = d['sample_id'][0]
        with self.assertRaises(ValueError): fit_pod(d,self.g,2)
        d = copy.deepcopy(self.data['train']);d['y'][:]=0
        with self.assertRaises(ValueError): fit_pod(d,self.g,2)
        d = copy.deepcopy(self.data['train']);d['y'][0,0,0]=1
        with self.assertRaises(ValueError): fit_pod(d,self.g,2)

    def test_training_fit_is_independent_of_validation(self):
        self.data['val']['y'] *= 100
        again=fit_pod(self.data['train'],self.g,4)
        for name in self.b: np.testing.assert_array_equal(self.b[name],again[name])

    def test_input_sensor_validation(self):
        c=self.data['train']['c']
        np.testing.assert_array_equal(encode_inputs(c,self.norm,2),((c[:,::2,::2]-1)/.2).reshape(12,-1))
        for stride in [0,3,1.5]:
            with self.assertRaises(ValueError): encode_inputs(c,self.norm,stride)
        with self.assertRaises(ValueError): encode_inputs(c,{'c_mean':1.,'c_std':0},2)
        with self.assertRaises(ValueError): encode_inputs(-c,self.norm,2)

    def test_coefficient_gradient_equals_physical_full_field_gradient(self):
        rank=3;part=self.data['val'];alpha=coordinates(part['y'],self.b,self.g,rank)
        z=torch.randn(len(alpha),rank,dtype=torch.float64,requires_grad=True)
        scale=torch.tensor(np.sqrt(self.b['eigenvalues'][:rank]))
        denominator=torch.tensor(self.g.l2_squared(part['y']))
        reduced=coefficient_loss(z,torch.tensor(alpha),scale,denominator)
        g_reduced=torch.autograd.grad(reduced,z,retain_graph=True)[0]
        fields=torch.tensor(self.b['mean'])+(z*scale)@torch.tensor(self.b['modes'][:,:rank].T)
        residual=fields-torch.tensor(part['y'].reshape(len(alpha),-1))
        full=(self.g.torch_loss('cpu',torch.float64)(residual)/denominator).mean()
        g_full=torch.autograd.grad(full,z)[0]
        np.testing.assert_allclose(g_reduced.numpy(),g_full.numpy(),rtol=1e-12,atol=1e-12)
        projection=reconstruct(alpha,self.b,rank)
        constant=np.mean(self.g.l2_squared(projection-part['y'])/denominator.numpy())
        self.assertAlmostEqual(float(full-reduced),constant,places=12)

    def assert_nested_equal(self,a,b):
        if isinstance(a,torch.Tensor): self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a: self.assert_nested_equal(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_nested_equal(x,y)
        else:self.assertEqual(a,b)

    def test_exact_resume_selection_reload_and_tamper_detection(self):
        arrays=prepare_arrays(self.data,self.b,self.g,self.norm,3,2)
        args=(self.config,arrays,np.sqrt(self.b['eigenvalues'][:3]),3,2,{'synthetic':True})
        uninterrupted=train_one(self.root/'continuous',*args)
        train_one(self.root/'resumed',*args,stop_after=2)
        resumed=train_one(self.root/'resumed',*args,resume=True)
        for key in uninterrupted:
            if key!='elapsed_seconds': self.assert_nested_equal(uninterrupted[key],resumed[key])
        result=finish_one(self.root/'resumed',resumed,self.config,arrays,self.b,3,self.data,self.g)
        self.assertEqual(result['optimizer_steps'],12)
        self.assertEqual(result['sample_visits'],48)
        self.assertEqual(validate_completed(self.root/'resumed'),result)
        (self.root/'resumed/history.json').write_text('[]')
        with self.assertRaises(ValueError): validate_completed(self.root/'resumed')

    def test_protocol_rejects_test_and_bad_rank(self):
        validate_config(self.config)
        for key,value in [('test_evaluations',1),('ranks',[4,4]),('learning_rate',float('nan'))]:
            c={**self.config,key:value}
            with self.assertRaises(ValueError):validate_config(c)


if __name__=='__main__':unittest.main()
