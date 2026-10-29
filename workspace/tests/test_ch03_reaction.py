import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from scipy import sparse
import torch
from neuralpde_opt.chapter3.reaction import *
from neuralpde_opt.chapter3.reference import analytic_errors,save_geometry
from neuralpde_opt.chapter3.reaction_data import read_split,verify_dataset,sha,dump
from neuralpde_opt.chapter3.operator_training import P1Geometry
from neuralpde_opt.chapter3.reaction_operators import ReactionOperator,fit_reaction_normalization

ROOT=Path(__file__).resolve().parents[1]
def config():return json.loads((ROOT/'configs/ch03_reaction_v1.json').read_text())

class Constant:
    def __init__(self,c=1,f=1):self.c,self.f=c,f
    def reaction(self,p):return np.full(p.shape[:-1],self.c)
    def source(self,p):return np.full(p.shape[:-1],self.f)

class ReactionTests(unittest.TestCase):
    def test_constant_reaction_is_mass_not_stiffness(self):
        fem=ReactionFEM(5,.01);matrix,load,_=fem.assemble(Constant(2))
        mesh=fem.mesh
        template=np.array([[2,1,1],[1,2,1],[1,1,2]])
        local=mesh.determinants[:,None,None]/24*template
        rows=np.broadcast_to(mesh.elements[:,:,None],local.shape)
        cols=np.broadcast_to(mesh.elements[:,None,:],local.shape)
        mass=sparse.coo_matrix((local.ravel(),(rows.ravel(),cols.ravel())),shape=matrix.shape).tocsr()
        np.testing.assert_allclose(matrix.toarray(),(.01*fem.stiffness+2*mass).toarray(),rtol=1e-13,atol=1e-15)
        self.assertAlmostEqual(load.sum(),1)

    def test_variable_matrix_spd(self):
        fem=ReactionFEM(6);a,_,_=fem.assemble(ReactionManufactured())
        reduced=a[fem.mesh.interior][:,fem.mesh.interior].toarray()
        np.testing.assert_allclose(reduced,reduced.T,atol=1e-15)
        self.assertGreater(np.linalg.eigvalsh(reduced).min(),0)

    def test_manufactured_convergence(self):
        f=ReactionManufactured()
        e=[analytic_errors(ReactionFEM(n).solve(f),f.exact,f.gradient) for n in [8,16,32]]
        self.assertGreater(np.log2(e[1]['relative_l2']/e[2]['relative_l2']),1.8)
        self.assertGreater(np.log2(e[1]['relative_h1_seminorm']/e[2]['relative_h1_seminorm']),.9)

    def test_residual_boundary_and_source_linearity(self):
        fem=ReactionFEM(12);s=fem.solve(Constant());t=fem.solve(Constant(f=3))
        np.testing.assert_allclose(t.values,3*s.values,rtol=1e-13,atol=1e-14)
        self.assertEqual(s.diagnostics['boundary_max'],0)
        self.assertLess(fem.independent_weak_residual(Constant(),s.values),1e-12)

    def test_reaction_increase_reduces_solution(self):
        fem=ReactionFEM(24)
        self.assertTrue(np.all(fem.solve(Constant(2)).values<=fem.solve(Constant(1)).values+1e-13))

    def test_invalid_coefficients_rejected(self):
        for e in [0,-1,np.inf,np.nan]:
            with self.assertRaises(ValueError):ReactionFEM(4,e)
        fem=ReactionFEM(4)
        for c in [0,-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):fem.solve(Constant(c))

    def test_config_validation(self):
        validate_config(config())
        for key,value in [('schema_version','ch03-fields-v1'),('source',2),('epsilon',0),('reference_n',129)]:
            c=config();c[key]=value
            with self.assertRaises(ValueError):validate_config(c)

    def test_streams_disjoint_append_stable_and_bounds(self):
        c=config();a=draw_reaction(c,'train',0);c['splits']['train']=256
        np.testing.assert_array_equal(a,draw_reaction(c,'train',0))
        self.assertFalse(np.array_equal(a,draw_reaction(c,'val',0)))
        points=make_mesh(2,16).nodes;f=ReactionField(a,c['reaction']);v=f.reaction(points)
        self.assertGreaterEqual(v.min(),.5);self.assertLessEqual(v.max(),1.5)
        np.testing.assert_array_equal(f.reaction(points[::2]),v[::2])
        with self.assertRaises(ValueError):draw_reaction(c,'test',0)

    def test_old_schema_and_test_loader_rejected(self):
        with self.assertRaises(ValueError):read_split('/does/not/exist','test')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);dump(p/'config.json',config());np.savez(p/'train.npz',a=[1],f=[1],y=[1],sample_id=['x'])
            with self.assertRaises(ValueError):read_split(p,'train')

    def test_normalization_train_only_and_constant_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_geometry(Path(tmp)/'g.npz',4);g=P1Geometry(Path(tmp)/'g.npz')
            c=np.stack([np.ones((5,5)),np.ones((5,5))*2]);y=c*.2
            data={'c':c,'y':y,'sample_id':np.array(['reaction-v1:train:0000','reaction-v1:train:0001'])}
            norm=fit_reaction_normalization(data,g)
            self.assertAlmostEqual(norm['c_mean'],1.5);self.assertAlmostEqual(norm['c_std'],.5)
            class Spy(torch.nn.Module):
                def forward(self,c,f):
                    assert torch.count_nonzero(f)==0
                    return c+f
            m=ReactionOperator(Spy(),norm);x=torch.tensor(c,dtype=torch.float32,requires_grad=True)
            m(x).sum().backward();self.assertTrue(torch.isfinite(x.grad).all())
            data['sample_id']=np.array(['reaction-v1:val:0000','reaction-v1:val:0001'])
            with self.assertRaises(ValueError):fit_reaction_normalization(data,g)
            with self.assertRaises(ValueError):ReactionOperator(Spy(),{'a_mean':1})

if __name__=='__main__':unittest.main()
