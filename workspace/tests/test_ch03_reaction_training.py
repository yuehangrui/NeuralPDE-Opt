import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from neuralpde_opt.chapter3 import reaction_training as r
from neuralpde_opt.chapter3.operator_training import checkpoint_digest

ROOT=Path(__file__).resolve().parents[1]
def config():return json.loads((ROOT/'configs/ch03_reaction_training_v1.json').read_text())

class ReactionTrainingTests(unittest.TestCase):
    def test_protocol_rejects_test_old_schema_and_formal_subsets(self):
        r.validate_config(config())
        for k,v in [('test_evaluations',1),('schema_version','ch03-fields-v1'),('train_limit',8),('microbatch_size',32)]:
            c=config();c[k]=v
            with self.assertRaises(ValueError):r.validate_config(c)

    def test_public_array_guard_blocks_test_and_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);np.savez(d/'train.npz',c=[1]);np.savez(d/'val.npz',c=[2])
            for name in ['test.npz','train_reference.npz']:np.savez(d/name,y=[100])
            with r.public_data_only(d) as opened:
                with np.load(d/'train.npz',allow_pickle=False) as z:self.assertEqual(z['c'][0],1)
                for name in ['test.npz','train_reference.npz']:
                    with self.assertRaises(ValueError):np.load(d/name,allow_pickle=False)
            self.assertEqual(len(opened),1)

    def test_uneven_gradient_accumulation_matches_full_batch(self):
        torch.manual_seed(5)
        c=torch.randn(7,3,3,dtype=torch.float64);y=torch.randn_like(c)
        class Affine(torch.nn.Module):
            def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(.7,dtype=torch.float64))
            def forward(self,c):return self.w*c
        m=Affine();n=copy.deepcopy(m)
        a=torch.optim.SGD(m.parameters(),lr=.02);b=torch.optim.SGD(n.parameters(),lr=.02)
        norm=lambda z:z.flatten(1).square().sum(1)
        v=r.batch_update(m,c,y,np.arange(7),7,a,norm,100)
        w=r.batch_update(n,c,y,np.arange(7),3,b,norm,100)
        self.assertAlmostEqual(v,w,places=14)
        torch.testing.assert_close(m.w,n.w,rtol=0,atol=1e-15)

    def test_plateau_and_cosine_resume_learning_rate(self):
        for spec in config()['models'][:2]:
            p=spec['training'];m=torch.nn.Linear(1,1)
            a=torch.optim.AdamW(m.parameters(),lr=p['learning_rate'])
            s=r.scheduler_for(a,p,10)
            for _ in range(3):
                a.zero_grad();m(torch.ones(1,1)).sum().backward();a.step()
                s.step(.5) if p['scheduler']=='plateau' else s.step()
            b=torch.optim.AdamW(m.parameters(),lr=p['learning_rate']);t=r.scheduler_for(b,p,10)
            b.load_state_dict(a.state_dict());t.load_state_dict(s.state_dict())
            a.step();b.step()
            s.step(.4) if p['scheduler']=='plateau' else s.step()
            t.step(.4) if p['scheduler']=='plateau' else t.step()
            self.assertEqual(a.param_groups[0]['lr'],b.param_groups[0]['lr'])

if __name__=='__main__':unittest.main()
