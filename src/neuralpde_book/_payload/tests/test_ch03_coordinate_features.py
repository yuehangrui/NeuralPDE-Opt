"""Fourier coordinates preserve query independence, gradients and old checkpoint keys."""
import json
from pathlib import Path
import unittest
import torch
from neuralpde_opt.chapter3.deeponet import DeepONet
from neuralpde_opt.chapter3.transformer import TransformerOperator
from neuralpde_opt.chapter3.experimental_data import validate_config


class CoordinateFeatureTests(unittest.TestCase):
    def test_fourier_queries_chunking_boundaries_and_gradients(self):
        torch.set_num_threads(2)
        for factory, kw in [(DeepONet, dict(branch_width=12,trunk_width=12,latent_dim=8)),
                            (TransformerOperator, dict(width=8,heads=2,layers=1))]:
            with self.subTest(model=factory.__name__):
                torch.manual_seed(4)
                model=factory(grid_size=9,sensor_stride=2,coordinate_modes=4,**kw).double()
                a=torch.rand(2,9,9,dtype=torch.float64,requires_grad=True)
                f=torch.rand_like(a,requires_grad=True)
                q=torch.tensor([[0.,.3],[.2,.4],[.7,.6],[1.,.2]],dtype=torch.float64,requires_grad=True)
                direct=model(a,f,q,query_chunk_size=None)
                chunked=model(a,f,q,query_chunk_size=1)
                torch.testing.assert_close(direct,chunked,atol=1e-12,rtol=1e-12)
                self.assertEqual(direct[:,[0,3]].abs().max().item(),0)
                grads=torch.autograd.grad(direct.sum(),(a,f,q),retain_graph=True)
                chunk_grads=torch.autograd.grad(chunked.sum(),(a,f,q))
                for g,h in zip(grads,chunk_grads):
                    self.assertTrue(torch.isfinite(g).all())
                    torch.testing.assert_close(g,h,atol=1e-11,rtol=1e-11)
                self.assertGreater(grads[1].abs().max().item(),0)

    def test_defaults_keep_checkpoint_keys_and_predictions(self):
        for factory in (DeepONet,TransformerOperator):
            torch.manual_seed(3)
            original=factory(grid_size=9,sensor_stride=2)
            torch.manual_seed(3)
            explicit=factory(grid_size=9,sensor_stride=2,coordinate_modes=0)
            explicit.load_state_dict(original.state_dict(),strict=True)
            a=torch.ones(1,9,9);f=torch.randn_like(a)
            self.assertTrue(torch.equal(original(a,f),explicit(a,f)))
            for bad in (-1,True,1.5):
                with self.assertRaises(ValueError):factory(coordinate_modes=bad)

    def test_full_data_scope_does_not_expand_legacy_limit(self):
        root=Path(__file__).parents[1]
        full=json.loads((root/'configs/ch03_full_data_v2.json').read_text())
        validate_config(full)
        full['splits']['train']=1025
        with self.assertRaises(ValueError):validate_config(full)
        full['splits']['train']=1024
        full['purpose']='chapter3_supervised_experiments_v1'
        with self.assertRaises(ValueError):validate_config(full)
