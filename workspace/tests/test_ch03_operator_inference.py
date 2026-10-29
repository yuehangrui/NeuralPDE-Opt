"""End-to-end raw-field inference and held-out resolution contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from neuralpde_opt.chapter3.operator_inference import read_inputs, read_queries, run_inference
from neuralpde_opt.chapter3.operator_training import run, sha256
from neuralpde_opt.chapter3.reference import save_geometry
from ch03_fixture_certificate import certify_fixture


class OperatorInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data = cls.root/'data'
        cls.data.mkdir()
        for name, n in (('observation', 4), ('reference', 8)):
            save_geometry(cls.data/(name+'_geometry.npz'), n)
        cls.fine_points = np.stack(np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 9),
                                             indexing='ij'), -1)
        for split in ('train', 'val', 'test'):
            fields = {}
            for suffix, n in (('', 5), ('_reference', 9)):
                x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing='ij')
                fields.update({'a'+suffix: np.stack([1+.1*x, 1.1+.1*x]),
                               'f'+suffix: np.stack([1+y, 2+y]),
                               'y'+suffix: np.stack([x*(1-x)*y*(1-y), 2*x*(1-x)*y*(1-y)])})
            np.savez(cls.data/(split+'.npz'), **fields,
                     sample_id=np.array([split+'0', split+'1']))
        certify_fixture(cls.data, {'train':2, 'val':2, 'test':2})
        config = {'purpose': 'chapter3_smoke_only_no_method_ranking',
                  'seed': 0, 'steps': 1, 'batch_size': 2, 'eval_every': 1,
                  'learning_rate': .001, 'cpu_threads': 1,
                  'models': [
                      {'name': 'fno', 'kwargs': {'modes': 2, 'width': 4, 'layers': 1}},
                      {'name': 'deeponet', 'kwargs': {'grid_size': 5, 'sensor_stride': 2,
                       'latent_dim': 4, 'branch_width': 4, 'trunk_width': 4, 'depth': 1}},
                      {'name': 'gno', 'kwargs': {'width': 4, 'layers': 1, 'radius': .6, 'node_stride': 2}},
                      {'name': 'transformer', 'kwargs': {'grid_size': 5, 'sensor_stride': 2,
                       'width': 4, 'layers': 1, 'heads': 1}}]}
        cls.config_path = cls.root/'config.json'
        cls.config_path.write_text(json.dumps(config))
        cls.runs = cls.root/'runs'
        run(cls.data, cls.config_path, cls.runs)
        with np.load(cls.data/'test.npz') as z:
            cls.input_path = cls.root/'unlabelled.npz'
            np.savez(cls.input_path, a=z['a'], f=z['f'], sample_id=z['sample_id'])
        cls.shared = cls.root/'shared.npz'
        np.savez(cls.shared, query_points=cls.fine_points)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_four_models_predict_raw_fields_without_labels(self):
        for name in ('fno', 'deeponet', 'gno', 'transformer'):
            with self.subTest(model=name):
                output = self.root/('pred-'+name)
                prediction = run_inference(self.runs/name, self.input_path, output, batch_size=2)
                with np.load(self.runs/name/'test_predictions.npz') as z:
                    np.testing.assert_allclose(prediction, z['y'], rtol=2e-6, atol=1e-9)
                self.assertTrue((output/'COMPLETE').exists())
                metadata = json.loads((output/'metadata.json').read_text())
                self.assertEqual(metadata['sample_count'], 2)
                for file, digest in json.loads((output/'manifest.json').read_text()).items():
                    self.assertEqual(sha256(output/file), digest)
                with self.assertRaises(FileExistsError):
                    run_inference(self.runs/name, self.input_path, output)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_reload_matches_training_after_default_tf32_reset(self):
        config = json.loads(self.config_path.read_text())
        config['models'] = config['models'][:1]
        path = self.root/'cuda-config.json'
        path.write_text(json.dumps(config))
        output = self.root/'cuda-run'
        run(self.data, path, output, device='cuda')
        # A fresh PyTorch process defaults to allowing TF32 in cuDNN.
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        predicted = run_inference(output/'fno', self.input_path,
                                  self.root/'cuda-predictions', device='cuda', batch_size=2)
        with np.load(output/'fno/test_predictions.npz') as z:
            np.testing.assert_array_equal(predicted, z['y'])
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

    def test_query_chunking_and_shared_per_sample_layouts(self):
        per_sample = self.root/'batched-points.npz'
        # An asymmetric interior query and boundary coordinates per sample.
        points = np.array([[[.21,.42], [0.,.7]], [[.63,.14], [1.,.3]]])
        np.savez(per_sample, query_points=points)
        for name in ('deeponet', 'transformer'):
            a = run_inference(self.runs/name, self.input_path, self.root/(name+'-fine'),
                              queries=self.shared, query_chunk_size=7, batch_size=2)
            self.assertEqual(a.shape, (2, 9, 9))
            b = run_inference(self.runs/name, self.input_path, self.root/(name+'-points'),
                              queries=per_sample, query_layout='per_sample', query_chunk_size=1)
            self.assertEqual(b.shape, (2, 2))
            self.assertTrue(np.array_equal(b[:, 1], np.zeros(2)))
            c = run_inference(self.runs/name, self.input_path, self.root/(name+'-points-wide'),
                              queries=per_sample, query_layout='per_sample', query_chunk_size=8)
            np.testing.assert_allclose(b, c, rtol=3e-5, atol=1e-9)
        for name in ('fno', 'gno'):
            with self.assertRaisesRegex(ValueError, 'separate coordinate queries'):
                run_inference(self.runs/name, self.input_path, self.root/(name+'-bad-query'),
                              queries=self.shared)

    def test_all_four_resolution_diagnostic_and_provenance(self):
        path = Path(__file__).resolve().parents[1]/'scripts/evaluate_ch03_operator_transfer.py'
        spec = importlib.util.spec_from_file_location('transfer_diagnostic', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = self.root/'transfer'
        report = module.evaluate_transfer(self.data, self.runs, output)
        self.assertEqual(set(report['models']), {'fno', 'deeponet', 'gno', 'transformer'})
        self.assertTrue((output/'COMPLETE').exists())
        for name in report['models']:
            self.assertTrue(np.isfinite(report['models'][name]['direct_fine']['mean']['relative_l2']))
        with self.assertRaises(FileExistsError):
            module.evaluate_transfer(self.data, self.runs, output)
        path = self.runs/'fno/provenance.json'
        original = path.read_bytes()
        try:
            path.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError, 'Run checksum mismatch: fno/provenance.json'):
                module.evaluate_transfer(self.data, self.runs, self.root/'bad-provenance')
        finally:
            path.write_bytes(original)

    def test_invalid_inputs_and_coordinate_layouts(self):
        invalid = self.root/'invalid.npz'
        for a in (np.array(1.), np.zeros((0,5,5)), np.ones((2,5,5), dtype=int),
                  -np.ones((2,5,5)), np.full((2,5,5), np.nan)):
            np.savez(invalid, a=a, f=a)
            with self.assertRaises(ValueError):
                read_inputs(invalid)
        np.savez(invalid, points=np.array([[1.1, .2]]))
        with self.assertRaises(ValueError):
            read_queries(invalid, 'shared', 2)
        with self.assertRaises(ValueError):
            read_queries(self.shared, 'per_sample', 2)
        with self.assertRaises(ValueError):
            run_inference(self.runs/'fno', self.input_path, self.root/'bad-batch', batch_size=True)


if __name__ == '__main__':
    unittest.main()
