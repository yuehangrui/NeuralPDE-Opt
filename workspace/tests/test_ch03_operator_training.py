import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from ch03_fixture_certificate import certify_fixture

from neuralpde_opt.chapter3 import operator_training as training
from neuralpde_opt.chapter3.operator_training import (
    P1Geometry, evaluate_arrays, fit_normalization, load_trained_model, run,
    sha256, verify_data_manifest,
)
from neuralpde_opt.chapter3.reference import save_geometry


class OperatorTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root/'data'
        self.data.mkdir()
        save_geometry(self.data/'observation_geometry.npz', 8)
        geometry = P1Geometry(self.data/'observation_geometry.npz')
        x, y = geometry.coordinates[..., 0], geometry.coordinates[..., 1]
        for split, offset in (('train', 0), ('val', .1), ('test', .2)):
            a = np.stack((1+x+offset, 3+x+offset, 2+2*x+offset))
            f = np.stack((1+y+offset, 1+2*y+offset, 2+y+offset))
            target = np.stack([scale*np.sin(np.pi*x)*np.sin(np.pi*y) for scale in (1., 1.5, .8)])
            target[:, 0, :] = target[:, -1, :] = 0
            target[:, :, 0] = target[:, :, -1] = 0
            np.savez(self.data/f'{split}.npz', a=a, f=f, y=target,
                     sample_id=np.array([split+str(i) for i in range(3)]))
        self.write_manifest()
        self.config = {'purpose': training.SMOKE_PURPOSE,
                       'seed': 0, 'steps': 2, 'batch_size': 2, 'eval_every': 1,
                       'checkpoint_every': 2, 'learning_rate': .001, 'cpu_threads': 1,
                       'models': [
                           {'name': 'fno', 'kwargs': {'modes': 2, 'width': 4, 'layers': 1,
                                                     'hard_boundary': True}},
                           {'name': 'deeponet', 'kwargs': {
                               'grid_size': 9, 'sensor_stride': 2, 'latent_dim': 4,
                               'branch_width': 4, 'trunk_width': 4, 'depth': 1,
                               'hard_boundary': True}},
                           {'name': 'gno', 'kwargs': {
                               'width': 4, 'layers': 1, 'radius': .4, 'node_stride': 2,
                               'hard_boundary': True}},
                           {'name': 'transformer', 'kwargs': {
                               'grid_size': 9, 'width': 4, 'heads': 1, 'layers': 1, 'sensor_stride': 2,
                               'hard_boundary': True}}]}
        self.config_path = self.root/'config.json'
        self.write_config()

    def write_manifest(self):
        certify_fixture(self.data, {'train':3, 'val':3, 'test':3})

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config))

    def assert_nested_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_exact_p1_metrics_and_train_statistics(self):
        path = self.root/'affine_geometry.npz'
        save_geometry(path, 4)
        geometry = P1Geometry(path)
        x, y = geometry.coordinates[..., 0], geometry.coordinates[..., 1]
        target = (x+2*y)[None]
        metrics = evaluate_arrays(2*target, target, geometry, ['affine'])
        self.assertAlmostEqual(metrics['mean']['absolute_l2'], np.sqrt(8/3), places=12)
        self.assertAlmostEqual(metrics['mean']['absolute_h1_seminorm'], np.sqrt(5), places=12)
        self.assertAlmostEqual(metrics['mean']['relative_l2'], 1., places=12)
        self.assertEqual(metrics['mean']['boundary_max'], 6.)
        for dtype, places in ((torch.float32, 5), (torch.float64, 12)):
            value = geometry.torch_loss('cpu', dtype)(torch.tensor(target, dtype=dtype))
            self.assertEqual(value.dtype, dtype)
            self.assertAlmostEqual(value.item(), 8/3, places=places)
        train = {'a': np.stack((1+x, 3+x)), 'f': np.stack((2*y, 2*y)),
                 'y': np.stack((x, x)), 'sample_id': np.array(['train0', 'train1'])}
        stats = fit_normalization(train, geometry)
        self.assertAlmostEqual(stats['a_mean'], 2.5, places=12)
        self.assertAlmostEqual(stats['a_std'], np.sqrt(1+1/12), places=12)
        self.assertAlmostEqual(stats['f_mean'], 1., places=12)
        self.assertAlmostEqual(stats['f_std'], np.sqrt(1/3), places=12)
        self.assertAlmostEqual(stats['y_rms'], np.sqrt(1/3), places=12)
        self.assertFalse(stats['output_centered'])
        self.assertEqual(stats['fit_sample_ids'], ['train0', 'train1'])

    def test_four_models_train_reload_and_safe_completed_skip(self):
        result = run(self.data, self.config_path, self.root/'output')
        for name in training.FACTORIES:
            directory = self.root/'output'/name
            stats = json.loads((directory/'normalization.json').read_text())
            expected = fit_normalization(training.read_split(self.data, 'train'),
                                         P1Geometry(self.data/'observation_geometry.npz'))
            self.assertEqual(stats, expected)
            self.assertEqual(result[name]['test_evaluations'], 1)
            self.assertEqual(result[name]['optimizer_steps'], 2)
            self.assertEqual(result[name]['test']['max']['boundary_max'], 0.)
            history = json.loads((directory/'history.json').read_text())
            selected = min(history, key=lambda row: row['val']['relative_l2'])['step']
            self.assertEqual(result[name]['selected_step'], selected)
            model = load_trained_model(directory)
            with np.load(self.data/'test.npz') as z, torch.no_grad():
                a = torch.tensor(z['a'], dtype=torch.float32)
                f = torch.tensor(z['f'], dtype=torch.float32)
                prediction = model(a, f).numpy()
                queries = torch.tensor([[.25, .375], [.5, .625]], dtype=torch.float32)
                if name in ('deeponet', 'transformer'):
                    queried = model(a, f, query_points=queries, query_chunk_size=1).numpy()
                    np.testing.assert_allclose(queried, prediction[:, [2, 4], [3, 5]],
                                               rtol=1e-5, atol=1e-6)
                else:
                    with self.assertRaisesRegex(ValueError, 'does not support coordinate queries'):
                        model(a, f, query_points=queries)
            with np.load(directory/'test_predictions.npz') as z:
                # Different inference batch sizes may choose different BLAS
                # kernels, with normal floating-point rounding differences.
                np.testing.assert_allclose(prediction, z['y'], rtol=1e-5, atol=1e-6)
            self.assertTrue((directory/'COMPLETE').exists())
        before = sha256(self.root/'output'/'summary.json')
        with mock.patch.object(training, 'run_one', side_effect=AssertionError('completed run retrained')):
            restored = run(self.data, self.config_path, self.root/'output', resume=True)
        self.assertEqual(restored, result)
        self.assertEqual(before, sha256(self.root/'output'/'summary.json'))
        with self.assertRaises(FileExistsError):
            run(self.data, self.config_path, self.root/'output')
        # Inference verifies the selected checkpoint and its interpretation,
        # even when only a single model directory is supplied.
        norm_path = self.root/'output/fno/normalization.json'
        norm_path.write_text(norm_path.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'artifact checksum mismatch: normalization.json'):
            load_trained_model(self.root/'output/fno')

    def test_resume_matches_continuous_all_models_exactly(self):
        # Interrupt between validation steps, then compare final optimizer,
        # weights, selected weights, RNG and deterministic metric histories.
        self.config.update(steps=4, eval_every=3, dtype='float64')
        for spec in self.config['models']:
            with self.subTest(model=spec['name']):
                config = {**self.config, 'models': [spec]}
                self.config_path.write_text(json.dumps(config))
                name = spec['name']
                continuous = self.root/(name+'-continuous')
                resumed = self.root/(name+'-resumed')
                run(self.data, self.config_path, continuous)
                partial = run(self.data, self.config_path, resumed, stop_after=2)
                self.assertEqual(partial['_interrupted'], name)
                self.assertFalse((resumed/'COMPLETE').exists())
                self.assertFalse((resumed/name/'COMPLETE').exists())
                run(self.data, self.config_path, resumed, resume=True)
                a = torch.load(continuous/name/'last.pt', weights_only=True)
                b = torch.load(resumed/name/'last.pt', weights_only=True)
                for key in ('model_state', 'optimizer_state', 'rng_state', 'best_model_state',
                            'best_step', 'best_value', 'optimizer_steps', 'validation_evaluations'):
                    self.assert_nested_equal(a[key], b[key])
                strip_time = lambda state: [{k: v for k, v in row.items() if k != 'elapsed_seconds'}
                                            for row in state['history']]
                self.assertEqual(strip_time(a), strip_time(b))
                for split in ('train', 'val', 'test'):
                    with (np.load(continuous/name/(split+'_predictions.npz')) as p,
                          np.load(resumed/name/(split+'_predictions.npz')) as q):
                        np.testing.assert_array_equal(p['y'], q['y'])
                self.assertEqual(next(load_trained_model(resumed/name).parameters()).dtype, torch.float64)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_resume_matches_continuous_all_models_exactly(self):
        # Exercise GPU optimizer/RNG restoration separately from the CPU
        # float64 contract. Interrupt before the next validation checkpoint.
        self.config.update(steps=3, eval_every=2, dtype='float32')
        for spec in self.config['models']:
            with self.subTest(model=spec['name']):
                config = {**self.config, 'models': [spec]}
                self.config_path.write_text(json.dumps(config))
                name = spec['name']
                continuous = self.root/(name+'-cuda-continuous')
                resumed = self.root/(name+'-cuda-resumed')
                run(self.data, self.config_path, continuous, device='cuda')
                partial = run(self.data, self.config_path, resumed, device='cuda', stop_after=1)
                self.assertEqual(partial['_interrupted'], name)
                self.assertFalse((resumed/'COMPLETE').exists())
                self.assertFalse((resumed/name/'COMPLETE').exists())
                run(self.data, self.config_path, resumed, device='cuda', resume=True)
                a = torch.load(continuous/name/'last.pt', map_location='cpu', weights_only=True)
                b = torch.load(resumed/name/'last.pt', map_location='cpu', weights_only=True)
                self.assertTrue(a['rng_state']['torch_cuda'])
                for key in ('model_state', 'optimizer_state', 'rng_state', 'best_model_state',
                            'best_step', 'best_value', 'optimizer_steps', 'validation_evaluations'):
                    self.assert_nested_equal(a[key], b[key])
                self.assertEqual(a['optimizer_steps'], 3)
                strip_time = lambda state: [{k: v for k, v in row.items() if k != 'elapsed_seconds'}
                                            for row in state['history']]
                self.assertEqual(strip_time(a), strip_time(b))
                for split in ('train', 'val', 'test'):
                    with (np.load(continuous/name/(split+'_predictions.npz')) as p,
                          np.load(resumed/name/(split+'_predictions.npz')) as q):
                        np.testing.assert_array_equal(p['y'], q['y'])

    def test_integrity_resume_mismatch_and_exclusive_lock(self):
        self.config['models'] = self.config['models'][:1]
        self.config['steps'] = 3
        self.write_config()
        output = self.root/'output'
        run(self.data, self.config_path, output, stop_after=1)
        changed = copy.deepcopy(self.config)
        changed['learning_rate'] = .002
        self.config_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'identity mismatch: config'):
            run(self.data, self.config_path, output, resume=True)
        self.write_config()
        original = training.source_fingerprints(self.config['models'])
        modified = {**original, 'operator_training.py': 'changed-source'}
        with mock.patch.object(training, 'source_fingerprints', return_value=modified):
            with self.assertRaisesRegex(ValueError, 'source_sha256'):
                run(self.data, self.config_path, output, resume=True)
        with training.exclusive_run_lock(output):
            with self.assertRaisesRegex(RuntimeError, 'run lock'):
                run(self.data, self.config_path, output, resume=True)
        # Manifest-consistent data replacement must still fail the run identity.
        with np.load(self.data/'val.npz') as z:
            arrays = {key: z[key].copy() for key in z.files}
        arrays['f'][0, 1, 1] += .1
        np.savez(self.data/'val.npz', **arrays)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'data_sha256'):
            run(self.data, self.config_path, output, resume=True)
        # COMPLETE does not bypass a broken input-file hash.
        with (self.data/'test.npz').open('ab') as handle:
            handle.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch: test.npz'):
            run(self.data, self.config_path, self.root/'different-output')
        self.assertFalse((output/'COMPLETE').exists())

    def test_old_best_checkpoint_and_subset_overfit_mode(self):
        # Legacy best.pt has only weights/step and no precision declaration.
        self.config['models'] = self.config['models'][:1]
        spec = self.config['models'][0]
        directory = self.root/'legacy'
        directory.mkdir()
        norm = fit_normalization(training.read_split(self.data, 'train'),
                                 P1Geometry(self.data/'observation_geometry.npz'))
        (directory/'config.json').write_text(json.dumps({'model': spec}))
        (directory/'normalization.json').write_text(json.dumps(norm))
        torch.save({'model_state': training.make_model(spec).state_dict(), 'step': 0}, directory/'best.pt')
        self.assertEqual(next(load_trained_model(directory).parameters()).dtype, torch.float32)
        self.config.update(purpose=training.OVERFIT_PURPOSE, train_subset_ids=['train1'])
        self.write_config()
        result = run(self.data, self.config_path, self.root/'overfit')
        self.assertEqual(result['fno']['training_sample_ids'], ['train1'])
        self.assertIn('lowest train', result['fno']['selection'])
        stats = json.loads((self.root/'overfit/fno/normalization.json').read_text())
        self.assertEqual(stats['fit_sample_ids'], ['train1'])

    def test_failed_export_resumes_without_repeating_committed_test(self):
        self.config['models'] = self.config['models'][:1]
        self.write_config()
        output = self.root/'output'
        original = training.dump_json
        def fail_metrics(path, value):
            if Path(path).name == 'metrics.json':
                raise OSError('simulated interrupted export')
            return original(path, value)
        with mock.patch.object(training, 'dump_json', side_effect=fail_metrics):
            with self.assertRaisesRegex(OSError, 'interrupted export'):
                run(self.data, self.config_path, output)
        self.assertFalse((output/'COMPLETE').exists())
        self.assertFalse((output/'fno'/'COMPLETE').exists())
        with mock.patch.object(training, 'predict', side_effect=AssertionError('repeated final prediction')):
            result = run(self.data, self.config_path, output, resume=True)
        self.assertEqual(result['fno']['test_evaluations'], 1)
        self.assertTrue((output/'COMPLETE').exists())

    def test_resume_rejects_corrupted_checkpoint_content(self):
        self.config['models'] = self.config['models'][:1]
        self.write_config()
        output = self.root/'output'
        run(self.data, self.config_path, output, stop_after=1)
        path = output/'fno'/'last.pt'
        state = torch.load(path, weights_only=True)
        state['optimizer_steps'] += 1
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, 'checkpoint checksum mismatch'):
            run(self.data, self.config_path, output, resume=True)
        self.assertFalse((output/'COMPLETE').exists())

    def test_integer_configuration_rejects_fractional_and_boolean_fields(self):
        for key in ('seed', 'steps', 'batch_size', 'eval_every', 'checkpoint_every', 'cpu_threads'):
            for value in (True, 1.5):
                with self.subTest(field=key, value=value):
                    config = {**self.config, key: value}
                    with self.assertRaisesRegex(ValueError, key+' must be an integer'):
                        training.validate_config(config)


    def test_relative_objective_and_explicit_experiment_limits(self):
        geometry = P1Geometry(self.data/'observation_geometry.npz')
        x, y = geometry.coordinates[..., 0], geometry.coordinates[..., 1]
        target = torch.tensor(np.stack((x+2*y, 100*(x+2*y))), dtype=torch.float64)
        prediction = (2*target).requires_grad_(True)
        norm = geometry.torch_loss('cpu', torch.float64)
        relative = training.objective_per_sample(prediction, target, norm, 1., 'relative_l2_squared')
        torch.testing.assert_close(relative, torch.ones(2, dtype=torch.float64), rtol=1e-14, atol=1e-14)
        relative.mean().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        absolute = training.objective_per_sample(prediction, target, norm, 1., 'absolute_l2_squared')
        self.assertAlmostEqual((absolute[1]/absolute[0]).item(), 10000., places=8)
        with self.assertRaisesRegex(ValueError, 'positive norm'):
            training.objective_per_sample(prediction, torch.zeros_like(target), norm, 1., 'relative_l2_squared')
        config = {**self.config, 'purpose': training.EXPERIMENT_PURPOSE, 'steps': 500,
                  'objective': 'relative_l2_squared', 'defer_test': True}
        training.validate_config(config)
        self.assertEqual(training.split_limits(config), (1024, 256, 256))
        for count in (499, 5001):
            with self.assertRaises(ValueError):
                training.validate_config({**config, 'steps': count})

    def test_objective_and_both_schedulers_resume_exactly(self):
        for schedule in ({'name': 'warmup_cosine', 'warmup_steps': 2, 'min_lr': 1e-5},
                         {'name': 'plateau', 'factor': .5, 'patience': 0, 'threshold': .9}):
            with self.subTest(scheduler=schedule['name']):
                config = {**self.config, 'models': self.config['models'][:1], 'steps': 4,
                          'dtype': 'float64', 'objective': 'relative_l2_squared',
                          'scheduler': schedule, 'defer_test': True}
                self.config_path.write_text(json.dumps(config))
                continuous = self.root/(schedule['name']+'-continuous')
                resumed = self.root/(schedule['name']+'-resumed')
                run(self.data, self.config_path, continuous)
                run(self.data, self.config_path, resumed, stop_after=2)
                run(self.data, self.config_path, resumed, resume=True)
                a = torch.load(continuous/'fno/last.pt', weights_only=True)
                b = torch.load(resumed/'fno/last.pt', weights_only=True)
                for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state',
                            'best_model_state', 'best_value', 'best_step'):
                    self.assert_nested_equal(a[key], b[key])
                self.assertEqual(a['scheduler_state']['optimizer_steps'], 4)
                self.assertEqual(a['scheduler_state']['metric_updates'], 5)
                self.assertLess(a['scheduler_state']['learning_rates'][0], config['learning_rate'])
                self.assertTrue((resumed/'FROZEN').exists())
                self.assertFalse((resumed/'COMPLETE').exists())

    def test_deferred_suite_freezes_without_test_then_finalizes_once(self):
        self.config['defer_test'] = True
        self.write_config()
        output = self.root/'deferred'
        original_read = training.read_split
        def forbid_test(data, split):
            self.assertNotEqual(split, 'test', 'deferred training opened test arrays')
            return original_read(data, split)
        with mock.patch.object(training, 'read_split', side_effect=forbid_test):
            frozen = run(self.data, self.config_path, output)
        best = {}
        for name in training.FACTORIES:
            self.assertNotIn('test', frozen[name])
            self.assertEqual(frozen[name]['test_evaluations'], 0)
            self.assertFalse((output/name/'test_predictions.npz').exists())
            self.assertTrue((output/name/'FROZEN').exists())
            self.assertFalse((output/name/'COMPLETE').exists())
            best[name] = sha256(output/name/'best.pt')
        marker = output/'transformer/FROZEN'
        marker_text = marker.read_text()
        marker.unlink()
        with mock.patch.object(training, 'read_split', side_effect=AssertionError('test opened too soon')):
            with self.assertRaisesRegex(ValueError, 'FROZEN'):
                training.finalize(self.data, output)
        marker.write_text(marker_text)
        calls = []
        def count_read(data, split):
            calls.append(split)
            return original_read(data, split)
        with mock.patch.object(training, 'read_split', side_effect=count_read):
            final = training.finalize(self.data, output)
        self.assertEqual(calls, ['test'])
        for name in training.FACTORIES:
            self.assertEqual(best[name], sha256(output/name/'best.pt'))
            self.assertEqual(final[name]['test_evaluations'], 1)
            self.assertIn('test', final[name])
            self.assertTrue((output/name/'test_predictions.npz').exists())
            self.assertTrue((output/name/'COMPLETE').exists())
        with (mock.patch.object(training, 'read_split', side_effect=AssertionError('reopened test')),
              mock.patch.object(training, 'predict', side_effect=AssertionError('repeated evaluation'))):
            self.assertEqual(training.finalize(self.data, output), final)
            self.assertEqual(run(self.data, self.config_path, output, resume=True), final)

    def test_finalization_export_failure_reuses_committed_test(self):
        self.config.update(models=self.config['models'][:1], defer_test=True)
        self.write_config()
        output = self.root/'deferred'
        run(self.data, self.config_path, output)
        original = training.dump_json
        def fail_metrics(path, value):
            if Path(path).name == 'metrics.json':
                raise OSError('simulated finalization export failure')
            return original(path, value)
        with mock.patch.object(training, 'dump_json', side_effect=fail_metrics):
            with self.assertRaisesRegex(OSError, 'finalization export failure'):
                training.finalize(self.data, output)
        self.assertTrue((output/'fno/test_commit.pt').exists())
        self.assertFalse((output/'COMPLETE').exists())
        with (mock.patch.object(training, 'read_split', side_effect=AssertionError('reopened test')),
              mock.patch.object(training, 'predict', side_effect=AssertionError('repeated evaluation'))):
            final = training.finalize(self.data, output)
        self.assertEqual(final['fno']['test_evaluations'], 1)

if __name__ == '__main__':
    unittest.main()
