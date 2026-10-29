import copy
import json
import subprocess
import sys
import importlib.util
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest import mock
import numpy as np
import torch
import test_ch03_operator_training as fixtures
from neuralpde_opt.chapter3 import operator_training as training


class AdaptedTrainingTests(unittest.TestCase):
    def fixture(self):
        f = fixtures.OperatorTrainingTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        f.config.update(steps=3, epoch_training={'epochs': 3}, dtype='float64',
                        defer_test=True, batch_size=1, eval_every=2,
                        objective='relative_l2_squared', scheduler={'name': 'constant'})
        f.config.pop('learning_rate')
        for spec in f.config['models']:
            spec['training'] = {'optimizer': 'AdamW', 'learning_rate': .001,
                                'batch_size': 2, 'weight_decay': 1e-5, 'grad_clip': .7,
                                'scheduler': 'plateau' if spec['name']=='fno' else 'cosine',
                                'patience': 0, 'factor': .3, 'min_lr': 1e-6}
        return f

    def test_minibatch_epochs_resume_selection_and_test_isolation_all_models(self):
        for name in training.FACTORIES:
            with self.subTest(model=name):
                f = self.fixture()
                f.config['models'] = [s for s in f.config['models'] if s['name']==name]
                f.write_config()
                original = training.read_split
                def safe_read(data, split):
                    self.assertNotEqual(split, 'test')
                    return original(data, split)
                with mock.patch.object(training, 'read_split', side_effect=safe_read):
                    training.run(f.data, f.config_path, f.root/'continuous')
                    training.run(f.data, f.config_path, f.root/'resumed', stop_after=1)
                    training.run(f.data, f.config_path, f.root/'resumed', resume=True)
                a = torch.load(f.root/'continuous'/name/'last.pt', weights_only=True)
                b = torch.load(f.root/'resumed'/name/'last.pt', weights_only=True)
                for key in ('model_state', 'optimizer_state', 'full_scheduler_state',
                            'scheduler_state', 'rng_state', 'best_model_state',
                            'selection_history', 'best_step', 'best_value'):
                    f.assert_nested_equal(a[key], b[key])
                self.assertEqual(a['optimizer_steps'], 6)  # 3 epochs * ceil(3/2)
                self.assertEqual(a['training_sample_exposures'], 9)
                self.assertEqual(a['best_step'], min(a['selection_history'],
                    key=lambda r:r['validation_mean_relative_l2'])['epoch'])
                metrics = json.loads((f.root/'continuous'/name/'metrics.json').read_text())
                self.assertEqual(metrics['objective'], 'relative_l2_squared')
                self.assertEqual(metrics['selection_metric'], 'mean_relative_l2')
                self.assertEqual(metrics['effective_batch_size'], 2)
                self.assertEqual(metrics['epochs_completed'], 3)
                self.assertEqual(metrics['validation_evaluations'], 3)
                self.assertEqual(metrics['test_evaluations'], 0)
                self.assertAlmostEqual(metrics['val']['mean']['relative_l2'], metrics['selection_value'], places=12)
                best = torch.load(f.root/'continuous'/name/'best.pt', weights_only=True)
                self.assertAlmostEqual(best['validation_mean_relative_l2'], metrics['selection_value'], places=12)

    def test_uneven_microbatches_preserve_clipped_adam_update(self):
        f = self.fixture()
        f.config.update(steps=1, epoch_training={'epochs': 1}, eval_every=1)
        f.config['models'] = f.config['models'][:1]
        f.config['models'][0]['training']['batch_size'] = 3
        f.config['batch_size'] = 2  # final microbatch has one item
        f.write_config()
        training.run(f.data, f.config_path, f.root/'micro')
        f.config['batch_size'] = 3
        f.write_config()
        training.run(f.data, f.config_path, f.root/'whole')
        a = torch.load(f.root/'micro/fno/last.pt', weights_only=True)
        b = torch.load(f.root/'whole/fno/last.pt', weights_only=True)
        for key in a['model_state']:
            torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=1e-11, atol=1e-12)

    def test_dimensionless_losses_and_gradients_are_unit_invariant(self):
        f = self.fixture()
        g = training.P1Geometry(f.data/'observation_geometry.npz')
        norm = g.torch_loss('cpu', torch.float64)
        y = torch.tensor(training.read_split(f.data, 'train')['y'], dtype=torch.float64)
        rms = float(torch.sqrt(norm(y).mean()))
        for objective in ('relative_l2_squared', 'absolute_l2_squared', 'scaled_nodal_mse'):
            grads, losses = [], []
            for scale in (1., 1e-3, 1e3):
                coefficient = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
                loss = training.objective_per_sample(coefficient*y*scale, y*scale,
                                                     norm, rms*scale, objective).mean()
                grads.append(torch.autograd.grad(loss, coefficient)[0])
                losses.append(loss)
            for i in (1, 2):
                torch.testing.assert_close(grads[i], grads[0], rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(losses[i], losses[0], rtol=1e-12, atol=1e-12)

    def test_conflicting_or_silently_ignored_settings_are_rejected(self):
        f = self.fixture()
        edits = [lambda c:c.update(legacy_full_epochs=3),
                 lambda c:c.update(learning_rate=.1),
                 lambda c:c.update(defer_test=False),
                 lambda c:c.update(scheduler={'name':'warmup_cosine'}),
                 lambda c:c['models'][0]['training'].update(batch_size=True),
                 lambda c:c['models'][0]['training'].update(grad_clip=0),
                 lambda c:c['models'][0]['training'].update(patence=10),
                 lambda c:c.update(epoch_training={'epochs': True})]
        for edit in edits:
            config = copy.deepcopy(f.config)
            edit(config)
            with self.assertRaises(ValueError):
                training.validate_config(config)

    def test_plateau_uses_configured_parameters(self):
        p = torch.nn.Parameter(torch.tensor(1.))
        opt = torch.optim.Adam([p], lr=.01)
        sch = training.epoch_scheduler(opt, {'scheduler':'plateau', 'patience':0,
                                            'factor':.3, 'min_lr':.0001}, 3)
        sch.step(1.)
        sch.step(2.)
        self.assertAlmostEqual(opt.param_groups[0]['lr'], .003)

    def test_subset_diagnostic_selects_train_without_test(self):
        f = self.fixture()
        f.config.update(purpose=training.OVERFIT_PURPOSE, train_subset_ids=['train0', 'train1'])
        f.config['models'] = f.config['models'][:1]
        f.write_config()
        result = training.run(f.data, f.config_path, f.root/'subset')['fno']
        self.assertEqual(result['selection_split'], 'train')
        self.assertEqual(result['training_sample_ids'], ['train0', 'train1'])
        selection = json.loads((f.root/'subset/fno/selection_history.json').read_text())
        self.assertEqual(result['selected_step'], min(selection, key=lambda r:r['train_mean_relative_l2'])['epoch'])

    def test_experiment_driver_defaults_to_train_val_only(self):
        f = self.fixture()
        f.config.update(steps=1, epoch_training={'epochs':1}, eval_every=1)
        f.config_path.write_text(json.dumps({'seeds':[0], 'training':f.config}))
        # Byte hashing is allowed, but opening this as a dataset would fail.
        (f.data/'test.npz').write_bytes(b'TEST ARRAYS MUST NOT BE OPENED')
        f.write_manifest()
        script = Path(__file__).resolve().parents[1]/'scripts/run_ch03_experiment.py'
        completed = subprocess.run([sys.executable, str(script), '--data', str(f.data),
                                   '--config', str(f.config_path), '--output', str(f.root/'driver'),
                                   '--device', 'cpu'], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
        self.assertTrue((f.root/'driver/ALL_MODELS_FROZEN').exists())
        for name in training.FACTORIES:
            directory = f.root/'driver/seed_0'/name
            self.assertTrue((directory/'FROZEN').exists())
            self.assertFalse((directory/'test_commit.pt').exists())

    def test_trunk_oracle_recovers_exact_representable_fields(self):
        f = self.fixture()
        g = training.P1Geometry(f.data/'observation_geometry.npz')
        script = Path(__file__).resolve().parents[1]/'scripts/diagnose_ch03_adaptation.py'
        spec = importlib.util.spec_from_file_location('adaptation_diagnosis', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        trunk = torch.nn.Linear(2, 2, bias=False).double()
        with torch.no_grad():
            trunk.weight.copy_(torch.eye(2, dtype=torch.float64))
        points = torch.tensor(g.coordinates.reshape(-1, 2), dtype=torch.float64)
        net = SimpleNamespace(trunk=trunk, grid_points=points,
                              bias=torch.nn.Parameter(torch.zeros((), dtype=torch.float64)),
                              hard_boundary=True, _coordinate_features=lambda x:x)
        x, y = g.coordinates[..., 0], g.coordinates[..., 1]
        b = 16*x*(1-x)*y*(1-y)
        target = np.stack((b*(x+2*y), b*(3-x)))
        result = module.trunk_projection(SimpleNamespace(model=net), {'train':target}, g)
        for row in result['tolerances'].values():
            self.assertEqual(row['rank'], 3)
            self.assertLess(row['oracle_relative_l2']['train'], 1e-12)


if __name__ == '__main__':
    unittest.main()
