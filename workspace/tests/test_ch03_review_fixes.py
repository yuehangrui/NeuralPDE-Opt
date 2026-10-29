"""Regressions for review R1/R2/R3/R7, using only temporary synthetic data."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock
import numpy as np
import torch
import test_ch03_operator_training as fixtures
from neuralpde_opt.chapter3 import operator_training as t

spec = importlib.util.spec_from_file_location('current_adaptation_diagnosis',
    Path(__file__).resolve().parents[1]/'scripts/diagnose_ch03_adaptation.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)

class ReviewFixTests(unittest.TestCase):
    def fixture(self):
        f = fixtures.OperatorTrainingTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        f.config.update(models=f.config['models'][:1],steps=4,eval_every=1,
                        dtype='float64',defer_test=True)
        f.write_config()
        return f

    def replace_labels(self,f,split,factor):
        with np.load(f.data/f'{split}.npz') as z:
            arrays = {k:z[k].copy() for k in z.files}
        arrays['y'] *= factor
        np.savez(f.data/f'{split}.npz',**arrays)

    def test_certificate_is_checked_without_loading_any_array(self):
        f = self.fixture()
        with mock.patch.object(np,'load',side_effect=AssertionError('No array loading')):
            self.assertEqual(t.verify_data_manifest(f.data)['workflow'],'ch03-resumable-fields-v1')

    def test_stale_complete_cannot_authorize_manifest_updated_labels(self):
        f = self.fixture()
        self.replace_labels(f,'train',2)
        manifest = json.loads((f.data/'manifest.json').read_text())
        manifest['train.npz'] = t.sha256(f.data/'train.npz')
        (f.data/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'COMPLETE manifest'):
            t.run(f.data,f.config_path,f.root/'bad')
        self.assertFalse((f.root/'bad').exists())

    def test_changed_audit_config_and_downgraded_marker_are_rejected(self):
        edits = ('audit_hash','audit_failed','config_identity','downgrade','arbitrary_text')
        for edit in edits:
            with self.subTest(edit=edit):
                f = self.fixture()
                marker = json.loads((f.data/'COMPLETE').read_text())
                if edit.startswith('audit'):
                    p = f.data/'reload_audit.json'
                    audit = json.loads(p.read_text())
                    audit['passed'] = edit != 'audit_failed'
                    audit['extra'] = 'changed'
                    p.write_text(json.dumps(audit))
                    if edit == 'audit_failed':
                        marker['reload_audit_sha256'] = t.sha256(p)
                elif edit == 'config_identity':
                    marker['config_sha256'] = '0'*64
                elif edit == 'downgrade':
                    (f.data/'COMPLETE').write_text('Pilot quality and serialized-data audit passed. No models trained.\n')
                else:
                    (f.data/'COMPLETE').write_text('synthetic test fixture\n')
                if edit not in ('downgrade','arbitrary_text'):
                    (f.data/'COMPLETE').write_text(json.dumps(marker))
                with self.assertRaises(ValueError):
                    t.verify_data_manifest(f.data)

    def test_explicit_legacy_pilot_metadata_remains_supported(self):
        f = self.fixture()
        cfg = json.loads((f.data/'config.json').read_text())
        cfg['purpose'] = 'pilot_only_no_training'
        (f.data/'config.json').write_text(json.dumps(cfg))
        (f.data/'identity.json').unlink()
        manifest = json.loads((f.data/'manifest.json').read_text())
        manifest.pop('identity.json')
        manifest['config.json'] = t.sha256(f.data/'config.json')
        (f.data/'manifest.json').write_text(json.dumps(manifest))
        (f.data/'COMPLETE').write_text('Pilot quality and serialized-data audit passed. No models trained.\n')
        self.assertEqual(t.verify_data_manifest(f.data)['workflow'],'legacy_pilot')

    def test_diagnosis_supports_seed_directory_subset_and_never_loads_test(self):
        f = self.fixture()
        f.config.update(purpose=t.OVERFIT_PURPOSE,train_subset_ids=['train1'])
        f.write_config()
        directory = f.root/'seed_0'
        t.run(f.data,f.config_path,directory)
        before = t.sha256(directory/'fno/best.pt')
        original = np.load
        def safe_load(path,*args,**kwargs):
            if Path(path).name in ('test.npz','test_predictions.npz'):
                raise AssertionError('Test arrays opened')
            return original(path,*args,**kwargs)
        with mock.patch.object(np,'load',side_effect=safe_load):
            result = diagnostic.diagnose(f.data,directory,f.root/'diagnosis.json')
        self.assertEqual(result['expected_model_count'],1)
        self.assertEqual(result['checked_model_count'],1)
        self.assertTrue(result['read_only_checks_passed'])
        saved = json.loads((directory/'fno/normalization.json').read_text())
        self.assertEqual(result['models']['fno']['normalization'],saved)
        self.assertNotEqual(saved,result['normalization'])
        self.assertEqual(before,t.sha256(directory/'fno/best.pt'))

    def test_diagnosis_rejects_stale_and_freshly_certified_wrong_dataset(self):
        for recertify in (False,True):
            with self.subTest(recertify=recertify):
                f = self.fixture()
                directory = f.root/'seed_0'
                t.run(f.data,f.config_path,directory)
                self.replace_labels(f,'train',10)
                self.replace_labels(f,'val',10)
                if recertify:
                    f.write_manifest()
                target = f.root/'bad.json'
                with self.assertRaisesRegex(ValueError,'identity mismatch|checksum mismatch'):
                    diagnostic.diagnose(f.data,directory,target)
                self.assertFalse(target.exists())

    def test_diagnosis_rejects_empty_and_incomplete_experiments(self):
        f = self.fixture()
        with self.assertRaises(ValueError):
            diagnostic.diagnose(f.data,f.root/'empty',f.root/'empty.json')
        directory = f.root/'experiment'
        directory.mkdir()
        t.run(f.data,f.config_path,directory/'seed_0')
        (directory/'experiment_config.json').write_text(json.dumps({'seeds':[0,1],'training':f.config}))
        with self.assertRaisesRegex(ValueError,'coverage mismatch'):
            diagnostic.diagnose(f.data,directory,f.root/'partial.json')
        (directory/'experiment_config.json').write_text(json.dumps({'seeds':[0],'training':f.config}))
        result = diagnostic.diagnose(f.data,directory,f.root/'complete.json')
        self.assertEqual(set(result['models']),{'seed_0/fno'})

    def test_legacy_subset_selects_and_schedules_training_mse(self):
        f = self.fixture()
        f.config.update(legacy_full_epochs=4,purpose=t.OVERFIT_PURPOSE,train_subset_ids=['train0'])
        f.config['models'][0]['legacy_training'] = {
            'optimizer':'Adam','learning_rate':.001,'batch_size':128,'scheduler':'plateau','patience':0}
        self.replace_labels(f,'val',-1)
        f.write_manifest()
        f.write_config()
        result = t.run(f.data,f.config_path,f.root/'run')['fno']
        selection = json.loads((f.root/'run/fno/selection_history.json').read_text())
        self.assertEqual(result['selection_split'],'train')
        self.assertEqual(result['selection_metric'],'training_mse')
        self.assertEqual(result['selected_step'],min(selection,key=lambda r:r['training_mse'])['epoch'])
        self.assertNotEqual(result['selected_step'],min(selection,key=lambda r:r['validation_mse'])['epoch'])
        self.assertEqual(result['selection_value'],result['full_scheduler_state']['best'])
        best = torch.load(f.root/'run/fno/best.pt',weights_only=True)
        self.assertEqual(best['selection_metric'],'training_mse')
        self.assertNotIn('validation_mean_relative_l2',best)
        self.assertEqual(best['selection_value'],result['selection_value'])
        t.run(f.data,f.config_path,f.root/'resume',stop_after=2)
        resumed = t.run(f.data,f.config_path,f.root/'resume',resume=True)['fno']
        self.assertEqual(result['selected_step'],resumed['selected_step'])
        self.assertEqual(result['selection_value'],resumed['selection_value'])

if __name__ == '__main__':
    unittest.main()
