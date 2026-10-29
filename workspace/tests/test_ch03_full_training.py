import copy
import json
import unittest
from unittest import mock
import torch
import numpy as np
import test_ch03_operator_training as fixtures
from neuralpde_opt.chapter3 import operator_training as training


class FullTrainingTests(unittest.TestCase):
    def fixture(self):
        f=fixtures.OperatorTrainingTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        return f

    def test_full_epoch_resume_and_deferred_selection_all_architectures(self):
        for name in training.FACTORIES:
            with self.subTest(model=name):
                f=self.fixture()
                spec=next(s for s in f.config['models'] if s['name']==name)
                spec['legacy_training']={'optimizer':'Adam' if name=='fno' else 'AdamW',
                    'batch_size':128 if name=='fno' else 1024,'learning_rate':.001,
                    'scheduler':'plateau' if name=='fno' else 'cosine',
                    'weight_decay':0. if name=='fno' else 1e-5,'grad_clip':1.}
                f.config.update(steps=4,legacy_full_epochs=4,dtype='float64',defer_test=True,
                                batch_size=2,models=[spec])
                f.write_config()
                original=training.read_split
                def safe_read(data,key):
                    self.assertNotEqual(key,'test')
                    return original(data,key)
                with mock.patch.object(training,'read_split',side_effect=safe_read):
                    training.run(f.data,f.config_path,f.root/'continuous')
                    training.run(f.data,f.config_path,f.root/'resumed',stop_after=2)
                    training.run(f.data,f.config_path,f.root/'resumed',resume=True)
                a=torch.load(f.root/'continuous'/name/'last.pt',weights_only=True)
                b=torch.load(f.root/'resumed'/name/'last.pt',weights_only=True)
                for key in ('model_state','optimizer_state','scheduler_state','full_scheduler_state',
                            'rng_state','best_step','best_value','selection_history'):
                    f.assert_nested_equal(a[key],b[key])
                history=json.loads((f.root/'continuous'/name/'selection_history.json').read_text())
                self.assertEqual([r['epoch'] for r in history],[1,2,3,4])
                self.assertEqual(a['best_step'],min(history,key=lambda r:r['validation_mse'])['epoch'])
                metrics=json.loads((f.root/'continuous'/name/'metrics.json').read_text())
                self.assertEqual(metrics['epochs_completed'],4)
                self.assertEqual(metrics['optimizer_steps'],4)
                self.assertEqual(metrics['test_evaluations'],0)
                self.assertEqual(metrics['objective'],'nodal_mse')

    def test_microbatch_accumulation_matches_full_batch_update(self):
        f=self.fixture()
        f.config.update(steps=1,legacy_full_epochs=1,dtype='float64',defer_test=True,batch_size=1)
        f.config['models']=f.config['models'][:1]
        f.config['models'][0]['legacy_training']={'optimizer':'Adam','batch_size':128,
            'learning_rate':.001,'scheduler':'plateau'}
        f.write_config()
        training.run(f.data,f.config_path,f.root/'micro')
        f.config['batch_size']=3
        f.write_config()
        training.run(f.data,f.config_path,f.root/'logical')
        a=torch.load(f.root/'micro/fno/last.pt',weights_only=True)
        b=torch.load(f.root/'logical/fno/last.pt',weights_only=True)
        for key in a['model_state']:
            torch.testing.assert_close(a['model_state'][key],b['model_state'][key],rtol=1e-12,atol=1e-12)

    def test_full_checkpoint_inference_batch_and_legacy_guard(self):
        from neuralpde_opt.chapter3.operator_inference import run_inference
        f=self.fixture()
        f.config.update(steps=1,legacy_full_epochs=1,dtype='float64',defer_test=True)
        f.config['models']=f.config['models'][:1]
        f.config['models'][0]['legacy_training']={'optimizer':'Adam','batch_size':128,
            'learning_rate':.001,'scheduler':'plateau'}
        f.write_config()
        training.run(f.data,f.config_path,f.root/'full')
        val=training.read_split(f.data,'val')
        inputs=f.root/'unlabelled.npz'
        np.savez(inputs,a=val['a'],f=val['f'],sample_id=val['sample_id'])
        large=run_inference(f.root/'full/fno',inputs,f.root/'large',batch_size=128)
        small=run_inference(f.root/'full/fno',inputs,f.root/'small',batch_size=1)
        np.testing.assert_allclose(large,small,rtol=1e-12,atol=1e-12)
        cfg=f.root/'full/fno/config.json'
        values=json.loads(cfg.read_text());values.pop('legacy_full_epochs');cfg.write_text(json.dumps(values))
        with self.assertRaisesRegex(ValueError,'1 to 8'):
            run_inference(f.root/'full/fno',inputs,f.root/'rejected',batch_size=128)


if __name__=='__main__':
    unittest.main()
