"""Verify a resumed optimization follows the uninterrupted CPU trajectory."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from neuralpde_opt import training


class ResumeTests(unittest.TestCase):
    def test_optimizer_resume_and_completed_skip(self):
        torch.set_num_threads(2)
        config = dict(name="resume_test", dtype="float64", learning_rate=.001,
                      adam_steps=4, lbfgs_steps=4, lbfgs_block=2, log_every=2,
                      quadrature_order=3, validation_order=4,
                      problems={"1": dict(width=8, depth=2, train_n=4,
                                          validation_n=6, weak_validation_n=8)})
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            original_save = training.save_json

            def interrupted_save(path, data):
                original_save(path, data)
                if Path(path).name == "history.json" and data[-1]["stage"] == "adam":
                    raise RuntimeError("Simulated interruption after checkpoint")

            with patch.object(training, "save_json", interrupted_save):
                with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
                    training.run_one(config, 1, "vpinn", 7, root/"resume", "cpu")
            resumed = training.run_one(config, 1, "vpinn", 7, root/"resume", "cpu")
            direct = training.run_one(config, 1, "vpinn", 7, root/"direct", "cpu")
            name = "poisson1d_vpinn_seed7/model.pt"
            a = torch.load(root/"resume"/name, weights_only=True)
            b = torch.load(root/"direct"/name, weights_only=True)
            for key in a:
                self.assertTrue(torch.allclose(a[key], b[key], atol=1e-12, rtol=1e-12), key)
            self.assertAlmostEqual(resumed["l2_rel"], direct["l2_rel"], places=12)
            self.assertEqual(resumed["objective_evaluations"], direct["objective_evaluations"])
            skipped = training.run_one(config, 1, "vpinn", 7, root/"resume", "cpu")
            self.assertEqual(skipped, resumed)
            with self.assertRaisesRegex(ValueError, "Configuration differs"):
                training.run_one(dict(config, adam_steps=5), 1, "vpinn", 7, root/"resume", "cpu")


if __name__ == "__main__":
    unittest.main()
