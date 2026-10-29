"""Small real-solver fixtures for gating, recovery and strict data integrity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from neuralpde_opt.chapter3.experimental_data import (
    _build_sample, audit_dataset, estimate, generate, validate_config)


def fixture_config():
    c = json.loads((Path(__file__).parents[1] / "configs/ch03_pilot.json").read_text())
    c.update(purpose="chapter3_supervised_experiments_fixture", seed=20261021,
             splits={"train": 2, "val": 1, "test": 1},
             audit_samples={"train": 1, "val": 1, "test": 1},
             observation_n=4, reference_n=8, audit_n=16)
    c["fields"]["a"]["modes"] = c["fields"]["f"]["modes"] = 2
    c["quality_gates"].update(reference_vs_audit_l2=.2, reference_vs_audit_h1=.5,
                              l2_refinement_ratio=.8, h1_refinement_ratio=.9,
                              quadrature_l2=.01, observation_reconstruction_l2=.3)
    return c


class ExperimentalDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(fixture_config()))
        self.output = self.root / "generated"

    def tearDown(self):
        self.temporary.cleanup()

    def complete(self):
        generate(self.config, self.output, "gate")
        return generate(self.config, self.output, "complete")

    def test_gate_is_required_and_partial_generation_resumes_from_records(self):
        with self.assertRaisesRegex(RuntimeError, "gate"):
            generate(self.config, self.output, "complete")
        partial = generate(self.config, self.output, "gate", max_new_samples=1)
        self.assertTrue(partial["paused"])
        self.assertFalse((self.output / "COMPLETE").exists())
        self.assertFalse((self.output / "GATE_PASSED").exists())
        saved = next((self.output / "samples").glob("*.npz"))
        content, modified = saved.read_bytes(), saved.stat().st_mtime_ns
        # A lost progress display cannot lose committed sample progress.
        (self.output / "checkpoint.json").unlink()
        gate = generate(self.config, self.output, "gate")
        self.assertTrue(gate["passed"])
        self.assertEqual(saved.read_bytes(), content)
        self.assertEqual(saved.stat().st_mtime_ns, modified)
        self.assertFalse((self.output / "train.npz").exists())
        report = generate(self.config, self.output, "complete")
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["independent_weak_residual_samples"], 4)
        rows = json.loads((self.output / "quality.json").read_text())["samples"]
        self.assertEqual(sum(r["full_audit"] for r in rows), 3)
        self.assertFalse(rows[1]["full_audit"])
        self.assertEqual(len(rows[1]["solves"]), 1)
        self.assertTrue(audit_dataset(self.output)["passed"])

    def test_configuration_is_immutable_on_resume(self):
        generate(self.config, self.output, "gate", max_new_samples=1)
        changed = fixture_config()
        changed["seed"] += 1
        self.config.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "immutable"):
            generate(self.config, self.output, "gate")

    def test_failed_gate_stops_before_unselected_samples(self):
        c = fixture_config()
        c["quality_gates"]["reference_vs_audit_l2"] = 0
        self.config.write_text(json.dumps(c))
        with self.assertRaisesRegex(RuntimeError, "quality gate failed"):
            generate(self.config, self.output, "gate")
        self.assertFalse((self.output / "GATE_PASSED").exists())
        self.assertFalse((self.output / "COMPLETE").exists())
        self.assertEqual(len(list((self.output / "samples").glob("*.npz"))), 1)

    def test_gate_reloads_labels_and_rejects_a_valid_checksum_with_bad_weak_residual(self):
        def wrong_serialized_label(*args, **kwargs):
            arrays, audits, row = _build_sample(*args, **kwargs)
            # This point is absent from the observation subgrid; all checksums
            # will be generated normally, so the independent weak check matters.
            arrays["y_reference"][1, 1] += .01
            return arrays, audits, row
        with patch("neuralpde_opt.chapter3.experimental_data._build_sample", side_effect=wrong_serialized_label):
            with self.assertRaisesRegex(RuntimeError, "full-audit gate failed"):
                generate(self.config, self.output, "gate")
        self.assertFalse((self.output / "GATE_PASSED").exists())
        reload_report = json.loads((self.output / "gate_reload_audit.json").read_text())
        self.assertTrue(any("independent reference residual" in failure for failure in reload_report["failures"]))

    def test_strict_audit_detects_corruption_and_untracked_file(self):
        self.assertTrue(self.complete()["passed"])
        unrelated = self.output / "unexpected.txt"
        unrelated.write_text("untracked")
        report = audit_dataset(self.output)
        self.assertFalse(report["passed"])
        self.assertIn("manifest file coverage", report["failures"])
        unrelated.unlink()
        path = next((self.output / "samples").glob("*.npz"))
        path.write_bytes(path.read_bytes() + b"corruption")
        report = audit_dataset(self.output)
        self.assertFalse(report["passed"])
        self.assertTrue(any("checksum" in failure for failure in report["failures"]))

    def test_raw_field_schema_and_flat_manifest_match_existing_consumers(self):
        self.assertTrue(self.complete()["passed"])
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertTrue(all(isinstance(value, str) and len(value) == 64 for value in manifest.values()))
        self.assertIn("train.npz", manifest)
        with np.load(self.output / "train.npz", allow_pickle=False) as sample:
            self.assertEqual(sample["a"].shape, (2, 5, 5))
            self.assertEqual(sample["y_reference"].shape, (2, 9, 9))
            self.assertTrue(np.array_equal(sample["y"], sample["y_reference"][:, ::2, ::2]))
            self.assertEqual(sample["a"].dtype, np.float64)

    def test_production_requires_real_audit_budget_and_approved_grid(self):
        c = fixture_config()
        c["purpose"] = "chapter3_supervised_experiments_v1"
        with self.assertRaises(ValueError):
            validate_config(c)
        c.update(splits={"train": 128, "val": 32, "test": 32},
                 audit_samples={"train": 16, "val": 8, "test": 8},
                 observation_n=64, reference_n=128, audit_n=256)
        validate_config(c)
        cost = estimate(c)
        self.assertEqual(cost["samples"], 192)
        self.assertEqual(cost["full_audit_samples"], 32)
        self.assertGreater(cost["recommended_free_bytes"], cost["retained_data_upper_estimate_bytes"])


if __name__ == "__main__":
    unittest.main()
