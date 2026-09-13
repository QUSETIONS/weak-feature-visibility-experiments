"""Contract tests for the v8 fresh-validation confirmation."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import unittest
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

try:
    import run_spectral_anchor_fresh_validation as confirm
except ModuleNotFoundError:
    confirm = None


class SpectralFreshValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        if confirm is None:
            self.fail("v8 fresh-validation runner is missing")

    def test_protocol_discloses_post_v7_sample_sizing_and_fresh_validation(self) -> None:
        protocol = confirm.load_protocol(confirm.DEFAULT_PROTOCOL_PATH)
        self.assertEqual(protocol["experiment_id"], "spectral-anchor-sae-v8-fresh-validation-confirmation")
        self.assertTrue(protocol["calibration_disclosure"]["sample_size_chosen_after_source_experiment"])
        self.assertTrue(protocol["calibration_disclosure"]["v7_is_exploratory"])
        self.assertTrue(protocol["calibration_disclosure"]["v8_validation_samples_are_fresh"])
        self.assertEqual(protocol["training"]["N_validation"], 500000)
        self.assertNotEqual(protocol["seeds"]["validation_rng_offset"], protocol["seeds"]["v7_validation_rng_offset"])
        self.assertEqual(protocol["certificate"]["family_count"], 48)
        self.assertEqual(protocol["expected_rows"], 48)

    def test_primary_readout_is_single_fixed_signed_pair_span(self) -> None:
        protocol = confirm.load_protocol(confirm.DEFAULT_PROTOCOL_PATH)
        self.assertEqual(protocol["primary_readout"]["id"], "signed-pair-span")
        self.assertEqual(protocol["primary_readout"]["kind"], "subspace")
        self.assertEqual(protocol["primary_readout"]["slots"], [0, 1])
        self.assertEqual(protocol["primary_readout"]["span_relative_singular_value_tolerance"], 0.05)
        self.assertTrue(protocol["comparison"]["one_primary_readout_per_model"])

    def test_runner_uses_familywise_delta_and_fresh_validation_seed(self) -> None:
        source = inspect.getsource(confirm.run)
        self.assertIn("family_count", source)
        self.assertIn("validation_rng_offset", source)
        self.assertIn("v7_validation_rng_offset", inspect.getsource(confirm.validate_protocol))
        self.assertIn("validation_selected", source)
        self.assertIn("calibration_disclosure", source)

    def test_validator_requires_all_primary_certificates(self) -> None:
        source = inspect.getsource(confirm.validate_rows)
        self.assertIn("all certificates", source)
        self.assertIn("familywise", source)
        self.assertIn("fresh_validation", source)
        self.assertIn("span_rank", source)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for training smoke test")
    def test_primary_span_is_unconstrained_after_warm_start(self) -> None:
        import numpy as np

        samples = np.random.default_rng(17).standard_normal((96, 4))
        anchor = confirm.estimate_spectral_anchor(samples)
        model, metrics = confirm.train_warm_start_sae(samples, anchor, 8, 2, 3, 32, 0.001, 31, {"id": "mse", "code_penalty": 0.0, "code_penalty_type": "none"}, "cpu")
        basis = confirm.signed_pair_basis(model, (0, 1), 0.05)
        self.assertIn(basis.shape[1], [1, 2])
        self.assertEqual(metrics["projection_hits"], 0)

    @unittest.skipUnless(
        (EXPERIMENTS / "results_spectral_anchor_fresh_validation" / "fresh_validation_raw.json").exists()
        and (EXPERIMENTS / "results_spectral_anchor_fresh_validation" / "fresh_validation_summary.json").exists(),
        "v8 artifact is generated after the confirmatory run",
    )
    def test_completed_artifact_passes_confirmatory_gate(self) -> None:
        rows = json.loads((EXPERIMENTS / "results_spectral_anchor_fresh_validation" / "fresh_validation_raw.json").read_text())
        summary = json.loads((EXPERIMENTS / "results_spectral_anchor_fresh_validation" / "fresh_validation_summary.json").read_text())
        protocol = confirm.load_protocol(confirm.DEFAULT_PROTOCOL_PATH)
        confirm.validate_rows(rows, protocol)
        self.assertEqual(len(rows), protocol["expected_rows"])
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["all_recovered"])
        self.assertTrue(summary["all_certified"])
        self.assertTrue(summary["fresh_validation_integrity"])


if __name__ == "__main__":
    unittest.main()
