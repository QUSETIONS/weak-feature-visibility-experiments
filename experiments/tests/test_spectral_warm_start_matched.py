"""Contract tests for the v6 paired spectral warm-start experiment."""

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
    import run_spectral_warm_start_matched as warm
except ModuleNotFoundError:
    warm = None


class SpectralWarmStartMatchedTest(unittest.TestCase):
    def setUp(self) -> None:
        if warm is None:
            self.fail("v6 matched warm-start runner is missing")

    def test_protocol_freezes_original_stress_conditions(self) -> None:
        protocol = warm.load_protocol(warm.DEFAULT_PROTOCOL_PATH)
        self.assertEqual(protocol["experiment_id"], "spectral-anchor-sae-v6-matched-warm-start")
        self.assertEqual(protocol["expected_rows"], 96)
        self.assertEqual(protocol["comparison"]["expected_pairs"], 48)
        self.assertEqual(protocol["geometry"], ["axis", "dense"])
        self.assertEqual([item["id"] for item in protocol["objectives"]], ["mse", "mse_l1", "mse_l2"])
        self.assertEqual([item["id"] for item in protocol["capacities"]], ["d16-k2", "d32-k4"])
        self.assertEqual({item["id"] for item in protocol["variants"]}, {"vanilla-random", "spectral-warm-start"})
        self.assertEqual(protocol["training"]["post_initialization_constraints"], "none")

    def test_fresh_replication_keeps_full_matrix_and_doubles_seeds(self) -> None:
        path = EXPERIMENTS / "spectral_warm_start_fresh_replication_protocol.json"
        protocol = warm.load_protocol(path)
        self.assertEqual(protocol["experiment_id"], "spectral-anchor-sae-v9-independent-fresh-seed-replication")
        self.assertEqual(protocol["seeds"]["count"], 8)
        self.assertEqual(protocol["comparison"]["expected_pairs"], 96)
        self.assertEqual(protocol["expected_rows"], 192)

    def test_certificate_settings_are_frozen_in_protocol(self) -> None:
        protocol = warm.load_protocol(warm.DEFAULT_PROTOCOL_PATH)
        self.assertAlmostEqual(protocol["certificate"]["delta"], 0.05)
        self.assertEqual(protocol["certificate"]["rank"], 1)
        self.assertAlmostEqual(protocol["certificate"]["cosine_threshold"], 0.80)

    def test_cosine_metric_clamps_float_roundoff(self) -> None:
        self.assertEqual(warm.clamp_cosine(1.0 + 2.0**-23), 1.0)
        self.assertEqual(warm.clamp_cosine(-2.0**-23), 0.0)

    def test_anchor_estimator_is_training_only_and_target_free(self) -> None:
        parameters = inspect.signature(warm.estimate_spectral_anchor).parameters
        self.assertNotIn("target", parameters)
        self.assertNotIn("geometry", parameters)

    def test_matched_seed_formula_is_explicitly_validated(self) -> None:
        source = inspect.getsource(warm.validate_rows)
        self.assertIn("data_seed", source)
        self.assertIn("train_seed", source)
        self.assertIn("validation_seed", source)
        self.assertIn("paired", source)

    def test_warm_start_initialization_has_signed_spectral_pair(self) -> None:
        source = inspect.getsource(warm.build_sae)
        self.assertIn("spectral-warm-start", source)
        self.assertIn("anchor_pair_slots", source)
        self.assertIn("training_spectral", source)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for initialization test")
    def test_warm_start_decoder_and_encoder_pair_are_target_free_anchor(self) -> None:
        import numpy as np
        import torch

        reference = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        reference = reference / np.linalg.norm(reference)
        model = warm.build_sae(4, 8, 2, 17, "spectral-warm-start", reference, "cpu")
        decoder = model.dec.weight.detach()
        encoder = model.enc.weight.detach()
        reference_tensor = torch.tensor(reference, dtype=decoder.dtype)
        self.assertGreater(float(torch.dot(decoder[:, 0], reference_tensor).abs()), 0.999)
        self.assertGreater(float(torch.dot(decoder[:, 1], reference_tensor).abs()), 0.999)
        self.assertLess(float(torch.dot(decoder[:, 0], decoder[:, 1])), -0.99)
        self.assertGreater(float(torch.dot(encoder[0], reference_tensor)), 0.99)
        self.assertLess(float(torch.dot(encoder[1], reference_tensor)), -0.99)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for training smoke test")
    def test_training_reports_initialization_and_no_projection(self) -> None:
        import numpy as np

        samples = np.random.default_rng(5).standard_normal((96, 4))
        reference = warm.estimate_spectral_anchor(samples)
        for variant in ["vanilla-random", "spectral-warm-start"]:
            model, metrics = warm.train_matched_sae(samples, reference, 8, 2, 3, 32, 0.001, 29, {"id": variant, "code_penalty": 0.0, "code_penalty_type": "none"}, "cpu")
            self.assertEqual(metrics["variant_id"], variant)
            self.assertIn("initial_anchor_cosine", metrics)
            self.assertIn("final_candidate_cosine", metrics)
            self.assertEqual(metrics["projection_hits"], 0)
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    @unittest.skipUnless(
        (EXPERIMENTS / "results_spectral_warm_start_matched" / "warm_start_raw.json").exists()
        and (EXPERIMENTS / "results_spectral_warm_start_matched" / "warm_start_summary.json").exists(),
        "v6 artifact is generated after the matched matrix run",
    )
    def test_completed_artifact_has_complete_pairs(self) -> None:
        rows = json.loads((EXPERIMENTS / "results_spectral_warm_start_matched" / "warm_start_raw.json").read_text())
        summary = json.loads((EXPERIMENTS / "results_spectral_warm_start_matched" / "warm_start_summary.json").read_text())
        protocol = warm.load_protocol(warm.DEFAULT_PROTOCOL_PATH)
        warm.validate_rows(rows, protocol)
        self.assertEqual(len(rows), protocol["expected_rows"])
        self.assertEqual(summary["pairs"], protocol["comparison"]["expected_pairs"])
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["pair_integrity"])


if __name__ == "__main__":
    unittest.main()
