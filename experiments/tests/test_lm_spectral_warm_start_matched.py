"""Contract tests for the real-residual matched warm-start protocol."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
import run_lm_spectral_warm_start_matched as lm
import run_spectral_warm_start_matched as synthetic


class LMMatchedWarmStartTest(unittest.TestCase):
    def test_protocol_freezes_two_model_full_matrix(self) -> None:
        protocol = lm.load_protocol(lm.DEFAULT_PROTOCOL)
        self.assertEqual([item["id"] for item in protocol["models"]], ["gpt2-small-l6", "pythia-70m-l3"])
        self.assertEqual(protocol["feature"]["directions"], ["pc1", "pc32", "dense"])
        self.assertEqual(protocol["feature"]["strengths"], [0.35, 0.55])
        self.assertEqual(protocol["seeds"]["count"], 8)
        self.assertEqual(protocol["comparison"]["expected_pairs_total"], 96)
        self.assertEqual(protocol["comparison"]["expected_rows_total"], 192)

    def test_scale_extension_protocol_is_a_complete_independent_matrix(self) -> None:
        path = EXPERIMENTS / "lm_spectral_warm_start_scale_extension_protocol.json"
        protocol = lm.load_protocol(path)
        self.assertEqual([item["id"] for item in protocol["models"]], ["pythia-160m-l6"])
        self.assertEqual(protocol["models"][0]["expected_hidden_size"], 768)
        self.assertEqual(protocol["comparison"]["expected_pairs_total"], 48)
        self.assertEqual(protocol["comparison"]["expected_rows_total"], 96)
        self.assertTrue(protocol["evaluation"]["gaussian_certificate_forbidden"])

    def test_scale_extension_budget_followup_is_frozen_at_6000_steps(self) -> None:
        path = EXPERIMENTS / "lm_spectral_warm_start_scale_extension_budget_protocol.json"
        protocol = lm.load_protocol(path)
        self.assertEqual(protocol["experiment_id"], "lm-residual-spectral-warm-start-scale-extension-budget-v1")
        self.assertEqual(protocol["sae"]["steps"], 6000)
        self.assertEqual(protocol["comparison"]["expected_pairs_total"], 48)
        self.assertEqual(protocol["parent_protocol"]["sha256"], "dff375051f6892d58a9572ff2f8e8e7d7b36ae7cbbf5f435f381fbe9ade44ffb")

    def test_real_text_loader_has_no_synthetic_fallback(self) -> None:
        source = inspect.getsource(lm.load_real_texts)
        self.assertNotIn("random digit", source.lower())
        self.assertIn("synthetic fallback is forbidden", source)

    def test_gaussian_certificate_is_forbidden(self) -> None:
        protocol = lm.load_protocol(lm.DEFAULT_PROTOCOL)
        self.assertTrue(protocol["evaluation"]["gaussian_certificate_forbidden"])
        source = inspect.getsource(lm.run_model)
        self.assertIn('"gaussian_certificate_asserted": False', source)

    def test_target_is_not_an_argument_to_training(self) -> None:
        parameters = inspect.signature(synthetic.train_matched_sae).parameters
        self.assertNotIn("target", parameters)
        source = inspect.getsource(lm.run_model)
        call = source[source.index("warm.train_matched_sae(") : source.index("columns = decoder_columns")]
        self.assertNotIn("target,", call)

    def test_seed_pair_is_variant_invariant(self) -> None:
        protocol = lm.load_protocol(lm.DEFAULT_PROTOCOL)
        first = lm.seed_triplet(protocol, 0, 0, 0, protocol["seeds"]["base"])
        second = lm.seed_triplet(protocol, 0, 0, 0, protocol["seeds"]["base"])
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 3)

    def test_direction_geometry(self) -> None:
        directions = lm.direction_vectors(32)
        self.assertAlmostEqual(float(np.linalg.norm(directions["pc1"])), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(directions["pc32"])), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(directions["dense"])), 1.0)
        self.assertEqual(float(directions["pc1"][0]), 1.0)
        self.assertEqual(float(directions["pc32"][-1]), 1.0)


if __name__ == "__main__":
    unittest.main()
