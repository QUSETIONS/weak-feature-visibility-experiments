"""Contract tests for the frozen Rayleigh-persistence ablation."""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

import run_frozen_rayleigh_persistence_ablation as rayleigh


class FrozenRayleighPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = rayleigh.load_protocol(rayleigh.DEFAULT_PROTOCOL_PATH)

    def test_protocol_is_frozen_and_powered(self) -> None:
        self.assertEqual(self.protocol["experiment_id"], "spectral-anchor-sae-v5-frozen-rayleigh-persistence-ablation")
        self.assertEqual(self.protocol["status"], "frozen-before-execution")
        self.assertEqual([mode["id"] for mode in self.protocol["modes"]], list(rayleigh.MODE_IDS))
        self.assertEqual(self.protocol["seeds"]["count"], 32)
        self.assertEqual(self.protocol["anchor"]["family_count"], 64)
        self.assertEqual(self.protocol["expected_rows"], 1152)
        self.assertIn("not an unconstrained-SAE", self.protocol["reporting_boundary"])

    def test_soft_budget_matches_hard_radius(self) -> None:
        hard = rayleigh.mode_spec_by_id(self.protocol, "hard_cap")
        soft = rayleigh.mode_spec_by_id(self.protocol, "soft_rayleigh")
        self.assertAlmostEqual(soft["soft_endpoint_gap_ratio_max"], math.sin(hard["anchor_trust_region_radians"]) ** 2, places=15)

    def test_canonical_v4_remains_separate(self) -> None:
        v4 = json.loads((EXPERIMENTS / "bounded_drift_spectral_anchor_sae_protocol.json").read_text())
        self.assertEqual(v4["experiment_id"], "spectral-anchor-sae-v4-bounded-drift")
        self.assertEqual(v4["expected_rows"], 48)
        self.assertAlmostEqual(v4["training"]["anchor_trust_region_radians"], 0.20)
        self.assertAlmostEqual(v4["training"]["residual_anchor_leakage_abs_max"], 0.10)

    def test_estimator_signature_is_target_free(self) -> None:
        parameters = inspect.signature(rayleigh.estimate_spectral_anchor).parameters
        self.assertNotIn("target", parameters)
        self.assertNotIn("geometry", parameters)

    def test_rayleigh_metrics_and_sign_invariance(self) -> None:
        moment = np.diag([3.0, 2.0, 0.5])
        reference, gap, _ = rayleigh.empirical_spectral_data(moment)
        top = np.array([1.0, 0.0, 0.0])
        exact = rayleigh.frozen_rayleigh_metrics(top, moment, reference, gap)
        flipped = rayleigh.frozen_rayleigh_metrics(-top, moment, reference, gap)
        orthogonal = rayleigh.frozen_rayleigh_metrics(np.array([0.0, 1.0, 0.0]), moment, reference, gap)
        self.assertAlmostEqual(exact["frozen_rayleigh_gap_ratio"], 0.0)
        self.assertEqual(exact, flipped)
        self.assertAlmostEqual(orthogonal["frozen_rayleigh_gap_ratio"], 1.0)

    def test_rayleigh_angle_bound_is_sharp_in_two_dimensions(self) -> None:
        angle = 0.31
        moment = np.diag([4.0, 1.5])
        anchor = np.array([math.cos(angle), math.sin(angle)])
        metrics = rayleigh.frozen_rayleigh_metrics(anchor, moment, 4.0, 2.5)
        self.assertAlmostEqual(metrics["frozen_rayleigh_gap_ratio"], math.sin(angle) ** 2, places=12)
        self.assertAlmostEqual(rayleigh.rayleigh_angle_sine_squared_upper_bound(metrics["frozen_rayleigh_gap_ratio"]), math.sin(angle) ** 2, places=12)

    def test_rayleigh_target_bound_reduces_to_spectral_bound_at_zero_gap(self) -> None:
        sine = 0.25
        self.assertAlmostEqual(rayleigh.rayleigh_persistence_cosine_lower_bound(sine, 0.0), math.sqrt(1.0 - sine * sine), places=12)
        self.assertEqual(rayleigh.rayleigh_persistence_cosine_lower_bound(sine, 1.0), 0.0)

    def test_familywise_reference_bound_is_nonvacuous(self) -> None:
        epsilon, sine, cosine = rayleigh.expected_spectral_bounds(self.protocol)
        self.assertGreater(epsilon, 0.0)
        self.assertLess(sine, 1.0)
        self.assertGreater(cosine, self.protocol["anchor"]["recovery_threshold"])
        hard = rayleigh.mode_spec_by_id(self.protocol, "hard_cap")
        self.assertGreater(rayleigh.bounded_drift_cosine_lower_bound(sine, hard["anchor_trust_region_radians"]), 0.80)

    def test_seed_pairing_excludes_mode(self) -> None:
        seed = self.protocol["seeds"]["base"]
        first = rayleigh.seed_bundle(self.protocol, seed, 0, 0, 0)
        second = rayleigh.seed_bundle(self.protocol, seed, 0, 0, 0)
        self.assertEqual(first, second)
        self.assertNotEqual(first["data_seed"], rayleigh.seed_bundle(self.protocol, seed, 1, 0, 0)["data_seed"])
        self.assertNotEqual(first["train_seed"], rayleigh.seed_bundle(self.protocol, seed, 0, 1, 0)["train_seed"])
        self.assertNotEqual(first["train_seed"], rayleigh.seed_bundle(self.protocol, seed, 0, 0, 1)["train_seed"])

    def test_expected_grid_has_no_duplicates(self) -> None:
        keys = rayleigh.expected_row_keys(self.protocol)
        self.assertEqual(len(keys), self.protocol["expected_rows"])

    def test_protocol_rejects_k_one(self) -> None:
        changed = json.loads(json.dumps(self.protocol))
        changed["capacities"][0]["k"] = 1
        with self.assertRaises(ValueError):
            rayleigh.validate_protocol(changed)

    def test_projection_to_cap(self) -> None:
        reference = np.array([1.0, 0.0, 0.0])
        projected = rayleigh.project_to_cap(np.array([0.0, 1.0, 0.0]), reference, 0.20)
        self.assertAlmostEqual(np.linalg.norm(projected), 1.0, places=12)
        self.assertLessEqual(math.acos(float(np.clip(projected @ reference, -1.0, 1.0))), 0.20 + 1e-12)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for tensor smoke tests")
    def test_three_training_modes_smoke(self) -> None:
        import torch

        rng = np.random.default_rng(7)
        target = np.array([1.0, 0.0, 0.0, 0.0])
        samples = rayleigh.sample_signal(rng, 128, target, 0.55)
        moment = rayleigh.training_second_moment(samples)
        eigenvalue, gap, reference = rayleigh.empirical_spectral_data(moment)
        for mode_id in rayleigh.MODE_IDS:
            model, metrics = rayleigh.train_persistence_mode(
                samples, reference, moment, eigenvalue, gap, 4, 2, 3, 32, 0.001, 11,
                {"code_penalty_type": "none", "code_penalty": 0.0},
                rayleigh.mode_spec_by_id(self.protocol, mode_id), 0.10, "cpu",
            )
            self.assertTrue(torch.isfinite(next(model.parameters())).all())
            self.assertLessEqual(metrics["residual_anchor_dot_max"], 0.10 + 1e-5)
            if mode_id == "hard_cap":
                self.assertLessEqual(metrics["anchor_drift_radians"], 0.20 + 1e-5)
            if mode_id == "unrestricted_anchor":
                self.assertEqual(metrics["rayleigh_penalty_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
