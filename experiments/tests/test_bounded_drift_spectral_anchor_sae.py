"""Contract tests for the bounded-drift spectral-anchor SAE family."""

import importlib.util
import inspect
import json
import sys
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

try:
    import run_bounded_drift_spectral_anchor_sae as anchor
except ModuleNotFoundError:
    anchor = None


class BoundedDriftSpectralAnchorTest(unittest.TestCase):
    def setUp(self) -> None:
        if anchor is None:
            self.fail("bounded-drift spectral-anchor runner is missing")

    def test_estimator_has_no_target_or_geometry_argument(self) -> None:
        parameters = inspect.signature(anchor.estimate_spectral_anchor).parameters
        self.assertNotIn("target", parameters)
        self.assertNotIn("geometry", parameters)

    def test_protocol_freezes_bounded_drift_matrix(self) -> None:
        protocol = anchor.load_protocol(anchor.DEFAULT_PROTOCOL_PATH)
        self.assertEqual(protocol["experiment_id"], "spectral-anchor-sae-v4-bounded-drift")
        self.assertTrue(protocol["training"]["anchor_trainable"])
        self.assertTrue(protocol["anchor"]["target_oracle_forbidden"])
        self.assertTrue(protocol["validation"]["selection_forbidden"])
        self.assertEqual(protocol["expected_rows"], 48)
        self.assertEqual(protocol["anchor"]["family_count"], 8)
        self.assertAlmostEqual(protocol["training"]["anchor_trust_region_radians"], 0.20)
        self.assertAlmostEqual(protocol["training"]["residual_anchor_leakage_abs_max"], 0.10)
        for capacity in protocol["capacities"]:
            self.assertEqual(
                capacity["total_dict_size"],
                capacity["residual_dict_size"] + protocol["training"]["anchor_active_slots"] + 1,
            )

    def test_projection_to_cap_enforces_angle_and_preserves_reference(self) -> None:
        import numpy as np

        reference = np.array([1.0, 0.0, 0.0])
        inside = np.array([0.99, 0.1, 0.0])
        outside = np.array([0.0, 1.0, 0.0])
        kept = anchor.project_to_cap(inside, reference, 0.20)
        clipped = anchor.project_to_cap(outside, reference, 0.20)
        self.assertGreaterEqual(float(np.dot(kept, reference)), float(np.dot(inside / np.linalg.norm(inside), reference)) - 1e-12)
        self.assertLessEqual(float(np.arccos(np.clip(np.dot(clipped, reference), -1.0, 1.0))), 0.20 + 1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(clipped)), 1.0, places=12)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for tensor projection test")
    def test_residual_projection_enforces_bounded_leakage_and_unit_norm(self) -> None:
        import torch

        anchor_vector = torch.tensor([1.0, 0.0, 0.0])
        weights = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        projected = anchor.project_residual_decoder_bounded(weights, anchor_vector, 0.10)
        dots = anchor_vector @ projected
        self.assertLessEqual(float(dots.abs().max()), 0.10 + 1e-6)
        self.assertTrue(torch.allclose(projected.norm(dim=0), torch.ones(2), atol=1e-6))

    def test_bounded_drift_cosine_bound_is_nonvacuous(self) -> None:
        bound = anchor.bounded_drift_cosine_lower_bound(0.339689, 0.20)
        self.assertAlmostEqual(bound, 0.854, delta=0.01)
        self.assertGreaterEqual(bound, 0.80)
        with self.assertRaises(ValueError):
            anchor.bounded_drift_cosine_lower_bound(0.99, 0.20)

    def test_runner_uses_familywise_delta_and_trust_region(self) -> None:
        source = inspect.getsource(anchor.run)
        self.assertIn("anchor_config[\"delta\"] / int(anchor_config[\"family_count\"])", source)
        self.assertIn("anchor_trust_region_radians", source)
        self.assertIn("residual_anchor_leakage_abs_max", source)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for training smoke test")
    def test_trainable_anchor_smoke_respects_constraints(self) -> None:
        import numpy as np
        import torch

        rng = np.random.default_rng(7)
        samples = rng.standard_normal((96, 4))
        reference = np.array([1.0, 0.0, 0.0, 0.0])
        anchors = anchor.make_signed_anchors(reference)
        model, _, drift, leakage = anchor.train_bounded_drift_sae(
            samples=samples,
            anchors=anchors,
            residual_dict_size=4,
            k=2,
            steps=3,
            batch=32,
            lr=0.001,
            seed=11,
            objective={"code_penalty_type": "none", "code_penalty": 0.0},
            device="cpu",
            trust_region_radians=0.20,
            residual_leakage_abs_max=0.10,
        )
        self.assertLessEqual(drift, 0.20 + 1e-5)
        self.assertLessEqual(leakage, 0.10 + 1e-5)
        self.assertTrue(torch.isfinite(next(model.parameters())).all())

    def test_completed_artifact_passes_bounded_drift_gate(self) -> None:
        artifact = json.loads((EXPERIMENTS / "results_bounded_drift_spectral_anchor_sae" / "bounded_drift_spectral_anchor_raw.json").read_text())
        summary = json.loads((EXPERIMENTS / "results_bounded_drift_spectral_anchor_sae" / "bounded_drift_spectral_anchor_summary.json").read_text())
        protocol = anchor.load_protocol(anchor.DEFAULT_PROTOCOL_PATH)
        anchor.validate_rows(artifact, protocol)
        self.assertEqual(len(artifact), 48)
        self.assertEqual(summary["anchor_recovery_rate"], 1.0)
        self.assertTrue(summary["evidence_eligible"])
        self.assertLessEqual(summary["maximum_anchor_drift_radians"], protocol["training"]["anchor_trust_region_radians"] + 1e-5)
        self.assertLessEqual(summary["maximum_residual_anchor_dot"], protocol["training"]["residual_anchor_leakage_abs_max"] + 1e-5)

    def test_manuscript_contains_bounded_drift_theorem_and_table(self) -> None:
        manuscript = (EXPERIMENTS.parent / "paper" / "main.tex").read_text()
        self.assertIn("Bounded-drift spectral-anchor recovery", manuscript)
        self.assertIn("tab:bounded_drift_anchor", manuscript)
        self.assertIn("all bounded-drift anchor-preserving objectives and capacities", manuscript)


if __name__ == "__main__":
    unittest.main()
