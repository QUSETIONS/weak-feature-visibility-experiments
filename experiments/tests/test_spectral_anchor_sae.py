"""Contract tests for a target-free spectral-anchor SAE recovery guarantee."""

import sys
import unittest
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

try:
    import run_spectral_anchor_sae as anchor
except ModuleNotFoundError:
    anchor = None


class SpectralAnchorSAETest(unittest.TestCase):
    def setUp(self) -> None:
        if anchor is None:
            self.fail("spectral-anchor SAE runner is missing")

    def test_anchor_estimator_has_no_target_direction_argument(self) -> None:
        import inspect

        parameters = inspect.signature(anchor.estimate_spectral_anchor).parameters
        self.assertNotIn("target", parameters)
        self.assertNotIn("geometry", parameters)

    def test_anchor_columns_are_fixed_and_opposite(self) -> None:
        import numpy as np

        direction = np.array([1.0, 0.0, 0.0])
        anchors = anchor.make_signed_anchors(direction)
        self.assertEqual(anchors.shape, (3, 2))
        self.assertTrue(np.allclose(anchors[:, 0], direction))
        self.assertTrue(np.allclose(anchors[:, 1], -direction))
        samples = np.array([[2.0, 3.0, 4.0], [-1.0, 5.0, 6.0]])
        self.assertTrue(np.allclose(anchor.signed_anchor_reconstruction(samples, direction), [[2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))

    def test_residual_objectives_cannot_update_anchor_columns(self) -> None:
        import inspect

        source = inspect.getsource(anchor.train_anchor_preserving_sae)
        self.assertIn("requires_grad_(False)", source)
        self.assertIn("anchor", source)
        self.assertIn("residual", source)
        self.assertIn("project_residual_decoder", source)

    def test_signed_anchor_respects_total_topk_budget(self) -> None:
        import inspect

        source = inspect.getsource(anchor.train_anchor_preserving_sae)
        self.assertIn("self.k - 1", source)

    def test_protocol_freezes_cross_objective_capacity_matrix(self) -> None:
        protocol = anchor.load_protocol(anchor.DEFAULT_PROTOCOL_PATH)
        self.assertEqual(protocol["experiment_id"], "spectral-anchor-sae-v3")
        self.assertEqual(protocol["anchor"]["selection_split"], "training_only")
        self.assertTrue(protocol["anchor"]["target_oracle_forbidden"])
        self.assertTrue(protocol["validation"]["selection_forbidden"])
        self.assertEqual(protocol["anchor"]["family_count"], 8)
        self.assertEqual(protocol["expected_rows"], 48)
        for capacity in protocol["capacities"]:
            self.assertEqual(capacity["total_dict_size"], capacity["residual_dict_size"] + protocol["training"]["anchor_active_slots"] + 1)

    def test_davis_kahan_bound_is_monotone_and_nonvacuous(self) -> None:
        loose = anchor.davis_kahan_sine_bound(covariance_error=0.20, eigengap=0.55)
        tight = anchor.davis_kahan_sine_bound(covariance_error=0.05, eigengap=0.55)
        self.assertAlmostEqual(loose, 0.20 / (0.55 - 0.20))
        self.assertAlmostEqual(tight, 0.05 / (0.55 - 0.05))
        self.assertGreater(loose, tight)
        self.assertLess(tight, 1.0)
        with self.assertRaises(ValueError):
            anchor.davis_kahan_sine_bound(covariance_error=0.55, eigengap=0.55)

    def test_runner_uses_familywise_delta(self) -> None:
        import inspect

        source = inspect.getsource(anchor.run)
        self.assertIn("family_count", source)
        self.assertIn("anchor_config[\"delta\"] / int(anchor_config[\"family_count\"])", source)

    def test_gaussian_wishart_bound_certifies_frozen_protocol(self) -> None:
        protocol = anchor.load_protocol(anchor.DEFAULT_PROTOCOL_PATH)
        epsilon = anchor.gaussian_wishart_operator_bound(
            dimension=protocol["feature"]["m"],
            sample_count=protocol["training"]["N_train"],
            covariance_norm=1.0 + protocol["feature"]["signal_strength"],
            delta=protocol["anchor"]["delta"],
        )
        sine = anchor.davis_kahan_sine_bound(epsilon, protocol["anchor"]["davis_kahan_eigengap"])
        self.assertLess(sine, (1.0 - protocol["anchor"]["recovery_threshold"] ** 2) ** 0.5)

    def test_cross_objective_rows_inherit_anchor_cosine(self) -> None:
        rows = [
            {"objective_id": "mse", "capacity_id": "d16", "anchor_cosine": 0.93},
            {"objective_id": "mse_l1", "capacity_id": "d32", "anchor_cosine": 0.93},
        ]
        anchor.validate_inherited_anchor_recovery(rows, threshold=0.80)
        rows[1]["anchor_cosine"] = 0.79
        with self.assertRaises(ValueError):
            anchor.validate_inherited_anchor_recovery(rows, threshold=0.80)

    def test_completed_artifact_passes_anchor_inheritance_gate(self) -> None:
        import json

        artifact = json.loads((EXPERIMENTS / "results_spectral_anchor_sae" / "spectral_anchor_raw.json").read_text())
        summary = json.loads((EXPERIMENTS / "results_spectral_anchor_sae" / "spectral_anchor_summary.json").read_text())
        protocol = anchor.load_protocol(anchor.DEFAULT_PROTOCOL_PATH)
        anchor.validate_rows(artifact, protocol)
        self.assertEqual(len(artifact), 48)
        self.assertEqual(summary["anchor_recovery_rate"], 1.0)
        self.assertTrue(summary["evidence_eligible"])
        self.assertGreaterEqual(summary["anchor_cosine_lower_bound"], protocol["anchor"]["recovery_threshold"])

    def test_manuscript_contains_anchor_preserving_theorem_and_table(self) -> None:
        manuscript = (EXPERIMENTS.parent / "paper" / "main.tex").read_text()
        self.assertIn("Objective/capacity-uniform spectral-anchor recovery", manuscript)
        self.assertIn("tab:spectral_anchor", manuscript)
        self.assertIn("all anchor-preserving objectives and capacities", manuscript)

    def test_manuscript_states_why_unconstrained_class_needs_a_linkage_assumption(self) -> None:
        manuscript = (EXPERIMENTS.parent / "paper" / "main.tex").read_text()
        self.assertIn("Why a structural constraint is necessary", manuscript)
        self.assertIn("no positive uniform alignment bound", manuscript)
        self.assertIn("does not rule out a data-dependent algorithm", manuscript)


if __name__ == "__main__":
    unittest.main()
