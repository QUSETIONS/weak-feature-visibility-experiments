"""Contract tests for the v5 spectral-anchor dose-response matrix."""

import importlib.util
import inspect
import json
import math
import sys
import unittest
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

try:
    import run_spectral_anchor_dose_response as dose
except ModuleNotFoundError:
    dose = None


class SpectralAnchorDoseResponseTest(unittest.TestCase):
    def setUp(self) -> None:
        if dose is None:
            self.fail("v5 dose-response runner is missing")

    def test_estimator_is_training_only_and_target_free(self) -> None:
        parameters = inspect.signature(dose.estimate_spectral_anchor).parameters
        self.assertNotIn("target", parameters)
        self.assertNotIn("geometry", parameters)

    def test_protocol_has_four_modes_and_nine_configs(self) -> None:
        protocol = dose.load_protocol(dose.DEFAULT_PROTOCOL_PATH)
        self.assertEqual(protocol["experiment_id"], "spectral-anchor-sae-v5-dose-response")
        self.assertEqual(protocol["expected_groups"], 8)
        self.assertEqual(protocol["expected_rows"], 72)
        self.assertEqual(len(protocol["configs"]), 9)
        self.assertEqual({config["mode"] for config in protocol["configs"]}, {"frozen", "hard", "soft", "unbounded"})
        self.assertTrue(protocol["anchor"]["target_oracle_forbidden"])
        self.assertTrue(protocol["validation"]["selection_forbidden"])
        self.assertEqual(protocol["training"]["residual_active_slots"], "k_minus_1")
        self.assertAlmostEqual(protocol["anchor"]["covariance_norm_upper_bound"], 1.55)
        self.assertAlmostEqual(protocol["anchor"]["davis_kahan_eigengap"], 0.55)
        self.assertAlmostEqual(protocol["anchor"]["delta"], 0.05)

    def test_protocol_contains_rho_and_eta_dose_response(self) -> None:
        protocol = dose.load_protocol(dose.DEFAULT_PROTOCOL_PATH)
        hard = [config for config in protocol["configs"] if config["mode"] == "hard"]
        self.assertEqual(sorted({config["rho_radians"] for config in hard}), [0.05, 0.1, 0.2, 0.4])
        frozen = [config for config in protocol["configs"] if config["mode"] == "frozen"]
        self.assertEqual(sorted({config["rho_radians"] for config in frozen}), [0.0])
        self.assertEqual(sorted({config["eta"] for config in hard}), [0.0, 0.1])
        self.assertTrue(any(config["mode"] == "soft" and config["anchor_soft_penalty"] > 0 for config in protocol["configs"]))
        self.assertTrue(any(config["mode"] == "unbounded" and config["rho_radians"] is None for config in protocol["configs"]))

    def test_only_hard_or_frozen_configs_get_deterministic_cosine_bound(self) -> None:
        protocol = dose.load_protocol(dose.DEFAULT_PROTOCOL_PATH)
        for config in protocol["configs"]:
            bound = dose.config_cosine_lower_bound(config, 0.339689)
            if config["mode"] in {"soft", "unbounded"}:
                self.assertIsNone(bound)
            else:
                self.assertGreaterEqual(bound, 0.0)

    def test_bounded_drift_cosine_bound_matches_triangle_inequality(self) -> None:
        bound = dose.bounded_drift_cosine_lower_bound(0.339689, 0.20)
        self.assertAlmostEqual(bound, 0.8543, delta=0.01)
        with self.assertRaises(ValueError):
            dose.bounded_drift_cosine_lower_bound(0.99, 0.20)

    def test_angular_distance_ignores_float32_self_dot_roundoff(self) -> None:
        self.assertEqual(dose.angular_distance_from_cosine(1.0 - 2.0**-23), 0.0)
        self.assertAlmostEqual(dose.angular_distance_from_cosine(math.cos(0.01)), 0.01, places=6)

    def test_soft_penalties_use_rho_and_eta_thresholds(self) -> None:
        self.assertAlmostEqual(dose.soft_anchor_penalty(1.0, 0.20, 5.0), 0.0)
        self.assertGreater(dose.soft_anchor_penalty(0.90, 0.20, 5.0), 0.0)
        self.assertAlmostEqual(dose.soft_residual_penalty([0.05, -0.10], 0.10, 5.0), 0.0)
        self.assertGreater(dose.soft_residual_penalty([0.20, -0.10], 0.10, 5.0), 0.0)

    def test_runner_records_projection_hits_and_anchor_groups(self) -> None:
        source = inspect.getsource(dose.train_dose_response_sae)
        self.assertIn("anchor_projection_hits", source)
        self.assertIn("residual_projection_hits", source)
        run_source = inspect.getsource(dose.run)
        self.assertIn("expected_groups", run_source)
        self.assertIn("validation_selected", run_source)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for tensor projection test")
    def test_residual_projection_respects_eta_and_unit_norm(self) -> None:
        import torch

        anchor = torch.tensor([1.0, 0.0, 0.0])
        weights = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        projected = dose.project_residual_decoder(weights, anchor, 0.10)
        self.assertLessEqual(float((anchor @ projected).abs().max()), 0.10 + 1e-6)
        self.assertTrue(torch.allclose(projected.norm(dim=0), torch.ones(2), atol=1e-6))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for frozen-anchor regression test")
    def test_frozen_non_axis_anchor_reports_zero_drift(self) -> None:
        import numpy as np

        samples = np.random.default_rng(19).standard_normal((48, 4))
        anchor = dose.make_signed_anchors(np.array([1.0, 2.0, 3.0, 4.0]))
        _, metrics = dose.train_dose_response_sae(
            samples,
            anchor,
            6,
            2,
            2,
            16,
            0.001,
            23,
            {"mode": "frozen", "rho_radians": 0.0, "eta": 0.0, "anchor_soft_penalty": 0.0, "residual_soft_penalty": 0.0},
            "cpu",
        )
        self.assertEqual(metrics["anchor_drift_radians"], 0.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for training smoke test")
    def test_train_modes_record_valid_constraints(self) -> None:
        import numpy as np
        import torch

        samples = np.random.default_rng(7).standard_normal((96, 4))
        anchor = dose.make_signed_anchors(np.array([1.0, 0.0, 0.0, 0.0]))
        for config in [
            {"mode": "frozen", "rho_radians": 0.0, "eta": 0.0, "anchor_soft_penalty": 0.0, "residual_soft_penalty": 0.0},
            {"mode": "hard", "rho_radians": 0.20, "eta": 0.10, "anchor_soft_penalty": 0.0, "residual_soft_penalty": 0.0},
            {"mode": "soft", "rho_radians": 0.20, "eta": 0.10, "anchor_soft_penalty": 5.0, "residual_soft_penalty": 5.0},
            {"mode": "unbounded", "rho_radians": None, "eta": None, "anchor_soft_penalty": 0.0, "residual_soft_penalty": 0.0},
        ]:
            model, metrics = dose.train_dose_response_sae(samples, anchor, 6, 2, 3, 32, 0.001, 11, config, "cpu")
            self.assertTrue(torch.isfinite(next(model.parameters())).all())
            self.assertGreaterEqual(metrics["anchor_projection_hits"], 0)
            self.assertGreaterEqual(metrics["residual_projection_hits"], 0)
            if config["mode"] == "hard":
                self.assertLessEqual(metrics["anchor_drift_radians"], 0.20 + 1e-5)
                self.assertLessEqual(metrics["residual_anchor_dot_max"], 0.10 + 1e-5)
            if config["mode"] == "frozen":
                self.assertLessEqual(metrics["anchor_drift_radians"], 1e-6)

    @unittest.skipUnless(
        (EXPERIMENTS / "results_dose_response" / "dose_response_raw.json").exists()
        and (EXPERIMENTS / "results_dose_response" / "dose_response_summary.json").exists(),
        "v5 complete artifact is generated after the matrix run",
    )
    def test_completed_artifact_passes_grouped_gate(self) -> None:
        artifact = json.loads((EXPERIMENTS / "results_dose_response" / "dose_response_raw.json").read_text())
        summary = json.loads((EXPERIMENTS / "results_dose_response" / "dose_response_summary.json").read_text())
        protocol = dose.load_protocol(dose.DEFAULT_PROTOCOL_PATH)
        dose.validate_rows(artifact, protocol)
        self.assertEqual(len(artifact), protocol["expected_rows"])
        self.assertEqual(summary["groups"], protocol["expected_groups"])
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["all_configs_have_group_rows"])

    def test_manuscript_contains_dose_response_boundary(self) -> None:
        manuscript = (EXPERIMENTS.parent / "paper" / "main.tex").read_text()
        self.assertIn("Dose-response and matched-compute baselines", manuscript)
        self.assertIn("tab:dose_response", manuscript)
        self.assertIn("mechanism-independent at this scale", manuscript)
        self.assertIn("0.734", manuscript)


if __name__ == "__main__":
    unittest.main()
