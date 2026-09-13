"""Contract tests for the v7 target-free readout audit."""

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
    import run_spectral_anchor_readout_audit as audit
except ModuleNotFoundError:
    audit = None


class SpectralReadoutAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        if audit is None:
            self.fail("v7 readout audit runner is missing")

    def test_protocol_freezes_v6_conditions_and_three_readouts(self) -> None:
        protocol = audit.load_protocol(audit.DEFAULT_PROTOCOL_PATH)
        self.assertEqual(protocol["experiment_id"], "spectral-anchor-sae-v7-readout-audit")
        self.assertEqual(protocol["source_protocol"], "spectral-anchor-sae-v6-matched-warm-start")
        self.assertEqual(protocol["expected_rows"], 48)
        self.assertEqual(protocol["comparison"]["expected_pairs"], 48)
        self.assertTrue(protocol["comparison"]["one_model_per_pair"])
        self.assertTrue(protocol["comparison"]["readouts_share_final_model"])
        self.assertEqual([item["id"] for item in protocol["readouts"]], ["usage-atom", "spectral-linked-atom", "signed-pair-span"])
        self.assertEqual(protocol["certificate"]["family_count"], 144)
        self.assertTrue(protocol["evaluation"]["target_is_never_passed_to_training"])

    def test_readout_selection_apis_are_target_free(self) -> None:
        for function_name in ["select_usage_atom", "select_spectral_linked_atom", "signed_pair_basis"]:
            parameters = inspect.signature(getattr(audit, function_name)).parameters
            self.assertNotIn("target", parameters)

    def test_subspace_alignment_and_projector_risk_are_rank_aware(self) -> None:
        import numpy as np

        basis = np.eye(3, 2)
        self.assertAlmostEqual(audit.subspace_cosine(basis, np.array([1.0, 0.0, 0.0])), 1.0)
        self.assertAlmostEqual(audit.subspace_cosine(basis, np.array([0.0, 0.0, 1.0])), 0.0)
        samples = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        target = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(audit.empirical_excess_risk_projector(samples, basis, target), 0.0)
        self.assertEqual(audit.subspace_rank(basis), 2)

    def test_validator_requires_shared_model_and_all_readouts(self) -> None:
        source = inspect.getsource(audit.validate_rows)
        self.assertIn("readouts", source)
        self.assertIn("readout_selection_is_training_only", source)
        self.assertIn("shared_model_id", source)
        self.assertIn("family_count", source)
        self.assertIn("validation_selected", source)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for readout selector test")
    def test_spectral_linked_atom_uses_training_anchor_not_target(self) -> None:
        import numpy as np
        import torch

        model = audit.build_sae(3, 4, 2, 7, "spectral-warm-start", np.array([1.0, 0.0, 0.0]), "cpu")
        index, vector = audit.select_spectral_linked_atom(model, np.array([1.0, 0.0, 0.0]), "cpu")
        self.assertIn(index, range(4))
        self.assertGreater(float(torch.tensor(vector).dot(torch.tensor([1.0, 0.0, 0.0])).abs()), 0.99)

    @unittest.skipUnless(
        (EXPERIMENTS / "results_spectral_anchor_readout_audit" / "readout_raw.json").exists()
        and (EXPERIMENTS / "results_spectral_anchor_readout_audit" / "readout_summary.json").exists(),
        "v7 artifact is generated after the readout audit run",
    )
    def test_completed_artifact_has_all_shared_model_readouts(self) -> None:
        rows = json.loads((EXPERIMENTS / "results_spectral_anchor_readout_audit" / "readout_raw.json").read_text())
        summary = json.loads((EXPERIMENTS / "results_spectral_anchor_readout_audit" / "readout_summary.json").read_text())
        protocol = audit.load_protocol(audit.DEFAULT_PROTOCOL_PATH)
        audit.validate_rows(rows, protocol)
        self.assertEqual(len(rows), protocol["expected_rows"])
        self.assertEqual(summary["pairs"], protocol["comparison"]["expected_pairs"])
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["shared_model_integrity"])


if __name__ == "__main__":
    unittest.main()
