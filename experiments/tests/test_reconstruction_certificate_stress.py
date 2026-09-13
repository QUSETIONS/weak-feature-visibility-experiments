"""Contract tests for the frozen cross-objective certificate stress test."""

import inspect
import json
import sys
import unittest
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

import run_reconstruction_certificate_stress as stress

from run_reconstruction_certificate_stress import (
    DEFAULT_PROTOCOL_PATH,
    certificate_matches_gate,
    expected_row_keys,
    load_protocol,
    validate_rows,
)


class ReconstructionCertificateStressTest(unittest.TestCase):
    def test_training_enforces_declared_decoder_column_normalization(self) -> None:
        source = inspect.getsource(stress.train_sae)
        self.assertIn("norm(dim=0", source)
        self.assertIn("copy_(", source)

    def test_protocol_freezes_corrected_projector_concentration(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
        self.assertEqual(protocol["experiment_id"], "reconstruction-certificate-objective-stress-v3")
        self.assertEqual(protocol["certificate"]["projector_difference_rank_factor"], 2)

    def test_protocol_freezes_objectives_capacities_and_seeds(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
        self.assertEqual([item["id"] for item in protocol["objectives"]], ["mse", "mse_l1", "mse_l2"])
        self.assertEqual([item["id"] for item in protocol["capacities"]], ["d16-k2", "d32-k4"])
        self.assertEqual(protocol["geometry"], ["axis", "dense"])
        self.assertEqual(protocol["seeds"]["count"], 4)
        self.assertEqual(protocol["expected_rows"], 48)

    def test_expected_keys_are_complete(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
        keys = expected_row_keys(protocol)
        self.assertEqual(len(keys), 48)
        self.assertEqual(len(keys), len(set(keys)))

    def test_incomplete_rows_are_rejected(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
        with self.assertRaises(ValueError):
            validate_rows([], protocol)

    def test_certificate_gate_is_recomputed_not_trusted(self) -> None:
        self.assertTrue(certificate_matches_gate({"cosine_lower_bound": 0.81, "certified": True}, 0.80))
        self.assertFalse(certificate_matches_gate({"cosine_lower_bound": 0.79, "certified": True}, 0.80))
        self.assertFalse(certificate_matches_gate({"cosine_lower_bound": 0.81, "certified": False}, 0.80))

    def test_resume_rows_cannot_change_frozen_protocol(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
        row = {
            "objective_id": "mse",
            "capacity_id": "d16-k2",
            "geometry": "axis",
            "seed": protocol["seeds"]["base"],
            "train_seed": 0,
            "data_seed": 0,
            "validation_seed": 0,
            "candidate_selection": "validation_cosine",
            "certificate": {"certified": True},
        }
        with self.assertRaises(ValueError):
            validate_rows([row], protocol)


if __name__ == "__main__":
    unittest.main()
