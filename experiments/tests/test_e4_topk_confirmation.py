#!/usr/bin/env python3
"""Contract tests for the held-out E4 TopK confirmation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

import run_e4_topk_confirmation as confirmation


class E4TopKConfirmationContractTest(unittest.TestCase):
    def test_frozen_protocol_has_independent_seed_grid(self) -> None:
        protocol = confirmation.load_protocol(confirmation.DEFAULT_PROTOCOL_PATH)
        self.assertEqual(protocol["experiment_id"], "E4-topk-heldout-confirmation")
        self.assertEqual(protocol["seeds"]["base"], 20260901)
        self.assertEqual(protocol["seeds"]["count"], 8)
        self.assertEqual(protocol["geometry"], ["axis", "dense"])
        self.assertEqual(
            [(c["id"], c["dict_size"], c["k"]) for c in protocol["topk_configs"]],
            [("d8-k1", 8, 1), ("d16-k2", 16, 2), ("d32-k4", 32, 4)],
        )

    def test_gate_requires_detector_split_and_no_axis_penalty(self) -> None:
        protocol = confirmation.load_protocol(confirmation.DEFAULT_PROTOCOL_PATH)
        rows = []
        for seed in range(protocol["seeds"]["base"], protocol["seeds"]["base"] + 8):
            rows.extend(
                [
                    {"kind": "detector", "seed": seed, "geometry": "axis", "label": 1.0, "cov": 1.0, "ind": 0.08},
                    {"kind": "detector", "seed": seed, "geometry": "dense", "label": 1.0, "cov": 1.0, "ind": 1.0},
                ]
            )
            for config in protocol["topk_configs"]:
                rows.extend(
                    [
                        {"kind": "training", "seed": seed, "geometry": "axis", "config_id": config["id"], "recovered": True, "decoder_cosine": 0.93},
                        {"kind": "training", "seed": seed, "geometry": "dense", "config_id": config["id"], "recovered": True, "decoder_cosine": 0.95},
                    ]
                )
        summary = confirmation.summarize(rows, protocol)
        self.assertTrue(summary["gates"]["all_pass"])

        failed = [dict(row) for row in rows]
        for row in failed:
            if row["kind"] == "training" and row["config_id"] == "d8-k1" and row["geometry"] == "axis":
                row["recovered"] = False
                row["decoder_cosine"] = 0.50
        failed_summary = confirmation.summarize(failed, protocol)
        self.assertFalse(failed_summary["gates"]["all_pass"])
        self.assertFalse(failed_summary["gates"]["per_config"]["d8-k1"]["pass"])

    def test_incomplete_rows_cannot_pass(self) -> None:
        protocol = confirmation.load_protocol(confirmation.DEFAULT_PROTOCOL_PATH)
        seed = protocol["seeds"]["base"]
        rows = confirmation.detector_rows_for_seed(seed, protocol)
        for config in protocol["topk_configs"]:
            for geometry in protocol["geometry"]:
                rows.append(
                    {
                        "kind": "training",
                        "seed": seed,
                        "geometry": geometry,
                        "config_id": config["id"],
                        "recovered": True,
                        "decoder_cosine": 0.95,
                    }
                )
        summary = confirmation.summarize(rows, protocol)
        self.assertFalse(summary["gates"]["complete"])
        self.assertFalse(summary["gates"]["all_pass"])

    def test_resume_rows_must_match_protocol(self) -> None:
        protocol = confirmation.load_protocol(confirmation.DEFAULT_PROTOCOL_PATH)
        row = confirmation.detector_rows_for_seed(protocol["seeds"]["base"], protocol)[0]
        row["lambda_eff"] = 0.1
        with self.assertRaises(ValueError):
            confirmation.validate_rows([row], protocol)


if __name__ == "__main__":
    unittest.main()
