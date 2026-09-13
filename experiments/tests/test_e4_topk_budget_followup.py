#!/usr/bin/env python3
"""Contract tests for the post-hoc d32-k4 budget diagnostic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

import run_e4_topk_budget_followup as followup


class E4TopKBudgetFollowupContractTest(unittest.TestCase):
    def test_protocol_freezes_posthoc_scope_and_checkpoints(self) -> None:
        protocol = followup.load_protocol(followup.DEFAULT_PROTOCOL_PATH)
        self.assertEqual(protocol["experiment_id"], "E4-topk-d32k4-budget-followup")
        self.assertEqual(protocol["status"], "posthoc-frozen-before-execution")
        self.assertEqual(protocol["geometry"], ["axis", "dense"])
        self.assertEqual(protocol["steps"], [4000, 8000, 16000])
        self.assertEqual(protocol["seeds"]["base"], 20260901)
        self.assertEqual(protocol["seeds"]["count"], 8)
        self.assertEqual(protocol["topk"], {"id": "d32-k4", "dict_size": 32, "k": 4})

    def test_summary_requires_complete_grid_and_reports_budget_gate(self) -> None:
        protocol = followup.load_protocol(followup.DEFAULT_PROTOCOL_PATH)
        rows = []
        for seed in range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"]):
            for geometry in protocol["geometry"]:
                for step in protocol["steps"]:
                    cosine = 0.70 if geometry == "dense" and step == 4000 else 0.90
                    rows.append(
                        {
                            "seed": seed,
                            "geometry": geometry,
                            "steps": step,
                            "decoder_cosine": cosine,
                            "recovered": cosine >= protocol["training"]["cosine_threshold"],
                        }
                    )
        summary = followup.summarize(rows, protocol)
        self.assertTrue(summary["gates"]["complete"])
        self.assertTrue(summary["gates"]["budget_evidence"]["pass"])

        incomplete = rows[:-1]
        incomplete_summary = followup.summarize(incomplete, protocol)
        self.assertFalse(incomplete_summary["gates"]["complete"])
        self.assertFalse(incomplete_summary["gates"]["all_pass"])


if __name__ == "__main__":
    unittest.main()
