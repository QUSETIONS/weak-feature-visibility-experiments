"""Regression tests for the residual bridge training/data contract."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]


class BridgeTrainingContractTest(unittest.TestCase):
    def test_train_loop_has_one_optimizer_step(self) -> None:
        tree = ast.parse((EXPERIMENTS / "run_gpt2_bridge.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_sae")
        calls = [
            node for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "opt"
            and node.func.attr == "step"
        ]
        self.assertEqual(len(calls), 1)

    def test_text_loader_fails_closed(self) -> None:
        source = (EXPERIMENTS / "run_gpt2_bridge.py").read_text()
        self.assertNotIn("fallback random digit strings", source)
        self.assertIn("synthetic text fallback is forbidden", source)


if __name__ == "__main__":
    unittest.main()
