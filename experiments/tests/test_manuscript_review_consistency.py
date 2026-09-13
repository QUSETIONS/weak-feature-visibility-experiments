"""Consistency contracts pinned by the 2026-09-05 adversarial review.

Every test pins one factual correction verified against
experiments/results_dose_response/dose_response_raw.json:
Welch t = 1.50 (loss mean not significant at n=8), soft-arm maximum leak
0.10125, unbounded worst-group cosine 0.990076 (beats four of the seven
constrained arms), and the canonical 7-point lambda grid
{0.30, 0.34, 0.38, 0.42, 0.48, 0.55, 0.65} used by the runners.
"""

from pathlib import Path
import unittest

EXPERIMENTS = Path(__file__).resolve().parents[1]
MAIN_TEX = EXPERIMENTS.parent / "paper" / "main.tex"


class ManuscriptReviewConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = MAIN_TEX.read_text()

    def test_leftover_credentials_sentence_removed(self) -> None:
        self.assertNotIn("Do not redistribute remote-host credentials", self.text)

    def test_dose_response_design_states_actual_nine_configs(self) -> None:
        # The old wording implies a crossed 4x2 hard grid = 12 mechanisms;
        # the real design is 2 frozen + 5 hard + 1 soft + 1 unbounded = 9.
        self.assertNotIn(
            "$\\rho\\in\\{0.05,0.10,0.20,0.40\\}$ with $\\eta\\in\\{0,0.10\\}$",
            self.text,
        )
        self.assertIn("$\\eta=0.10$ for $\\rho\\in\\{0.05,0.10,0.40\\}$", self.text)
        self.assertIn("$(\\rho,\\eta)\\in\\{(0.20,0),(0.20,0.10)\\}$", self.text)
        self.assertIn("$9\\times 8=72$ rows", self.text)

    def test_leak_cap_claim_true_for_soft_arm(self) -> None:
        self.assertNotIn("caps leakage at $\\le 0.101$", self.text)
        self.assertIn("0.10125", self.text)
        self.assertIn("at or below $0.102$", self.text)

    def test_lambda_grid_enumerated_not_ellided(self) -> None:
        self.assertNotIn("$\\{0.30,\\ldots,0.65\\}$", self.text)
        self.assertIn("$\\{0.30,0.34,0.38,0.42,0.48,0.55,0.65\\}$", self.text)

    def test_worst_group_cosine_claim_not_float_fragile(self) -> None:
        self.assertNotIn("stays above $0.9788$", self.text)
        self.assertIn("at or above $0.9788$", self.text)

    def test_monotonicity_claim_scoped_to_eta_series(self) -> None:
        self.assertNotIn(
            "tighter trust regions monotonically improve the worst-group cosine",
            self.text,
        )
        self.assertIn("within the $\\eta=0.10$ hard series", self.text)

    def test_unbounded_cosine_comparison_disclosed(self) -> None:
        self.assertIn("beats four of the seven constrained arms", self.text)

    def test_loss_contrast_significance_reported(self) -> None:
        self.assertIn("Welch $t", self.text)
        self.assertIn("not significant", self.text)

    def test_artifact_evidence_scope_disclosed(self) -> None:
        self.assertIn("evidence\\_eligible=false", self.text)


if __name__ == "__main__":
    unittest.main()
