"""Contract tests for objective-agnostic reconstruction certification."""

import json
import math
import sys
import unittest
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

from run_reconstruction_certificate import (
    build_artifact_metadata,
    certificate_bound,
    certify_decoder_output,
    certify_projection,
    certify_validation,
    run_validation_case,
    validation_tolerance,
    make_cases,
    validate_case,
)


class ReconstructionCertificateTest(unittest.TestCase):
    def test_axis_and_dense_are_certified_by_same_risk_gap(self) -> None:
        cases = make_cases()
        axis = cases["axis"]
        dense = cases["dense"]
        for case in cases.values():
            validate_case(case)
            result = certify_projection(
                case["target_cosine"],
                case["signal_strength"],
                case["risk_excess"],
                case["noise_variance"],
            )
            self.assertTrue(result["certified"])
            self.assertLessEqual(result["observed_sin"], result["angle_sin_bound"])
        self.assertEqual(axis["detector_C_ind"], 0.0)
        self.assertGreater(dense["detector_C_ind"], axis["detector_C_ind"])

    def test_certificate_fails_when_risk_gap_is_too_small(self) -> None:
        result = certify_projection(0.0, 1.0, 0.01, 1.0)
        self.assertFalse(result["certified"])
        self.assertLess(result["angle_sin_bound"], 1.0)

    def test_uninformative_projection_bound_is_not_certified(self) -> None:
        result = certify_projection(0.0, 1.0, 1.01, 1.0)
        self.assertFalse(result["certified"])

    def test_bound_matches_projection_identity(self) -> None:
        bound = certificate_bound(signal_strength=0.8, risk_excess=0.2, noise_variance=1.0)
        self.assertAlmostEqual(bound, 0.25)

    def test_finite_sample_certificate_is_objective_agnostic(self) -> None:
        tolerance = validation_tolerance(1.0, 1.0, 100000, 0.05)
        result = certify_validation(
            empirical_excess=1.0 - 0.9**2,
            signal_strength=1.0,
            noise_variance=1.0,
            sample_count=100000,
            delta=0.05,
            cosine_threshold=0.80,
            independent_validation=True,
        )
        self.assertGreater(tolerance, 0.0)
        self.assertTrue(result["certified"])
        self.assertGreaterEqual(result["cosine_lower_bound"], 0.80)
        self.assertAlmostEqual(result["confidence"], 0.95)

    def test_finite_sample_certificate_rejects_unrecovered_projection(self) -> None:
        result = certify_validation(
            empirical_excess=1.0,
            signal_strength=1.0,
            noise_variance=1.0,
            sample_count=1000000,
            delta=0.05,
            cosine_threshold=0.80,
            independent_validation=True,
        )
        self.assertFalse(result["certified"])

    def test_certificate_requires_an_independent_validation_split(self) -> None:
        with self.assertRaises(ValueError):
            certify_validation(
                empirical_excess=0.1,
                signal_strength=1.0,
                noise_variance=1.0,
                sample_count=100000,
                delta=0.05,
                cosine_threshold=0.80,
                independent_validation=False,
            )

    def test_negative_empirical_gap_is_allowed_after_concentration_correction(self) -> None:
        result = certify_validation(
            empirical_excess=-0.01,
            signal_strength=1.0,
            noise_variance=1.0,
            sample_count=100000,
            delta=0.05,
            cosine_threshold=0.80,
            independent_validation=True,
        )
        self.assertTrue(result["certified"])
        self.assertGreater(result["tolerance"], 0.01)
        with self.assertRaises(ValueError):
            certify_validation(
                empirical_excess=-1.0,
                signal_strength=1.0,
                noise_variance=1.0,
                sample_count=100000,
                delta=0.05,
                cosine_threshold=0.80,
                independent_validation=True,
            )

    def test_rank_controls_finite_sample_tolerance(self) -> None:
        rank_one = validation_tolerance(1.0, 1.0, 10000, 0.05, decoder_rank=1)
        rank_four = validation_tolerance(1.0, 1.0, 10000, 0.05, decoder_rank=4)
        t = math.log(2.0 / 0.05)
        self.assertAlmostEqual(rank_one, 2.0 * 2.0 * (math.sqrt(4.0 * t / 10000.0) + t / 10000.0))
        self.assertGreater(rank_four, rank_one)
        with self.assertRaises(ValueError):
            validation_tolerance(1.0, 1.0, 10000, 0.05, decoder_rank=0)

    def test_independent_synthetic_validation_covers_axis_and_dense(self) -> None:
        cases = make_cases()
        for name, case in cases.items():
            result = run_validation_case(name, case, sample_count=20000, seed=20260831)
            self.assertTrue(result["candidate_frozen_before_validation"])
            self.assertTrue(result["independent_validation"])
            self.assertGreaterEqual(result["cosine_lower_bound"], 0.80)

    def test_artifact_tolerances_match_current_projector_bound(self) -> None:
        artifact = json.loads((EXPERIMENTS / "results_reconstruction_certificate" / "certificate.json").read_text())
        axis = artifact["cases"]["axis"]
        held_out = axis["held_out_certificate"]
        confidence = held_out["confidence"]
        expected = validation_tolerance(
            axis["case"]["signal_strength"],
            axis["case"]["noise_variance"],
            artifact["validation_split"]["sample_count"],
            1.0 - confidence,
            decoder_rank=axis["case"]["decoder_rank"],
        )
        self.assertAlmostEqual(held_out["tolerance"], expected)

    def test_artifact_marks_synthetic_demo_as_non_evidence(self) -> None:
        metadata = build_artifact_metadata()
        self.assertTrue(metadata["synthetic_validation_demo"])
        self.assertFalse(metadata["evidence_eligible"])
        self.assertFalse(metadata["multi_objective_training"])

    def test_arbitrary_decoder_output_gets_the_same_certificate(self) -> None:
        result = certify_decoder_output(
            signal_strength=1.0,
            empirical_excess=0.05,
            noise_variance=1.0,
            sample_count=100000,
            delta=0.05,
            cosine_threshold=0.80,
            decoder_rank=1,
            independent_validation=True,
        )
        self.assertTrue(result["certified"])
        self.assertEqual(result["certificate_scope"], "decoder_output")

    def test_certificate_does_not_depend_on_objective_label(self) -> None:
        cases = make_cases()
        labels = ["TopK", "L1 sparse", "nonnegative sparse", "Gated-like"]
        for label in labels:
            case = dict(cases["axis"], objective=label)
            result = certify_projection(
                case["target_cosine"],
                case["signal_strength"],
                case["risk_excess"],
                case["noise_variance"],
            )
            self.assertTrue(result["certified"])

    def test_rank_cannot_exceed_ambient_dimension(self) -> None:
        case = dict(make_cases()["axis"], decoder_rank=9, dimension=8)
        with self.assertRaises(ValueError):
            validate_case(case)


if __name__ == "__main__":
    unittest.main()
