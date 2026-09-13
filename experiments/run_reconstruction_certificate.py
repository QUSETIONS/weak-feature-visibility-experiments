"""Objective-agnostic reconstruction certificates for a target direction.

The certificate is conditional: an objective/optimizer may produce any decoder
subspace, but a held-out excess-risk bound certifies its target alignment.  No
training run is claimed by this analytic runner.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from typing import Any


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "results_reconstruction_certificate"
CLAIM = (
    "For an isotropic-noise signal model, any decoder subspace whose held-out "
    "reconstruction excess risk is certified has an objective-independent "
    "target-alignment bound."
)


def certificate_bound(signal_strength: float, risk_excess: float, noise_variance: float) -> float:
    """Return the certified squared sine bound; noise cancels at fixed rank."""
    del noise_variance
    if signal_strength <= 0 or risk_excess < 0:
        raise ValueError("signal strength must be positive and excess risk nonnegative")
    return risk_excess / signal_strength


def validation_tolerance(
    signal_strength: float,
    noise_variance: float,
    sample_count: int,
    delta: float,
    decoder_rank: int = 1,
) -> float:
    """Return a Gaussian quadratic-form deviation envelope.

    The compared projections have the same rank, so isotropic noise cancels in
    expectation.  It still controls finite-sample fluctuations.
    """
    if signal_strength <= 0 or noise_variance < 0:
        raise ValueError("signal strength must be positive and noise variance nonnegative")
    if sample_count < 1 or not 0.0 < delta < 1.0 or decoder_rank < 1:
        raise ValueError("sample count/rank must be positive and delta must lie in (0, 1)")
    t = math.log(2.0 / delta)
    scale = noise_variance + signal_strength
    rank_term = math.sqrt(float(decoder_rank))
    return 2.0 * scale * (rank_term * math.sqrt(4.0 * t / sample_count) + t / sample_count)


def certify_projection(
    target_cosine: float,
    signal_strength: float,
    risk_excess: float,
    noise_variance: float,
) -> dict[str, Any]:
    """Compare observed target angle with the risk-derived certificate."""
    if not -1.0 <= target_cosine <= 1.0:
        raise ValueError("target cosine must lie in [-1, 1]")
    bound_sq = certificate_bound(signal_strength, risk_excess, noise_variance)
    observed_sin = math.sqrt(max(0.0, 1.0 - target_cosine * target_cosine))
    informative = 0.0 <= bound_sq <= 1.0
    angle_sin_bound = math.sqrt(min(1.0, max(0.0, bound_sq)))
    return {
        "target_cosine": target_cosine,
        "observed_sin": observed_sin,
        "angle_sin_bound": angle_sin_bound,
        "bound_squared": bound_sq,
        "informative": informative,
        "certified": informative and observed_sin <= angle_sin_bound + 1e-12,
    }


def certify_validation(
    empirical_excess: float,
    signal_strength: float,
    noise_variance: float,
    sample_count: int,
    delta: float,
    decoder_rank: int = 1,
    cosine_threshold: float = 0.80,
    independent_validation: bool = False,
) -> dict[str, Any]:
    """Certify alignment from held-out excess risk with probability 1-delta."""
    if not independent_validation:
        raise ValueError("the risk estimate must come from an independent validation split")
    if not 0.0 < cosine_threshold <= 1.0:
        raise ValueError("cosine threshold must lie in (0, 1]")
    tolerance = validation_tolerance(
        signal_strength,
        noise_variance,
        sample_count,
        delta,
        decoder_rank=decoder_rank,
    )
    if empirical_excess < -tolerance:
        raise ValueError("empirical excess risk is below its concentration envelope")
    bound_sq = min(1.0, max(0.0, (empirical_excess + tolerance) / signal_strength))
    cosine_lower_bound = math.sqrt(max(0.0, 1.0 - bound_sq))
    return {
        "empirical_excess": empirical_excess,
        "tolerance": tolerance,
        "angle_sin_bound": math.sqrt(bound_sq),
        "bound_squared": bound_sq,
        "cosine_lower_bound": cosine_lower_bound,
        "confidence": 1.0 - delta,
        "certified": cosine_lower_bound >= cosine_threshold,
        "independent_validation": independent_validation,
    }


def build_artifact_metadata() -> dict[str, Any]:
    """Declare the synthetic demo's evidence boundary explicitly."""
    return {
        "synthetic_validation_demo": True,
        "evidence_eligible": False,
        "multi_objective_training": False,
        "analytic_identity_included": True,
        "held_out_validation_included": True,
    }


def certify_decoder_output(
    signal_strength: float,
    empirical_excess: float,
    noise_variance: float,
    sample_count: int,
    delta: float,
    cosine_threshold: float,
    decoder_rank: int,
    independent_validation: bool,
) -> dict[str, Any]:
    """Certify any frozen decoder output without using its objective label."""
    result = certify_validation(
        empirical_excess=empirical_excess,
        signal_strength=signal_strength,
        noise_variance=noise_variance,
        sample_count=sample_count,
        delta=delta,
        decoder_rank=decoder_rank,
        cosine_threshold=cosine_threshold,
        independent_validation=independent_validation,
    )
    result["certificate_scope"] = "decoder_output"
    return result


def make_cases() -> dict[str, dict[str, Any]]:
    """Return axis/dense examples with the same objective-independent gap."""
    return {
        "axis": {
            "target_cosine": 0.95,
            "signal_strength": 1.0,
            "risk_excess": 1.0 - 0.95**2,
            "noise_variance": 1.0,
            "dimension": 8,
            "decoder_rank": 1,
            "detector_C_ind": 0.0,
            "objective": "frozen candidate A",
        },
        "dense": {
            "target_cosine": 0.90,
            "signal_strength": 1.0,
            "risk_excess": 1.0 - 0.90**2,
            "noise_variance": 1.0,
            "dimension": 8,
            "decoder_rank": 1,
            "detector_C_ind": 0.875,
            "objective": "frozen candidate B",
        },
    }


def target_direction(case_name: str, dimension: int) -> np.ndarray:
    """Return axis or dense target direction for the synthetic validation demo."""
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    if case_name == "axis":
        direction = np.zeros(dimension)
        direction[0] = 1.0
        return direction
    if case_name == "dense":
        return np.full(dimension, 1.0 / math.sqrt(dimension))
    raise ValueError(f"unknown certificate case: {case_name}")


def candidate_direction(target: np.ndarray, target_cosine: float) -> np.ndarray:
    """Construct a frozen candidate with the requested target cosine."""
    if not 0.0 < target_cosine <= 1.0:
        raise ValueError("target cosine must lie in (0, 1]")
    probe = np.zeros_like(target)
    probe[0] = 1.0
    if abs(float(np.dot(probe, target))) > 0.9:
        probe[:] = 0.0
        probe[1] = 1.0
    orthogonal = probe - np.dot(probe, target) * target
    orthogonal /= np.linalg.norm(orthogonal)
    return target_cosine * target + math.sqrt(1.0 - target_cosine**2) * orthogonal


def empirical_excess_risk(
    samples: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
) -> float:
    """Estimate candidate risk minus the rank-matched target-containing oracle."""
    candidate_projection = samples @ candidate
    target_projection = samples @ target
    candidate_residual = np.sum(samples * samples, axis=1) - candidate_projection**2
    oracle_residual = np.sum(samples * samples, axis=1) - target_projection**2
    return float(np.mean(candidate_residual - oracle_residual))


def run_validation_case(
    case_name: str,
    case: dict[str, Any],
    sample_count: int = 100_000,
    seed: int = 20260831,
    delta: float = 0.05,
    cosine_threshold: float = 0.80,
) -> dict[str, Any]:
    """Run an independent synthetic validation for one frozen candidate."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    target = target_direction(case_name, int(case["dimension"]))
    candidate = candidate_direction(target, float(case["target_cosine"]))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, math.sqrt(float(case["noise_variance"])), size=(sample_count, len(target)))
    signal = rng.normal(0.0, math.sqrt(float(case["signal_strength"])), size=sample_count)
    samples = noise + signal[:, None] * target[None, :]
    empirical_excess = empirical_excess_risk(samples, candidate, target)
    certificate = certify_validation(
        empirical_excess=empirical_excess,
        signal_strength=float(case["signal_strength"]),
        noise_variance=float(case["noise_variance"]),
        sample_count=sample_count,
        delta=delta,
        decoder_rank=int(case["decoder_rank"]),
        cosine_threshold=cosine_threshold,
        independent_validation=True,
    )
    certificate["observed_target_cosine_for_audit_only"] = float(np.dot(candidate, target))
    certificate["validation_seed"] = seed
    certificate["candidate_frozen_before_validation"] = True
    return certificate


def validate_case(case: dict[str, Any]) -> None:
    required = {
        "target_cosine",
        "signal_strength",
        "risk_excess",
        "noise_variance",
        "detector_C_ind",
        "objective",
    }
    missing = required - case.keys()
    if missing:
        raise ValueError(f"certificate case missing fields: {sorted(missing)}")
    if case["signal_strength"] <= 0 or case["risk_excess"] < 0 or case["noise_variance"] < 0:
        raise ValueError("certificate scales must be nonnegative with positive signal")
    if case.get("dimension", 0) < 2 or case.get("decoder_rank", 0) < 1:
        raise ValueError("certificate dimension/rank must support a nontrivial projection")
    if case["decoder_rank"] > case["dimension"]:
        raise ValueError("decoder rank cannot exceed ambient dimension")
    if not -1.0 <= case["target_cosine"] <= 1.0:
        raise ValueError("certificate target cosine must lie in [-1, 1]")
    if not isinstance(case["objective"], str) or not case["objective"].strip():
        raise ValueError("certificate objective label must be nonempty")
    result = certify_projection(
        case["target_cosine"],
        case["signal_strength"],
        case["risk_excess"],
        case["noise_variance"],
    )
    expected = 1.0 - case["target_cosine"] ** 2
    if not math.isclose(result["bound_squared"], expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("risk certificate does not match the projection identity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    cases = make_cases()
    for case in cases.values():
        validate_case(case)
    results = {
        name: {
            "case": case,
            "analytic_certificate": certify_projection(
                case["target_cosine"],
                case["signal_strength"],
                case["risk_excess"],
                case["noise_variance"],
            ),
            "held_out_certificate": run_validation_case(name, case),
        }
        for name, case in cases.items()
    }
    args.out.mkdir(parents=True, exist_ok=True)
    raw = {
        "claim_tested": CLAIM,
        "status": "certified_under_assumptions",
        "assumptions": [
            "held-out risk excess is valid for the same-rank decoder projection",
            "signal covariance is rank one with positive strength",
            "noise is isotropic and independent of the target direction",
        ],
        **build_artifact_metadata(),
        "validation_split": {
            "independent": True,
            "sample_count": 100000,
            "seed": 20260831,
            "candidate_frozen_before_validation": True,
        },
        "cases": results,
    }
    (args.out / "certificate.json").write_text(json.dumps(raw, indent=2) + "\n")
    summary = {
        "claim_tested": CLAIM,
        "status": "certified_under_assumptions",
        "case_count": len(results),
        "objective_agnostic": True,
        "detector_geometry_agnostic": True,
        **build_artifact_metadata(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
