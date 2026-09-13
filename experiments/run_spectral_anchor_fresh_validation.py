"""Fresh-validation confirmation for the v7 signed-pair readout."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import run_reconstruction_certificate as certificate
import run_reconstruction_certificate_stress as stress
import run_spectral_anchor_readout_audit as audit
import run_spectral_warm_start_matched as warm

ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL_PATH = ROOT / "spectral_anchor_fresh_validation_protocol.json"
DEFAULT_OUTPUT_PATH = ROOT / "results_spectral_anchor_fresh_validation"
RAW_NAME = "fresh_validation_raw.json"
SUMMARY_NAME = "fresh_validation_summary.json"

estimate_spectral_anchor = warm.estimate_spectral_anchor
signed_pair_basis = audit.signed_pair_basis


def validate_protocol(protocol: dict[str, Any]) -> None:
    required = {
        "experiment_id", "status", "purpose", "evidence_role", "calibration_disclosure",
        "feature", "geometry", "objectives", "capacities", "warm_start", "primary_readout",
        "seeds", "training", "certificate", "comparison", "expected_rows",
        "evidence_status",
    }
    missing = required - protocol.keys()
    if missing:
        raise ValueError(f"v8 protocol missing keys: {sorted(missing)}")
    if protocol["experiment_id"] != "spectral-anchor-sae-v8-fresh-validation-confirmation":
        raise ValueError("unexpected v8 protocol id")
    if protocol["status"] != "frozen-before-execution":
        raise ValueError("v8 protocol must be frozen before execution")
    disclosure = protocol["calibration_disclosure"]
    if disclosure.get("source_experiment") != "spectral-anchor-sae-v7-readout-audit":
        raise ValueError("v8 source experiment is not v7")
    if disclosure.get("sample_size_chosen_after_source_experiment") is not True or disclosure.get("v7_is_exploratory") is not True or disclosure.get("v8_validation_samples_are_fresh") is not True:
        raise ValueError("v8 calibration disclosure is incomplete")
    feature = protocol["feature"]
    if feature.get("m") != 8 or float(feature.get("signal_strength", 0.0)) <= 0.0:
        raise ValueError("v8 feature settings are invalid")
    if protocol["geometry"] != ["axis", "dense"]:
        raise ValueError("v8 geometry is not frozen")
    if [item["id"] for item in protocol["objectives"]] != ["mse", "mse_l1", "mse_l2"]:
        raise ValueError("v8 objective order is not frozen")
    if [item["id"] for item in protocol["capacities"]] != ["d16-k2", "d32-k4"]:
        raise ValueError("v8 capacity order is not frozen")
    warm_config = protocol["warm_start"]
    if warm_config.get("variant_id") != "spectral-warm-start" or warm_config.get("initialization") != "training_spectral_signed_pair" or warm_config.get("post_initialization_constraints") != "none":
        raise ValueError("v8 warm-start condition changed")
    primary = protocol["primary_readout"]
    if primary.get("id") != "signed-pair-span" or primary.get("kind") != "subspace" or primary.get("slots") != [0, 1]:
        raise ValueError("v8 primary readout must be the fixed signed pair span")
    if not 0.0 < float(primary.get("span_relative_singular_value_tolerance", 0.0)) < 1.0:
        raise ValueError("v8 span rank tolerance is invalid")
    seeds = protocol["seeds"]
    if int(seeds.get("base", 0)) != 20261101 or int(seeds.get("count", 0)) != 4:
        raise ValueError("v8 training seeds must match v6")
    if int(seeds["validation_rng_offset"]) == int(seeds["v7_validation_rng_offset"]):
        raise ValueError("v8 validation seeds must be fresh relative to v7")
    training = protocol["training"]
    if int(training.get("N_train", 0)) != 20000 or int(training.get("N_validation", 0)) != 500000 or int(training.get("steps", 0)) != 4000:
        raise ValueError("v8 sample/training budget is not frozen")
    if training.get("warm_start_uses_training_split_only") is not True or training.get("validation_is_never_used_for_training_or_selection") is not True or training.get("post_initialization_constraints") != "none":
        raise ValueError("v8 training selection/constraint contract failed")
    cert = protocol["certificate"]
    if int(cert.get("family_count", 0)) != 48 or int(cert.get("span_rank_max", 0)) != 2 or not 0.0 < float(cert.get("delta", 0.0)) < 1.0:
        raise ValueError("v8 certificate family/rank settings are invalid")
    if cert.get("independent_validation") is not True or float(cert.get("noise_variance", 0.0)) <= 0.0:
        raise ValueError("v8 certificate must use independent validation")
    if not 0.0 < float(cert.get("cosine_threshold", 0.0)) <= 1.0:
        raise ValueError("v8 certificate cosine threshold is invalid")
    comparison = protocol["comparison"]
    if comparison.get("expected_models") != 48 or comparison.get("one_primary_readout_per_model") is not True or comparison.get("fresh_validation_only_change_from_v7_training") is not True:
        raise ValueError("v8 comparison contract is invalid")
    expected = len(protocol["objectives"]) * len(protocol["capacities"]) * len(protocol["geometry"]) * int(seeds["count"])
    if int(protocol["expected_rows"]) != expected or expected != 48:
        raise ValueError("v8 expected row count is inconsistent")


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    validate_protocol(protocol)
    return protocol


def train_warm_start_sae(
    samples: np.ndarray,
    training_spectral_anchor: np.ndarray,
    dict_size: int,
    k: int,
    steps: int,
    batch: int,
    lr: float,
    seed: int,
    objective: dict[str, Any],
    device: str,
) -> tuple[Any, dict[str, Any]]:
    """Train the v6 spectral warm-start with no post-initialization constraints."""
    return warm.train_matched_sae(
        samples,
        training_spectral_anchor,
        dict_size,
        k,
        steps,
        batch,
        lr,
        seed,
        {**objective, "variant_id": "spectral-warm-start"},
        device,
    )


def _target_cosine(basis: np.ndarray, target: np.ndarray) -> float:
    return warm.clamp_cosine(float(np.linalg.norm(np.asarray(basis).T @ np.asarray(target))))


def expected_keys(protocol: dict[str, Any]) -> set[tuple[str, str, str, int]]:
    seeds = range(int(protocol["seeds"]["base"]), int(protocol["seeds"]["base"]) + int(protocol["seeds"]["count"]))
    return {
        (objective["id"], capacity["id"], geometry, seed)
        for objective in protocol["objectives"]
        for capacity in protocol["capacities"]
        for geometry in protocol["geometry"]
        for seed in seeds
    }


def row_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (row["objective_id"], row["capacity_id"], row["geometry"], int(row["seed"]))


def validate_rows(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    validate_protocol(protocol)
    expected = expected_keys(protocol)
    if len(rows) != int(protocol["expected_rows"]):
        raise ValueError("v8 artifact has incomplete rows")
    seen: set[tuple[str, str, str, int]] = set()
    familywise_delta = float(protocol["certificate"]["delta"]) / int(protocol["certificate"]["family_count"])
    confidence = 1.0 - familywise_delta
    for row in rows:
        key = row_key(row)
        if key in seen or key not in expected:
            raise ValueError(f"duplicate or unexpected v8 row: {key}")
        seen.add(key)
        shared_model_id = ":".join(map(str, key))
        if row.get("shared_model_id") != shared_model_id or row.get("primary_readout") != "signed-pair-span":
            raise ValueError("v8 shared model or primary readout changed")
        if row.get("fresh_validation") is not True or row.get("validation_selected") is not False or row.get("target_used_for_training") is not False:
            raise ValueError("v8 fresh validation or target-free training contract failed")
        if int(row.get("projection_hits", -1)) != 0:
            raise ValueError("v8 post-initialization projection was used")
        if int(row.get("validation_seed")) == int(protocol["seeds"]["v7_validation_rng_offset"]):
            raise ValueError("v8 reused a v7 validation seed")
        span = row.get("span_readout")
        if not isinstance(span, dict) or span.get("selection") != "fixed_training_spectral_warm_start_pair_slots" or span.get("kind") != "subspace":
            raise ValueError("v8 span readout metadata is invalid")
        rank = int(span.get("span_rank", 0))
        if rank < 1 or rank > int(protocol["certificate"]["span_rank_max"]):
            raise ValueError("v8 span_rank is outside the frozen range")
        for field in ("target_cosine", "empirical_excess"):
            if not math.isfinite(float(span[field])):
                raise ValueError(f"v8 span field is not finite: {field}")
        if not 0.0 <= float(span["target_cosine"]) <= 1.0:
            raise ValueError("v8 span target cosine is outside [0, 1]")
        cert = span.get("certificate")
        if not isinstance(cert, dict) or cert.get("independent_validation") is not True:
            raise ValueError("v8 span certificate is not independent")
        if not math.isclose(float(cert["confidence"]), confidence, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("v8 certificate does not use family-wise delta")
        if int(cert.get("decoder_rank", rank)) != rank:
            raise ValueError("v8 certificate rank does not equal span rank")
        threshold = float(protocol["certificate"]["cosine_threshold"])
        if bool(span["recovered"]) != bool(float(span["target_cosine"]) >= threshold):
            raise ValueError("v8 recovery flag disagrees with span cosine")
        if bool(span["certified"]) != bool(cert["certified"]):
            raise ValueError("v8 certificate flag disagrees with certificate")
    if seen != expected:
        raise ValueError("v8 artifact has missing rows")
    if not all(row["span_readout"]["certificate"]["certified"] for row in rows):
        raise ValueError("v8 confirmatory gate requires all certificates to pass")


def atomic_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(protocol_path: Path, out_dir: Path, device: str) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to execute v8") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / RAW_NAME
    rows: list[dict[str, Any]] = []
    m = int(protocol["feature"]["m"])
    signal_strength = float(protocol["feature"]["signal_strength"])
    targets = stress.directions(m)
    threshold = float(protocol["certificate"]["cosine_threshold"])
    familywise_delta = float(protocol["certificate"]["delta"]) / int(protocol["certificate"]["family_count"])
    span_tolerance = float(protocol["primary_readout"]["span_relative_singular_value_tolerance"])
    for seed in range(int(protocol["seeds"]["base"]), int(protocol["seeds"]["base"]) + int(protocol["seeds"]["count"])):
        for objective_index, objective in enumerate(protocol["objectives"]):
            for capacity_index, capacity in enumerate(protocol["capacities"]):
                for geometry_index, geometry in enumerate(protocol["geometry"]):
                    data_seed = int(protocol["seeds"]["training_rng_offset"]) + seed * 1000 + objective_index * 100 + capacity_index * 10 + geometry_index
                    train_seed = int(protocol["seeds"]["torch_seed_offset"]) + seed * 100 + objective_index * 10 + capacity_index
                    validation_seed = int(protocol["seeds"]["validation_rng_offset"]) + seed * 1000 + objective_index * 100 + capacity_index * 10 + geometry_index
                    target = targets[geometry]
                    train = stress.sample_signal(np.random.default_rng(data_seed), int(protocol["training"]["N_train"]), target, signal_strength)
                    validation = stress.sample_signal(np.random.default_rng(validation_seed), int(protocol["training"]["N_validation"]), target, signal_strength)
                    training_spectral = estimate_spectral_anchor(train)
                    started = time.perf_counter()
                    model, metrics = train_warm_start_sae(
                        train,
                        training_spectral,
                        int(capacity["dict_size"]),
                        int(capacity["k"]),
                        int(protocol["training"]["steps"]),
                        int(protocol["training"]["batch"]),
                        float(protocol["training"]["lr"]),
                        train_seed,
                        objective,
                        device,
                    )
                    basis = signed_pair_basis(model, (0, 1), span_tolerance)
                    span_rank = int(basis.shape[1])
                    target_cosine = _target_cosine(basis, target)
                    empirical_excess = audit.empirical_excess_risk_projector(validation, basis, target)
                    cert = certificate.certify_validation(
                        empirical_excess=empirical_excess,
                        signal_strength=signal_strength,
                        noise_variance=float(protocol["certificate"]["noise_variance"]),
                        sample_count=int(protocol["training"]["N_validation"]),
                        delta=familywise_delta,
                        decoder_rank=span_rank,
                        cosine_threshold=threshold,
                        independent_validation=bool(protocol["certificate"]["independent_validation"]),
                    )
                    span = {
                        "kind": "subspace",
                        "selection": protocol["primary_readout"]["selection"],
                        "span_rank": span_rank,
                        "target_cosine": target_cosine,
                        "empirical_excess": empirical_excess,
                        "recovered": bool(target_cosine >= threshold),
                        "certificate": {**cert, "empirical_excess": empirical_excess, "decoder_rank": span_rank},
                        "certified": bool(cert["certified"]),
                    }
                    singular_values = np.linalg.svd(model.dec.weight.detach().cpu().numpy()[:, [0, 1]], compute_uv=False)
                    span["pair_singular_values"] = [float(value) for value in singular_values]
                    span["pair_relative_second_singular_value"] = float(singular_values[1] / singular_values[0])
                    key = (objective["id"], capacity["id"], geometry, seed)
                    row = {
                        "shared_model_id": ":".join(map(str, key)),
                        "objective_id": objective["id"],
                        "capacity_id": capacity["id"],
                        "geometry": geometry,
                        "seed": seed,
                        "data_seed": data_seed,
                        "train_seed": train_seed,
                        "validation_seed": validation_seed,
                        "training_variant": "spectral-warm-start",
                        "primary_readout": "signed-pair-span",
                        "span_readout": span,
                        "objective_loss": float(metrics["objective_loss"]),
                        "projection_hits": int(metrics["projection_hits"]),
                        "fresh_validation": True,
                        "validation_selected": False,
                        "target_used_for_training": False,
                        "elapsed_sec": time.perf_counter() - started,
                    }
                    rows.append(row)
                    atomic_write(raw_path, rows)
                    print(f"[fresh-validation] model={row['shared_model_id']} span={target_cosine:.3f} rank={span_rank} cert={cert['certified']}", flush=True)
    validate_rows(rows, protocol)
    spans = [row["span_readout"] for row in rows]
    summary = {
        "experiment_id": protocol["experiment_id"],
        "status": "completed",
        "rows": len(rows),
        "expected_rows": protocol["expected_rows"],
        "models": len(rows),
        "expected_models": protocol["comparison"]["expected_models"],
        "complete": True,
        "all_recovered": bool(all(span["recovered"] for span in spans)),
        "all_certified": bool(all(span["certified"] for span in spans)),
        "fresh_validation_integrity": bool(all(row["fresh_validation"] and row["validation_seed"] != protocol["seeds"]["v7_validation_rng_offset"] for row in rows)),
        "recovery_rate": float(np.mean([span["recovered"] for span in spans])),
        "certificate_pass_rate": float(np.mean([span["certified"] for span in spans])),
        "minimum_target_cosine": float(min(span["target_cosine"] for span in spans)),
        "mean_target_cosine": float(np.mean([span["target_cosine"] for span in spans])),
        "rank_counts": {str(rank): sum(span["span_rank"] == rank for span in spans) for rank in range(1, 3)},
        "familywise_delta": familywise_delta,
        "familywise_confidence": 1.0 - float(protocol["certificate"]["delta"]),
        "evidence_eligible": bool(all(span["recovered"] and span["certified"] for span in spans)),
        "evidence_scope": "fresh_validation_confirmation_of_v7_primary_readout",
        "calibration_disclosure": protocol["calibration_disclosure"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path),
    }
    atomic_write(out_dir / SUMMARY_NAME, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.out, args.device), indent=2))


if __name__ == "__main__":
    main()
