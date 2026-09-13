"""Target-free readout audit for unconstrained spectral warm-start SAEs."""

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
import run_spectral_warm_start_matched as warm

ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL_PATH = ROOT / "spectral_anchor_readout_audit_protocol.json"
DEFAULT_OUTPUT_PATH = ROOT / "results_spectral_anchor_readout_audit"
RAW_NAME = "readout_raw.json"
SUMMARY_NAME = "readout_summary.json"

build_sae = warm.build_sae


def validate_protocol(protocol: dict[str, Any]) -> None:
    required = {
        "experiment_id", "status", "source_protocol", "feature", "geometry", "objectives",
        "capacities", "warm_start", "readouts", "seeds", "training", "certificate",
        "evaluation", "comparison", "expected_rows", "evidence_status",
    }
    missing = required - protocol.keys()
    if missing:
        raise ValueError(f"v7 protocol missing keys: {sorted(missing)}")
    if protocol["experiment_id"] != "spectral-anchor-sae-v7-readout-audit":
        raise ValueError("unexpected v7 protocol id")
    if protocol["status"] != "frozen-before-execution":
        raise ValueError("v7 protocol must be frozen before execution")
    if protocol["source_protocol"] != "spectral-anchor-sae-v6-matched-warm-start":
        raise ValueError("v7 must reuse the v6 warm-start conditions")
    if int(protocol["feature"].get("m", 0)) != 8 or float(protocol["feature"].get("signal_strength", 0.0)) <= 0.0:
        raise ValueError("v7 feature settings are invalid")
    if protocol["geometry"] != ["axis", "dense"]:
        raise ValueError("v7 geometry must be exactly axis and dense")
    if [item["id"] for item in protocol["objectives"]] != ["mse", "mse_l1", "mse_l2"]:
        raise ValueError("v7 objective order is not frozen")
    if [item["id"] for item in protocol["capacities"]] != ["d16-k2", "d32-k4"]:
        raise ValueError("v7 capacity order is not frozen")
    warm_config = protocol["warm_start"]
    if warm_config.get("variant_id") != "spectral-warm-start" or warm_config.get("initialization") != "training_spectral_signed_pair":
        raise ValueError("v7 must use the v6 spectral warm-start")
    if warm_config.get("anchor_pair_slots") != [0, 1] or warm_config.get("post_initialization_constraints") != "none":
        raise ValueError("v7 signed pair slots/constraint scope changed")
    readouts = protocol["readouts"]
    if [item["id"] for item in readouts] != ["usage-atom", "spectral-linked-atom", "signed-pair-span"]:
        raise ValueError("v7 readout order is not frozen")
    if [item["kind"] for item in readouts] != ["atom", "atom", "subspace"]:
        raise ValueError("v7 readout kinds are invalid")
    if readouts[2].get("slots") != [0, 1]:
        raise ValueError("v7 signed-pair readout must use slots 0 and 1")
    seeds = protocol["seeds"]
    if int(seeds.get("count", 0)) != 4 or int(seeds.get("base", 0)) != 20261101:
        raise ValueError("v7 seeds must match v6")
    training = protocol["training"]
    for key in ("N_train", "N_validation", "steps", "batch"):
        if int(training.get(key, 0)) <= 0:
            raise ValueError(f"v7 training field must be positive: {key}")
    if training.get("candidate_selection") != "training_split_activation_usage_only":
        raise ValueError("v7 usage readout must match the original selector")
    if training.get("validation_is_never_used_for_selection") is not True or training.get("warm_start_uses_training_split_only") is not True:
        raise ValueError("v7 training/readout selection must exclude validation")
    if training.get("post_initialization_constraints") != "none":
        raise ValueError("v7 cannot constrain training after initialization")
    cert = protocol["certificate"]
    expected_family_count = int(protocol["comparison"]["expected_pairs"]) * len(readouts)
    if int(cert.get("family_count", 0)) != expected_family_count:
        raise ValueError("v7 family count must cover every pair/readout")
    if not 0.0 < float(cert.get("delta", 0.0)) < 1.0 or int(cert.get("atom_rank", 0)) != 1 or int(cert.get("span_rank_max", 0)) != 2:
        raise ValueError("v7 certificate rank/confidence settings are invalid")
    tolerance = float(cert.get("span_relative_singular_value_tolerance", 0.0))
    if not 0.0 < tolerance < 1.0:
        raise ValueError("v7 span singular-value tolerance must lie in (0, 1)")
    if cert.get("independent_validation") is not True or float(cert.get("noise_variance", 0.0)) <= 0.0:
        raise ValueError("v7 certificate must use independent validation and positive noise")
    if protocol["comparison"].get("one_model_per_pair") is not True or protocol["comparison"].get("readouts_share_final_model") is not True:
        raise ValueError("v7 readouts must share one final model per pair")
    if protocol["comparison"].get("readout_selection_is_training_only") is not True:
        raise ValueError("v7 readout selection must be training-only")
    expected_pairs = len(protocol["objectives"]) * len(protocol["capacities"]) * len(protocol["geometry"]) * int(seeds["count"])
    if int(protocol["comparison"]["expected_pairs"]) != expected_pairs or int(protocol["expected_rows"]) != expected_pairs:
        raise ValueError("v7 expected pair/row count is inconsistent")


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    validate_protocol(protocol)
    return protocol


def select_usage_atom(model: Any, training_samples: np.ndarray, device: str) -> tuple[int, np.ndarray]:
    """Apply the original training activation-usage selector."""
    return stress.select_candidate(model, training_samples, device)


def select_spectral_linked_atom(model: Any, training_spectral_anchor: np.ndarray, device: str) -> tuple[int, np.ndarray]:
    """Select the decoder atom closest to the training-only spectral anchor."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for spectral readout") from exc
    reference = torch.as_tensor(training_spectral_anchor, dtype=torch.float32, device=device)
    reference = reference / (reference.norm() + 1e-12)
    with torch.no_grad():
        columns = model.dec.weight / (model.dec.weight.norm(dim=0, keepdim=True) + 1e-12)
        scores = (columns.T @ reference).abs()
        index = int(torch.argmax(scores).item())
        vector = columns[:, index].detach().cpu().numpy()
    return index, vector


def subspace_rank(vectors: np.ndarray, relative_singular_value_tolerance: float = 0.05) -> int:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("subspace vectors must be a nonempty matrix")
    if not 0.0 < relative_singular_value_tolerance < 1.0:
        raise ValueError("relative singular-value tolerance must lie in (0, 1)")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values[0] <= 1e-12:
        raise ValueError("subspace vectors are numerically zero")
    return int(np.sum(singular_values >= relative_singular_value_tolerance * singular_values[0]))


def orthonormal_basis(vectors: np.ndarray, relative_singular_value_tolerance: float = 0.05) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 1e-12:
        raise ValueError("subspace vectors are numerically zero")
    rank = int(np.sum(singular_values >= relative_singular_value_tolerance * singular_values[0]))
    if rank < 1:
        raise ValueError("subspace rank is zero")
    return u[:, :rank]


def signed_pair_basis(
    model: Any,
    slots: tuple[int, int] = (0, 1),
    relative_singular_value_tolerance: float = 0.05,
) -> np.ndarray:
    """Build a stable basis for the fixed warm-start pair slots without a target."""
    weights = model.dec.weight.detach().cpu().numpy()[:, list(slots)]
    normalized = weights / (np.linalg.norm(weights, axis=0, keepdims=True) + 1e-12)
    return orthonormal_basis(normalized, relative_singular_value_tolerance)


def subspace_cosine(basis: np.ndarray, direction: np.ndarray) -> float:
    q = orthonormal_basis(basis, 1e-10)
    vector = np.asarray(direction, dtype=np.float64)
    vector = vector / np.linalg.norm(vector)
    return warm.clamp_cosine(float(np.linalg.norm(q.T @ vector)))


def _target_containing_basis(target: np.ndarray, rank: int) -> np.ndarray:
    vector = np.asarray(target, dtype=np.float64)
    vector = vector / np.linalg.norm(vector)
    if not 1 <= rank <= vector.size:
        raise ValueError("oracle rank must lie within the ambient dimension")
    columns = [vector]
    for index in range(vector.size):
        if len(columns) >= rank:
            break
        basis_vector = np.zeros(vector.size, dtype=np.float64)
        basis_vector[index] = 1.0
        candidate = basis_vector.copy()
        for column in columns:
            candidate -= float(candidate @ column) * column
        norm = np.linalg.norm(candidate)
        if norm > 1e-10:
            columns.append(candidate / norm)
    return np.stack(columns, axis=1)


def empirical_excess_risk_projector(samples: np.ndarray, basis: np.ndarray, target: np.ndarray) -> float:
    q = orthonormal_basis(basis, 1e-10)
    oracle = _target_containing_basis(target, q.shape[1])
    candidate_energy = np.sum((samples @ q) ** 2, axis=1)
    oracle_energy = np.sum((samples @ oracle) ** 2, axis=1)
    return float(np.mean(oracle_energy - candidate_energy))


def _atom_basis(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    value = value / np.linalg.norm(value)
    return value[:, None]


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
    readouts = {item["id"]: item for item in protocol["readouts"]}
    capacities = {item["id"]: item for item in protocol["capacities"]}
    expected = expected_keys(protocol)
    if len(rows) != int(protocol["expected_rows"]):
        raise ValueError("v7 artifact row count is incomplete")
    seen: set[tuple[str, str, str, int]] = set()
    family_count = int(protocol["certificate"]["family_count"])
    confidence = 1.0 - float(protocol["certificate"]["delta"]) / family_count
    for row in rows:
        key = row_key(row)
        if key in seen or key not in expected:
            raise ValueError(f"duplicate or unexpected v7 row: {key}")
        seen.add(key)
        if row.get("shared_model_id") != ":".join(map(str, key)):
            raise ValueError("v7 shared_model_id is inconsistent")
        if row.get("readout_selection_is_training_only") is not True or row.get("validation_selected") is not False:
            raise ValueError("v7 readout selection leaked validation information")
        if row.get("target_used_for_training") is not False or int(row.get("projection_hits", -1)) != 0:
            raise ValueError("v7 training used a target or post-initialization projection")
        if row["capacity_id"] not in capacities:
            raise ValueError("v7 row references an unknown capacity")
        row_readouts = row.get("readouts")
        if not isinstance(row_readouts, dict) or set(row_readouts) != set(readouts):
            raise ValueError("v7 row does not contain every frozen readout")
        for readout_id, result in row_readouts.items():
            config = readouts[readout_id]
            if result.get("selection") != config["selection"] or result.get("kind") != config["kind"]:
                raise ValueError("v7 readout metadata changed")
            rank = int(result.get("rank", 0))
            if rank < 1 or rank > int(protocol["certificate"]["span_rank_max"]):
                raise ValueError("v7 readout rank is outside the frozen range")
            if config["kind"] == "atom" and rank != 1:
                raise ValueError("v7 atom readout must have rank one")
            if not 0.0 <= float(result["target_cosine"]) <= 1.0:
                raise ValueError("v7 readout target cosine is outside [0, 1]")
            cert = result.get("certificate")
            if not isinstance(cert, dict) or cert.get("independent_validation") is not True:
                raise ValueError("v7 readout certificate is not independent")
            if not math.isclose(float(cert["confidence"]), confidence, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("v7 readout certificate does not use family-wise delta")
            threshold = float(protocol["evaluation"]["cosine_threshold"])
            if bool(result["recovered"]) != bool(float(result["target_cosine"]) >= threshold):
                raise ValueError("v7 readout recovery flag disagrees with target cosine")
            if bool(result["certified"]) != bool(cert["certified"]):
                raise ValueError("v7 readout certificate flag is inconsistent")
    if seen != expected:
        raise ValueError("v7 artifact has missing rows")
    if protocol["comparison"].get("readouts_share_final_model") is not True or protocol["comparison"].get("readout_selection_is_training_only") is not True:
        raise ValueError("v7 shared readouts are not frozen as training-only")


def atomic_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(protocol_path: Path, out_dir: Path, device: str) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to execute v7") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / RAW_NAME
    rows: list[dict[str, Any]] = []
    m = int(protocol["feature"]["m"])
    signal_strength = float(protocol["feature"]["signal_strength"])
    targets = stress.directions(m)
    threshold = float(protocol["evaluation"]["cosine_threshold"])
    family_delta = float(protocol["certificate"]["delta"]) / int(protocol["certificate"]["family_count"])
    span_tolerance = float(protocol["certificate"]["span_relative_singular_value_tolerance"])
    readout_config = {item["id"]: item for item in protocol["readouts"]}
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
                    training_spectral = warm.estimate_spectral_anchor(train)
                    started = time.perf_counter()
                    model, train_metrics = warm.train_matched_sae(
                        train,
                        training_spectral,
                        int(capacity["dict_size"]),
                        int(capacity["k"]),
                        int(protocol["training"]["steps"]),
                        int(protocol["training"]["batch"]),
                        float(protocol["training"]["lr"]),
                        train_seed,
                        {**objective, "variant_id": "spectral-warm-start"},
                        device,
                    )
                    usage_index, usage_vector = select_usage_atom(model, train, device)
                    spectral_index, spectral_vector = select_spectral_linked_atom(model, training_spectral, device)
                    pair_basis = signed_pair_basis(model, (0, 1), span_tolerance)
                    candidates: dict[str, tuple[np.ndarray, int | None]] = {
                        "usage-atom": (_atom_basis(usage_vector), usage_index),
                        "spectral-linked-atom": (_atom_basis(spectral_vector), spectral_index),
                        "signed-pair-span": (pair_basis, None),
                    }
                    readout_results: dict[str, Any] = {}
                    for readout_id, (basis, selected_atom) in candidates.items():
                        rank = int(basis.shape[1])
                        target_cosine = subspace_cosine(basis, target)
                        empirical_excess = empirical_excess_risk_projector(validation, basis, target)
                        cert = certificate.certify_validation(
                            empirical_excess=empirical_excess,
                            signal_strength=signal_strength,
                            noise_variance=float(protocol["certificate"]["noise_variance"]),
                            sample_count=int(protocol["training"]["N_validation"]),
                            delta=family_delta,
                            decoder_rank=rank,
                            cosine_threshold=threshold,
                            independent_validation=bool(protocol["certificate"]["independent_validation"]),
                        )
                        config = readout_config[readout_id]
                        result = {
                            "kind": config["kind"],
                            "selection": config["selection"],
                            "selected_atom": selected_atom,
                            "rank": rank,
                            "target_cosine": target_cosine,
                            "recovered": bool(target_cosine >= threshold),
                            "certificate": {**cert, "empirical_excess": empirical_excess},
                            "certified": bool(cert["certified"]),
                        }
                        if readout_id == "signed-pair-span":
                            singular_values = np.linalg.svd(model.dec.weight.detach().cpu().numpy()[:, [0, 1]], compute_uv=False)
                            result["pair_singular_values"] = [float(value) for value in singular_values]
                            result["pair_relative_second_singular_value"] = float(singular_values[1] / singular_values[0])
                        readout_results[readout_id] = result
                    shared_model_id = ":".join([objective["id"], capacity["id"], geometry, str(seed)])
                    row = {
                        "objective_id": objective["id"],
                        "capacity_id": capacity["id"],
                        "geometry": geometry,
                        "seed": seed,
                        "shared_model_id": shared_model_id,
                        "data_seed": data_seed,
                        "train_seed": train_seed,
                        "validation_seed": validation_seed,
                        "training_variant": "spectral-warm-start",
                        "objective_loss": float(train_metrics["objective_loss"]),
                        "projection_hits": int(train_metrics["projection_hits"]),
                        "readout_selection_is_training_only": True,
                        "validation_selected": False,
                        "target_used_for_training": False,
                        "readouts": readout_results,
                        "elapsed_sec": time.perf_counter() - started,
                    }
                    rows.append(row)
                    atomic_write(raw_path, rows)
                    print(
                        f"[readout] model={shared_model_id} usage={readout_results['usage-atom']['target_cosine']:.3f} "
                        f"spectral={readout_results['spectral-linked-atom']['target_cosine']:.3f} "
                        f"span={readout_results['signed-pair-span']['target_cosine']:.3f} rank={readout_results['signed-pair-span']['rank']}",
                        flush=True,
                    )
    validate_rows(rows, protocol)
    readout_summary: dict[str, Any] = {}
    for readout in protocol["readouts"]:
        results = [row["readouts"][readout["id"]] for row in rows]
        readout_summary[readout["id"]] = {
            "kind": readout["kind"],
            "rows": len(results),
            "recovery_rate": float(np.mean([result["recovered"] for result in results])),
            "certificate_pass_rate": float(np.mean([result["certified"] for result in results])),
            "minimum_target_cosine": float(min(result["target_cosine"] for result in results)),
            "mean_target_cosine": float(np.mean([result["target_cosine"] for result in results])),
            "rank_counts": {str(rank): sum(int(result["rank"]) == rank for result in results) for rank in range(1, 3)},
        }
    summary = {
        "experiment_id": protocol["experiment_id"],
        "status": "completed",
        "rows": len(rows),
        "expected_rows": protocol["expected_rows"],
        "pairs": len({row["shared_model_id"] for row in rows}),
        "expected_pairs": protocol["comparison"]["expected_pairs"],
        "complete": True,
        "shared_model_integrity": True,
        "familywise_confidence": 1.0 - float(protocol["certificate"]["delta"]),
        "readout_summary": readout_summary,
        "evidence_eligible": True,
        "evidence_scope": "target_free_readout_ablation",
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
