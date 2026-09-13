"""Paired vanilla versus spectral signed-pair warm-start SAE stress test."""

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

ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL_PATH = ROOT / "spectral_warm_start_matched_protocol.json"
DEFAULT_OUTPUT_PATH = ROOT / "results_spectral_warm_start_matched"
RAW_NAME = "warm_start_raw.json"
SUMMARY_NAME = "warm_start_summary.json"


def validate_protocol(protocol: dict[str, Any]) -> None:
    required = {
        "experiment_id",
        "status",
        "feature",
        "geometry",
        "objectives",
        "capacities",
        "variants",
        "seeds",
        "training",
        "comparison",
        "evaluation",
        "certificate",
        "expected_rows",
        "evidence_status",
    }
    missing = required - protocol.keys()
    if missing:
        raise ValueError(f"protocol is missing keys: {sorted(missing)}")
    if protocol["status"] != "frozen-before-execution":
        raise ValueError("protocol must be frozen before execution")
    feature = protocol["feature"]
    if feature.get("m") != 8 or feature.get("signal_strength", 0) <= 0:
        raise ValueError("protocol must use m=8 and positive signal strength")
    if protocol["geometry"] != ["axis", "dense"]:
        raise ValueError("geometry must be exactly [axis, dense]")
    if [item["id"] for item in protocol["objectives"]] != ["mse", "mse_l1", "mse_l2"]:
        raise ValueError("objective order is not frozen as expected")
    if [item["id"] for item in protocol["capacities"]] != ["d16-k2", "d32-k4"]:
        raise ValueError("capacity order is not frozen as expected")
    variants = protocol["variants"]
    if [item["id"] for item in variants] != ["vanilla-random", "spectral-warm-start"]:
        raise ValueError("variant order is not frozen as expected")
    if variants[0].get("initialization") != "random_decoder_encoder":
        raise ValueError("vanilla variant must use random decoder/encoder initialization")
    warm_variant = variants[1]
    if warm_variant.get("initialization") != "training_spectral_signed_pair" or warm_variant.get("anchor_pair_slots") != [0, 1]:
        raise ValueError("warm-start variant must use the frozen signed spectral pair slots")
    seeds = protocol["seeds"]
    if seeds.get("count", 0) < 2 or seeds.get("base", 0) <= 0:
        raise ValueError("protocol needs at least two positive seeds")
    training = protocol["training"]
    for key in ("N_train", "N_validation", "steps", "batch"):
        if training.get(key, 0) <= 0:
            raise ValueError(f"training field {key} must be positive: {key}")
    if training.get("decoder_columns_unit_norm") is not True:
        raise ValueError("decoder columns must remain unit norm")
    if training.get("candidate_selection") != "training_split_activation_usage_only":
        raise ValueError("candidate selection must use the original training-only rule")
    if training.get("candidate_selection_tie_break") != "lowest_atom_index":
        raise ValueError("candidate-selection tie break must be lowest atom index")
    if training.get("validation_is_never_used_for_selection") is not True:
        raise ValueError("validation split must be excluded from selection")
    if training.get("warm_start_uses_training_split_only") is not True:
        raise ValueError("warm start must use the training split only")
    if training.get("post_initialization_constraints") != "none":
        raise ValueError("v6 must have no post-initialization constraints")
    comparison = protocol["comparison"]
    if comparison.get("paired_conditions") != "same data_seed, train_seed, validation_seed per row key":
        raise ValueError("paired comparison contract is not frozen")
    if comparison.get("only_variant_difference") != "initialization":
        raise ValueError("the paired comparison must differ only by initialization")
    if comparison.get("warm_start_target_oracle_forbidden") is not True:
        raise ValueError("warm start cannot use the target oracle")
    evaluation = protocol["evaluation"]
    if not 0.0 < float(evaluation.get("cosine_threshold", 0.0)) <= 1.0:
        raise ValueError("cosine threshold must lie in (0, 1]")
    certificate_config = protocol["certificate"]
    if not 0.0 < float(certificate_config.get("delta", 0.0)) < 1.0 or int(certificate_config.get("rank", 0)) != 1:
        raise ValueError("v6 certificate must use a valid rank-one confidence setting")
    if float(certificate_config.get("noise_variance", 0.0)) <= 0.0 or certificate_config.get("independent_validation") is not True:
        raise ValueError("v6 certificate must declare positive noise and independent validation")
    if float(certificate_config.get("cosine_threshold", 0.0)) != float(evaluation["cosine_threshold"]):
        raise ValueError("evaluation and certificate cosine thresholds must match")
    expected = len(variants) * len(protocol["objectives"]) * len(protocol["capacities"]) * len(protocol["geometry"]) * int(seeds["count"])
    if int(protocol["expected_rows"]) != expected:
        raise ValueError(f"expected_rows must be {expected}")
    if int(comparison["expected_pairs"]) != expected // len(variants):
        raise ValueError("expected pair count is inconsistent with the matrix")


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    validate_protocol(protocol)
    return protocol


def estimate_spectral_anchor(training_samples: np.ndarray) -> np.ndarray:
    """Estimate a leading training-second-moment direction without target or geometry."""
    if training_samples.ndim != 2 or training_samples.shape[0] < 2:
        raise ValueError("training samples must be a nontrivial matrix")
    second_moment = training_samples.T @ training_samples / training_samples.shape[0]
    values, vectors = np.linalg.eigh(second_moment)
    anchor = vectors[:, int(np.argmax(values))]
    norm = np.linalg.norm(anchor)
    if norm <= 0:
        raise ValueError("spectral anchor has zero norm")
    anchor = anchor / norm
    pivot = int(np.argmax(np.abs(anchor)))
    return anchor if anchor[pivot] >= 0 else -anchor


def clamp_cosine(value: float) -> float:
    """Clamp floating-point cosine measurements to their mathematical range."""
    return max(0.0, min(1.0, float(value)))


class MatchedTopKSAE:
    """Placeholder type for documentation; build_sae returns the torch module below."""


def build_sae(
    m: int,
    dict_size: int,
    k: int,
    seed: int,
    variant_id: str,
    training_spectral_anchor: np.ndarray,
    device: str,
) -> Any:
    """Build the original TopK SAE, changing only its initialization variant."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("torch is required for the matched warm-start matrix") from exc
    if m < 2 or dict_size < 2 or not 1 <= k <= dict_size:
        raise ValueError("invalid SAE dimensions")
    if variant_id not in {"vanilla-random", "spectral-warm-start"}:
        raise ValueError(f"unknown initialization variant: {variant_id}")

    class TopKSAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.Linear(m, dict_size, bias=True)
            self.dec = nn.Linear(dict_size, m, bias=False)
            self.k = k
            nn.init.orthogonal_(self.dec.weight)

        def forward(self, x: Any) -> tuple[Any, Any]:
            pre = F.relu(self.enc(x))
            values, indices = torch.topk(pre, self.k, dim=1)
            code = torch.zeros_like(pre).scatter(1, indices, values)
            return self.dec(code), code

    torch.manual_seed(seed)
    model = TopKSAE().to(device)
    if variant_id == "spectral-warm-start":
        anchor_pair_slots = [0, 1]
        training_spectral = torch.as_tensor(training_spectral_anchor, dtype=torch.float32, device=device)
        training_spectral = training_spectral / (training_spectral.norm() + 1e-12)
        with torch.no_grad():
            model.dec.weight[:, anchor_pair_slots[0]].copy_(training_spectral)
            model.dec.weight[:, anchor_pair_slots[1]].copy_(-training_spectral)
            model.enc.weight[anchor_pair_slots[0]].copy_(training_spectral)
            model.enc.weight[anchor_pair_slots[1]].copy_(-training_spectral)
            model.enc.bias[anchor_pair_slots].zero_()
    return model


def _unit_decoder_columns(model: Any) -> Any:
    return model.dec.weight / (model.dec.weight.norm(dim=0, keepdim=True) + 1e-8)


def _decoder_cosines(model: Any, reference: np.ndarray, device: str) -> np.ndarray:
    import torch

    ref = torch.as_tensor(reference, dtype=torch.float32, device=device)
    ref = ref / (ref.norm() + 1e-12)
    with torch.no_grad():
        columns = _unit_decoder_columns(model)
        return (columns.T @ ref).abs().detach().cpu().numpy()


def train_matched_sae(
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
    """Train one original-style SAE with no post-initialization constraint."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("torch is required for matched warm-start training") from exc
    variant_id = str(objective.get("variant_id", objective.get("id", "")))
    model = build_sae(samples.shape[1], dict_size, k, seed, variant_id, training_spectral_anchor, device)
    reference = np.asarray(training_spectral_anchor, dtype=np.float64)
    reference = reference / np.linalg.norm(reference)
    initial_cosines = _decoder_cosines(model, reference, device)
    initial_anchor_cosine = clamp_cosine(float(np.max(initial_cosines)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(samples, dtype=torch.float32, device=device)
    n = x.shape[0]
    last_loss = 0.0
    model.train()
    for _ in range(steps):
        indices = torch.randint(0, n, (min(batch, n),), device=device)
        reconstruction, code = model(x[indices])
        loss = F.mse_loss(reconstruction, x[indices])
        penalty_type = objective["code_penalty_type"]
        penalty = float(objective["code_penalty"])
        if penalty_type == "l1":
            loss = loss + penalty * code.abs().mean()
        elif penalty_type == "l2":
            loss = loss + penalty * code.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            weights = model.dec.weight
            model.dec.weight.copy_(weights / (weights.norm(dim=0, keepdim=True) + 1e-8))
        last_loss = float(loss.item())
    candidate_atom, candidate = stress.select_candidate(model, samples, device)
    final_cosines = _decoder_cosines(model, reference, device)
    metrics: dict[str, Any] = {
        "variant_id": variant_id,
        "initial_anchor_cosine": initial_anchor_cosine,
        "final_candidate_cosine": clamp_cosine(float(abs(np.dot(candidate, reference)))),
        "candidate_atom": candidate_atom,
        "projection_hits": 0,
        "objective_loss": last_loss,
        "anchor_pair_final_cosine_min": clamp_cosine(float(np.min(final_cosines[:2]))),
        "anchor_pair_final_cosine_max": clamp_cosine(float(np.max(final_cosines[:2]))),
        "final_reference_cosine_max": clamp_cosine(float(np.max(final_cosines))),
    }
    return model, metrics


def expected_pair_keys(protocol: dict[str, Any]) -> set[tuple[str, str, str, int]]:
    seeds = range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"])
    return {
        (objective["id"], capacity["id"], geometry, seed)
        for objective in protocol["objectives"]
        for capacity in protocol["capacities"]
        for geometry in protocol["geometry"]
        for seed in seeds
    }


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (row["variant_id"], row["objective_id"], row["capacity_id"], row["geometry"], int(row["seed"]))


def pair_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (row["objective_id"], row["capacity_id"], row["geometry"], int(row["seed"]))


def validate_rows(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    validate_protocol(protocol)
    variants = {item["id"]: item for item in protocol["variants"]}
    objectives = {item["id"]: item for item in protocol["objectives"]}
    capacities = {item["id"]: item for item in protocol["capacities"]}
    seeds = protocol["seeds"]
    expected = {
        (variant_id, *key)
        for variant_id in variants
        for key in expected_pair_keys(protocol)
    }
    seen: set[tuple[str, str, str, str, int]] = set()
    required = {
        "variant_id",
        "variant_initialization",
        "objective_id",
        "capacity_id",
        "geometry",
        "seed",
        "pair_id",
        "train_seed",
        "data_seed",
        "validation_seed",
        "candidate_selection",
        "candidate_atom",
        "target_cosine",
        "final_candidate_cosine",
        "initial_anchor_cosine",
        "certificate",
        "objective_loss",
        "projection_hits",
        "validation_selected",
    }
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"matched row missing fields: {sorted(missing)}")
        key = row_key(row)
        if key in seen:
            raise ValueError(f"duplicate matched row: {key}")
        seen.add(key)
        if key not in expected:
            raise ValueError(f"matched row is outside frozen protocol: {key}")
        if row["variant_id"] not in variants or row["objective_id"] not in objectives or row["capacity_id"] not in capacities:
            raise ValueError("matched row references an unknown variant, objective, or capacity")
        if row["variant_initialization"] != variants[row["variant_id"]]["initialization"]:
            raise ValueError("matched row initialization does not match the frozen variant")
        geometry_index = protocol["geometry"].index(row["geometry"])
        objective_index = next(i for i, item in enumerate(protocol["objectives"]) if item["id"] == row["objective_id"])
        capacity_index = next(i for i, item in enumerate(protocol["capacities"]) if item["id"] == row["capacity_id"])
        seed = int(row["seed"])
        expected_data = int(seeds["training_rng_offset"]) + seed * 1000 + objective_index * 100 + capacity_index * 10 + geometry_index
        expected_train = int(seeds["torch_seed_offset"]) + seed * 100 + objective_index * 10 + capacity_index
        expected_validation = int(seeds["validation_rng_offset"]) + seed * 1000 + objective_index * 100 + capacity_index * 10 + geometry_index
        if int(row["data_seed"]) != expected_data or int(row["train_seed"]) != expected_train or int(row["validation_seed"]) != expected_validation:
            raise ValueError(f"matched seeds do not match the frozen protocol for {key}")
        if row["pair_id"] != ":".join([row["objective_id"], row["capacity_id"], row["geometry"], str(seed)]):
            raise ValueError("matched pair id is inconsistent")
        if row["candidate_selection"] != protocol["training"]["candidate_selection"]:
            raise ValueError("matched candidate selection does not match the original training-only rule")
        if not isinstance(row["candidate_atom"], int) or row["candidate_atom"] < 0 or row["candidate_atom"] >= capacities[row["capacity_id"]]["dict_size"]:
            raise ValueError("matched candidate atom is outside the dictionary")
        if not 0.0 <= float(row["target_cosine"]) <= 1.0 or not 0.0 <= float(row["final_candidate_cosine"]) <= 1.0 or not 0.0 <= float(row["initial_anchor_cosine"]) <= 1.0:
            raise ValueError("matched cosine must lie in [0, 1]")
        if bool(row["validation_selected"]):
            raise ValueError("validation data was used for selection")
        if int(row["projection_hits"]) != 0:
            raise ValueError("v6 post-initialization constraints must be absent")
        cert = row["certificate"]
        if not isinstance(cert, dict) or cert.get("independent_validation") is not True or "cosine_lower_bound" not in cert or "certified" not in cert:
            raise ValueError(f"matched certificate is incomplete for {key}")
        threshold = float(protocol["evaluation"]["cosine_threshold"])
        if bool(row["recovered"]) != bool(float(row["target_cosine"]) >= threshold):
            raise ValueError(f"matched recovery flag disagrees with target cosine for {key}")
        if not isinstance(row["objective_loss"], (int, float)) or not math.isfinite(float(row["objective_loss"])):
            raise ValueError("matched objective loss must be finite")
    if seen != expected:
        raise ValueError(f"matched artifact has missing or unexpected rows: expected {len(expected)}, got {len(seen)}")
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(pair_key(row), []).append(row)
    if set(grouped) != expected_pair_keys(protocol):
        raise ValueError("paired artifact does not cover the frozen pair keys")
    for key, pair in grouped.items():
        if {row["variant_id"] for row in pair} != set(variants) or len(pair) != len(variants):
            raise ValueError(f"paired rows are incomplete for {key}")
        triplets = {(int(row["data_seed"]), int(row["train_seed"]), int(row["validation_seed"])) for row in pair}
        if len(triplets) != 1:
            raise ValueError(f"paired rows do not share data/train/validation seeds for {key}")


def atomic_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _bootstrap_mean_ci(values: list[float], resamples: int, seed: int) -> list[float]:
    """Percentile bootstrap over independent base-seed clusters."""
    if not values:
        raise ValueError("cannot bootstrap an empty value list")
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    draws = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _clustered_paired_summary(paired_deltas: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    """Aggregate repeated objective/capacity/geometry rows within each base seed."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in paired_deltas:
        seed = int(str(item["pair_id"]).rsplit(":", 1)[-1])
        grouped.setdefault(seed, []).append(item)
    expected_per_seed = len(protocol["objectives"]) * len(protocol["capacities"]) * len(protocol["geometry"])
    if any(len(items) != expected_per_seed for items in grouped.values()):
        raise ValueError("clustered paired summary found an incomplete base-seed cluster")
    cluster_rows = []
    for seed, items in sorted(grouped.items()):
        cluster_rows.append({
            "seed": seed,
            "conditions": len(items),
            "mean_delta_recovery_warm_minus_vanilla": float(np.mean([
                float(item["warm_recovered"]) - float(item["vanilla_recovered"]) for item in items
            ])),
            "mean_delta_target_cosine_warm_minus_vanilla": float(np.mean([
                item["delta_target_cosine_warm_minus_vanilla"] for item in items
            ])),
            "mean_delta_objective_loss_warm_minus_vanilla": float(np.mean([
                item["delta_objective_loss_warm_minus_vanilla"] for item in items
            ])),
        })
    analysis = protocol.get("analysis", {})
    resamples = int(analysis.get("bootstrap_resamples", 10000))
    bootstrap_seed = int(analysis.get("bootstrap_seed", 20260910))
    output: dict[str, Any] = {
        "primary_unit": "base_seed_cluster",
        "clusters": len(cluster_rows),
        "conditions_per_cluster": expected_per_seed,
        "cluster_rows": cluster_rows,
    }
    for key in (
        "mean_delta_recovery_warm_minus_vanilla",
        "mean_delta_target_cosine_warm_minus_vanilla",
        "mean_delta_objective_loss_warm_minus_vanilla",
    ):
        values = [float(row[key]) for row in cluster_rows]
        output[key] = float(np.mean(values))
        output[key + "_bootstrap_ci95"] = _bootstrap_mean_ci(values, resamples, bootstrap_seed + len(output))
    return output


def run(protocol_path: Path, out_dir: Path, device: str) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to execute the matched warm-start matrix") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / RAW_NAME
    rows: list[dict[str, Any]] = []
    m = int(protocol["feature"]["m"])
    signal_strength = float(protocol["feature"]["signal_strength"])
    target_by_geometry = stress.directions(m)
    threshold = float(protocol["evaluation"]["cosine_threshold"])
    for seed in range(int(protocol["seeds"]["base"]), int(protocol["seeds"]["base"]) + int(protocol["seeds"]["count"])):
        for objective_index, objective in enumerate(protocol["objectives"]):
            for capacity_index, capacity in enumerate(protocol["capacities"]):
                for geometry_index, geometry in enumerate(protocol["geometry"]):
                    data_seed = int(protocol["seeds"]["training_rng_offset"]) + seed * 1000 + objective_index * 100 + capacity_index * 10 + geometry_index
                    train_seed = int(protocol["seeds"]["torch_seed_offset"]) + seed * 100 + objective_index * 10 + capacity_index
                    validation_seed = int(protocol["seeds"]["validation_rng_offset"]) + seed * 1000 + objective_index * 100 + capacity_index * 10 + geometry_index
                    target = target_by_geometry[geometry]
                    train_samples = stress.sample_signal(np.random.default_rng(data_seed), int(protocol["training"]["N_train"]), target, signal_strength)
                    validation_samples = stress.sample_signal(np.random.default_rng(validation_seed), int(protocol["training"]["N_validation"]), target, signal_strength)
                    training_spectral_anchor = estimate_spectral_anchor(train_samples)
                    pair_id = ":".join([objective["id"], capacity["id"], geometry, str(seed)])
                    for variant in protocol["variants"]:
                        started = time.perf_counter()
                        model, metrics = train_matched_sae(
                            train_samples,
                            training_spectral_anchor,
                            int(capacity["dict_size"]),
                            int(capacity["k"]),
                            int(protocol["training"]["steps"]),
                            int(protocol["training"]["batch"]),
                            float(protocol["training"]["lr"]),
                            train_seed,
                            {**objective, "variant_id": variant["id"]},
                            device,
                        )
                        candidate_atom = int(metrics["candidate_atom"])
                        candidate_atom_vector = stress.select_candidate(model, train_samples, device)[1]
                        empirical_excess = certificate.empirical_excess_risk(validation_samples, candidate_atom_vector, target)
                        cert = certificate.certify_validation(
                            empirical_excess=empirical_excess,
                            signal_strength=signal_strength,
                            noise_variance=float(protocol["certificate"]["noise_variance"]),
                            sample_count=int(protocol["training"]["N_validation"]),
                            delta=float(protocol["certificate"]["delta"]),
                            decoder_rank=int(protocol["certificate"]["rank"]),
                            cosine_threshold=threshold,
                            independent_validation=bool(protocol["certificate"]["independent_validation"]),
                        )
                        target_cosine = clamp_cosine(float(abs(np.dot(candidate_atom_vector, target))))
                        row = {
                            "variant_id": variant["id"],
                            "variant_initialization": variant["initialization"],
                            "objective_id": objective["id"],
                            "capacity_id": capacity["id"],
                            "geometry": geometry,
                            "seed": seed,
                            "pair_id": pair_id,
                            "train_seed": train_seed,
                            "data_seed": data_seed,
                            "validation_seed": validation_seed,
                            "candidate_selection": protocol["training"]["candidate_selection"],
                            "candidate_atom": candidate_atom,
                            "target_cosine": target_cosine,
                            "initial_anchor_cosine": float(metrics["initial_anchor_cosine"]),
                            "final_candidate_cosine": float(metrics["final_candidate_cosine"]),
                            "final_reference_cosine_max": float(metrics["final_reference_cosine_max"]),
                            "anchor_pair_final_cosine_min": float(metrics["anchor_pair_final_cosine_min"]),
                            "anchor_pair_final_cosine_max": float(metrics["anchor_pair_final_cosine_max"]),
                            "certificate": {**cert, "empirical_excess": empirical_excess},
                            "objective_loss": float(metrics["objective_loss"]),
                            "recovered": bool(target_cosine >= threshold),
                            "projection_hits": int(metrics["projection_hits"]),
                            "validation_selected": False,
                            "target_used_for_training": False,
                            "elapsed_sec": time.perf_counter() - started,
                        }
                        rows.append(row)
                        atomic_write(raw_path, rows)
                        print(f"[warm-start] pair={pair_id} variant={variant['id']} cosine={target_cosine:.3f} candidate={candidate_atom}", flush=True)
    validate_rows(rows, protocol)
    by_variant: dict[str, list[dict[str, Any]]] = {variant["id"]: [row for row in rows if row["variant_id"] == variant["id"]] for variant in protocol["variants"]}
    variant_summary: dict[str, Any] = {}
    for variant in protocol["variants"]:
        variant_rows = by_variant[variant["id"]]
        variant_summary[variant["id"]] = {
            "initialization": variant["initialization"],
            "rows": len(variant_rows),
            "pairs": len({row["pair_id"] for row in variant_rows}),
            "recovery_rate": float(np.mean([row["recovered"] for row in variant_rows])),
            "certificate_pass_rate": float(np.mean([row["certificate"]["certified"] for row in variant_rows])),
            "mean_target_cosine": float(np.mean([row["target_cosine"] for row in variant_rows])),
            "minimum_target_cosine": float(min(row["target_cosine"] for row in variant_rows)),
            "mean_initial_anchor_cosine": float(np.mean([row["initial_anchor_cosine"] for row in variant_rows])),
            "mean_final_candidate_cosine_to_training_spectral": float(np.mean([row["final_candidate_cosine"] for row in variant_rows])),
            "mean_objective_loss": float(np.mean([row["objective_loss"] for row in variant_rows])),
            "projection_hits": int(sum(row["projection_hits"] for row in variant_rows)),
        }
    by_pair = {key: group for key, group in ((key, [row for row in rows if pair_key(row) == key]) for key in expected_pair_keys(protocol))}
    paired_deltas = []
    for key, pair in sorted(by_pair.items()):
        vanilla = next(row for row in pair if row["variant_id"] == "vanilla-random")
        warm = next(row for row in pair if row["variant_id"] == "spectral-warm-start")
        paired_deltas.append({
            "pair_id": vanilla["pair_id"],
            "delta_target_cosine_warm_minus_vanilla": warm["target_cosine"] - vanilla["target_cosine"],
            "delta_objective_loss_warm_minus_vanilla": warm["objective_loss"] - vanilla["objective_loss"],
            "vanilla_recovered": vanilla["recovered"],
            "warm_recovered": warm["recovered"],
        })
    clustered_paired = _clustered_paired_summary(paired_deltas, protocol)
    summary = {
        "experiment_id": protocol["experiment_id"],
        "status": "completed",
        "rows": len(rows),
        "expected_rows": protocol["expected_rows"],
        "pairs": len(paired_deltas),
        "expected_pairs": protocol["comparison"]["expected_pairs"],
        "complete": True,
        "pair_integrity": True,
        "variant_summary": variant_summary,
        "paired_deltas": paired_deltas,
        "clustered_paired_analysis": clustered_paired,
        "evidence_eligible": True,
        "evidence_scope": "paired_initialization_ablation",
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
