#!/usr/bin/env python3
"""Paired target-free Rayleigh-persistence ablation for a trainable SAE anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL_PATH = ROOT / "frozen_rayleigh_persistence_ablation_protocol.json"
DEFAULT_OUTPUT_PATH = ROOT / "results_frozen_rayleigh_persistence_ablation"
RAW_NAME = "frozen_rayleigh_persistence_raw.json"
SUMMARY_NAME = "frozen_rayleigh_persistence_summary.json"
MODE_IDS = ("hard_cap", "soft_rayleigh", "unrestricted_anchor")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _unique_ids(items: list[dict[str, Any]], name: str) -> None:
    identifiers = [str(item.get("id", "")) for item in items]
    if not identifiers or len(identifiers) != len(set(identifiers)) or any(not value for value in identifiers):
        raise ValueError(f"{name} identifiers must be nonempty and unique")


def validate_protocol(protocol: dict[str, Any]) -> None:
    required = {
        "experiment_id", "status", "parent_protocol", "purpose", "feature", "geometry",
        "objectives", "capacities", "modes", "seeds", "training", "anchor",
        "validation", "gates", "expected_rows", "evidence_status", "reporting_boundary",
    }
    missing = required.difference(protocol)
    if missing:
        raise ValueError(f"protocol missing fields: {sorted(missing)}")
    if protocol["experiment_id"] != "spectral-anchor-sae-v5-frozen-rayleigh-persistence-ablation":
        raise ValueError("unexpected experiment identifier")
    if protocol["status"] != "frozen-before-execution":
        raise ValueError("protocol must be frozen before execution")
    if protocol["geometry"] != ["axis", "dense"]:
        raise ValueError("geometry must be exactly [axis, dense]")
    feature, training, anchor, seeds = (protocol[key] for key in ("feature", "training", "anchor", "seeds"))
    if int(feature.get("m", 0)) < 2 or float(feature.get("signal_strength", 0.0)) <= 0.0:
        raise ValueError("feature settings are invalid")
    _unique_ids(protocol["objectives"], "objective")
    _unique_ids(protocol["capacities"], "capacity")
    _unique_ids(protocol["modes"], "mode")
    if tuple(mode["id"] for mode in protocol["modes"]) != MODE_IDS:
        raise ValueError(f"modes must be ordered exactly as {MODE_IDS}")
    if int(seeds.get("count", 0)) < 2:
        raise ValueError("at least two seed-level runs are required")
    if not all(int(training.get(key, 0)) > 0 for key in ("N_train", "N_validation", "steps", "batch")):
        raise ValueError("training sizes and steps must be positive")
    if float(training.get("lr", 0.0)) <= 0.0:
        raise ValueError("learning rate must be positive")
    if not training.get("anchor_trainable") or training.get("residual_active_slots") != "k_minus_1":
        raise ValueError("v5 requires a trainable anchor with k_minus_1 residual slots")
    if int(training.get("anchor_active_slots", 0)) != 1 or not training.get("signed_pair"):
        raise ValueError("v5 requires one signed-anchor active slot")
    if not training.get("validation_is_never_used_for_selection"):
        raise ValueError("validation may not select an anchor or model")
    leakage = float(training.get("residual_anchor_leakage_abs_max", -1.0))
    if not 0.0 <= leakage < 1.0:
        raise ValueError("residual leakage cap must lie in [0, 1)")
    if anchor.get("estimator") != "leading_eigenvector_of_training_second_moment":
        raise ValueError("anchor must be a training-second-moment leading eigenvector")
    if anchor.get("selection_split") != "training_only" or not anchor.get("target_oracle_forbidden"):
        raise ValueError("anchor must be training-only and target-free")
    if not anchor.get("frozen_training_second_moment_required") or not protocol["validation"].get("selection_forbidden"):
        raise ValueError("training moment must be frozen and validation selection forbidden")
    if not 0.0 < float(anchor.get("delta", 0.0)) < 1.0:
        raise ValueError("anchor confidence must lie in (0, 1)")
    if not 0.0 < float(anchor.get("recovery_threshold", 0.0)) <= 1.0:
        raise ValueError("recovery threshold must lie in (0, 1]")
    for capacity in protocol["capacities"]:
        if int(capacity.get("k", 0)) < 2:
            raise ValueError("signed-anchor constructions require k >= 2")
        expected_size = int(capacity.get("residual_dict_size", -1)) + int(training["anchor_active_slots"]) + 1
        if int(capacity.get("total_dict_size", 0)) != expected_size:
            raise ValueError("capacity does not account for the signed anchor pair")
    family_count = int(seeds["count"]) * len(protocol["geometry"])
    if int(anchor.get("family_count", 0)) != family_count:
        raise ValueError("family count must equal seed_count times geometry_count")
    expected_rows = len(protocol["modes"]) * len(protocol["objectives"]) * len(protocol["capacities"]) * len(protocol["geometry"]) * int(seeds["count"])
    if int(protocol["expected_rows"]) != expected_rows:
        raise ValueError("expected_rows does not match the frozen factorial matrix")
    modes = {mode["id"]: mode for mode in protocol["modes"]}
    hard, soft, unrestricted = (modes[key] for key in MODE_IDS)
    radius = hard.get("anchor_trust_region_radians")
    if radius is None or not 0.0 < float(radius) < math.pi / 2 or float(hard.get("rayleigh_penalty_weight", -1.0)) != 0.0 or hard.get("soft_endpoint_gap_ratio_max") is not None:
        raise ValueError("hard_cap mode semantics are invalid")
    if soft.get("anchor_trust_region_radians") is not None or float(soft.get("rayleigh_penalty_weight", 0.0)) <= 0.0 or soft.get("soft_endpoint_gap_ratio_max") is None:
        raise ValueError("soft_rayleigh mode semantics are invalid")
    if unrestricted.get("anchor_trust_region_radians") is not None or float(unrestricted.get("rayleigh_penalty_weight", -1.0)) != 0.0 or unrestricted.get("soft_endpoint_gap_ratio_max") is not None:
        raise ValueError("unrestricted_anchor mode semantics are invalid")
    if not math.isclose(float(soft["soft_endpoint_gap_ratio_max"]), math.sin(float(radius)) ** 2, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("soft endpoint budget must equal sin(hard_cap_radius)^2")
    boundary = str(protocol["reporting_boundary"])
    if "unrestricted_anchor" not in boundary or "not an unconstrained-SAE" not in boundary:
        raise ValueError("reporting boundary must prevent theorem escalation")


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    validate_protocol(protocol)
    return protocol


def mode_spec_by_id(protocol: dict[str, Any], mode_id: str) -> dict[str, Any]:
    for mode in protocol["modes"]:
        if mode["id"] == mode_id:
            return mode
    raise ValueError(f"unknown mode: {mode_id!r}")


def directions(m: int) -> dict[str, np.ndarray]:
    axis = np.zeros(m, dtype=np.float64)
    axis[0] = 1.0
    return {"axis": axis, "dense": np.full(m, 1.0 / math.sqrt(m), dtype=np.float64)}


def sample_signal(rng: np.random.Generator, n: int, direction: np.ndarray, signal_strength: float) -> np.ndarray:
    return rng.standard_normal((n, direction.size)) + rng.normal(0.0, math.sqrt(signal_strength), size=n)[:, None] * direction[None, :]


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if value.ndim != 1 or not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("vector must be nonzero and finite")
    return value / norm


def training_second_moment(samples: np.ndarray) -> np.ndarray:
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("training samples must be a nontrivial matrix")
    moment = samples.T @ samples / samples.shape[0]
    return 0.5 * (moment + moment.T)


def empirical_spectral_data(second_moment: np.ndarray) -> tuple[float, float, np.ndarray]:
    moment = np.asarray(second_moment, dtype=np.float64)
    if moment.ndim != 2 or moment.shape[0] != moment.shape[1] or moment.shape[0] < 2:
        raise ValueError("second moment must be square with dimension at least two")
    values, vectors = np.linalg.eigh(0.5 * (moment + moment.T))
    eigenvalue, eigengap = float(values[-1]), float(values[-1] - values[-2])
    if not math.isfinite(eigenvalue) or not math.isfinite(eigengap) or eigengap <= 0.0:
        raise ValueError("frozen training moment must have a positive empirical eigengap")
    anchor = normalize_vector(vectors[:, -1])
    pivot = int(np.argmax(np.abs(anchor)))
    return eigenvalue, eigengap, anchor if anchor[pivot] >= 0.0 else -anchor


def estimate_spectral_anchor(training_samples: np.ndarray) -> np.ndarray:
    """Estimate a deterministic-sign leading training-second-moment direction."""
    return empirical_spectral_data(training_second_moment(training_samples))[2]


def make_signed_anchors(direction: np.ndarray) -> np.ndarray:
    unit = normalize_vector(direction)
    return np.stack([unit, -unit], axis=1)


def frozen_rayleigh_metrics(anchor: np.ndarray, second_moment: np.ndarray, reference_eigenvalue: float, empirical_eigengap: float) -> dict[str, float]:
    """Return target-free Rayleigh metrics for a unit anchor and frozen moment."""
    unit = normalize_vector(anchor)
    moment = np.asarray(second_moment, dtype=np.float64)
    moment = 0.5 * (moment + moment.T)
    if not math.isfinite(reference_eigenvalue) or not math.isfinite(empirical_eigengap) or empirical_eigengap <= 0.0:
        raise ValueError("reference eigenvalue and empirical eigengap must be finite and positive")
    rayleigh = float(unit @ moment @ unit)
    deficit = float(reference_eigenvalue - rayleigh)
    if deficit < -1e-7:
        raise ValueError("Rayleigh deficit is negative beyond numerical tolerance")
    deficit = max(0.0, deficit)
    return {"frozen_rayleigh_final": rayleigh, "frozen_rayleigh_gap": deficit, "frozen_rayleigh_gap_ratio": deficit / empirical_eigengap}


def rayleigh_angle_sine_squared_upper_bound(rayleigh_gap_ratio: float) -> float:
    if not math.isfinite(rayleigh_gap_ratio) or rayleigh_gap_ratio < -1e-10:
        raise ValueError("Rayleigh gap ratio must be finite and nonnegative")
    return min(1.0, max(0.0, rayleigh_gap_ratio))


def rayleigh_persistence_cosine_lower_bound(spectral_sine_bound: float, rayleigh_gap_ratio: float) -> float:
    if not 0.0 <= spectral_sine_bound < 1.0:
        raise ValueError("spectral sine bound must lie in [0, 1)")
    q = rayleigh_angle_sine_squared_upper_bound(rayleigh_gap_ratio)
    return max(0.0, math.sqrt(1.0 - spectral_sine_bound * spectral_sine_bound) * math.sqrt(1.0 - q) - spectral_sine_bound * math.sqrt(q))


def bounded_drift_cosine_lower_bound(spectral_sine_bound: float, trust_region_radians: float) -> float:
    if not 0.0 <= spectral_sine_bound < 1.0 or not 0.0 <= trust_region_radians < math.pi / 2:
        raise ValueError("invalid spectral error or trust region")
    return max(0.0, math.cos(math.asin(spectral_sine_bound) + trust_region_radians))


def gaussian_wishart_operator_bound(dimension: int, sample_count: int, covariance_norm: float, delta: float) -> float:
    if dimension < 1 or sample_count < 1 or covariance_norm <= 0.0 or not 0.0 < delta < 1.0:
        raise ValueError("invalid Gaussian-Wishart bound inputs")
    deviation = (math.sqrt(dimension) + math.sqrt(2.0 * math.log(2.0 / delta))) / math.sqrt(sample_count)
    if deviation >= 1.0:
        raise ValueError("Gaussian-Wishart envelope is vacuous at this sample count")
    return covariance_norm * (2.0 * deviation + deviation * deviation)


def davis_kahan_sine_bound(covariance_error: float, eigengap: float) -> float:
    if covariance_error < 0.0 or eigengap <= 0.0 or 2.0 * covariance_error >= eigengap:
        raise ValueError("covariance error must lie in [0, eigengap/2)")
    return covariance_error / (eigengap - covariance_error)


def expected_spectral_bounds(protocol: dict[str, Any]) -> tuple[float, float, float]:
    anchor, training, feature = (protocol[key] for key in ("anchor", "training", "feature"))
    epsilon = gaussian_wishart_operator_bound(int(feature["m"]), int(training["N_train"]), float(anchor["covariance_norm_upper_bound"]), float(anchor["delta"]) / int(anchor["family_count"]))
    sine = davis_kahan_sine_bound(epsilon, float(anchor["davis_kahan_eigengap"]))
    return epsilon, sine, math.sqrt(max(0.0, 1.0 - sine * sine))


def seed_bundle(protocol: dict[str, Any], seed: int, geometry_index: int, objective_index: int, capacity_index: int) -> dict[str, int | str]:
    seeds = protocol["seeds"]
    return {
        "data_seed": int(seeds["data_rng_offset"]) + 10_000 * int(seed) + 100 * int(geometry_index),
        "validation_seed": int(seeds["validation_rng_offset"]) + 10_000 * int(seed) + 100 * int(geometry_index),
        "train_seed": int(seeds["torch_rng_offset"]) + 10_000 * int(seed) + 100 * int(objective_index) + 10 * int(capacity_index),
        "seed_formula_version": str(seeds["seed_formula_version"]),
    }


def _fallback_orthogonal(anchor: Any) -> Any:
    import torch

    index = int(torch.argmin(anchor.abs()).item())
    basis = torch.zeros_like(anchor)
    basis[index] = 1.0
    value = basis - anchor * torch.dot(basis, anchor)
    return value / (value.norm() + 1e-12)


def project_residual_decoder_bounded(weights: Any, anchor_vector: Any, leakage_abs_max: float) -> Any:
    import torch

    if not 0.0 <= leakage_abs_max < 1.0:
        raise ValueError("leakage cap must lie in [0, 1)")
    anchor = anchor_vector / (anchor_vector.norm() + 1e-12)
    columns = weights / (weights.norm(dim=0, keepdim=True) + 1e-12)
    dots = (anchor[:, None] * columns).sum(dim=0)
    clipped = dots.clamp(-float(leakage_abs_max), float(leakage_abs_max))
    orthogonal = columns - anchor[:, None] * dots[None, :]
    orth_norm = orthogonal.norm(dim=0, keepdim=True)
    fallback = _fallback_orthogonal(anchor)
    orth_unit = orthogonal / (orth_norm + 1e-12)
    orth_unit = torch.where(orth_norm > 1e-8, orth_unit, fallback[:, None])
    return orth_unit * torch.sqrt(torch.clamp(1.0 - clipped[None, :] ** 2, min=0.0)) + anchor[:, None] * clipped[None, :]


def project_to_cap(vector: np.ndarray, reference: np.ndarray, cap_radians: float) -> np.ndarray:
    if not 0.0 <= cap_radians < math.pi:
        raise ValueError("cap must lie in [0, pi)")
    value, ref = normalize_vector(vector), normalize_vector(reference)
    cosine = float(np.clip(value @ ref, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= cap_radians + 1e-14:
        return value
    orthogonal = value - cosine * ref
    orth_norm = float(np.linalg.norm(orthogonal))
    if orth_norm <= 1e-12:
        basis = np.zeros_like(ref)
        basis[int(np.argmin(np.abs(ref)))] = 1.0
        orthogonal = basis - float(basis @ ref) * ref
        orth_norm = float(np.linalg.norm(orthogonal))
    return math.cos(cap_radians) * ref + math.sin(cap_radians) * orthogonal / orth_norm


def _project_to_cap_torch(vector: Any, reference: Any, cap_radians: float) -> Any:
    import torch

    value = vector / (vector.norm() + 1e-12)
    ref = reference / (reference.norm() + 1e-12)
    cosine = torch.clamp(torch.dot(value, ref), -1.0, 1.0)
    angle = torch.acos(cosine)
    if float(angle.item()) <= cap_radians:
        return value
    orthogonal = value - cosine * ref
    orth_norm = orthogonal.norm()
    if float(orth_norm.item()) <= 1e-8:
        orthogonal = _fallback_orthogonal(ref)
        orth_norm = orthogonal.norm()
    return math.cos(cap_radians) * ref + math.sin(cap_radians) * orthogonal / (orth_norm + 1e-12)


def _configure_determinism(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _mode_certificate(mode: dict[str, Any], spectral_sine_bound: float, rayleigh_gap_ratio: float) -> float | None:
    if mode["id"] == "hard_cap":
        return bounded_drift_cosine_lower_bound(spectral_sine_bound, float(mode["anchor_trust_region_radians"]))
    if mode["id"] == "soft_rayleigh":
        return rayleigh_persistence_cosine_lower_bound(spectral_sine_bound, rayleigh_gap_ratio)
    if mode["id"] == "unrestricted_anchor":
        return None
    raise ValueError(f"unknown mode: {mode['id']}")


def train_persistence_mode(
    samples: np.ndarray,
    reference_anchor: np.ndarray,
    frozen_second_moment: np.ndarray,
    frozen_eigenvalue: float,
    frozen_eigengap: float,
    residual_dict_size: int,
    k: int,
    steps: int,
    batch: int,
    lr: float,
    seed: int,
    objective: dict[str, Any],
    mode: dict[str, Any],
    residual_leakage_abs_max: float,
    device: str,
) -> tuple[Any, dict[str, float]]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("torch is required for the Rayleigh-persistence ablation") from exc
    if k < 2:
        raise ValueError("signed-anchor construction requires k >= 2")
    _configure_determinism(torch, seed)
    m = int(samples.shape[1])
    reference = torch.tensor(reference_anchor, dtype=torch.float32, device=device)
    reference = reference / (reference.norm() + 1e-12)
    moment = torch.tensor(frozen_second_moment, dtype=torch.float32, device=device)
    moment = 0.5 * (moment + moment.T)

    class PersistenceSAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("anchor_reference", reference.detach().clone())
            self.anchor_direction = nn.Parameter(reference.detach().clone())
            self.residual_encoder = nn.Linear(m, residual_dict_size, bias=True)
            self.residual_decoder = nn.Linear(residual_dict_size, m, bias=False)
            self.k = int(k)

        def current_anchor(self) -> Any:
            return self.anchor_direction / (self.anchor_direction.norm() + 1e-12)

        def forward(self, x: Any) -> tuple[Any, Any]:
            anchor = self.current_anchor()
            signed = torch.stack([anchor, -anchor], dim=1)
            anchor_code = torch.relu(x @ signed)
            residual_pre = torch.relu(self.residual_encoder(x))
            active = min(self.k - 1, residual_pre.shape[1])
            values, indices = torch.topk(residual_pre, active, dim=1)
            residual_code = torch.zeros_like(residual_pre).scatter(1, indices, values)
            return anchor_code @ signed.T + self.residual_decoder(residual_code), residual_code

    model = PersistenceSAE().to(device)
    with torch.no_grad():
        model.residual_decoder.weight.copy_(project_residual_decoder_bounded(model.residual_decoder.weight, model.current_anchor(), residual_leakage_abs_max))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(samples, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    metrics = {"base_objective_loss": 0.0, "code_penalty_loss": 0.0, "rayleigh_penalty_loss": 0.0, "objective_loss": 0.0}
    for _ in range(int(steps)):
        indices = torch.randint(0, x.shape[0], (min(int(batch), x.shape[0]),), generator=generator, device="cpu").to(device)
        reconstruction, residual = model(x[indices])
        base_loss = F.mse_loss(reconstruction, x[indices])
        code_penalty = torch.zeros((), dtype=base_loss.dtype, device=device)
        if objective["code_penalty_type"] == "l1":
            code_penalty = float(objective["code_penalty"]) * residual.abs().mean()
        elif objective["code_penalty_type"] == "l2":
            code_penalty = float(objective["code_penalty"]) * residual.square().mean()
        rayleigh_penalty = torch.zeros((), dtype=base_loss.dtype, device=device)
        if mode["id"] == "soft_rayleigh":
            anchor = model.current_anchor()
            deficit = torch.relu(torch.as_tensor(frozen_eigenvalue, dtype=base_loss.dtype, device=device) - anchor @ moment @ anchor)
            rayleigh_penalty = float(mode["rayleigh_penalty_weight"]) * (deficit / float(frozen_eigengap)).square()
        loss = base_loss + code_penalty + rayleigh_penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if mode["id"] == "hard_cap":
                projected = _project_to_cap_torch(model.anchor_direction, model.anchor_reference, float(mode["anchor_trust_region_radians"]))
                model.anchor_direction.copy_(projected)
            model.residual_decoder.weight.copy_(project_residual_decoder_bounded(model.residual_decoder.weight, model.current_anchor(), residual_leakage_abs_max))
        metrics = {
            "base_objective_loss": float(base_loss.item()),
            "code_penalty_loss": float(code_penalty.item()),
            "rayleigh_penalty_loss": float(rayleigh_penalty.item()),
            "objective_loss": float(loss.item()),
        }
    with torch.no_grad():
        current = model.current_anchor()
        signed_cosine = float(torch.clamp(current.dot(model.anchor_reference), -1.0, 1.0).item())
        metrics.update({
            "anchor_reference_to_final_cosine": signed_cosine,
            "anchor_drift_radians": float(torch.acos(torch.clamp(current.dot(model.anchor_reference), -1.0, 1.0)).item()),
            "residual_anchor_dot_max": float((current[:, None] * model.residual_decoder.weight).sum(dim=0).abs().max().item()),
        })
    return model, metrics


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (str(row["mode_id"]), str(row["objective_id"]), str(row["capacity_id"]), str(row["geometry"]), int(row["seed"]))


def expected_row_keys(protocol: dict[str, Any]) -> set[tuple[str, str, str, str, int]]:
    return {
        (mode["id"], objective["id"], capacity["id"], geometry, seed)
        for mode in protocol["modes"]
        for objective in protocol["objectives"]
        for capacity in protocol["capacities"]
        for geometry in protocol["geometry"]
        for seed in range(int(protocol["seeds"]["base"]), int(protocol["seeds"]["base"]) + int(protocol["seeds"]["count"]))
    }


def validate_rows(rows: list[dict[str, Any]], protocol: dict[str, Any], require_complete: bool = False) -> None:
    validate_protocol(protocol)
    expected = expected_row_keys(protocol)
    seen: set[tuple[str, str, str, str, int]] = set()
    modes = {mode["id"]: mode for mode in protocol["modes"]}
    tolerance = float(protocol["gates"]["numeric_tolerance"])
    threshold = float(protocol["anchor"]["recovery_threshold"])
    epsilon, sine_bound, reference_bound = expected_spectral_bounds(protocol)
    if reference_bound < threshold:
        raise ValueError("reference spectral gate is below recovery threshold")
    required = {
        "mode_id", "objective_id", "capacity_id", "geometry", "seed", "data_seed", "validation_seed", "train_seed", "seed_formula_version",
        "mode_index", "objective_index", "capacity_index", "geometry_index", "anchor_source", "anchor_estimator", "frozen_training_second_moment",
        "validation_selected", "anchors_trainable", "anchor_update_rule", "hard_cap_radians", "rayleigh_penalty_weight", "soft_endpoint_gap_ratio_max",
        "anchor_reference_cosine", "anchor_cosine", "anchor_reference_to_final_cosine", "anchor_drift_radians", "frozen_rayleigh_reference",
        "frozen_rayleigh_final", "frozen_rayleigh_gap", "frozen_empirical_eigengap", "frozen_rayleigh_gap_ratio", "spectral_covariance_error_bound",
        "spectral_anchor_sine_bound", "reference_anchor_cosine_lower_bound", "mode_anchor_cosine_lower_bound", "residual_anchor_dot_max",
        "residual_anchor_leakage_abs_max", "base_objective_loss", "code_penalty_loss", "rayleigh_penalty_loss", "objective_loss",
        "validation_reconstruction_loss", "observed_recovered", "persistence_gate_pass", "row_integrity_pass", "elapsed_sec",
    }
    for row in rows:
        if required.difference(row):
            raise ValueError(f"row is missing fields: {sorted(required.difference(row))}")
        key = row_key(row)
        if key not in expected or key in seen:
            raise ValueError(f"unexpected or duplicate row: {key}")
        seen.add(key)
        mode_id, objective_id, capacity_id, geometry, seed = key
        objective_index = next(index for index, item in enumerate(protocol["objectives"]) if item["id"] == objective_id)
        capacity_index = next(index for index, item in enumerate(protocol["capacities"]) if item["id"] == capacity_id)
        geometry_index = protocol["geometry"].index(geometry)
        mode_index = next(index for index, item in enumerate(protocol["modes"]) if item["id"] == mode_id)
        seed_info = seed_bundle(protocol, seed, geometry_index, objective_index, capacity_index)
        if any(row[field] != seed_info[field] for field in seed_info):
            raise ValueError("row seed provenance does not match protocol")
        if (row["mode_index"], row["objective_index"], row["capacity_index"], row["geometry_index"]) != (mode_index, objective_index, capacity_index, geometry_index):
            raise ValueError("row indexes do not match protocol")
        if row["anchor_source"] != "training_spectral_only" or row["anchor_estimator"] != "leading_eigenvector_of_training_second_moment" or not row["frozen_training_second_moment"] or row["validation_selected"] or not row["anchors_trainable"]:
            raise ValueError("training-only target-free provenance failed")
        numeric = [value for name, value in row.items() if name not in {"mode_anchor_cosine_lower_bound", "hard_cap_radians", "soft_endpoint_gap_ratio_max"} and isinstance(value, (int, float))]
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("row contains nonfinite values")
        if abs(float(row["spectral_covariance_error_bound"]) - epsilon) > tolerance or abs(float(row["spectral_anchor_sine_bound"]) - sine_bound) > tolerance or abs(float(row["reference_anchor_cosine_lower_bound"]) - reference_bound) > tolerance:
            raise ValueError("spectral bound differs from protocol")
        gap = float(row["frozen_rayleigh_gap"])
        eigengap = float(row["frozen_empirical_eigengap"])
        if eigengap <= 0.0 or gap < -tolerance:
            raise ValueError("invalid Rayleigh gap or eigengap")
        if abs(float(row["frozen_rayleigh_reference"]) - float(row["frozen_rayleigh_final"]) - gap) > tolerance or abs(float(row["frozen_rayleigh_gap_ratio"]) - gap / eigengap) > tolerance:
            raise ValueError("Rayleigh identities failed")
        if float(row["residual_anchor_dot_max"]) > float(protocol["training"]["residual_anchor_leakage_abs_max"]) + tolerance:
            raise ValueError("residual leakage cap exceeded")
        if bool(row["observed_recovered"]) != bool(float(row["anchor_cosine"]) >= threshold) or not bool(row["row_integrity_pass"]):
            raise ValueError("row flags disagree with metrics")
        mode = modes[mode_id]
        if mode_id == "hard_cap":
            radius = float(mode["anchor_trust_region_radians"])
            certificate = bounded_drift_cosine_lower_bound(sine_bound, radius)
            if row["anchor_update_rule"] != mode["anchor_update_rule"] or row["hard_cap_radians"] is None or abs(float(row["hard_cap_radians"]) - radius) > tolerance or float(row["anchor_drift_radians"]) > radius + tolerance or not row["persistence_gate_pass"] or abs(float(row["mode_anchor_cosine_lower_bound"]) - certificate) > tolerance:
                raise ValueError("hard-cap gate failed")
        elif mode_id == "soft_rayleigh":
            budget = float(mode["soft_endpoint_gap_ratio_max"])
            ratio = float(row["frozen_rayleigh_gap_ratio"])
            certificate = rayleigh_persistence_cosine_lower_bound(sine_bound, ratio)
            if row["anchor_update_rule"] != mode["anchor_update_rule"] or row["hard_cap_radians"] is not None or ratio > budget + tolerance or not row["persistence_gate_pass"] or abs(float(row["mode_anchor_cosine_lower_bound"]) - certificate) > tolerance or certificate < threshold:
                raise ValueError("soft-Rayleigh gate failed")
        elif row["anchor_update_rule"] != mode["anchor_update_rule"] or row["hard_cap_radians"] is not None or row["soft_endpoint_gap_ratio_max"] is not None or row["mode_anchor_cosine_lower_bound"] is not None or abs(float(row["rayleigh_penalty_loss"])) > tolerance or row["persistence_gate_pass"]:
            raise ValueError("unrestricted-anchor integrity gate failed")
    if require_complete and seen != expected:
        raise ValueError("artifact is incomplete")


def by_mode_summary(rows: list[dict[str, Any]], mode_id: str) -> dict[str, Any]:
    subset = [row for row in rows if row["mode_id"] == mode_id]
    values = lambda field: [float(row[field]) for row in subset]
    return {
        "rows": len(subset),
        "observed_recovery_rate": float(np.mean([row["observed_recovered"] for row in subset])),
        "minimum_target_cosine": float(min(values("anchor_cosine"))),
        "mean_target_cosine": float(np.mean(values("anchor_cosine"))),
        "maximum_anchor_drift_radians": float(max(values("anchor_drift_radians"))),
        "maximum_rayleigh_gap_ratio": float(max(values("frozen_rayleigh_gap_ratio"))),
        "maximum_residual_anchor_dot": float(max(values("residual_anchor_dot_max"))),
        "mean_validation_reconstruction_loss": float(np.mean(values("validation_reconstruction_loss"))),
        "persistence_gate_pass_rate": float(np.mean([row["persistence_gate_pass"] for row in subset])),
    }


def paired_contrast(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    key = lambda row: (row["objective_id"], row["capacity_id"], row["geometry"], row["seed"])
    left_rows = {key(row): row for row in rows if row["mode_id"] == left}
    right_rows = {key(row): row for row in rows if row["mode_id"] == right}
    if set(left_rows) != set(right_rows):
        raise ValueError("paired contrast has mismatched cells")
    result: dict[str, Any] = {"paired_cells": len(left_rows)}
    for metric in ("anchor_cosine", "validation_reconstruction_loss", "anchor_drift_radians", "frozen_rayleigh_gap_ratio", "objective_loss"):
        differences = [float(left_rows[item][metric]) - float(right_rows[item][metric]) for item in left_rows]
        result[f"mean_{metric}_difference"] = float(np.mean(differences))
    return result


def summarize(rows: list[dict[str, Any]], protocol: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    validate_rows(rows, protocol, require_complete=True)
    counts = Counter(row["mode_id"] for row in rows)
    epsilon, sine, lower = expected_spectral_bounds(protocol)
    gates = {
        "complete": True, "row_schema": True, "seed_provenance": True, "pairing": True,
        "training_only_anchor": True, "residual_leakage": True, "reference_spectral_gate": True,
        "hard_cap": True, "soft_rayleigh": True, "unrestricted_anchor_integrity": True,
    }
    gates["all_protocol_gates_pass"] = all(gates.values())
    return {
        "metadata": metadata,
        "counts": {"rows": len(rows), "expected_rows": protocol["expected_rows"], "independent_training_moments": protocol["anchor"]["family_count"], "by_mode": dict(counts)},
        "reference_spectral_certificate": {"family_count": protocol["anchor"]["family_count"], "covariance_error_bound": epsilon, "anchor_sine_bound": sine, "reference_anchor_cosine_lower_bound": lower},
        "by_mode": {mode_id: by_mode_summary(rows, mode_id) for mode_id in MODE_IDS},
        "paired_contrasts": {
            "soft_rayleigh_minus_hard_cap": paired_contrast(rows, "soft_rayleigh", "hard_cap"),
            "unrestricted_anchor_minus_hard_cap": paired_contrast(rows, "unrestricted_anchor", "hard_cap"),
            "unrestricted_anchor_minus_soft_rayleigh": paired_contrast(rows, "unrestricted_anchor", "soft_rayleigh"),
        },
        "gates": gates,
        "evidence_eligible": gates["all_protocol_gates_pass"],
        "reporting_boundary": protocol["reporting_boundary"],
    }


def run_cell(protocol: dict[str, Any], mode: dict[str, Any], mode_index: int, objective: dict[str, Any], objective_index: int, capacity: dict[str, Any], capacity_index: int, geometry: str, geometry_index: int, seed: int, device: str, spectral_bounds: tuple[float, float, float]) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to execute the Rayleigh-persistence ablation") from exc
    training = protocol["training"]
    seed_info = seed_bundle(protocol, seed, geometry_index, objective_index, capacity_index)
    target = directions(int(protocol["feature"]["m"]))[geometry]
    train = sample_signal(np.random.default_rng(int(seed_info["data_seed"])), int(training["N_train"]), target, float(protocol["feature"]["signal_strength"]))
    frozen_moment = training_second_moment(train)
    eigenvalue, eigengap, reference = empirical_spectral_data(frozen_moment)
    started = time.perf_counter()
    model, metrics = train_persistence_mode(train, reference, frozen_moment, eigenvalue, eigengap, int(capacity["residual_dict_size"]), int(capacity["k"]), int(training["steps"]), int(training["batch"]), float(training["lr"]), int(seed_info["train_seed"]), objective, mode, float(training["residual_anchor_leakage_abs_max"]), device)
    validation = sample_signal(np.random.default_rng(int(seed_info["validation_seed"])), int(training["N_validation"]), target, float(protocol["feature"]["signal_strength"]))
    with torch.no_grad():
        x = torch.tensor(validation, dtype=torch.float32, device=device)
        reconstruction, _ = model(x)
        validation_loss = float(((reconstruction - x).square().mean()).item())
        final_anchor = model.current_anchor().detach().cpu().numpy()
    rayleigh = frozen_rayleigh_metrics(final_anchor, frozen_moment, eigenvalue, eigengap)
    certificate = _mode_certificate(mode, spectral_bounds[1], rayleigh["frozen_rayleigh_gap_ratio"])
    if mode["id"] == "hard_cap":
        persistence = metrics["anchor_drift_radians"] <= float(mode["anchor_trust_region_radians"]) + float(protocol["gates"]["numeric_tolerance"])
    elif mode["id"] == "soft_rayleigh":
        persistence = rayleigh["frozen_rayleigh_gap_ratio"] <= float(mode["soft_endpoint_gap_ratio_max"]) + float(protocol["gates"]["numeric_tolerance"])
    else:
        persistence = False
    final_cosine = float(abs(final_anchor @ target))
    return {
        "mode_id": mode["id"], "objective_id": objective["id"], "capacity_id": capacity["id"], "geometry": geometry, "seed": seed,
        **seed_info, "mode_index": mode_index, "objective_index": objective_index, "capacity_index": capacity_index, "geometry_index": geometry_index,
        "anchor_source": "training_spectral_only", "anchor_estimator": "leading_eigenvector_of_training_second_moment", "frozen_training_second_moment": True,
        "validation_selected": False, "anchors_trainable": True, "anchor_update_rule": mode["anchor_update_rule"],
        "hard_cap_radians": mode["anchor_trust_region_radians"], "rayleigh_penalty_weight": float(mode["rayleigh_penalty_weight"]), "soft_endpoint_gap_ratio_max": mode["soft_endpoint_gap_ratio_max"],
        "anchor_reference_cosine": float(abs(reference @ target)), "anchor_cosine": final_cosine,
        "anchor_reference_to_final_cosine": metrics["anchor_reference_to_final_cosine"], "anchor_drift_radians": metrics["anchor_drift_radians"],
        "frozen_rayleigh_reference": eigenvalue, **rayleigh, "frozen_empirical_eigengap": eigengap,
        "spectral_covariance_error_bound": spectral_bounds[0], "spectral_anchor_sine_bound": spectral_bounds[1], "reference_anchor_cosine_lower_bound": spectral_bounds[2],
        "mode_anchor_cosine_lower_bound": certificate, "residual_anchor_dot_max": metrics["residual_anchor_dot_max"], "residual_anchor_leakage_abs_max": float(training["residual_anchor_leakage_abs_max"]),
        "base_objective_loss": metrics["base_objective_loss"], "code_penalty_loss": metrics["code_penalty_loss"], "rayleigh_penalty_loss": metrics["rayleigh_penalty_loss"], "objective_loss": metrics["objective_loss"],
        "validation_reconstruction_loss": validation_loss, "observed_recovered": final_cosine >= float(protocol["anchor"]["recovery_threshold"]),
        "persistence_gate_pass": persistence, "row_integrity_pass": True, "elapsed_sec": time.perf_counter() - started,
    }


def run(protocol_path: Path, out_dir: Path, device: str, resume: bool) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to execute the Rayleigh-persistence ablation") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path, summary_path = out_dir / RAW_NAME, out_dir / SUMMARY_NAME
    if raw_path.exists() and not resume:
        raise FileExistsError(f"{raw_path} exists; pass --resume or choose a new output directory")
    rows = json.loads(raw_path.read_text()) if raw_path.exists() else []
    validate_rows(rows, protocol)
    seen = {row_key(row) for row in rows}
    spectral_bounds = expected_spectral_bounds(protocol)
    if spectral_bounds[2] < float(protocol["anchor"]["recovery_threshold"]):
        raise RuntimeError("reference spectral certificate is below recovery threshold")
    for seed in range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"]):
        for geometry_index, geometry in enumerate(protocol["geometry"]):
            for objective_index, objective in enumerate(protocol["objectives"]):
                for capacity_index, capacity in enumerate(protocol["capacities"]):
                    for mode_index, mode in enumerate(protocol["modes"]):
                        key = (mode["id"], objective["id"], capacity["id"], geometry, seed)
                        if key in seen:
                            continue
                        row = run_cell(protocol, mode, mode_index, objective, objective_index, capacity, capacity_index, geometry, geometry_index, seed, device, spectral_bounds)
                        rows.append(row)
                        seen.add(key)
                        atomic_write_json(raw_path, rows)
                        print(f"[rayleigh-v5] {key} cosine={row['anchor_cosine']:.3f} gap_ratio={row['frozen_rayleigh_gap_ratio']:.5f}", flush=True)
    metadata = {
        "experiment_id": protocol["experiment_id"], "status": "completed", "protocol_path": str(protocol_path), "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(Path(__file__)), "device": device, "torch_version": str(torch.__version__),
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()), "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    result = summarize(rows, protocol, metadata)
    atomic_write_json(summary_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.out, args.device, args.resume)["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
