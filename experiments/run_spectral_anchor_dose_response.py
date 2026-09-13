"""Dose-response and matched-compute baselines for spectral anchors."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL_PATH = ROOT / "dose_response_spectral_anchor_protocol.json"
DEFAULT_OUTPUT_PATH = ROOT / "results_dose_response"
RAW_NAME = "dose_response_raw.json"
SUMMARY_NAME = "dose_response_summary.json"


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    required = {
        "experiment_id", "status", "feature", "geometry", "objectives", "capacities",
        "configs", "seeds", "training", "anchor", "validation", "expected_groups", "expected_rows",
    }
    missing = required - protocol.keys()
    if missing:
        raise ValueError(f"protocol missing fields: {sorted(missing)}")
    if protocol["experiment_id"] != "spectral-anchor-sae-v5-dose-response":
        raise ValueError("unexpected dose-response protocol id")
    if protocol["status"] != "frozen-before-execution":
        raise ValueError("dose-response protocol must be frozen before execution")
    feature = protocol["feature"]
    if int(feature.get("m", 0)) != 8 or float(feature.get("signal_strength", 0.0)) <= 0:
        raise ValueError("dose-response protocol must use m=8 and positive signal strength")
    if protocol["geometry"] != ["axis", "dense"]:
        raise ValueError("dose-response geometry must be exactly axis and dense")
    if len(protocol["objectives"]) != 1 or len(protocol["capacities"]) != 1:
        raise ValueError("dose-response uses one matched objective and capacity")
    if protocol["objectives"][0]["id"] != "mse" or protocol["capacities"][0]["id"] != "d32-k4":
        raise ValueError("dose-response objective/capacity are not frozen as expected")
    configs = protocol["configs"]
    if len(configs) != 9 or len({config["id"] for config in configs}) != 9:
        raise ValueError("dose-response must contain nine unique configurations")
    modes = {config["mode"] for config in configs}
    if modes != {"frozen", "hard", "soft", "unbounded"}:
        raise ValueError("dose-response must cover frozen, hard, soft, and unbounded modes")
    for config in configs:
        mode = config["mode"]
        rho = config.get("rho_radians")
        eta = config.get("eta")
        if mode == "unbounded":
            if rho is not None or eta is not None:
                raise ValueError("unbounded configuration cannot define rho or eta")
        else:
            if rho is None or not 0.0 <= float(rho) < math.pi / 2:
                raise ValueError("bounded configuration needs an acute rho")
            if eta is None or not 0.0 <= float(eta) < 1.0:
                raise ValueError("bounded configuration needs eta in [0, 1)")
        if mode == "frozen" and float(rho or 0.0) != 0.0:
            raise ValueError("frozen configuration must use rho=0")
        if mode in {"soft", "unbounded"} and float(config.get("residual_soft_penalty", 0.0)) < 0:
            raise ValueError("soft residual penalty must be nonnegative")
    seeds = protocol["seeds"]
    if int(seeds.get("count", 0)) < 2 or int(seeds.get("base", 0)) <= 0:
        raise ValueError("dose-response needs at least two positive seeds")
    training = protocol["training"]
    for key in ("N_train", "N_validation", "steps", "batch"):
        if int(training.get(key, 0)) <= 0:
            raise ValueError(f"training field {key} must be positive")
    if training.get("candidate_selection") != "fixed_signed_training_spectral_anchor":
        raise ValueError("candidate selection must remain the fixed training spectral anchor")
    if training.get("residual_active_slots") != "k_minus_1":
        raise ValueError("dose-response must reserve one total TopK slot")
    if training.get("validation_is_never_used_for_selection") is not True:
        raise ValueError("validation cannot select a model")
    if protocol["anchor"].get("selection_split") != "training_only" or not protocol["anchor"].get("target_oracle_forbidden"):
        raise ValueError("anchor must be target-free and training-only")
    expected_groups = int(seeds["count"]) * len(protocol["geometry"])
    expected_rows = expected_groups * len(configs) * len(protocol["objectives"]) * len(protocol["capacities"])
    if int(protocol["expected_groups"]) != expected_groups or int(protocol["expected_rows"]) != expected_rows:
        raise ValueError("dose-response expected group/row count is inconsistent")
    if int(protocol["anchor"].get("family_count", 0)) != expected_groups:
        raise ValueError("family count must cover every independent anchor group")
    return protocol


def directions(m: int) -> dict[str, np.ndarray]:
    axis = np.zeros(m, dtype=np.float64)
    axis[0] = 1.0
    return {"axis": axis, "dense": np.full(m, 1.0 / math.sqrt(m), dtype=np.float64)}


def sample_signal(rng: np.random.Generator, n: int, target: np.ndarray, signal_strength: float) -> np.ndarray:
    noise = rng.standard_normal((n, target.size))
    signal = rng.normal(0.0, math.sqrt(signal_strength), size=n)
    return noise + signal[:, None] * target[None, :]


def estimate_spectral_anchor(training_samples: np.ndarray) -> np.ndarray:
    """Estimate the leading training-second-moment eigenvector without a target."""
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


def make_signed_anchors(direction: np.ndarray) -> np.ndarray:
    unit = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(unit)
    if unit.ndim != 1 or norm <= 0:
        raise ValueError("anchor direction must be a nonzero vector")
    unit = unit / norm
    return np.stack([unit, -unit], axis=1)


def gaussian_wishart_operator_bound(dimension: int, sample_count: int, covariance_norm: float, delta: float) -> float:
    if dimension < 1 or sample_count < 1 or covariance_norm <= 0 or not 0.0 < delta < 1.0:
        raise ValueError("invalid Gaussian-Wishart bound inputs")
    q = (math.sqrt(dimension) + math.sqrt(2.0 * math.log(2.0 / delta))) / math.sqrt(sample_count)
    if q >= 1.0:
        raise ValueError("Gaussian-Wishart envelope is vacuous at this sample count")
    return covariance_norm * (2.0 * q + q * q)


def davis_kahan_sine_bound(covariance_error: float, eigengap: float) -> float:
    if covariance_error < 0 or eigengap <= 0 or 2.0 * covariance_error >= eigengap:
        raise ValueError("covariance error must lie in [0, eigengap/2)")
    return covariance_error / (eigengap - covariance_error)


def angular_distance_from_cosine(cosine: float, roundoff_tolerance: float = 2.0**-22) -> float:
    """Convert cosine to angle while suppressing float32 self-dot roundoff."""
    value = max(-1.0, min(1.0, float(cosine)))
    if 1.0 - value <= roundoff_tolerance:
        return 0.0
    return math.acos(value)


def bounded_drift_cosine_lower_bound(spectral_sine_bound: float, trust_region_radians: float) -> float:
    if not 0.0 <= spectral_sine_bound < 1.0 or trust_region_radians < 0.0:
        raise ValueError("invalid spectral error or trust region")
    total_angle = math.asin(spectral_sine_bound) + trust_region_radians
    if total_angle >= math.pi / 2:
        raise ValueError("bounded-drift cosine lower bound is non-positive")
    return math.cos(total_angle)


def config_cosine_lower_bound(config: dict[str, Any], spectral_sine_bound: float) -> float | None:
    """Return a deterministic bound only when the configuration enforces a cap."""
    if config.get("mode") not in {"frozen", "hard"}:
        return None
    rho = float(config.get("rho_radians") or 0.0)
    return bounded_drift_cosine_lower_bound(spectral_sine_bound, rho)


def soft_anchor_penalty(cosine: Any, rho_radians: float, strength: float) -> Any:
    """Penalize only anchor angles outside the requested soft trust region."""
    if not 0.0 <= float(rho_radians) < math.pi / 2 or float(strength) < 0.0:
        raise ValueError("invalid soft anchor penalty parameters")
    threshold = math.cos(float(rho_radians))
    if hasattr(cosine, "clamp_min"):
        return float(strength) * (cosine.new_tensor(threshold) - cosine).clamp_min(0.0).square()
    return float(strength) * max(0.0, threshold - float(cosine)) ** 2


def soft_residual_penalty(dots: Any, eta: float, strength: float) -> Any:
    """Penalize only residual columns whose absolute leakage exceeds eta."""
    if not 0.0 <= float(eta) < 1.0 or float(strength) < 0.0:
        raise ValueError("invalid soft residual penalty parameters")
    if hasattr(dots, "clamp_min"):
        return float(strength) * (dots.abs() - float(eta)).clamp_min(0.0).square().mean()
    values = [float(value) for value in dots]
    if not values:
        raise ValueError("residual dots cannot be empty")
    return float(strength) * sum(max(0.0, abs(value) - float(eta)) ** 2 for value in values) / len(values)


def _fallback_orthogonal(anchor: Any) -> Any:
    import torch

    index = int(torch.argmin(anchor.abs()).item())
    basis = torch.zeros_like(anchor)
    basis[index] = 1.0
    value = basis - anchor * torch.dot(basis, anchor)
    return value / (value.norm() + 1e-12)


def project_residual_decoder(weights: Any, anchor_vector: Any, leakage_abs_max: float) -> Any:
    """Normalize columns and clip their signed anchor component."""
    import torch

    if leakage_abs_max < 0.0 or leakage_abs_max >= 1.0:
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


def _residual_projection_needed(weights: Any, anchor_vector: Any, leakage_abs_max: float) -> bool:
    anchor = anchor_vector / (anchor_vector.norm() + 1e-12)
    columns = weights / (weights.norm(dim=0, keepdim=True) + 1e-12)
    max_dot = float((anchor[:, None] * columns).sum(dim=0).abs().max().item())
    return max_dot > leakage_abs_max + 1e-7


def _project_to_cap_torch(vector: Any, reference: Any, cap_radians: float) -> tuple[Any, bool]:
    import torch

    value = vector / (vector.norm() + 1e-12)
    ref = reference / (reference.norm() + 1e-12)
    cosine = torch.clamp(torch.dot(value, ref), -1.0, 1.0)
    angle = torch.acos(cosine)
    if float(angle.item()) <= cap_radians + 1e-12:
        return value, False
    orthogonal = value - cosine * ref
    orth_norm = orthogonal.norm()
    if float(orth_norm.item()) <= 1e-8:
        orthogonal = _fallback_orthogonal(ref)
        orth_norm = orthogonal.norm()
    projected = math.cos(cap_radians) * ref + math.sin(cap_radians) * (orthogonal / (orth_norm + 1e-12))
    return projected, True


def train_dose_response_sae(
    samples: np.ndarray,
    anchors: np.ndarray,
    residual_dict_size: int,
    k: int,
    steps: int,
    batch: int,
    lr: float,
    seed: int,
    config: dict[str, Any],
    device: str,
) -> tuple[Any, dict[str, float | int]]:
    """Train one matched-compute dose-response configuration."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("torch is required for the dose-response matrix") from exc

    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("anchors must be a signed pair")
    mode = str(config["mode"])
    eta = config.get("eta")
    rho = config.get("rho_radians")
    anchor_trainable = mode != "frozen"
    m = samples.shape[1]
    reference = torch.tensor(anchors[:, 0], dtype=torch.float32, device=device)
    reference = reference / (reference.norm() + 1e-12)

    class DoseResponseSAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("anchor_reference", reference.detach().clone())
            self.anchor_direction = nn.Parameter(reference.detach().clone(), requires_grad=anchor_trainable)
            self.residual_encoder = nn.Linear(m, residual_dict_size, bias=True)
            self.residual_decoder = nn.Linear(residual_dict_size, m, bias=False)
            self.k = k

        def current_anchor(self) -> Any:
            return self.anchor_direction / (self.anchor_direction.norm() + 1e-12)

        def forward(self, x: Any) -> tuple[Any, Any]:
            anchor = self.current_anchor()
            signed = torch.stack([anchor, -anchor], dim=1)
            anchor_code = torch.relu(x @ signed)
            residual_pre = torch.relu(self.residual_encoder(x))
            active = min(max(self.k - 1, 1), residual_pre.shape[1])
            residual_values, residual_indices = torch.topk(residual_pre, active, dim=1)
            residual_code = torch.zeros_like(residual_pre).scatter(1, residual_indices, residual_values)
            reconstruction = anchor_code @ signed.T + self.residual_decoder(residual_code)
            return reconstruction, residual_code

    torch.manual_seed(seed)
    model = DoseResponseSAE().to(device)
    if eta is not None and mode in {"frozen", "hard"}:
        with torch.no_grad():
            model.residual_decoder.weight.copy_(project_residual_decoder(model.residual_decoder.weight, model.current_anchor(), float(eta)))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    x = torch.tensor(samples, dtype=torch.float32, device=device)
    last_loss = 0.0
    anchor_projection_hits = 0
    residual_projection_hits = 0
    for _ in range(steps):
        indices = torch.randint(0, x.shape[0], (min(batch, x.shape[0]),), device=device)
        reconstruction, residual = model(x[indices])
        loss = F.mse_loss(reconstruction, x[indices])
        if config.get("anchor_soft_penalty", 0.0) > 0.0:
            current = model.current_anchor()
            anchor_dot = torch.dot(current, model.anchor_reference)
            loss = loss + soft_anchor_penalty(anchor_dot, float(rho), float(config["anchor_soft_penalty"]))
        if config.get("residual_soft_penalty", 0.0) > 0.0:
            current = model.current_anchor()
            normalized_weights = model.residual_decoder.weight / (model.residual_decoder.weight.norm(dim=0, keepdim=True) + 1e-12)
            leakage = (current[:, None] * normalized_weights).sum(dim=0)
            loss = loss + soft_residual_penalty(leakage, float(eta), float(config["residual_soft_penalty"]))
        if config.get("code_penalty_type", "none") == "l1":
            loss = loss + float(config["code_penalty"]) * residual.abs().mean()
        elif config.get("code_penalty_type", "none") == "l2":
            loss = loss + float(config["code_penalty"]) * residual.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if mode == "frozen":
                model.anchor_direction.copy_(model.anchor_reference)
            elif mode == "hard":
                current, hit = _project_to_cap_torch(model.anchor_direction, model.anchor_reference, float(rho))
                anchor_projection_hits += int(hit)
                model.anchor_direction.copy_(current)
            current = model.current_anchor()
            if eta is not None and mode in {"frozen", "hard"}:
                needed = _residual_projection_needed(model.residual_decoder.weight, current, float(eta))
                residual_projection_hits += int(needed)
                model.residual_decoder.weight.copy_(project_residual_decoder(model.residual_decoder.weight, current, float(eta)))
            else:
                weights = model.residual_decoder.weight
                model.residual_decoder.weight.copy_(weights / (weights.norm(dim=0, keepdim=True) + 1e-8))
        last_loss = float(loss.item())
    with torch.no_grad():
        current = model.current_anchor()
        drift = angular_distance_from_cosine(float(torch.dot(current, model.anchor_reference).item()))
        normalized_weights = model.residual_decoder.weight / (model.residual_decoder.weight.norm(dim=0, keepdim=True) + 1e-12)
        leakage = float((current[:, None] * normalized_weights).sum(dim=0).abs().max().item())
    metrics: dict[str, float | int] = {
        "anchor_drift_radians": drift,
        "residual_anchor_dot_max": leakage,
        "anchor_projection_hits": anchor_projection_hits,
        "residual_projection_hits": residual_projection_hits,
        "anchor_projection_hit_fraction": anchor_projection_hits / max(steps, 1),
        "residual_projection_hit_fraction": residual_projection_hits / max(steps, 1),
        "anchor_trainable": int(anchor_trainable),
        "last_loss": last_loss,
    }
    return model, metrics


def expected_row_keys(protocol: dict[str, Any]) -> set[tuple[str, str, str, str, int]]:
    return {
        (config["id"], objective["id"], capacity["id"], geometry, seed)
        for config in protocol["configs"]
        for objective in protocol["objectives"]
        for capacity in protocol["capacities"]
        for geometry in protocol["geometry"]
        for seed in range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"])
    }


def validate_rows(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    expected = expected_row_keys(protocol)
    seen = {(row["config_id"], row["objective_id"], row["capacity_id"], row["geometry"], int(row["seed"])) for row in rows}
    if seen != expected or len(rows) != int(protocol["expected_rows"]):
        raise ValueError("dose-response artifact has missing, unexpected, or duplicate rows")
    configs = {config["id"]: config for config in protocol["configs"]}
    for row in rows:
        config = configs[row["config_id"]]
        if row["anchor_source"] != "training_spectral_only" or row["validation_selected"]:
            raise ValueError("dose-response selection leakage")
        if row["anchor_cosine"] < 0.0 or row["anchor_cosine"] > 1.0:
            raise ValueError("anchor cosine must lie in [0, 1]")
        if not math.isfinite(float(row["anchor_drift_radians"])) or not math.isfinite(float(row["residual_anchor_dot_max"])):
            raise ValueError("dose-response metrics must be finite")
        if config["mode"] == "frozen" and row["anchor_drift_radians"] > 1e-5:
            raise ValueError("frozen anchor moved")
        if config["mode"] == "hard":
            if row["anchor_drift_radians"] > float(config["rho_radians"]) + 1e-5:
                raise ValueError("hard anchor exceeded rho")
            if row["residual_anchor_dot_max"] > float(config["eta"]) + 1e-5:
                raise ValueError("hard residual exceeded eta")
        if config["mode"] == "frozen" and row["residual_anchor_dot_max"] > float(config["eta"]) + 1e-5:
            raise ValueError("frozen residual exceeded eta")
        if row["anchor_projection_hits"] < 0 or row["residual_projection_hits"] < 0:
            raise ValueError("projection hit counts must be nonnegative")


def atomic_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(protocol_path: Path, out_dir: Path, device: str) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to execute dose-response training") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / RAW_NAME
    rows: list[dict[str, Any]] = []
    m = int(protocol["feature"]["m"])
    strength = float(protocol["feature"]["signal_strength"])
    training = protocol["training"]
    target_by_geometry = directions(m)
    anchor_config = protocol["anchor"]
    familywise_delta = float(anchor_config["delta"]) / int(anchor_config["family_count"])
    covariance_error_bound = gaussian_wishart_operator_bound(m, int(training["N_train"]), float(anchor_config["covariance_norm_upper_bound"]), familywise_delta)
    spectral_sine_bound = davis_kahan_sine_bound(covariance_error_bound, float(anchor_config["davis_kahan_eigengap"]))
    spectral_cosine_lower_bound = math.sqrt(max(0.0, 1.0 - spectral_sine_bound * spectral_sine_bound))
    for seed in range(int(protocol["seeds"]["base"]), int(protocol["seeds"]["base"]) + int(protocol["seeds"]["count"])):
        for geometry_index, geometry in enumerate(protocol["geometry"]):
            data_seed = int(protocol["seeds"]["training_rng_offset"]) + seed * 100 + geometry_index
            validation_seed = int(protocol["seeds"]["validation_rng_offset"]) + seed * 100 + geometry_index
            target = target_by_geometry[geometry]
            train = sample_signal(np.random.default_rng(data_seed), int(training["N_train"]), target, strength)
            validation = sample_signal(np.random.default_rng(validation_seed), int(training["N_validation"]), target, strength)
            reference = estimate_spectral_anchor(train)
            anchors = make_signed_anchors(reference)
            reference_cosine = float(abs(np.dot(reference, target)))
            for config in protocol["configs"]:
                for objective in protocol["objectives"]:
                    for capacity in protocol["capacities"]:
                        started = time.perf_counter()
                        train_seed = int(protocol["seeds"]["torch_seed_offset"]) + seed * 100
                        model, metrics = train_dose_response_sae(
                            train,
                            anchors,
                            int(capacity["residual_dict_size"]),
                            int(capacity["k"]),
                            int(training["steps"]),
                            int(training["batch"]),
                            float(training["lr"]),
                            train_seed,
                            {**objective, **config},
                            device,
                        )
                        with torch.no_grad():
                            validation_tensor = torch.tensor(validation, dtype=torch.float32, device=device)
                            reconstruction, _ = model(validation_tensor)
                            validation_loss = float(((reconstruction - validation_tensor).square().mean()).item())
                            final_anchor = model.current_anchor().detach().cpu().numpy()
                        final_cosine = float(abs(np.dot(final_anchor, target)))
                        rho = config.get("rho_radians")
                        config_lower_bound = None if rho is None else bounded_drift_cosine_lower_bound(spectral_sine_bound, float(rho))
                        row = {
                            "config_id": config["id"],
                            "config_mode": config["mode"],
                            "objective_id": objective["id"],
                            "capacity_id": capacity["id"],
                            "geometry": geometry,
                            "seed": seed,
                            "anchor_group": f"{geometry}:{seed}",
                            "data_seed": data_seed,
                            "train_seed": train_seed,
                            "validation_seed": validation_seed,
                            "anchor_reference_cosine": reference_cosine,
                            "anchor_cosine": final_cosine,
                            "recovered": bool(final_cosine >= float(protocol["anchor"]["recovery_threshold"])),
                            "spectral_sine_bound": spectral_sine_bound,
                            "spectral_cosine_lower_bound": spectral_cosine_lower_bound,
                            "config_cosine_lower_bound": config_lower_bound,
                            "covariance_error_bound": covariance_error_bound,
                            "rho_radians": rho,
                            "eta": config.get("eta"),
                            "anchor_source": "training_spectral_only",
                            "validation_selected": False,
                            "anchor_trainable": bool(metrics["anchor_trainable"]),
                            "anchor_projection_hits": int(metrics["anchor_projection_hits"]),
                            "residual_projection_hits": int(metrics["residual_projection_hits"]),
                            "anchor_projection_hit_fraction": float(metrics["anchor_projection_hit_fraction"]),
                            "residual_projection_hit_fraction": float(metrics["residual_projection_hit_fraction"]),
                            "anchor_drift_radians": float(metrics["anchor_drift_radians"]),
                            "residual_anchor_dot_max": float(metrics["residual_anchor_dot_max"]),
                            "objective_loss": float(metrics["last_loss"]),
                            "validation_reconstruction_loss": validation_loss,
                            "elapsed_sec": time.perf_counter() - started,
                        }
                        rows.append(row)
                        atomic_write(raw_path, rows)
                        print(f"[dose-response] group={geometry}:{seed} config={config['id']} cosine={final_cosine:.3f} drift={row['anchor_drift_radians']:.3f} hits={row['anchor_projection_hits']}/{training['steps']}", flush=True)
    validate_rows(rows, protocol)
    config_summary: dict[str, Any] = {}
    for config in protocol["configs"]:
        config_rows = [row for row in rows if row["config_id"] == config["id"]]
        config_summary[config["id"]] = {
            "mode": config["mode"],
            "rows": len(config_rows),
            "groups": len({row["anchor_group"] for row in config_rows}),
            "recovery_rate": float(np.mean([row["recovered"] for row in config_rows])),
            "minimum_anchor_cosine": float(min(row["anchor_cosine"] for row in config_rows)),
            "mean_anchor_cosine": float(np.mean([row["anchor_cosine"] for row in config_rows])),
            "maximum_anchor_drift_radians": float(max(row["anchor_drift_radians"] for row in config_rows)),
            "maximum_residual_anchor_dot": float(max(row["residual_anchor_dot_max"] for row in config_rows)),
            "mean_anchor_projection_hit_fraction": float(np.mean([row["anchor_projection_hit_fraction"] for row in config_rows])),
            "mean_residual_projection_hit_fraction": float(np.mean([row["residual_projection_hit_fraction"] for row in config_rows])),
        }
    group_summary = []
    for group in sorted({row["anchor_group"] for row in rows}):
        group_rows = [row for row in rows if row["anchor_group"] == group]
        group_summary.append({
            "anchor_group": group,
            "reference_cosine": float(group_rows[0]["anchor_reference_cosine"]),
            "configs": len({row["config_id"] for row in group_rows}),
            "minimum_final_cosine": float(min(row["anchor_cosine"] for row in group_rows)),
            "maximum_drift_radians": float(max(row["anchor_drift_radians"] for row in group_rows)),
            "recovered_configs": int(sum(row["recovered"] for row in group_rows)),
        })
    summary = {
        "experiment_id": protocol["experiment_id"],
        "status": "completed",
        "rows": len(rows),
        "expected_rows": protocol["expected_rows"],
        "groups": len(group_summary),
        "expected_groups": protocol["expected_groups"],
        "complete": True,
        "all_configs_have_group_rows": all(item["groups"] == protocol["expected_groups"] for item in config_summary.values()),
        "config_summary": config_summary,
        "group_summary": group_summary,
        "spectral_sine_bound": spectral_sine_bound,
        "spectral_cosine_lower_bound": spectral_cosine_lower_bound,
        "covariance_error_bound": covariance_error_bound,
        "evidence_eligible": False,
        "evidence_scope": protocol["anchor"]["evidence_scope"],
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
