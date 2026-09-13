"""Bounded-drift, target-free spectral-anchor SAE recovery."""

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
DEFAULT_PROTOCOL_PATH = ROOT / "bounded_drift_spectral_anchor_sae_protocol.json"
DEFAULT_OUTPUT_PATH = ROOT / "results_bounded_drift_spectral_anchor_sae"
RAW_NAME = "bounded_drift_spectral_anchor_raw.json"
SUMMARY_NAME = "bounded_drift_spectral_anchor_summary.json"


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    required = {
        "experiment_id", "feature", "geometry", "objectives", "capacities",
        "seeds", "training", "anchor", "validation", "expected_rows",
    }
    missing = required - protocol.keys()
    if missing:
        raise ValueError(f"protocol missing fields: {sorted(missing)}")
    if protocol["experiment_id"] != "spectral-anchor-sae-v4-bounded-drift":
        raise ValueError("unexpected bounded-drift protocol id")
    if not protocol["training"].get("anchor_trainable"):
        raise ValueError("v4 requires a trainable anchor")
    if protocol["anchor"].get("selection_split") != "training_only" or not protocol["anchor"].get("target_oracle_forbidden"):
        raise ValueError("anchor must be target-free and training-only")
    if not protocol["training"].get("validation_is_never_used_for_selection") or not protocol["validation"].get("selection_forbidden"):
        raise ValueError("validation cannot select the anchor or model")
    if protocol["anchor"].get("estimator") != "leading_eigenvector_of_training_second_moment":
        raise ValueError("anchor must use the training second moment")
    if protocol["training"].get("residual_active_slots") != "k_minus_1":
        raise ValueError("the signed anchor must reserve one total TopK slot")
    if not 0.0 < float(protocol["training"].get("anchor_trust_region_radians", 0.0)) < math.pi / 2:
        raise ValueError("trust region must be a nontrivial acute angle")
    if not 0.0 < float(protocol["training"].get("residual_anchor_leakage_abs_max", 0.0)) < 1.0:
        raise ValueError("residual leakage cap must lie in (0, 1)")
    for capacity in protocol["capacities"]:
        if int(capacity["total_dict_size"]) != int(capacity["residual_dict_size"]) + int(protocol["training"]["anchor_active_slots"]) + 1:
            raise ValueError("capacity does not account for the signed anchor pair")
    expected_family_count = int(protocol["seeds"].get("count", 0)) * len(protocol["geometry"])
    if int(protocol["anchor"].get("family_count", 0)) != expected_family_count:
        raise ValueError("family count must cover every seed-geometry anchor")
    return protocol


def directions(m: int) -> dict[str, np.ndarray]:
    axis = np.zeros(m, dtype=np.float64)
    axis[0] = 1.0
    return {"axis": axis, "dense": np.full(m, 1.0 / math.sqrt(m), dtype=np.float64)}


def sample_signal(rng: np.random.Generator, n: int, direction: np.ndarray, signal_strength: float) -> np.ndarray:
    return rng.standard_normal((n, direction.size)) + rng.normal(0.0, math.sqrt(signal_strength), size=n)[:, None] * direction[None, :]


def estimate_spectral_anchor(training_samples: np.ndarray) -> np.ndarray:
    """Estimate a training-only spectral anchor without target or geometry inputs."""
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


def project_to_cap(vector: np.ndarray, reference: np.ndarray, cap_radians: float) -> np.ndarray:
    """Project a unit direction into the spherical cap around a reference direction."""
    if cap_radians < 0.0 or cap_radians >= math.pi:
        raise ValueError("cap must be in [0, pi)")
    value = np.asarray(vector, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    value = value / np.linalg.norm(value)
    ref = ref / np.linalg.norm(ref)
    cosine = float(np.clip(np.dot(value, ref), -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= cap_radians + 1e-14:
        return value
    orthogonal = value - cosine * ref
    orth_norm = np.linalg.norm(orthogonal)
    if orth_norm <= 1e-12:
        basis = np.zeros_like(ref)
        basis[int(np.argmin(np.abs(ref)))] = 1.0
        orthogonal = basis - float(np.dot(basis, ref)) * ref
        orth_norm = np.linalg.norm(orthogonal)
    orthogonal = orthogonal / orth_norm
    return math.cos(cap_radians) * ref + math.sin(cap_radians) * orthogonal


def bounded_drift_cosine_lower_bound(spectral_sine_bound: float, trust_region_radians: float) -> float:
    """Use the angular triangle inequality for a trainable bounded-drift anchor."""
    if not 0.0 <= spectral_sine_bound < 1.0 or trust_region_radians < 0.0:
        raise ValueError("invalid spectral error or trust region")
    total_angle = math.asin(spectral_sine_bound) + trust_region_radians
    if total_angle >= math.pi / 2:
        raise ValueError("bounded-drift cosine lower bound is non-positive")
    return math.cos(total_angle)


def _fallback_orthogonal(anchor: Any) -> Any:
    import torch

    index = int(torch.argmin(anchor.abs()).item())
    basis = torch.zeros_like(anchor)
    basis[index] = 1.0
    value = basis - anchor * torch.dot(basis, anchor)
    return value / (value.norm() + 1e-12)


def project_residual_decoder_bounded(weights: Any, anchor_vector: Any, leakage_abs_max: float) -> Any:
    """Normalize columns while clipping their signed anchor component to a cap."""
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
        fallback = _fallback_orthogonal(ref)
        orthogonal = fallback
        orth_norm = orthogonal.norm()
    return math.cos(cap_radians) * ref + math.sin(cap_radians) * (orthogonal / (orth_norm + 1e-12))


def train_bounded_drift_sae(
    samples: np.ndarray,
    anchors: np.ndarray,
    residual_dict_size: int,
    k: int,
    steps: int,
    batch: int,
    lr: float,
    seed: int,
    objective: dict[str, Any],
    device: str,
    trust_region_radians: float,
    residual_leakage_abs_max: float,
) -> tuple[Any, float, float, float]:
    """Train an anchor that may move inside a spherical cap with bounded residual leakage."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("torch is required for bounded-drift spectral-anchor training") from exc

    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("anchors must be a signed pair")
    m = samples.shape[1]
    reference = torch.tensor(anchors[:, 0], dtype=torch.float32, device=device)
    reference = reference / (reference.norm() + 1e-12)

    class BoundedDriftSAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("anchor_reference", reference.detach().clone())
            self.anchor_direction = nn.Parameter(reference.detach().clone())
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
    model = BoundedDriftSAE().to(device)
    with torch.no_grad():
        model.residual_decoder.weight.copy_(project_residual_decoder_bounded(model.residual_decoder.weight, model.current_anchor(), residual_leakage_abs_max))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    x = torch.tensor(samples, dtype=torch.float32, device=device)
    last_loss = 0.0
    for _ in range(steps):
        indices = torch.randint(0, x.shape[0], (min(batch, x.shape[0]),), device=device)
        reconstruction, residual = model(x[indices])
        loss = F.mse_loss(reconstruction, x[indices])
        if objective["code_penalty_type"] == "l1":
            loss = loss + float(objective["code_penalty"]) * residual.abs().mean()
        elif objective["code_penalty_type"] == "l2":
            loss = loss + float(objective["code_penalty"]) * residual.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            current = _project_to_cap_torch(model.anchor_direction, model.anchor_reference, trust_region_radians)
            model.anchor_direction.copy_(current)
            model.residual_decoder.weight.copy_(project_residual_decoder_bounded(model.residual_decoder.weight, current, residual_leakage_abs_max))
        last_loss = float(loss.item())
    with torch.no_grad():
        current = model.current_anchor()
        drift = float(torch.acos(torch.clamp(torch.dot(current, model.anchor_reference), -1.0, 1.0)).item())
        leakage = float((current[:, None] * model.residual_decoder.weight).sum(dim=0).abs().max().item())
    return model, last_loss, drift, leakage


def expected_row_keys(protocol: dict[str, Any]) -> set[tuple[str, str, str, int]]:
    return {
        (objective["id"], capacity["id"], geometry, seed)
        for seed in range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"])
        for objective in protocol["objectives"]
        for capacity in protocol["capacities"]
        for geometry in protocol["geometry"]
    }


def validate_rows(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    expected = expected_row_keys(protocol)
    seen = {(row["objective_id"], row["capacity_id"], row["geometry"], int(row["seed"])) for row in rows}
    if seen != expected or len(rows) != int(protocol["expected_rows"]):
        raise ValueError("artifact has missing, unexpected, or incomplete rows")
    threshold = float(protocol["anchor"]["recovery_threshold"])
    trust = float(protocol["training"]["anchor_trust_region_radians"])
    leakage_cap = float(protocol["training"]["residual_anchor_leakage_abs_max"])
    for row in rows:
        if row["anchor_source"] != "training_spectral_only" or row["validation_selected"]:
            raise ValueError("anchor selection leakage")
        if not row["anchors_trainable"]:
            raise ValueError("anchor was not trainable")
        if row["anchor_drift_radians"] > trust + 1e-5:
            raise ValueError("anchor exceeded trust region")
        if row["residual_anchor_dot_max"] > leakage_cap + 1e-5:
            raise ValueError("residual anchor leakage exceeded cap")
        if row["anchor_cosine_lower_bound"] < threshold:
            raise ValueError("bounded-drift certificate is below recovery threshold")
        if row["anchor_cosine"] < threshold:
            raise ValueError("trained bounded-drift anchor missed recovery threshold")


def atomic_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(protocol_path: Path, out_dir: Path, device: str) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to execute bounded-drift spectral-anchor stress") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / RAW_NAME
    rows: list[dict[str, Any]] = []
    m = int(protocol["feature"]["m"])
    strength = float(protocol["feature"]["signal_strength"])
    anchor_config = protocol["anchor"]
    training_config = protocol["training"]
    familywise_delta = anchor_config["delta"] / int(anchor_config["family_count"])
    covariance_error_bound = gaussian_wishart_operator_bound(m, int(training_config["N_train"]), float(anchor_config["covariance_norm_upper_bound"]), familywise_delta)
    anchor_sine_bound = davis_kahan_sine_bound(covariance_error_bound, float(anchor_config["davis_kahan_eigengap"]))
    anchor_cosine_lower_bound = bounded_drift_cosine_lower_bound(anchor_sine_bound, float(training_config["anchor_trust_region_radians"]))
    if anchor_cosine_lower_bound < float(anchor_config["recovery_threshold"]):
        raise RuntimeError("bounded-drift spectral gate cannot certify recovery at this sample size")
    target_by_geometry = directions(m)
    for seed in range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"]):
        for geometry_index, geometry in enumerate(protocol["geometry"]):
            data_seed = int(protocol["seeds"]["training_rng_offset"]) + seed * 100 + geometry_index
            train = sample_signal(np.random.default_rng(data_seed), int(training_config["N_train"]), target_by_geometry[geometry], strength)
            reference = estimate_spectral_anchor(train)
            anchors = make_signed_anchors(reference)
            reference_cosine = float(abs(np.dot(reference, target_by_geometry[geometry])))
            for objective_index, objective in enumerate(protocol["objectives"]):
                for capacity_index, capacity in enumerate(protocol["capacities"]):
                    started = time.perf_counter()
                    train_seed = int(protocol["seeds"]["torch_seed_offset"]) + seed * 100 + objective_index * 10 + capacity_index
                    model, loss, drift, leakage = train_bounded_drift_sae(train, anchors, int(capacity["residual_dict_size"]), int(capacity["k"]), int(training_config["steps"]), int(training_config["batch"]), float(training_config["lr"]), train_seed, objective, device, float(training_config["anchor_trust_region_radians"]), float(training_config["residual_anchor_leakage_abs_max"]))
                    validation_seed = int(protocol["seeds"]["validation_rng_offset"]) + seed * 100 + geometry_index
                    validation = sample_signal(np.random.default_rng(validation_seed), int(training_config["N_validation"]), target_by_geometry[geometry], strength)
                    with torch.no_grad():
                        validation_tensor = torch.tensor(validation, dtype=torch.float32, device=device)
                        reconstruction, _ = model(validation_tensor)
                        validation_loss = float(((reconstruction - validation_tensor).square().mean()).item())
                        final_anchor = model.current_anchor().detach().cpu().numpy()
                    final_cosine = float(abs(np.dot(final_anchor, target_by_geometry[geometry])))
                    row = {
                        "objective_id": objective["id"], "capacity_id": capacity["id"], "geometry": geometry, "seed": seed,
                        "data_seed": data_seed, "train_seed": train_seed, "validation_seed": validation_seed,
                        "anchor_reference_cosine": reference_cosine, "anchor_cosine": final_cosine,
                        "anchor_cosine_lower_bound": anchor_cosine_lower_bound, "covariance_error_bound": covariance_error_bound,
                        "anchor_sine_bound": anchor_sine_bound, "anchor_drift_radians": drift,
                        "residual_anchor_dot_max": leakage, "residual_anchor_leakage_abs_max": float(training_config["residual_anchor_leakage_abs_max"]),
                        "anchor_source": "training_spectral_only", "validation_selected": False, "anchors_trainable": True,
                        "objective_loss": loss, "validation_reconstruction_loss": validation_loss, "elapsed_sec": time.perf_counter() - started,
                    }
                    rows.append(row)
                    atomic_write(raw_path, rows)
                    print(f"[bounded-drift] seed={seed} geometry={geometry} objective={objective['id']} capacity={capacity['id']} ref={reference_cosine:.3f} final={final_cosine:.3f} drift={drift:.3f}", flush=True)
    validate_rows(rows, protocol)
    summary = {
        "experiment_id": protocol["experiment_id"], "status": "completed", "rows": len(rows), "expected_rows": protocol["expected_rows"],
        "anchor_recovery_rate": float(np.mean([row["anchor_cosine"] >= protocol["anchor"]["recovery_threshold"] for row in rows])),
        "minimum_anchor_cosine": float(min(row["anchor_cosine"] for row in rows)),
        "maximum_anchor_drift_radians": float(max(row["anchor_drift_radians"] for row in rows)),
        "maximum_residual_anchor_dot": float(max(row["residual_anchor_dot_max"] for row in rows)),
        "anchor_cosine_lower_bound": anchor_cosine_lower_bound, "covariance_error_bound": covariance_error_bound,
        "evidence_eligible": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_path": str(protocol_path),
    }
    atomic_write(out_dir / SUMMARY_NAME, summary)
    return summary


def gaussian_wishart_operator_bound(dimension: int, sample_count: int, covariance_norm: float, delta: float) -> float:
    if dimension < 1 or sample_count < 1 or covariance_norm <= 0 or not 0.0 < delta < 1.0:
        raise ValueError("invalid Gaussian-Wishart bound inputs")
    deviation = (math.sqrt(dimension) + math.sqrt(2.0 * math.log(2.0 / delta))) / math.sqrt(sample_count)
    if deviation >= 1.0:
        raise ValueError("Gaussian-Wishart envelope is vacuous at this sample count")
    return covariance_norm * (2.0 * deviation + deviation * deviation)


def davis_kahan_sine_bound(covariance_error: float, eigengap: float) -> float:
    if covariance_error < 0 or eigengap <= 0 or 2.0 * covariance_error >= eigengap:
        raise ValueError("covariance error must lie in [0, eigengap/2)")
    return covariance_error / (eigengap - covariance_error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.out, args.device), indent=2))


if __name__ == "__main__":
    main()
