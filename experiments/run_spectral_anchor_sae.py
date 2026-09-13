"""Target-free spectral-anchor SAE recovery across residual objectives.

The leading training-split covariance direction is frozen into a signed decoder pair.
Residual SAE objectives train only the complementary decoder. Therefore the anchor's
recovery is invariant to the residual objective and residual dictionary capacity.
"""

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
DEFAULT_PROTOCOL_PATH = ROOT / "spectral_anchor_sae_protocol.json"
DEFAULT_OUTPUT_PATH = ROOT / "results_spectral_anchor_sae"
RAW_NAME = "spectral_anchor_raw.json"
SUMMARY_NAME = "spectral_anchor_summary.json"


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    required = {"experiment_id", "feature", "geometry", "objectives", "capacities", "seeds", "training", "anchor", "validation", "expected_rows"}
    missing = required - protocol.keys()
    if missing:
        raise ValueError(f"protocol missing fields: {sorted(missing)}")
    if protocol["experiment_id"] != "spectral-anchor-sae-v3":
        raise ValueError("unexpected spectral-anchor protocol id")
    if protocol["anchor"]["selection_split"] != "training_only" or not protocol["anchor"]["target_oracle_forbidden"]:
        raise ValueError("spectral anchor must be target-free and training-only")
    if not protocol["training"]["anchor_columns_fixed"] or not protocol["validation"]["selection_forbidden"]:
        raise ValueError("anchors must be fixed and validation cannot select them")
    if protocol["anchor"].get("estimator") != "leading_eigenvector_of_training_second_moment":
        raise ValueError("spectral anchor must use the frozen training second moment")
    if protocol["training"].get("residual_active_slots") != "k_minus_1":
        raise ValueError("signed anchor must reserve one total TopK slot")
    for capacity in protocol["capacities"]:
        if int(capacity["total_dict_size"]) != int(capacity["residual_dict_size"]) + int(protocol["training"]["anchor_active_slots"]) + 1:
            raise ValueError("capacity does not account for the signed anchor pair")
    if not 0.0 < float(protocol["anchor"].get("delta", 0.0)) < 1.0:
        raise ValueError("anchor protocol needs a valid spectral confidence level")
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
    """Estimate the leading training-second-moment eigenvector without a target oracle."""
    if training_samples.ndim != 2 or training_samples.shape[0] < 2:
        raise ValueError("training samples must be a nontrivial matrix")
    second_moment = training_samples.T @ training_samples / training_samples.shape[0]
    values, vectors = np.linalg.eigh(second_moment)
    anchor = vectors[:, int(np.argmax(values))]
    norm = np.linalg.norm(anchor)
    if norm <= 0:
        raise ValueError("spectral anchor has zero norm")
    anchor = anchor / norm
    # Sign convention is deterministic and does not use the target direction.
    pivot = int(np.argmax(np.abs(anchor)))
    return anchor if anchor[pivot] >= 0 else -anchor


def make_signed_anchors(direction: np.ndarray) -> np.ndarray:
    unit = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(unit)
    if unit.ndim != 1 or norm <= 0:
        raise ValueError("anchor direction must be a nonzero vector")
    unit = unit / norm
    return np.stack([unit, -unit], axis=1)


def signed_anchor_reconstruction(samples: np.ndarray, direction: np.ndarray) -> np.ndarray:
    '''Exactly reconstruct the signed-anchor projection with one nonnegative slot.'''
    unit = np.asarray(direction, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    return (samples @ unit)[:, None] * unit[None, :]


def gaussian_wishart_operator_bound(dimension: int, sample_count: int, covariance_norm: float, delta: float) -> float:
    '''Explicit Gaussian sample-second-moment operator-norm bound.'''
    if dimension < 1 or sample_count < 1 or covariance_norm <= 0 or not 0.0 < delta < 1.0:
        raise ValueError("invalid Gaussian-Wishart bound inputs")
    deviation = (math.sqrt(dimension) + math.sqrt(2.0 * math.log(2.0 / delta))) / math.sqrt(sample_count)
    if deviation >= 1.0:
        raise ValueError("Gaussian-Wishart envelope is vacuous at this sample count")
    return covariance_norm * (2.0 * deviation + deviation * deviation)


def davis_kahan_sine_bound(covariance_error: float, eigengap: float) -> float:
    """Return a conservative simple-eigenvector bound after eigengap perturbation."""
    if covariance_error < 0 or eigengap <= 0 or 2.0 * covariance_error >= eigengap:
        raise ValueError("covariance error must lie in [0, eigengap/2)")
    return covariance_error / (eigengap - covariance_error)


def project_residual_decoder(weights: Any, anchor: Any) -> Any:
    '''Project decoder columns to the anchor orthogonal complement and normalize.'''
    residual = weights - anchor[:, None] * (anchor @ weights)[None, :]
    return residual / (residual.norm(dim=0, keepdim=True) + 1e-8)


def train_anchor_preserving_sae(samples: np.ndarray, anchors: np.ndarray, residual_dict_size: int, k: int, steps: int, batch: int, lr: float, seed: int, objective: dict[str, Any], device: str) -> tuple[Any, float, float]:
    """Train residual SAE while anchors are excluded from optimizer updates."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("torch is required for spectral-anchor SAE training") from exc

    m = samples.shape[1]

    class AnchorPreservingSAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("anchor", torch.tensor(anchors, dtype=torch.float32))
            self.anchor.requires_grad_(False)
            self.anchor_encoder = nn.Linear(m, 2, bias=False)
            with torch.no_grad():
                self.anchor_encoder.weight.copy_(self.anchor.T)
            self.anchor_encoder.weight.requires_grad_(False)
            self.residual_encoder = nn.Linear(m, residual_dict_size, bias=True)
            self.residual_decoder = nn.Linear(residual_dict_size, m, bias=False)
            with torch.no_grad():
                self.residual_decoder.weight.copy_(project_residual_decoder(self.residual_decoder.weight, self.anchor[:, 0]))
            self.k = k

        def forward(self, x: Any) -> tuple[Any, Any]:
            anchor_code = torch.relu(self.anchor_encoder(x))
            residual_pre = torch.relu(self.residual_encoder(x))
            residual_values, residual_indices = torch.topk(residual_pre, min(self.k - 1, residual_pre.shape[1]), dim=1)
            residual_code = torch.zeros_like(residual_pre).scatter(1, residual_indices, residual_values)
            reconstruction = anchor_code @ self.anchor.T + self.residual_decoder(residual_code)
            return reconstruction, residual_code

    torch.manual_seed(seed)
    model = AnchorPreservingSAE().to(device)
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
            weights = model.residual_decoder.weight
            model.residual_decoder.weight.copy_(project_residual_decoder(weights, model.anchor[:, 0]))
        last_loss = float(loss.item())
    with torch.no_grad():
        residual_anchor_dot_max = float((model.anchor[:, 0] @ model.residual_decoder.weight).abs().max().item())
    return model, last_loss, residual_anchor_dot_max


def expected_row_keys(protocol: dict[str, Any]) -> set[tuple[str, str, str, int]]:
    return {(objective["id"], capacity["id"], geometry, seed) for seed in range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"]) for objective in protocol["objectives"] for capacity in protocol["capacities"] for geometry in protocol["geometry"]}


def validate_inherited_anchor_recovery(rows: list[dict[str, Any]], threshold: float) -> None:
    if not rows:
        raise ValueError("rows are required")
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        grouped.setdefault((row.get("geometry", ""), int(row.get("seed", -1))), []).append(float(row["anchor_cosine"]))
    for key, values in grouped.items():
        if any(value < threshold for value in values):
            raise ValueError(f"anchor recovery failed for {key}")
        if max(values) - min(values) > 1e-12:
            raise ValueError(f"anchor changed across residual objectives/capacities for {key}")


def validate_rows(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    expected = expected_row_keys(protocol)
    seen = {(row["objective_id"], row["capacity_id"], row["geometry"], int(row["seed"])) for row in rows}
    if seen != expected:
        raise ValueError("artifact has missing or unexpected rows")
    if len(rows) != protocol["expected_rows"]:
        raise ValueError("artifact row count mismatches protocol")
    threshold = float(protocol["anchor"]["recovery_threshold"])
    validate_inherited_anchor_recovery(rows, threshold)
    for row in rows:
        if row["anchor_source"] != "training_spectral_only" or row["validation_selected"]:
            raise ValueError("anchor selection leakage")
        if not row["anchors_fixed"]:
            raise ValueError("anchor was not fixed")
        if row["residual_anchor_dot_max"] > 1e-6:
            raise ValueError("residual decoder is not orthogonal to anchor")
        if row["anchor_cosine_lower_bound"] < threshold:
            raise ValueError("spectral anchor certificate is below recovery threshold")


def atomic_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(protocol_path: Path, out_dir: Path, device: str) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to execute spectral-anchor stress") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / RAW_NAME
    rows: list[dict[str, Any]] = []
    m = int(protocol["feature"]["m"])
    strength = float(protocol["feature"]["signal_strength"])
    anchor_config = protocol["anchor"]
    familywise_delta = anchor_config["delta"] / int(anchor_config["family_count"])
    covariance_error_bound = gaussian_wishart_operator_bound(m, int(protocol["training"]["N_train"]), float(anchor_config["covariance_norm_upper_bound"]), familywise_delta)
    anchor_sine_bound = davis_kahan_sine_bound(covariance_error_bound, float(anchor_config["davis_kahan_eigengap"]))
    anchor_cosine_lower_bound = math.sqrt(max(0.0, 1.0 - anchor_sine_bound * anchor_sine_bound))
    if anchor_cosine_lower_bound < float(anchor_config["recovery_threshold"]):
        raise RuntimeError("frozen spectral gate cannot certify anchor recovery at this sample size")
    target_by_geometry = directions(m)
    for seed in range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"]):
        for geometry_index, geometry in enumerate(protocol["geometry"]):
            data_seed = int(protocol["seeds"]["training_rng_offset"]) + seed * 100 + geometry_index
            train = sample_signal(np.random.default_rng(data_seed), int(protocol["training"]["N_train"]), target_by_geometry[geometry], strength)
            anchor = estimate_spectral_anchor(train)
            anchors = make_signed_anchors(anchor)
            anchor_cosine = float(abs(np.dot(anchor, target_by_geometry[geometry])))
            for objective_index, objective in enumerate(protocol["objectives"]):
                for capacity_index, capacity in enumerate(protocol["capacities"]):
                    started = time.perf_counter()
                    train_seed = int(protocol["seeds"]["torch_seed_offset"]) + seed * 100 + objective_index * 10 + capacity_index
                    model, loss, residual_anchor_dot_max = train_anchor_preserving_sae(train, anchors, int(capacity["residual_dict_size"]), int(capacity["k"]), int(protocol["training"]["steps"]), int(protocol["training"]["batch"]), float(protocol["training"]["lr"]), train_seed, objective, device)
                    validation_seed = int(protocol["seeds"]["validation_rng_offset"]) + seed * 100 + geometry_index
                    validation = sample_signal(np.random.default_rng(validation_seed), int(protocol["training"]["N_validation"]), target_by_geometry[geometry], strength)
                    with torch.no_grad():
                        validation_tensor = torch.tensor(validation, dtype=torch.float32, device=device)
                        reconstruction, _ = model(validation_tensor)
                        validation_loss = float(((reconstruction - validation_tensor).square().mean()).item())
                    row = {"objective_id": objective["id"], "capacity_id": capacity["id"], "geometry": geometry, "seed": seed, "data_seed": data_seed, "train_seed": train_seed, "validation_seed": validation_seed, "anchor_cosine": anchor_cosine, "anchor_cosine_lower_bound": anchor_cosine_lower_bound, "covariance_error_bound": covariance_error_bound, "residual_anchor_dot_max": residual_anchor_dot_max, "anchor_source": "training_spectral_only", "validation_selected": False, "anchors_fixed": True, "objective_loss": loss, "validation_reconstruction_loss": validation_loss, "elapsed_sec": time.perf_counter() - started}
                    rows.append(row)
                    atomic_write(raw_path, rows)
                    print(f"[anchor] seed={seed} geometry={geometry} objective={objective['id']} capacity={capacity['id']} cosine={anchor_cosine:.3f}", flush=True)
    validate_rows(rows, protocol)
    summary = {"experiment_id": protocol["experiment_id"], "status": "completed", "rows": len(rows), "expected_rows": protocol["expected_rows"], "anchor_recovery_rate": float(np.mean([row["anchor_cosine"] >= protocol["anchor"]["recovery_threshold"] for row in rows])), "anchor_cosine_lower_bound": anchor_cosine_lower_bound, "covariance_error_bound": covariance_error_bound, "evidence_eligible": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_path": str(protocol_path)}
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
