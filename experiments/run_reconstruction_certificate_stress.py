"""Cross-objective stress test for the reconstruction certificate.

The protocol freezes objective styles, capacities, geometry, seeds, and a
training-only candidate-selection rule.  Validation is independent and is
never used to select an atom or objective.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import run_reconstruction_certificate as certificate


DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("reconstruction_certificate_stress_protocol.json")
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "results_reconstruction_certificate_stress"
RAW_NAME = "stress_raw.json"
SUMMARY_NAME = "stress_summary.json"


def validate_protocol(protocol: dict[str, Any]) -> None:
    required = {"experiment_id", "status", "feature", "geometry", "objectives", "capacities", "seeds", "training", "certificate", "expected_rows"}
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
    objectives = protocol["objectives"]
    capacities = protocol["capacities"]
    if [item["id"] for item in objectives] != ["mse", "mse_l1", "mse_l2"]:
        raise ValueError("objective order is not frozen as expected")
    if [item["id"] for item in capacities] != ["d16-k2", "d32-k4"]:
        raise ValueError("capacity order is not frozen as expected")
    if any(item["code_penalty_type"] == "none" and item["code_penalty"] != 0.0 for item in objectives):
        raise ValueError("MSE objective must have zero code penalty")
    if any(item["code_penalty_type"] not in {"none", "l1", "l2"} for item in objectives):
        raise ValueError("unknown objective penalty type")
    seeds = protocol["seeds"]
    if seeds.get("count", 0) < 2 or seeds.get("base", 0) <= 0:
        raise ValueError("protocol needs at least two positive held-out seeds")
    training = protocol["training"]
    if training.get("N_train", 0) <= 0 or training.get("N_validation", 0) <= 0 or training.get("steps", 0) <= 0:
        raise ValueError("training and validation sizes/steps must be positive")
    if training.get("candidate_selection") != "training_split_activation_usage_only":
        raise ValueError("candidate selection must use training activation usage only")
    if training.get("candidate_selection_tie_break") != "lowest_atom_index":
        raise ValueError("candidate-selection tie break must be lowest atom index")
    if training.get("validation_is_never_used_for_selection") is not True:
        raise ValueError("validation split must be excluded from candidate selection")
    cert = protocol["certificate"]
    if cert.get("rank") != 1 or not 0.0 < cert.get("delta", 0) < 1.0 or not 0.0 < cert.get("cosine_threshold", 0) <= 1.0:
        raise ValueError("invalid certificate settings")
    expected = len(objectives) * len(capacities) * len(protocol["geometry"]) * seeds["count"]
    if protocol["expected_rows"] != expected:
        raise ValueError(f"expected_rows must be {expected}")


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    validate_protocol(protocol)
    return protocol


def expected_row_keys(protocol: dict[str, Any]) -> set[tuple[str, str, str, int]]:
    seeds = range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"])
    return {
        (objective["id"], capacity["id"], geometry, seed)
        for objective in protocol["objectives"]
        for capacity in protocol["capacities"]
        for geometry in protocol["geometry"]
        for seed in seeds
    }


def row_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (row["objective_id"], row["capacity_id"], row["geometry"], int(row["seed"]))


def certificate_matches_gate(certificate_row: dict[str, Any], cosine_threshold: float) -> bool:
    """Recompute the certificate decision from its reported lower bound."""
    if "cosine_lower_bound" not in certificate_row or "certified" not in certificate_row:
        raise ValueError("certificate row is missing its gate fields")
    return bool(certificate_row["certified"]) == bool(
        float(certificate_row["cosine_lower_bound"]) >= cosine_threshold
    )


def validate_rows(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    validate_protocol(protocol)
    expected = expected_row_keys(protocol)
    if len(rows) != len(expected):
        raise ValueError(f"incomplete stress artifact: expected {len(expected)} rows, got {len(rows)}")
    objective_ids = {item["id"] for item in protocol["objectives"]}
    capacity_by_id = {item["id"]: item for item in protocol["capacities"]}
    seed_cfg = protocol["seeds"]
    seen: set[tuple[str, str, str, int]] = set()
    required = {
        "objective_id", "capacity_id", "geometry", "seed", "train_seed", "data_seed",
        "validation_seed", "candidate_selection", "candidate_atom", "target_cosine",
        "certificate", "objective_loss",
    }
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"stress row missing fields: {sorted(missing)}")
        key = row_key(row)
        if key in seen:
            raise ValueError(f"duplicate stress row: {key}")
        seen.add(key)
        if key not in expected:
            raise ValueError(f"stress row is outside frozen protocol: {key}")
        if row["objective_id"] not in objective_ids or row["capacity_id"] not in capacity_by_id:
            raise ValueError("stress row references an unknown objective or capacity")
        geometry_index = protocol["geometry"].index(row["geometry"])
        objective_index = next(i for i, item in enumerate(protocol["objectives"]) if item["id"] == row["objective_id"])
        capacity_index = next(i for i, item in enumerate(protocol["capacities"]) if item["id"] == row["capacity_id"])
        seed = int(row["seed"])
        expected_data = int(seed_cfg["training_rng_offset"]) + seed * 1000 + objective_index * 100 + capacity_index * 10 + geometry_index
        expected_train = int(seed_cfg["torch_seed_offset"]) + seed * 100 + objective_index * 10 + capacity_index
        expected_validation = int(seed_cfg["validation_rng_offset"]) + seed * 1000 + objective_index * 100 + capacity_index * 10 + geometry_index
        if int(row["data_seed"]) != expected_data or int(row["train_seed"]) != expected_train or int(row["validation_seed"]) != expected_validation:
            raise ValueError(f"stress seeds do not match frozen protocol for {key}")
        if row["candidate_selection"] != protocol["training"]["candidate_selection"]:
            raise ValueError("stress candidate selection does not match frozen training-only rule")
        if not isinstance(row["candidate_atom"], int) or row["candidate_atom"] < 0 or row["candidate_atom"] >= capacity_by_id[row["capacity_id"]]["dict_size"]:
            raise ValueError("stress candidate atom is outside dictionary")
        target_cosine = float(row["target_cosine"])
        if not 0.0 <= target_cosine <= 1.0:
            raise ValueError("stress target cosine must lie in [0, 1]")
        cert = row["certificate"]
        if not isinstance(cert, dict) or cert.get("independent_validation") is not True or "cosine_lower_bound" not in cert or "certified" not in cert:
            raise ValueError(f"stress certificate is incomplete for {key}")
        if not certificate_matches_gate(cert, float(protocol["certificate"]["cosine_threshold"])):
            raise ValueError(f"stress certificate gate disagrees with its lower bound for {key}")
        recovered = bool(row["recovered"])
        expected_recovered = target_cosine >= float(protocol["certificate"]["cosine_threshold"])
        if recovered != expected_recovered:
            raise ValueError(f"stress recovery flag disagrees with target cosine for {key}")
        if not isinstance(row["objective_loss"], (int, float)):
            raise ValueError("stress objective loss must be numeric")
    if seen != expected:
        raise ValueError("stress artifact has missing or unexpected rows")


def directions(m: int) -> dict[str, np.ndarray]:
    axis = np.zeros(m, dtype=np.float64)
    axis[0] = 1.0
    return {"axis": axis, "dense": np.full(m, 1.0 / np.sqrt(m), dtype=np.float64)}


def sample_signal(rng: np.random.Generator, n: int, target: np.ndarray, signal_strength: float) -> np.ndarray:
    noise = rng.standard_normal((n, len(target)))
    signal = rng.normal(0.0, np.sqrt(signal_strength), size=n)
    return noise + signal[:, None] * target[None, :]


def train_sae(samples: np.ndarray, m: int, dict_size: int, k: int, steps: int, batch: int, lr: float, seed: int, objective: dict[str, Any], device: str) -> tuple[Any, float]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("torch is required for the objective stress training matrix") from exc

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
    return model, last_loss


def select_candidate(model: Any, samples: np.ndarray, device: str) -> tuple[int, np.ndarray]:
    import torch
    with torch.no_grad():
        x = torch.tensor(samples, dtype=torch.float32, device=device)
        model.eval()
        _, code = model(x)
        usage = (code > 0).float().mean(dim=0).cpu().numpy()
    candidate_atom = int(np.argmax(usage))
    decoder = model.dec.weight.detach().cpu().numpy()[:, candidate_atom]
    norm = np.linalg.norm(decoder)
    if norm <= 1e-12:
        raise RuntimeError("selected decoder atom has zero norm")
    return candidate_atom, decoder / norm


def atomic_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(protocol_path: Path, out_dir: Path, device: str) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to execute the objective stress matrix") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / RAW_NAME
    rows: list[dict[str, Any]] = []
    m = int(protocol["feature"]["m"])
    signal_strength = float(protocol["feature"]["signal_strength"])
    target_by_geometry = directions(m)
    for seed in range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"]):
        for objective_index, objective in enumerate(protocol["objectives"]):
            for capacity_index, capacity in enumerate(protocol["capacities"]):
                for geometry_index, geometry in enumerate(protocol["geometry"]):
                    data_seed = int(protocol["seeds"]["training_rng_offset"]) + seed * 1000 + objective_index * 100 + capacity_index * 10 + geometry_index
                    train_seed = int(protocol["seeds"]["torch_seed_offset"]) + seed * 100 + objective_index * 10 + capacity_index
                    validation_seed = int(protocol["seeds"]["validation_rng_offset"]) + seed * 1000 + objective_index * 100 + capacity_index * 10 + geometry_index
                    target = target_by_geometry[geometry]
                    train_rng = np.random.default_rng(data_seed)
                    validation_rng = np.random.default_rng(validation_seed)
                    train_samples = sample_signal(train_rng, int(protocol["training"]["N_train"]), target, signal_strength)
                    validation_samples = sample_signal(validation_rng, int(protocol["training"]["N_validation"]), target, signal_strength)
                    started = time.perf_counter()
                    model, objective_loss = train_sae(
                        train_samples,
                        m,
                        int(capacity["dict_size"]),
                        int(capacity["k"]),
                        int(protocol["training"]["steps"]),
                        int(protocol["training"]["batch"]),
                        float(protocol["training"]["lr"]),
                        train_seed,
                        objective,
                        device,
                    )
                    candidate_atom, candidate = select_candidate(model, train_samples, device)
                    empirical_excess = certificate.empirical_excess_risk(validation_samples, candidate, target)
                    cert = certificate.certify_validation(
                        empirical_excess=empirical_excess,
                        signal_strength=signal_strength,
                        noise_variance=1.0,
                        sample_count=int(protocol["training"]["N_validation"]),
                        delta=float(protocol["certificate"]["delta"]),
                        decoder_rank=int(protocol["certificate"]["rank"]),
                        cosine_threshold=float(protocol["certificate"]["cosine_threshold"]),
                        independent_validation=True,
                    )
                    target_cosine = float(abs(np.dot(candidate, target)))
                    row = {
                        "objective_id": objective["id"],
                        "capacity_id": capacity["id"],
                        "geometry": geometry,
                        "seed": seed,
                        "train_seed": train_seed,
                        "data_seed": data_seed,
                        "validation_seed": validation_seed,
                        "candidate_selection": protocol["training"]["candidate_selection"],
                        "candidate_atom": candidate_atom,
                        "target_cosine": target_cosine,
                        "certificate": {**cert, "empirical_excess": empirical_excess},
                        "objective_loss": objective_loss,
                        "recovered": bool(target_cosine >= float(protocol["certificate"]["cosine_threshold"])),
                        "elapsed_sec": time.perf_counter() - started,
                    }
                    rows.append(row)
                    atomic_write(raw_path, rows)
                    print(f"[stress] seed={seed} objective={objective['id']} capacity={capacity['id']} geometry={geometry} cosine={target_cosine:.3f} cert={cert['certified']}", flush=True)
    validate_rows(rows, protocol)
    summary = {
        "experiment_id": protocol["experiment_id"],
        "status": "completed",
        "rows": len(rows),
        "expected_rows": protocol["expected_rows"],
        "certificate_pass_rate": float(np.mean([row["certificate"]["certified"] for row in rows])),
        "recovery_rate": float(np.mean([row["recovered"] for row in rows])),
        "all_certificate_pass": bool(all(row["certificate"]["certified"] for row in rows)),
        "evidence_eligible": bool(all(row["certificate"]["certified"] for row in rows)),
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
