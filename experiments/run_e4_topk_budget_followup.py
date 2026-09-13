#!/usr/bin/env python3
"""Post-hoc d32-k4 training-budget diagnostic for the E4 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import run_v2 as e4

DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("e4_topk_budget_followup_protocol.json")
RAW_NAME = "e4_topk_budget_followup_raw.json"
SUMMARY_NAME = "e4_topk_budget_followup_summary.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + chr(10))
    temporary.replace(path)


def validate_protocol(protocol: dict[str, Any]) -> None:
    required = {"experiment_id", "status", "geometry", "feature", "topk", "training", "steps", "seeds", "analysis", "gates"}
    missing = required.difference(protocol)
    if missing:
        raise ValueError(f"protocol is missing keys: {sorted(missing)}")
    if protocol["experiment_id"] != "E4-topk-d32k4-budget-followup":
        raise ValueError("unexpected experiment identifier")
    if protocol["status"] != "posthoc-frozen-before-execution":
        raise ValueError("protocol must be frozen before execution")
    if protocol["geometry"] != ["axis", "dense"]:
        raise ValueError("geometry must be exactly [axis, dense]")
    if protocol["feature"]["m"] != 8 or protocol["feature"]["p"] <= 0 or protocol["feature"]["lambda_eff"] <= 0:
        raise ValueError("invalid feature settings")
    topk = protocol["topk"]
    if topk != {"id": "d32-k4", "dict_size": 32, "k": 4}:
        raise ValueError("budget follow-up is restricted to d32-k4")
    training = protocol["training"]
    if training["N_train"] <= 0 or training["batch"] <= 0 or training["lr"] <= 0 or not 0 < training["cosine_threshold"] <= 1:
        raise ValueError("invalid training settings")
    if protocol["steps"] != [4000, 8000, 16000]:
        raise ValueError("checkpoint steps must be exactly [4000, 8000, 16000]")
    if protocol["seeds"]["base"] != 20260901 or protocol["seeds"]["count"] != 8:
        raise ValueError("budget follow-up seed grid must be the original held-out seeds")


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    validate_protocol(protocol)
    return protocol


def confidence_interval(values: list[float], protocol: dict[str, Any], salt: int) -> dict[str, Any]:
    values_array = np.asarray(values, dtype=float)
    if len(values_array) == 0:
        return {"mean": None, "ci95": None, "n": 0, "values": []}
    rng = np.random.default_rng(protocol["analysis"]["bootstrap_seed"] + salt)
    indices = rng.integers(0, len(values_array), size=(int(protocol["analysis"]["bootstrap_resamples"]), len(values_array)))
    means = values_array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values_array.mean()), "ci95": [float(low), float(high)], "n": int(len(values_array)), "values": [float(value) for value in values_array]}


def geometry_directions(m: int) -> dict[str, np.ndarray]:
    axis, dense = e4.axis_dense(m)
    return {"axis": axis, "dense": dense}


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["seed"], row["geometry"], row["steps"])


def expected_training_seed(seed: int, protocol: dict[str, Any]) -> int:
    return int(protocol["seeds"]["torch_seed_offset"]) + int(seed) * 100 + 2


def expected_data_seed(seed: int, geometry: str, protocol: dict[str, Any]) -> int:
    geometry_index = protocol["geometry"].index(geometry)
    return int(protocol["seeds"]["training_rng_offset"]) + int(seed) * 1000 + 20 + geometry_index


def validate_rows(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    validate_protocol(protocol)
    expected_seeds = set(range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"]))
    expected_steps = set(protocol["steps"])
    threshold = float(protocol["training"]["cosine_threshold"])
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if row.get("seed") not in expected_seeds or row.get("geometry") not in protocol["geometry"] or row.get("steps") not in expected_steps:
            raise ValueError("row lies outside the frozen follow-up grid")
        key = row_key(row)
        if key in seen:
            raise ValueError(f"duplicate checkpoint row: {key!r}")
        seen.add(key)
        if int(row.get("dict_size", -1)) != 32 or int(row.get("k", -1)) != 4:
            raise ValueError("row TopK configuration is not d32-k4")
        if not np.isclose(float(row.get("lambda_eff")), float(protocol["feature"]["lambda_eff"]), rtol=0.0, atol=1e-12):
            raise ValueError("row lambda_eff does not match the frozen protocol")
        if int(row.get("data_seed", -1)) != expected_data_seed(int(row["seed"]), row["geometry"], protocol):
            raise ValueError("row data seed does not match the frozen protocol")
        if int(row.get("train_seed", -1)) != expected_training_seed(int(row["seed"]), protocol):
            raise ValueError("row train seed does not match the frozen protocol")
        cosine = float(row.get("decoder_cosine"))
        if bool(row.get("recovered")) != bool(cosine >= threshold):
            raise ValueError("row recovery flag disagrees with the frozen cosine threshold")


def train_checkpoints(seed: int, geometry: str, protocol: dict[str, Any], device: str) -> list[dict[str, Any]]:
    if e4.torch is None:
        raise RuntimeError("torch is required to execute the budget follow-up")
    feature, topk, training = protocol["feature"], protocol["topk"], protocol["training"]
    m, p = int(feature["m"]), float(feature["p"])
    direction = geometry_directions(m)[geometry]
    data_seed = expected_data_seed(seed, geometry, protocol)
    train_seed = expected_training_seed(seed, protocol)
    rng = np.random.default_rng(data_seed)
    amplitude = float(feature["lambda_eff"]) / np.sqrt(p)
    samples, _ = e4.bernoulli_h(rng, 1, int(training["N_train"]), m, amplitude, p, direction)

    torch = e4.torch
    torch.manual_seed(train_seed)
    model = e4.TopKSAE(m, int(topk["dict_size"]), int(topk["k"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["lr"]))
    inputs = torch.tensor(samples[0], dtype=torch.float32, device=device)
    checkpoints = set(protocol["steps"])
    max_steps = max(checkpoints)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for step in range(1, max_steps + 1):
        batch = torch.randint(0, inputs.shape[0], (int(training["batch"]),), device=device)
        reconstruction = model(inputs[batch])
        loss = torch.nn.functional.mse_loss(reconstruction, inputs[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            decoder = model.dec.weight
            model.dec.weight.copy_(decoder / (decoder.norm(dim=0, keepdim=True) + 1e-8))
        if step in checkpoints:
            cosine = float(e4.decoder_cosine(model, direction))
            rows.append({
                "seed": int(seed),
                "geometry": geometry,
                "steps": int(step),
                "dict_size": int(topk["dict_size"]),
                "k": int(topk["k"]),
                "lambda_eff": float(feature["lambda_eff"]),
                "data_seed": int(data_seed),
                "train_seed": int(train_seed),
                "decoder_cosine": cosine,
                "recovered": bool(cosine >= float(training["cosine_threshold"])),
                "elapsed_sec": time.perf_counter() - started,
            })
    return rows


def summarize(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    by_step: dict[str, dict[str, dict[str, Any]]] = {}
    for step_index, step in enumerate(protocol["steps"]):
        by_step[str(step)] = {}
        for geometry_index, geometry in enumerate(protocol["geometry"]):
            subset = [row for row in rows if row["steps"] == step and row["geometry"] == geometry]
            by_step[str(step)][geometry] = {
                "recovery": confidence_interval([float(row["recovered"]) for row in subset], protocol, 10 + step_index * 10 + geometry_index),
                "cosine": confidence_interval([float(row["decoder_cosine"]) for row in subset], protocol, 100 + step_index * 10 + geometry_index),
            }

    expected = {(seed, geometry, step) for seed in range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"]) for geometry in protocol["geometry"] for step in protocol["steps"]}
    actual = {row_key(row) for row in rows}
    complete = len(rows) == len(expected) and actual == expected
    final = by_step[str(max(protocol["steps"]))]
    axis_recovery = final["axis"]["recovery"]["mean"]
    dense_recovery = final["dense"]["recovery"]["mean"]
    axis_cosine = final["axis"]["cosine"]["mean"]
    dense_cosine = final["dense"]["cosine"]["mean"]
    recovery_gap = None if axis_recovery is None or dense_recovery is None else axis_recovery - dense_recovery
    cosine_gap = None if axis_cosine is None or dense_cosine is None else axis_cosine - dense_cosine
    first_dense = by_step[str(min(protocol["steps"]))]["dense"]["recovery"]["mean"]
    gates = protocol["gates"]
    budget_evidence = {
        "axis_recovery": axis_recovery is not None and axis_recovery >= gates["minimum_recovery_rate_at_16000"],
        "dense_recovery": dense_recovery is not None and dense_recovery >= gates["minimum_recovery_rate_at_16000"],
        "recovery_gap": recovery_gap is not None and abs(recovery_gap) <= gates["max_absolute_recovery_gap_at_16000"],
        "axis_cosine_penalty": cosine_gap is not None and cosine_gap >= -gates["max_axis_cosine_penalty_at_16000"],
        "dense_recovery_gain": first_dense is not None and dense_recovery is not None and dense_recovery - first_dense >= gates["minimum_dense_recovery_gain"],
    }
    budget_evidence["pass"] = all(budget_evidence.values())
    return {
        "by_step": by_step,
        "gates": {
            "complete": complete,
            "missing_rows": [list(key) for key in sorted(expected - actual, key=str)],
            "unexpected_rows": [list(key) for key in sorted(actual - expected, key=str)],
            "budget_evidence": budget_evidence,
            "all_pass": complete and budget_evidence["pass"],
        },
        "final_axis_minus_dense_recovery": recovery_gap,
        "final_axis_minus_dense_cosine": cosine_gap,
        "counts": {"rows": len(rows), "expected_rows": len(expected)},
    }


def run(protocol_path: Path, out_dir: Path, device: str, resume: bool) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if e4.torch is None:
        raise RuntimeError("torch is required to execute the budget follow-up")
    if device.startswith("cuda") and not e4.torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path, summary_path = out_dir / RAW_NAME, out_dir / SUMMARY_NAME
    if raw_path.exists() and not resume:
        raise FileExistsError(f"{raw_path} exists; pass --resume or choose a new --out directory")
    rows: list[dict[str, Any]] = json.loads(raw_path.read_text()) if raw_path.exists() else []
    validate_rows(rows, protocol)
    seen = {row_key(row) for row in rows}
    for seed in range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"]):
        for geometry in protocol["geometry"]:
            expected_rows = {(seed, geometry, step) for step in protocol["steps"]}
            if expected_rows.issubset(seen):
                continue
            if expected_rows.intersection(seen):
                raise ValueError("resume refuses a partial checkpoint trajectory; choose a clean output directory")
            for row in train_checkpoints(seed, geometry, protocol, device):
                rows.append(row)
                seen.add(row_key(row))
                atomic_write_json(raw_path, rows)
                print(f"[budget] seed={seed} geometry={geometry} steps={row['steps']} cosine={row['decoder_cosine']:.3f} recovered={row['recovered']}", flush=True)
    result = summarize(rows, protocol)
    result["metadata"] = {
        "experiment_id": protocol["experiment_id"],
        "status": "completed",
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "device": device,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(summary_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--out", type=Path, default=Path("results_e4_topk_budget_followup"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.protocol, args.out, args.device, args.resume)
    print(json.dumps(result["gates"], indent=2, sort_keys=True))
    if not result["gates"]["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
