#!/usr/bin/env python3
"""Held-out TopK confirmation for the E4 detector-versus-recovery result.

This script reuses the centered-Bernoulli generator, detector statistics, TopK
implementation, and decoder-cosine metric from run_v2.py. It changes only the
held-out seed range and the frozen capacity/sparsity grid.
"""

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


DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("e4_topk_confirmation_protocol.json")
RAW_NAME = "e4_topk_confirmation_raw.json"
SUMMARY_NAME = "e4_topk_confirmation_summary.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_protocol(protocol: dict[str, Any]) -> None:
    required = {
        "experiment_id",
        "status",
        "geometry",
        "feature",
        "detector",
        "topk_configs",
        "training",
        "seeds",
        "analysis",
        "gates",
    }
    missing = required.difference(protocol)
    if missing:
        raise ValueError(f"protocol is missing keys: {sorted(missing)}")
    if protocol["experiment_id"] != "E4-topk-heldout-confirmation":
        raise ValueError("unexpected experiment identifier")
    if protocol["status"] != "frozen-before-execution":
        raise ValueError("protocol must be frozen before execution")
    if protocol["geometry"] != ["axis", "dense"]:
        raise ValueError("geometry must be exactly [axis, dense]")

    feature = protocol["feature"]
    detector = protocol["detector"]
    training = protocol["training"]
    seeds = protocol["seeds"]
    if feature["m"] != 8 or feature["p"] <= 0 or feature["lambda_eff"] <= 0:
        raise ValueError("invalid feature settings")
    if detector["N"] <= 0 or detector["trials"] <= 0 or detector["null_draws"] <= 0:
        raise ValueError("invalid detector settings")
    if training["N_train"] <= 0 or training["steps"] <= 0 or training["batch"] <= 0:
        raise ValueError("invalid training settings")
    if not 0 < training["cosine_threshold"] <= 1:
        raise ValueError("invalid cosine recovery threshold")
    if seeds["count"] < 2:
        raise ValueError("confirmation requires at least two held-out seeds")

    configs = protocol["topk_configs"]
    identifiers = [config["id"] for config in configs]
    if not configs or len(identifiers) != len(set(identifiers)):
        raise ValueError("TopK configuration identifiers must be nonempty and unique")
    for config in configs:
        if config["dict_size"] < feature["m"]:
            raise ValueError(f"dictionary is not overcomplete: {config}")
        if not 0 < config["k"] <= config["dict_size"]:
            raise ValueError(f"invalid TopK configuration: {config}")


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    validate_protocol(protocol)
    return protocol


def confidence_interval(values: list[float], protocol: dict[str, Any], salt: int) -> dict[str, Any]:
    values_array = np.asarray(values, dtype=float)
    if len(values_array) == 0:
        return {"mean": None, "ci95": None, "n": 0, "values": []}
    rng = np.random.default_rng(protocol["analysis"]["bootstrap_seed"] + salt)
    resamples = int(protocol["analysis"]["bootstrap_resamples"])
    indices = rng.integers(0, len(values_array), size=(resamples, len(values_array)))
    means = values_array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values_array.mean()),
        "ci95": [float(low), float(high)],
        "n": int(len(values_array)),
        "values": [float(value) for value in values_array],
    }


def geometry_directions(m: int) -> dict[str, np.ndarray]:
    axis, dense = e4.axis_dense(m)
    return {"axis": axis, "dense": dense}


def detector_rows_for_seed(seed: int, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    feature = protocol["feature"]
    detector = protocol["detector"]
    seeds = protocol["seeds"]
    m = int(feature["m"])
    p = float(feature["p"])
    rng = np.random.default_rng(seed + int(seeds["detector_rng_offset"]))
    directions = geometry_directions(m)
    h0, z0 = e4.bernoulli_h(
        rng,
        int(detector["null_draws"]),
        int(detector["N"]),
        m,
        0.0,
        p,
        directions["axis"],
    )
    thresholds = {
        "label": e4.quantile_thresh(e4.label_stat_z(h0, z0), detector["alpha"]),
        "cov": e4.quantile_thresh(e4.cov_stat(h0), detector["alpha"]),
        "ind": e4.quantile_thresh(e4.ind_stat(h0), detector["alpha"]),
    }
    amplitude = float(feature["lambda_eff"]) / np.sqrt(p)
    rows = []
    for geometry in protocol["geometry"]:
        direction = directions[geometry]
        h1, z1 = e4.bernoulli_h(
            rng,
            int(detector["trials"]),
            int(detector["N"]),
            m,
            amplitude,
            p,
            direction,
        )
        rows.append(
            {
                "kind": "detector",
                "seed": seed,
                "geometry": geometry,
                "lambda_eff": float(feature["lambda_eff"]),
                "C_ind": float(e4.geom_C(direction)),
                "label": float(e4.power(e4.label_stat_z(h1, z1), thresholds["label"])),
                "cov": float(e4.power(e4.cov_stat(h1), thresholds["cov"])),
                "ind": float(e4.power(e4.ind_stat(h1), thresholds["ind"])),
                "thresholds": {key: float(value) for key, value in thresholds.items()},
            }
        )
    return rows


def training_row(
    seed: int,
    config_index: int,
    config: dict[str, Any],
    geometry: str,
    protocol: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    feature = protocol["feature"]
    training = protocol["training"]
    seed_settings = protocol["seeds"]
    m = int(feature["m"])
    p = float(feature["p"])
    geometry_index = protocol["geometry"].index(geometry)
    data_seed = (
        int(seed_settings["training_rng_offset"])
        + int(seed) * 1000
        + config_index * 10
        + geometry_index
    )
    rng = np.random.default_rng(data_seed)
    direction = geometry_directions(m)[geometry]
    amplitude = float(feature["lambda_eff"]) / np.sqrt(p)
    samples, _ = e4.bernoulli_h(rng, 1, int(training["N_train"]), m, amplitude, p, direction)

    # Pair initialization and minibatch sequence across geometry for each seed/config.
    train_seed = int(seed_settings["torch_seed_offset"]) + int(seed) * 100 + config_index
    started = time.perf_counter()
    model = e4.train_sae(
        samples[0],
        m,
        int(config["dict_size"]),
        int(config["k"]),
        int(training["steps"]),
        int(training["batch"]),
        float(training["lr"]),
        train_seed,
        device,
    )
    cosine = float(e4.decoder_cosine(model, direction))
    return {
        "kind": "training",
        "seed": int(seed),
        "geometry": geometry,
        "config_id": config["id"],
        "dict_size": int(config["dict_size"]),
        "k": int(config["k"]),
        "lambda_eff": float(feature["lambda_eff"]),
        "data_seed": int(data_seed),
        "train_seed": int(train_seed),
        "decoder_cosine": cosine,
        "recovered": bool(cosine >= float(training["cosine_threshold"])),
        "elapsed_sec": time.perf_counter() - started,
    }


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["kind"], row["seed"], row["geometry"], row.get("config_id"))


def validate_rows(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    """Reject malformed or protocol-incompatible rows before resume."""
    validate_protocol(protocol)
    expected_seeds = set(
        range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"])
    )
    config_by_id = {config["id"]: config for config in protocol["topk_configs"]}
    directions = geometry_directions(int(protocol["feature"]["m"]))
    expected_lambda = float(protocol["feature"]["lambda_eff"])
    expected_threshold = float(protocol["training"]["cosine_threshold"])
    seen: set[tuple[Any, ...]] = set()

    for row in rows:
        if row.get("kind") not in {"detector", "training"}:
            raise ValueError(f"unknown row kind: {row.get('kind')!r}")
        if row.get("seed") not in expected_seeds:
            raise ValueError(f"row seed is outside the frozen protocol: {row.get('seed')!r}")
        if row.get("geometry") not in protocol["geometry"]:
            raise ValueError(f"row geometry is outside the frozen protocol: {row.get('geometry')!r}")
        key = row_key(row)
        if key in seen:
            raise ValueError(f"duplicate row key: {key!r}")
        seen.add(key)
        if not np.isclose(float(row.get("lambda_eff")), expected_lambda, rtol=0.0, atol=1e-12):
            raise ValueError(f"row lambda_eff does not match the frozen protocol: {row.get('lambda_eff')!r}")

        geometry = row["geometry"]
        if row["kind"] == "detector":
            required = {"label", "cov", "ind", "C_ind", "thresholds"}
            if not required.issubset(row):
                raise ValueError(f"detector row is missing fields: {sorted(required.difference(row))}")
            expected_c = float(e4.geom_C(directions[geometry]))
            if not np.isclose(float(row["C_ind"]), expected_c, rtol=0.0, atol=1e-12):
                raise ValueError(f"detector geometry coefficient does not match protocol: {row['C_ind']!r}")
            if set(row["thresholds"]) != {"label", "cov", "ind"}:
                raise ValueError("detector thresholds do not match the frozen detector set")
        else:
            config_id = row.get("config_id")
            if config_id not in config_by_id:
                raise ValueError(f"unknown TopK configuration: {config_id!r}")
            config = config_by_id[config_id]
            for field in ("dict_size", "k"):
                if int(row.get(field, -1)) != int(config[field]):
                    raise ValueError(f"training {field} does not match configuration {config_id}")
            config_index = next(index for index, item in enumerate(protocol["topk_configs"]) if item["id"] == config_id)
            geometry_index = protocol["geometry"].index(geometry)
            expected_data_seed = (
                int(protocol["seeds"]["training_rng_offset"])
                + int(row["seed"]) * 1000
                + config_index * 10
                + geometry_index
            )
            expected_train_seed = (
                int(protocol["seeds"]["torch_seed_offset"])
                + int(row["seed"]) * 100
                + config_index
            )
            if int(row.get("data_seed", -1)) != expected_data_seed or int(row.get("train_seed", -1)) != expected_train_seed:
                raise ValueError(f"training seeds do not match the frozen protocol for {config_id}/{geometry}")
            cosine = float(row.get("decoder_cosine"))
            recovered = bool(row.get("recovered"))
            if recovered != bool(cosine >= expected_threshold):
                raise ValueError("training recovery flag disagrees with the frozen cosine threshold")


def mean_for(metric: str, rows: list[dict[str, Any]], protocol: dict[str, Any], salt: int) -> dict[str, Any]:
    return confidence_interval([float(row[metric]) for row in rows], protocol, salt)


def summarize(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    detector_rows = [row for row in rows if row["kind"] == "detector"]
    training_rows = [row for row in rows if row["kind"] == "training"]

    detector: dict[str, Any] = {}
    salt = 1
    for geometry in protocol["geometry"]:
        subset = [row for row in detector_rows if row["geometry"] == geometry]
        detector[geometry] = {}
        for metric in ("label", "cov", "ind"):
            detector[geometry][metric] = mean_for(metric, subset, protocol, salt)
            salt += 1

    detector_settings = protocol["detector"]
    detector_gate = {
        "axis_label": detector["axis"]["label"]["mean"] is not None
        and detector["axis"]["label"]["mean"] >= detector_settings["label_cov_min_power"],
        "dense_label": detector["dense"]["label"]["mean"] is not None
        and detector["dense"]["label"]["mean"] >= detector_settings["label_cov_min_power"],
        "axis_cov": detector["axis"]["cov"]["mean"] is not None
        and detector["axis"]["cov"]["mean"] >= detector_settings["label_cov_min_power"],
        "dense_cov": detector["dense"]["cov"]["mean"] is not None
        and detector["dense"]["cov"]["mean"] >= detector_settings["label_cov_min_power"],
        "axis_ind": detector["axis"]["ind"]["mean"] is not None
        and detector["axis"]["ind"]["mean"] <= detector_settings["axis_ind_max_power"],
        "dense_ind": detector["dense"]["ind"]["mean"] is not None
        and detector["dense"]["ind"]["mean"] >= detector_settings["dense_ind_min_power"],
    }
    detector_gate["pass"] = all(detector_gate.values())

    per_config: dict[str, Any] = {}
    gates = protocol["gates"]
    for config_index, config in enumerate(protocol["topk_configs"]):
        summary_by_geometry: dict[str, Any] = {}
        for geometry_index, geometry in enumerate(protocol["geometry"]):
            subset = [
                row
                for row in training_rows
                if row["config_id"] == config["id"] and row["geometry"] == geometry
            ]
            summary_by_geometry[geometry] = {
                "recovery": mean_for("recovered", subset, protocol, 100 + config_index * 10 + geometry_index),
                "cosine": mean_for("decoder_cosine", subset, protocol, 200 + config_index * 10 + geometry_index),
            }
        axis_recovery = summary_by_geometry["axis"]["recovery"]["mean"]
        dense_recovery = summary_by_geometry["dense"]["recovery"]["mean"]
        axis_cosine = summary_by_geometry["axis"]["cosine"]["mean"]
        dense_cosine = summary_by_geometry["dense"]["cosine"]["mean"]
        recovery_gap = None if axis_recovery is None or dense_recovery is None else axis_recovery - dense_recovery
        cosine_gap = None if axis_cosine is None or dense_cosine is None else axis_cosine - dense_cosine
        config_gate = {
            "axis_recovery": axis_recovery is not None and axis_recovery >= gates["minimum_recovery_rate"],
            "dense_recovery": dense_recovery is not None and dense_recovery >= gates["minimum_recovery_rate"],
            "recovery_gap": recovery_gap is not None and abs(recovery_gap) <= gates["max_absolute_recovery_gap"],
            "axis_cosine_penalty": cosine_gap is not None and cosine_gap >= -gates["max_axis_cosine_penalty"],
        }
        config_gate["pass"] = all(config_gate.values())
        per_config[config["id"]] = {
            "config": config,
            "axis": summary_by_geometry["axis"],
            "dense": summary_by_geometry["dense"],
            "axis_minus_dense_recovery": recovery_gap,
            "axis_minus_dense_cosine": cosine_gap,
            "gate_details": config_gate,
            "pass": config_gate["pass"],
        }

    expected_keys = {
        ("detector", seed, geometry, None)
        for seed in range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"])
        for geometry in protocol["geometry"]
    }
    expected_keys.update(
        ("training", seed, geometry, config["id"])
        for seed in range(protocol["seeds"]["base"], protocol["seeds"]["base"] + protocol["seeds"]["count"])
        for config in protocol["topk_configs"]
        for geometry in protocol["geometry"]
    )
    actual_keys = {row_key(row) for row in rows}
    complete = len(rows) == len(expected_keys) and actual_keys == expected_keys
    all_pass = complete and detector_gate["pass"] and all(config["pass"] for config in per_config.values())
    return {
        "detector": detector,
        "training": per_config,
        "gates": {
            "complete": complete,
            "missing_rows": [list(key) for key in sorted(expected_keys - actual_keys, key=str)],
            "unexpected_rows": [list(key) for key in sorted(actual_keys - expected_keys, key=str)],
            "detector": detector_gate,
            "per_config": per_config,
            "all_pass": all_pass,
        },
        "counts": {
            "detector_rows": len(detector_rows),
            "training_rows": len(training_rows),
            "expected_rows": len(expected_keys),
        },
    }


def run(protocol_path: Path, out_dir: Path, device: str, resume: bool) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if e4.torch is None:
        raise RuntimeError("torch is required to execute the TopK confirmation")
    if device.startswith("cuda") and not e4.torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / RAW_NAME
    summary_path = out_dir / SUMMARY_NAME
    if raw_path.exists() and not resume:
        raise FileExistsError(f"{raw_path} exists; pass --resume or choose a new --out directory")
    rows: list[dict[str, Any]] = json.loads(raw_path.read_text()) if raw_path.exists() else []
    validate_rows(rows, protocol)
    seen = {row_key(row) for row in rows}

    seed_settings = protocol["seeds"]
    for seed in range(seed_settings["base"], seed_settings["base"] + seed_settings["count"]):
        for row in detector_rows_for_seed(seed, protocol):
            if row_key(row) not in seen:
                rows.append(row)
                seen.add(row_key(row))
                atomic_write_json(raw_path, rows)
        for config_index, config in enumerate(protocol["topk_configs"]):
            for geometry in protocol["geometry"]:
                key = ("training", seed, geometry, config["id"])
                if key in seen:
                    continue
                row = training_row(seed, config_index, config, geometry, protocol, device)
                rows.append(row)
                seen.add(key)
                atomic_write_json(raw_path, rows)
                print(
                    f"[confirmation] seed={seed} config={config['id']} geometry={geometry} "
                    f"cosine={row['decoder_cosine']:.3f} recovered={row['recovered']}",
                    flush=True,
                )

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
    parser.add_argument("--out", type=Path, default=Path("results_e4_topk_confirmation"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.protocol, args.out, args.device, args.resume)
    print(json.dumps(result["gates"], indent=2, sort_keys=True))
    if not result["gates"]["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
