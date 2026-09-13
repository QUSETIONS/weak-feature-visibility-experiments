#!/usr/bin/env python3
"""Matched spectral warm-start validation on empirical LM residual backgrounds.

The target is used only for held-out evaluation.  Warm-start estimation uses the
injected training split, and the two variants share data, initialization RNG,
minibatches, optimizer, and compute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_spectral_warm_start_matched as warm

DEFAULT_PROTOCOL = ROOT / "lm_spectral_warm_start_matched_protocol.json"
DEFAULT_OUT = ROOT / "results_lm_spectral_warm_start_matched"


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(protocol: dict[str, Any]) -> None:
    required = {
        "experiment_id", "status", "models", "corpus", "residuals", "feature",
        "sae", "variants", "seeds", "evaluation", "comparison", "analysis",
        "data_quality_gates", "evidence_status", "reporting_boundary",
    }
    missing = required - protocol.keys()
    if missing:
        raise ValueError(f"LM protocol is missing fields: {sorted(missing)}")
    if protocol["status"] != "frozen-before-execution":
        raise ValueError("LM protocol must be frozen before execution")
    model_ids = [item["id"] for item in protocol["models"]]
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError("the protocol must contain at least one uniquely named model")
    if any(int(item.get("expected_hidden_size", 0)) <= 0 for item in protocol["models"]):
        raise ValueError("every model must freeze a positive expected hidden size")
    if protocol["feature"]["directions"] != ["pc1", "pc32", "dense"]:
        raise ValueError("directions must be pc1, pc32, dense")
    if protocol["feature"]["strengths"] != [0.35, 0.55]:
        raise ValueError("strengths must be 0.35 and 0.55")
    if protocol["corpus"].get("synthetic_fallback_forbidden") is not True:
        raise ValueError("synthetic corpus fallback must be forbidden")
    if [item["id"] for item in protocol["variants"]] != ["vanilla-random", "spectral-warm-start"]:
        raise ValueError("variant order is not frozen")
    sae = protocol["sae"]
    if sae.get("post_initialization_constraints") != "none":
        raise ValueError("LM warm start must be unconstrained after initialization")
    if protocol["evaluation"].get("gaussian_certificate_forbidden") is not True:
        raise ValueError("the Gaussian certificate must not be asserted on empirical residuals")
    n_pairs_model = len(protocol["feature"]["directions"]) * len(protocol["feature"]["strengths"]) * int(protocol["seeds"]["count"])
    if int(protocol["comparison"]["expected_pairs_per_model"]) != n_pairs_model:
        raise ValueError("expected pair count per model is inconsistent")
    if int(protocol["comparison"]["expected_rows_per_model"]) != n_pairs_model * len(protocol["variants"]):
        raise ValueError("expected row count per model is inconsistent")


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    validate_protocol(protocol)
    return protocol


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def direction_vectors(m: int) -> dict[str, np.ndarray]:
    pc1 = np.zeros(m, dtype=np.float64)
    pc1[0] = 1.0
    pc32 = np.zeros(m, dtype=np.float64)
    pc32[-1] = 1.0
    return {"pc1": pc1, "pc32": pc32, "dense": unit(np.ones(m, dtype=np.float64))}


def seed_triplet(protocol: dict[str, Any], model_index: int, direction_index: int, strength_index: int, seed: int) -> tuple[int, int, int]:
    offsets = protocol["seeds"]
    suffix = seed * 1000 + model_index * 100 + direction_index * 10 + strength_index
    return (
        int(offsets["data_rng_offset"]) + suffix,
        int(offsets["torch_rng_offset"]) + suffix,
        int(offsets["validation_rng_offset"]) + suffix,
    )


def load_real_texts(protocol: dict[str, Any], text_file: Path | None = None) -> tuple[list[str], dict[str, Any]]:
    minimum = int(protocol["corpus"]["minimum_nonempty_documents"])
    if text_file is not None:
        texts = [line.strip() for line in text_file.read_text(errors="ignore").splitlines() if line.strip()]
        if len(texts) < minimum:
            raise RuntimeError(f"real text file has only {len(texts)} nonempty lines; need {minimum}")
        return texts, {
            "source_type": "local_real_text_file",
            "path": str(text_file.resolve()),
            "sha256": sha256_file(text_file),
            "nonempty_documents": len(texts),
            "synthetic_fallback": False,
        }
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required; synthetic fallback is forbidden") from exc
    failures = []
    for dataset_name, config_name in protocol["corpus"]["dataset_candidates"]:
        try:
            dataset = load_dataset(dataset_name, config_name, split=protocol["corpus"]["split"])
            texts = [str(value).strip() for value in dataset["text"] if str(value).strip()]
            if len(texts) < minimum:
                failures.append(f"{dataset_name}/{config_name}: only {len(texts)} nonempty documents")
                continue
            return texts, {
                "source_type": "huggingface_dataset",
                "dataset_name": dataset_name,
                "config_name": config_name,
                "split": protocol["corpus"]["split"],
                "fingerprint": getattr(dataset, "_fingerprint", None),
                "nonempty_documents": len(texts),
                "synthetic_fallback": False,
            }
        except Exception as exc:  # fail closed after exhausting named real sources
            failures.append(f"{dataset_name}/{config_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("no approved real WikiText source could be loaded; " + " | ".join(failures))


def tokenize_and_split(tokenizer: Any, texts: list[str], protocol: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    residuals = protocol["residuals"]
    seq_len = int(residuals["sequence_length"])
    n_train = int(residuals["train_sequences"])
    n_validation = int(residuals["validation_sequences"])
    needed = (n_train + n_validation) * seq_len
    token_ids: list[int] = []
    for text in texts:
        token_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(token_ids) >= needed + seq_len:
            break
    usable = (len(token_ids) // seq_len) * seq_len
    if usable < needed:
        raise RuntimeError(f"real corpus yielded {usable} tokens; {needed} are required and repetition is forbidden")
    sequences = np.asarray(token_ids[:usable], dtype=np.int64).reshape(-1, seq_len)
    rng = np.random.default_rng(int(protocol["corpus"]["sequence_shuffle_seed"]))
    order = rng.permutation(sequences.shape[0])[: n_train + n_validation]
    train_indices = order[:n_train]
    validation_indices = order[n_train:]
    if set(train_indices.tolist()) & set(validation_indices.tolist()):
        raise RuntimeError("train and validation sequence indices overlap")
    train = sequences[train_indices]
    validation = sequences[validation_indices]
    train_content = {hashlib.sha256(row.tobytes()).hexdigest() for row in train}
    validation_content = {hashlib.sha256(row.tobytes()).hexdigest() for row in validation}
    metadata = {
        "available_tokens": len(token_ids),
        "used_tokens": needed,
        "sequence_length": seq_len,
        "train_sequences": n_train,
        "validation_sequences": n_validation,
        "sequence_indices_disjoint": True,
        "exact_content_overlap_sequences": len(train_content & validation_content),
        "train_sequence_index_sha256": hashlib.sha256(train_indices.tobytes()).hexdigest(),
        "validation_sequence_index_sha256": hashlib.sha256(validation_indices.tobytes()).hexdigest(),
    }
    return train, validation, metadata


def resolve_model(model_spec: dict[str, Any], model_dir: Path | None, device: str) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    source = str(model_dir.resolve()) if model_dir is not None else model_spec["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(source).to(device)
    model.eval()
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size != int(model_spec["expected_hidden_size"]):
        raise RuntimeError(f"hidden size {hidden_size} does not match frozen value {model_spec['expected_hidden_size']}")
    return model, tokenizer, {
        "source": source,
        "hf_id": model_spec["hf_id"],
        "config_name_or_path": getattr(model.config, "_name_or_path", None),
        "commit_hash": getattr(model.config, "_commit_hash", None),
        "hidden_size": hidden_size,
        "torch_version": torch.__version__,
    }


def collect_residuals(model: Any, sequences: np.ndarray, layer: int, batch: int, device: str) -> np.ndarray:
    import torch

    chunks = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch):
            tokens = torch.as_tensor(sequences[start : start + batch], dtype=torch.long, device=device)
            output = model(tokens, output_hidden_states=True)
            hidden = output.hidden_states[layer].detach().float().cpu().numpy()
            chunks.append(hidden.reshape(-1, hidden.shape[-1]))
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def fit_whitening_projection(hidden: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = hidden.mean(axis=0, dtype=np.float64)
    covariance = np.zeros((hidden.shape[1], hidden.shape[1]), dtype=np.float64)
    for start in range(0, len(hidden), 32768):
        centered = hidden[start : start + 32768].astype(np.float64) - mean
        covariance += centered.T @ centered
    covariance /= max(len(hidden) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 1e-8, None)
    vectors = vectors[:, order]
    projection = vectors[:, :m] / np.sqrt(values[:m]).reshape(1, -1)
    return mean, projection, values[:m]


def apply_projection(hidden: np.ndarray, mean: np.ndarray, projection: np.ndarray) -> np.ndarray:
    output = np.empty((len(hidden), projection.shape[1]), dtype=np.float32)
    for start in range(0, len(hidden), 32768):
        output[start : start + 32768] = (hidden[start : start + 32768].astype(np.float64) - mean) @ projection
    return output


def prepare_residual_cache(protocol: dict[str, Any], model_spec: dict[str, Any], model_dir: Path | None, text_file: Path | None, device: str, cache_path: Path) -> dict[str, Any]:
    model, tokenizer, model_meta = resolve_model(model_spec, model_dir, device)
    texts, corpus_meta = load_real_texts(protocol, text_file)
    train_sequences, validation_sequences, split_meta = tokenize_and_split(tokenizer, texts, protocol)
    batch = int(protocol["residuals"]["extraction_batch"])
    layer = int(model_spec["layer"])
    hidden_train = collect_residuals(model, train_sequences, layer, batch, device)
    mean, projection, eigenvalues = fit_whitening_projection(hidden_train, int(protocol["residuals"]["sae_dimensions"]))
    z_train = apply_projection(hidden_train, mean, projection)
    del hidden_train
    hidden_validation = collect_residuals(model, validation_sequences, layer, batch, device)
    z_validation = apply_projection(hidden_validation, mean, projection)
    del hidden_validation, model
    covariance_diag = np.var(z_train.astype(np.float64), axis=0, ddof=1)
    metadata = {
        "model": model_meta,
        "corpus": corpus_meta,
        "split": split_meta,
        "train_rows": len(z_train),
        "validation_rows": len(z_validation),
        "train_whitened_variance_min": float(covariance_diag.min()),
        "train_whitened_variance_max": float(covariance_diag.max()),
        "pca_eigenvalue_min_retained": float(eigenvalues.min()),
        "pca_eigenvalue_max_retained": float(eigenvalues.max()),
        "real_corpus": corpus_meta.get("synthetic_fallback") is False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, z_train=z_train, z_validation=z_validation, mean=mean, projection=projection, eigenvalues=eigenvalues, metadata=np.asarray(json.dumps(metadata)))
    return {"z_train": z_train, "z_validation": z_validation, "metadata": metadata}


def load_or_prepare_cache(protocol: dict[str, Any], model_spec: dict[str, Any], model_dir: Path | None, text_file: Path | None, device: str, cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"].item()))
        if metadata["model"]["hidden_size"] != model_spec["expected_hidden_size"] or metadata["real_corpus"] is not True:
            raise RuntimeError("existing residual cache fails the frozen provenance gates")
        return {"z_train": data["z_train"], "z_validation": data["z_validation"], "metadata": metadata}
    return prepare_residual_cache(protocol, model_spec, model_dir, text_file, device, cache_path)


def inject(base: np.ndarray, target: np.ndarray, strength: float, prevalence: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = rng.binomial(1, prevalence, size=(len(base), 1)).astype(np.float32)
    amplitude = strength / math.sqrt(prevalence)
    samples = base.astype(np.float32, copy=True) + np.float32(amplitude) * (labels - np.float32(prevalence)) * target.astype(np.float32).reshape(1, -1)
    return samples, labels.reshape(-1)


def decoder_columns(model: Any) -> np.ndarray:
    matrix = model.dec.weight.detach().cpu().numpy().astype(np.float64)
    return matrix / (np.linalg.norm(matrix, axis=0, keepdims=True) + 1e-12)


def span_cosine(columns: np.ndarray, slots: list[int], target: np.ndarray) -> float:
    basis, singular, _ = np.linalg.svd(columns[:, slots], full_matrices=False)
    keep = singular > max(singular[0] * 0.05, 1e-12)
    return float(np.linalg.norm(basis[:, keep].T @ unit(target))) if np.any(keep) else 0.0


def evaluate_model(model: Any, samples: np.ndarray, device: str, batch: int = 4096) -> dict[str, float]:
    import torch

    squared_error = 0.0
    scalar_count = 0
    active_total = 0.0
    row_count = 0
    usage = None
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch):
            value = torch.as_tensor(samples[start : start + batch], dtype=torch.float32, device=device)
            reconstruction, code = model(value)
            squared_error += float((reconstruction - value).square().sum().item())
            scalar_count += int(value.numel())
            active_total += float((code > 0).sum().item())
            row_count += int(len(value))
            part = (code > 0).sum(dim=0).detach().cpu().numpy()
            usage = part if usage is None else usage + part
    return {
        "reconstruction_mse": squared_error / scalar_count,
        "active_l0": active_total / row_count,
        "dead_atom_fraction": float(np.mean(usage == 0)),
    }


def row_key(row: dict[str, Any]) -> tuple[str, str, float, int, str]:
    return (row["model_id"], row["direction"], float(row["strength"]), int(row["seed"]), row["variant_id"])


def validate_model_rows(rows: list[dict[str, Any]], protocol: dict[str, Any], model_id: str) -> None:
    expected = {
        (model_id, direction, float(strength), seed, variant["id"])
        for direction in protocol["feature"]["directions"]
        for strength in protocol["feature"]["strengths"]
        for seed in range(int(protocol["seeds"]["base"]), int(protocol["seeds"]["base"]) + int(protocol["seeds"]["count"]))
        for variant in protocol["variants"]
    }
    seen = set()
    for row in rows:
        key = row_key(row)
        if key in seen or key not in expected:
            raise ValueError(f"duplicate or unexpected LM result row: {key}")
        seen.add(key)
        for metric in ("oracle_target_cosine", "spectral_readout_target_cosine", "heldout_reconstruction_mse", "heldout_active_l0", "heldout_dead_atom_fraction"):
            if not math.isfinite(float(row[metric])):
                raise ValueError(f"non-finite metric {metric} in {key}")
        if bool(row["recovered"]) != bool(float(row["oracle_target_cosine"]) >= float(protocol["evaluation"]["cosine_threshold"])):
            raise ValueError(f"recovery flag mismatch in {key}")
        if row["gaussian_certificate_asserted"] is not False or row["target_used_for_training"] is not False:
            raise ValueError(f"claim-boundary violation in {key}")
    if seen != expected:
        raise ValueError(f"incomplete LM result: got {len(seen)} rows, expected {len(expected)}")
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    for pair_id, pair in by_pair.items():
        if len(pair) != 2 or {row["variant_id"] for row in pair} != {"vanilla-random", "spectral-warm-start"}:
            raise ValueError(f"incomplete pair {pair_id}")
        matched = {(row["data_seed"], row["train_seed"], row["validation_seed"]) for row in pair}
        if len(matched) != 1:
            raise ValueError(f"seed mismatch within pair {pair_id}")


def bootstrap_ci(values: list[float], resamples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize_model(rows: list[dict[str, Any]], protocol: dict[str, Any], model_id: str, metadata: dict[str, Any], protocol_path: Path) -> dict[str, Any]:
    validate_model_rows(rows, protocol, model_id)
    variants = {}
    for variant in protocol["variants"]:
        subset = [row for row in rows if row["variant_id"] == variant["id"]]
        variants[variant["id"]] = {
            "rows": len(subset),
            "recovered": int(sum(row["recovered"] for row in subset)),
            "recovery_rate": float(np.mean([row["recovered"] for row in subset])),
            "mean_oracle_target_cosine": float(np.mean([row["oracle_target_cosine"] for row in subset])),
            "mean_spectral_readout_target_cosine": float(np.mean([row["spectral_readout_target_cosine"] for row in subset])),
            "mean_heldout_reconstruction_mse": float(np.mean([row["heldout_reconstruction_mse"] for row in subset])),
        }
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], {})[row["variant_id"]] = row
    seed_clusters: dict[int, list[dict[str, float]]] = {}
    improvements = 0
    for pair in pairs.values():
        vanilla, spectral = pair["vanilla-random"], pair["spectral-warm-start"]
        delta = {
            "recovery": float(spectral["recovered"]) - float(vanilla["recovered"]),
            "cosine": float(spectral["oracle_target_cosine"]) - float(vanilla["oracle_target_cosine"]),
            "mse": float(spectral["heldout_reconstruction_mse"]) - float(vanilla["heldout_reconstruction_mse"]),
        }
        improvements += delta["cosine"] > 0
        seed_clusters.setdefault(int(vanilla["seed"]), []).append(delta)
    cluster_rows = []
    expected_conditions = len(protocol["feature"]["directions"]) * len(protocol["feature"]["strengths"])
    for seed, deltas in sorted(seed_clusters.items()):
        if len(deltas) != expected_conditions:
            raise ValueError(f"seed cluster {seed} is incomplete")
        cluster_rows.append({
            "seed": seed,
            "conditions": len(deltas),
            "mean_delta_recovery": float(np.mean([item["recovery"] for item in deltas])),
            "mean_delta_cosine": float(np.mean([item["cosine"] for item in deltas])),
            "mean_delta_heldout_mse": float(np.mean([item["mse"] for item in deltas])),
        })
    analysis = protocol["analysis"]
    resamples = int(analysis["bootstrap_resamples"])
    boot_seed = int(analysis["bootstrap_seed"])
    clustered = {"primary_unit": "base_seed_cluster_within_model", "clusters": len(cluster_rows), "conditions_per_cluster": expected_conditions, "rows": cluster_rows}
    for index, key in enumerate(("mean_delta_recovery", "mean_delta_cosine", "mean_delta_heldout_mse")):
        values = [row[key] for row in cluster_rows]
        clustered[key] = float(np.mean(values))
        clustered[key + "_bootstrap_ci95"] = bootstrap_ci(values, resamples, boot_seed + index)
    quality = {
        "real_corpus": metadata["real_corpus"] is True,
        "sequence_indices_disjoint": metadata["split"]["sequence_indices_disjoint"] is True,
        "hidden_size_matches": metadata["model"]["hidden_size"] == next(item["expected_hidden_size"] for item in protocol["models"] if item["id"] == model_id),
        "complete_pairs": len(pairs) == int(protocol["comparison"]["expected_pairs_per_model"]),
        "finite_metrics": True,
    }
    return {
        "experiment_id": protocol["experiment_id"],
        "model_id": model_id,
        "status": "completed",
        "rows": len(rows),
        "pairs": len(pairs),
        "pair_integrity": True,
        "data_quality": quality,
        "data_metadata": metadata,
        "variant_summary": variants,
        "paired_cosine_improvements": improvements,
        "clustered_paired_analysis": clustered,
        "gaussian_certificate_asserted": False,
        "evidence_eligible_model_specific": all(quality.values()),
        "reporting_boundary": protocol["reporting_boundary"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_model(protocol_path: Path, out_root: Path, model_id: str, model_dir: Path | None, text_file: Path | None, device: str) -> dict[str, Any]:
    import torch

    protocol = load_protocol(protocol_path)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    model_index = next((index for index, item in enumerate(protocol["models"]) if item["id"] == model_id), None)
    if model_index is None:
        raise ValueError(f"unknown model id: {model_id}")
    model_spec = protocol["models"][model_index]
    out_dir = out_root / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_or_prepare_cache(protocol, model_spec, model_dir, text_file, device, out_dir / "residual_cache.npz")
    z_train = np.asarray(cache["z_train"], dtype=np.float32)
    z_validation = np.asarray(cache["z_validation"], dtype=np.float32)
    raw_path = out_dir / "matched_raw.json"
    rows = json.loads(raw_path.read_text()) if raw_path.exists() else []
    completed = {row_key(row) for row in rows}
    directions = direction_vectors(int(protocol["residuals"]["sae_dimensions"]))
    sae = protocol["sae"]
    threshold = float(protocol["evaluation"]["cosine_threshold"])
    for seed in range(int(protocol["seeds"]["base"]), int(protocol["seeds"]["base"]) + int(protocol["seeds"]["count"])):
        for direction_index, direction_name in enumerate(protocol["feature"]["directions"]):
            target = directions[direction_name]
            for strength_index, strength in enumerate(protocol["feature"]["strengths"]):
                data_seed, train_seed, validation_seed = seed_triplet(protocol, model_index, direction_index, strength_index, seed)
                train_rng = np.random.default_rng(data_seed)
                validation_rng = np.random.default_rng(validation_seed)
                train_indices = train_rng.integers(0, len(z_train), size=int(sae["N_train"]))
                validation_indices = validation_rng.integers(0, len(z_validation), size=int(sae["N_validation"]))
                train_samples, _ = inject(z_train[train_indices], target, float(strength), float(protocol["feature"]["prevalence"]), train_rng)
                validation_samples, _ = inject(z_validation[validation_indices], target, float(strength), float(protocol["feature"]["prevalence"]), validation_rng)
                spectral_anchor = warm.estimate_spectral_anchor(train_samples)
                pair_id = f"{model_id}:{direction_name}:{float(strength):.2f}:{seed}"
                for variant in protocol["variants"]:
                    key = (model_id, direction_name, float(strength), seed, variant["id"])
                    if key in completed:
                        continue
                    started = time.perf_counter()
                    model, training_metrics = warm.train_matched_sae(
                        train_samples,
                        spectral_anchor,
                        int(sae["dict_size"]),
                        int(sae["k"]),
                        int(sae["steps"]),
                        int(sae["batch"]),
                        float(sae["lr"]),
                        train_seed,
                        {**sae["objective"], "variant_id": variant["id"]},
                        device,
                    )
                    columns = decoder_columns(model)
                    target_cosines = np.abs(columns.T @ unit(target))
                    spectral_cosines = np.abs(columns.T @ unit(spectral_anchor))
                    spectral_atom = int(np.argmax(spectral_cosines))
                    heldout = evaluate_model(model, validation_samples, device)
                    row = {
                        "model_id": model_id,
                        "model_layer": int(model_spec["layer"]),
                        "direction": direction_name,
                        "strength": float(strength),
                        "seed": seed,
                        "pair_id": pair_id,
                        "variant_id": variant["id"],
                        "variant_initialization": variant["initialization"],
                        "data_seed": data_seed,
                        "train_seed": train_seed,
                        "validation_seed": validation_seed,
                        "oracle_target_atom": int(np.argmax(target_cosines)),
                        "oracle_target_cosine": float(np.max(target_cosines)),
                        "spectral_readout_atom": spectral_atom,
                        "spectral_readout_target_cosine": float(target_cosines[spectral_atom]),
                        "training_spectral_target_cosine": float(abs(np.dot(unit(spectral_anchor), unit(target)))),
                        "signed_pair_span_target_cosine": span_cosine(columns, [0, 1], target) if variant["id"] == "spectral-warm-start" else None,
                        "recovered": bool(float(np.max(target_cosines)) >= threshold),
                        "heldout_reconstruction_mse": heldout["reconstruction_mse"],
                        "heldout_active_l0": heldout["active_l0"],
                        "heldout_dead_atom_fraction": heldout["dead_atom_fraction"],
                        "final_training_objective": float(training_metrics["objective_loss"]),
                        "initial_anchor_cosine": float(training_metrics["initial_anchor_cosine"]),
                        "final_reference_cosine_max": float(training_metrics["final_reference_cosine_max"]),
                        "post_initialization_constraints": "none",
                        "target_used_for_training": False,
                        "validation_used_for_training_or_selection": False,
                        "gaussian_certificate_asserted": False,
                        "elapsed_sec": time.perf_counter() - started,
                    }
                    rows.append(row)
                    completed.add(key)
                    atomic_write(raw_path, rows)
                    print(f"[lm-warm] {pair_id} {variant['id']} cosine={row['oracle_target_cosine']:.3f} mse={row['heldout_reconstruction_mse']:.5f}", flush=True)
    summary = summarize_model(rows, protocol, model_id, cache["metadata"], protocol_path)
    atomic_write(out_dir / "matched_summary.json", summary)
    return summary


def aggregate(protocol_path: Path, out_root: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    summaries = []
    total_rows = 0
    total_pairs = 0
    for model in protocol["models"]:
        path = out_root / model["id"] / "matched_summary.json"
        if not path.exists():
            raise RuntimeError(f"missing completed model summary: {path}")
        summary = json.loads(path.read_text())
        summaries.append(summary)
        total_rows += int(summary["rows"])
        total_pairs += int(summary["pairs"])
    eligible = (
        total_rows == int(protocol["comparison"]["expected_rows_total"])
        and total_pairs == int(protocol["comparison"]["expected_pairs_total"])
        and all(item["evidence_eligible_model_specific"] for item in summaries)
    )
    result = {
        "experiment_id": protocol["experiment_id"],
        "status": "completed" if eligible else "incomplete_or_failed_gate",
        "models": {item["model_id"]: item for item in summaries},
        "rows": total_rows,
        "pairs": total_pairs,
        "pair_integrity": all(item["pair_integrity"] for item in summaries),
        "evidence_eligible": eligible,
        "pooled_result_role": "descriptive_only",
        "gaussian_certificate_asserted": False,
        "reporting_boundary": protocol["reporting_boundary"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(protocol_path),
    }
    atomic_write(out_root / "matched_summary_all_models.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.aggregate_only:
        print(json.dumps(aggregate(args.protocol, args.out), indent=2))
        return
    if args.model is None:
        parser.error("--model is required unless --aggregate-only is used")
    print(json.dumps(run_model(args.protocol, args.out, args.model, args.model_dir, args.text_file, args.device), indent=2))


if __name__ == "__main__":
    main()
