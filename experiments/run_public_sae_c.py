#!/usr/bin/env python3
"""Geometry of a public GPT-2-small residual SAE (no training).

Loads a pretrained decoder (Bloom GPT2-Small-SAEs, layer 6 residual) and reports
per-atom C = 1 - ||w||_4^4 in residual coordinates, and — if GPT-2 residuals
are available — after projection into the same 32-d PCA-whitened slice used
by the injection bridge.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

SEARCH_DIRS = (
    Path("experiments/weights"),
    Path("/mnt/liuzelin/weak_feature_visibility/experiments/weights"),
    Path("/mnt/e1_runs/hf_cache/hub"),
    Path("/mnt/e2_runs/work/e2_real_experiment_code_v5_scheme_aligned_20260619_0118/.hf_cache/hub"),
    Path(os.path.expanduser("~/.cache/huggingface/hub")),
)

CANDIDATE_FILES = (
    "sae_weights.safetensors",
    "final_sparse_autoencoder_gpt2-small_blocks.6.hook_resid_pre_24576.pt",
    "W_dec.npy",
    "w_dec.npy",
)


def unit_cols(D):
    D = np.asarray(D, dtype=np.float64)
    n = np.linalg.norm(D, axis=0, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return D / n


def atom_C(D):
    cols = unit_cols(D)
    return 1.0 - np.sum(cols**4, axis=0)


def _as_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _looks_like_dec(arr):
    if arr is None or not hasattr(arr, "shape") or len(arr.shape) != 2:
        return False
    a, b = int(arr.shape[0]), int(arr.shape[1])
    return 768 in (a, b) and max(a, b) >= 1024


def _orient_dec(arr):
    """Return D with shape (d_model, n_atoms)."""
    arr = np.asarray(_as_numpy(arr), dtype=np.float64)
    if arr.shape[0] == 768 and arr.shape[1] != 768:
        return arr
    if arr.shape[1] == 768 and arr.shape[0] != 768:
        return arr.T
    if arr.shape[0] >= arr.shape[1]:
        return arr
    return arr.T


def _extract_from_mapping(obj, depth=0):
    if depth > 6 or obj is None:
        return None
    if hasattr(obj, "items"):
        keys = list(obj.keys())
        for k in keys:
            ks = str(k).lower()
            if ks in ("w_dec", "decoder.weight", "dec.weight", "ae.w_dec"):
                v = obj[k]
                if _looks_like_dec(v):
                    return v
        for k in keys:
            if "dec" in str(k).lower() and "weight" in str(k).lower():
                v = obj[k]
                if _looks_like_dec(v):
                    return v
        for k in keys:
            hit = _extract_from_mapping(obj[k], depth + 1)
            if hit is not None:
                return hit
    if hasattr(obj, "state_dict"):
        try:
            return _extract_from_mapping(obj.state_dict(), depth + 1)
        except Exception:
            return None
    return None


def load_decoder(path: Path):
    path = Path(path)
    if path.suffix == ".npy":
        return _orient_dec(np.load(path)), str(path)
    if path.suffix == ".npz":
        z = np.load(path)
        for k in ("W_dec", "w_dec", "decoder"):
            if k in z:
                return _orient_dec(z[k]), str(path)
        return _orient_dec(z[z.files[0]]), str(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("safetensors is required to read " + str(path)) from exc
        sd = load_file(str(path), device="cpu")
        hit = _extract_from_mapping(sd)
        if hit is None:
            raise RuntimeError("no decoder matrix in " + str(path) + " keys=" + str(list(sd.keys())))
        return _orient_dec(hit), str(path)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to read " + str(path)) from exc
    obj = torch.load(str(path), map_location="cpu")
    hit = _extract_from_mapping(obj)
    if hit is None and _looks_like_dec(obj):
        hit = obj
    if hit is None:
        raise RuntimeError("no decoder matrix in " + str(path))
    return _orient_dec(hit), str(path)


def find_weights(explicit: Path | None):
    searched = []
    if explicit is not None:
        searched.append(str(explicit))
        if explicit.is_file():
            return explicit, searched
    for root in SEARCH_DIRS:
        if not root.exists():
            searched.append(str(root) + " (missing)")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
            names = set(filenames)
            for cand in CANDIDATE_FILES:
                if cand in names:
                    p = Path(dirpath) / cand
                    searched.append(str(p))
                    return p, searched
            for fn in filenames:
                low = fn.lower()
                if "blocks.6" in low and ("sae" in low or "resid" in low) and low.endswith((".pt", ".safetensors", ".npy")):
                    p = Path(dirpath) / fn
                    searched.append(str(p))
                    return p, searched
            if root.name in ("weights",) or "GPT2-Small-SAEs" in dirpath:
                continue
            # do not walk entire hf hub if we already listed candidate files
            if "hub" in str(root) and "models--" not in dirpath:
                # still allow one level of models--*
                pass
    return None, searched


def summarize_C(c):
    c = np.asarray(c, dtype=np.float64)
    return {
        "n_atoms": int(c.size),
        "C_min": float(np.min(c)),
        "C_p1": float(np.quantile(c, 0.01)),
        "C_p10": float(np.quantile(c, 0.10)),
        "C_median": float(np.median(c)),
        "C_mean": float(np.mean(c)),
        "frac_C_lt_0.05": float(np.mean(c < 0.05)),
        "frac_C_lt_0.20": float(np.mean(c < 0.20)),
        "frac_C_lt_0.50": float(np.mean(c < 0.50)),
        "n_C_lt_0.05": int(np.sum(c < 0.05)),
        "n_C_lt_0.20": int(np.sum(c < 0.20)),
    }


def maybe_whiten_stats(D, device):
    """Project decoder atoms into the GPT-2 layer-6 32-d whitened slice."""
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_gpt2_bridge import (
            GPT2Model,
            GPT2TokenizerFast,
            apply_whiten,
            collect_residuals,
            fit_whiten,
            load_texts,
        )
    except Exception as exc:
        return {"whitened": False, "reason": "import_failed:" + repr(exc)}
    local_snap = "/mnt/e2_runs/work/e2_real_experiment_code_v5_scheme_aligned_20260619_0118/.hf_cache/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    model_id = local_snap if os.path.isdir(local_snap) else "gpt2"
    try:
        tok = GPT2TokenizerFast.from_pretrained(model_id)
        model = GPT2Model.from_pretrained(model_id).to(device)
        texts = load_texts()
        H, _ = collect_residuals(model, tok, texts, 128, 4000, 16, 6, device)
        mu, W = fit_whiten(H)
    except Exception as exc:
        return {"whitened": False, "reason": "gpt2_failed:" + repr(exc)}
    # Decoder atoms are residual directions. The bridge injects in
    # Z = (H - mu) @ W; the linear map on a direction is W[:, :32].T @ d.
    Wp = W[:, :32]
    proj = (Wp.T @ D).T
    cols = unit_cols(proj.T)  # (32, n_atoms)
    c = atom_C(cols)
    e0 = np.zeros(32)
    e0[0] = 1.0
    elast = np.zeros(32)
    elast[-1] = 1.0
    dots0 = np.abs(cols.T @ e0)
    dotsl = np.abs(cols.T @ elast)
    out = summarize_C(c)
    out["whitened"] = True
    out["max_cosine_to_e0"] = float(np.max(dots0))
    out["max_cosine_to_elast"] = float(np.max(dotsl))
    out["n_cosine_e0_gt_0.5"] = int(np.sum(dots0 > 0.5))
    out["n_cosine_elast_gt_0.5"] = int(np.sum(dotsl > 0.5))
    out["slice_m"] = 32
    out["n_tokens"] = int(H.shape[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/results_public_sae"))
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--skip-whiten", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    path, searched = find_weights(args.weights)
    report = {
        "loaded": False,
        "searched": searched[:80],
        "elapsed_sec": None,
        "repo": "jbloom/GPT2-Small-SAEs-Reformatted",
        "hook": "blocks.6.hook_resid_pre",
    }
    if path is None:
        report["elapsed_sec"] = time.time() - t0
        (args.out / "public_sae_c.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return
    print("[pub] loading", path, flush=True)
    D, src = load_decoder(path)
    print("[pub] D", D.shape, "from", src, flush=True)
    c = atom_C(D)
    max_abs = np.max(np.abs(unit_cols(D)), axis=0)
    report.update(
        {
            "loaded": True,
            "source": src,
            "shape": [int(D.shape[0]), int(D.shape[1])],
            "residual": summarize_C(c),
            "max_coord_abs_min": float(np.min(max_abs)),
            "max_coord_abs_median": float(np.median(max_abs)),
            "max_coord_abs_max": float(np.max(max_abs)),
            "n_max_coord_gt_0.5": int(np.sum(max_abs > 0.5)),
            "n_max_coord_gt_0.9": int(np.sum(max_abs > 0.9)),
        }
    )
    if not args.skip_whiten:
        device = args.device
        try:
            import torch

            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"
        print("[pub] whitening slice", flush=True)
        report["pca32"] = maybe_whiten_stats(D, device)
    report["elapsed_sec"] = time.time() - t0
    (args.out / "public_sae_c.json").write_text(json.dumps(report, indent=2))
    slim = {k: report[k] for k in report if k != "searched"}
    print(json.dumps(slim, indent=2), flush=True)


if __name__ == "__main__":
    main()
