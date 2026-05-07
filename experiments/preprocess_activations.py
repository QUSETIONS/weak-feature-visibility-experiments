"""Preprocess cached activations for robustness experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pca_whiten(x: np.ndarray, eps: float, max_fit_tokens: int, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    fit = xc
    if max_fit_tokens and fit.shape[0] > max_fit_tokens:
        fit = fit[rng.choice(fit.shape[0], size=max_fit_tokens, replace=False)]
    cov = (fit.T @ fit) / fit.shape[0]
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, eps)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    whitener = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    xw = xc @ whitener
    info = {
        "preprocess": "pca_whiten",
        "eps": eps,
        "fit_tokens": int(fit.shape[0]),
        "eig_min": float(vals.min()),
        "eig_max": float(vals.max()),
        "eig_median": float(np.median(vals)),
        "output_mean_abs": float(np.abs(xw.mean(axis=0)).mean()),
        "output_std_mean": float(xw.std(axis=0).mean()),
    }
    return xw.astype(np.float16), info


def diag_standardize(x: np.ndarray, eps: float) -> tuple[np.ndarray, dict[str, object]]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    xs = (x - mean) / np.maximum(std, eps)
    return xs.astype(np.float16), {
        "preprocess": "diag_standardize",
        "eps": eps,
        "output_mean_abs": float(np.abs(xs.mean(axis=0)).mean()),
        "output_std_mean": float(xs.std(axis=0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-path", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--mode", choices=["pca_whiten", "diag_standardize"], default="pca_whiten")
    parser.add_argument("--max-output-tokens", type=int, default=0)
    parser.add_argument("--max-fit-tokens", type=int, default=50000)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260503)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    x = np.load(args.activation_path).astype(np.float32)
    if args.max_output_tokens and x.shape[0] > args.max_output_tokens:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(x.shape[0], size=args.max_output_tokens, replace=False))
        x = x[idx]
    if args.mode == "pca_whiten":
        xp, info = pca_whiten(x, args.eps, args.max_fit_tokens, args.seed)
    else:
        xp, info = diag_standardize(x, args.eps)
    path = outdir / f"{Path(args.activation_path).stem}_{args.mode}.npy"
    np.save(path, xp)
    info.update(
        {
            "activation_path": args.activation_path,
            "output_path": str(path),
            "tokens": int(xp.shape[0]),
            "d_model": int(xp.shape[1]),
            "output_global_mean": float(xp.astype(np.float32).mean()),
            "output_global_std": float(xp.astype(np.float32).std()),
        }
    )
    with (outdir / "preprocess_summary.json").open("w") as f:
        json.dump(info, f, indent=2, sort_keys=True)
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
