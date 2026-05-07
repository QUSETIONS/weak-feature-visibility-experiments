"""Quantify the two main abstractions behind the detector-law bridge.

This script audits, on a fresh GPT-2 residual-stream cache:

1. the local isotropy reduction used by the quartic coefficient after a
   train/holdout whitening split; and
2. the second-order feature-independence condition from Proposition 17 using a
   pretrained SAE latent basis.

The goal is not to prove exact equality on real activations. It is to replace
the abstraction with measured residuals and therefore answer how far the bridge
is from the raw GPT-2 activation geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from pretrained_sae_natural_event_audit import collect_token_labels, unit_label_direction


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_sae(sae_dir: Path, device: torch.device) -> dict[str, torch.Tensor]:
    weights = load_file(str(sae_dir / "sae_weights.safetensors"), device=str(device))
    required = {"W_enc", "W_dec", "b_enc", "b_dec"}
    missing = required.difference(weights)
    if missing:
        raise KeyError(f"Missing SAE tensors: {sorted(missing)}")
    return {k: weights[k].float() for k in required}


def pca_whiten_fit(x_train: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    xc = x_train - mean
    cov = (xc.T @ xc) / xc.shape[0]
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, eps)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    whitener = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return mean, whitener.astype(np.float32)


def covariance_stats(x: np.ndarray) -> dict[str, float]:
    xc = x - x.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / xc.shape[0]
    d = cov.shape[0]
    diag = np.diag(cov)
    off = cov - np.diag(diag)
    ident = np.eye(d, dtype=np.float64)
    dev = cov - ident
    eigvals = np.linalg.eigvalsh(cov)
    return {
        "tokens": int(x.shape[0]),
        "d_model": int(d),
        "mean_diag": float(diag.mean()),
        "median_diag": float(np.median(diag)),
        "max_abs_diag_dev": float(np.max(np.abs(diag - 1.0))),
        "diag_std": float(diag.std()),
        "offdiag_rms": float(np.linalg.norm(off, ord="fro") / np.sqrt(d * (d - 1))),
        "offdiag_max_abs": float(np.max(np.abs(off))),
        "cov_fro_dev_from_identity": float(np.linalg.norm(dev, ord="fro") / np.linalg.norm(ident, ord="fro")),
        "cov_spectral_dev_from_identity": float(np.max(np.abs(eigvals - 1.0))),
        "eig_min": float(eigvals.min()),
        "eig_median": float(np.median(eigvals)),
        "eig_max": float(eigvals.max()),
    }


@torch.no_grad()
def latent_moments(
    x: np.ndarray,
    weights: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    d_sae = int(weights["W_enc"].shape[1])
    count = x.shape[0]
    sums = torch.zeros(d_sae, device=device)
    sumsq = torch.zeros(d_sae, device=device)
    positive = torch.zeros(d_sae, device=device)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        xb = torch.from_numpy(np.asarray(x[start:end], dtype=np.float32)).to(device)
        acts = torch.relu((xb - weights["b_dec"]) @ weights["W_enc"] + weights["b_enc"])
        sums += acts.sum(dim=0)
        sumsq += acts.pow(2).sum(dim=0)
        positive += (acts > 0).sum(dim=0)
    mean = (sums / count).detach().cpu().numpy()
    var = (sumsq / count - (sums / count).pow(2)).clamp_min(0.0).detach().cpu().numpy()
    firing = (positive / count).detach().cpu().numpy()
    return var.astype(np.float64), firing.astype(np.float64)


@torch.no_grad()
def encode_subset(
    x: np.ndarray,
    weights: dict[str, torch.Tensor],
    feature_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    w_enc = weights["W_enc"].index_select(1, torch.as_tensor(feature_ids, device=device, dtype=torch.long))
    b_enc = weights["b_enc"].index_select(0, torch.as_tensor(feature_ids, device=device, dtype=torch.long))
    out = np.empty((x.shape[0], feature_ids.size), dtype=np.float32)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        xb = torch.from_numpy(np.asarray(x[start:end], dtype=np.float32)).to(device)
        acts = torch.relu((xb - weights["b_dec"]) @ w_enc + b_enc)
        out[start:end] = acts.detach().cpu().numpy()
    return out


def second_order_audit(
    x_train: np.ndarray,
    x_test: np.ndarray,
    x_test_whitened: np.ndarray,
    whitener: np.ndarray,
    weights: dict[str, torch.Tensor],
    top_features: int,
    min_firing_rate: float,
    batch_size: int,
    device: torch.device,
    feature_ids: np.ndarray | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    var_train, firing_train = latent_moments(x_train, weights, batch_size, device)
    live = np.flatnonzero(firing_train >= min_firing_rate)
    if live.size == 0:
        raise RuntimeError("No SAE features passed the firing-rate threshold.")
    if feature_ids is not None:
        chosen = np.asarray(sorted(set(int(v) for v in feature_ids.tolist())), dtype=np.int64)
        chosen = chosen[(chosen >= 0) & (chosen < var_train.size)]
        chosen = chosen[firing_train[chosen] >= min_firing_rate]
        if chosen.size == 0:
            raise RuntimeError("No requested feature ids passed the firing-rate threshold.")
        selection_mode = "explicit_feature_ids"
    else:
        order = live[np.argsort(var_train[live])[::-1]]
        chosen = order[: min(top_features, order.size)]
        selection_mode = "top_variance_live_features"

    acts_test = encode_subset(x_test, weights, chosen, batch_size, device)
    acts_test = acts_test.astype(np.float64)
    acts_centered = acts_test - acts_test.mean(axis=0, keepdims=True)
    sigma_a = (acts_centered.T @ acts_centered) / acts_centered.shape[0]

    w_dec = weights["W_dec"].index_select(0, torch.as_tensor(chosen, device=device, dtype=torch.long))
    decoder = w_dec.detach().cpu().numpy().astype(np.float64)
    decoder_norm = np.linalg.norm(decoder, axis=1)

    diag_sigma = np.diag(np.diag(sigma_a))
    off_sigma = sigma_a - diag_sigma
    cov_diag = decoder.T @ diag_sigma @ decoder
    cov_full = decoder.T @ sigma_a @ decoder
    cov_resid = decoder.T @ off_sigma @ decoder

    whitened_decoder = decoder @ whitener
    whitened_decoder_norm = np.linalg.norm(whitened_decoder, axis=1, keepdims=True)
    whitened_decoder_unit = whitened_decoder / np.maximum(whitened_decoder_norm, 1e-12)
    local_scores = x_test_whitened @ whitened_decoder_unit.T
    local_var = np.var(local_scores, axis=0)

    abs_off = np.abs(off_sigma)
    np.fill_diagonal(abs_off, 0.0)
    bound = float(np.sum(abs_off * np.outer(decoder_norm, decoder_norm)))

    denom = max(float(np.linalg.norm(cov_diag, ord="fro")), 1e-12)
    total = max(float(np.linalg.norm(cov_full, ord="fro")), 1e-12)
    resid = float(np.linalg.norm(cov_resid, ord="fro"))

    upper = np.triu_indices_from(off_sigma, k=1)
    pair_abs = np.abs(off_sigma[upper])

    total_var = float(var_train.sum())
    selected_var = float(var_train[chosen].sum())
    feature_rows: list[dict[str, object]] = []
    geom = 1.0 - np.sum((decoder / np.maximum(decoder_norm[:, None], 1e-12)) ** 4, axis=1)
    for local_i, feature in enumerate(chosen[: min(64, chosen.size)]):
        feature_rows.append(
            {
                "feature": int(feature),
                "train_var": float(var_train[feature]),
                "train_firing_rate": float(firing_train[feature]),
                "decoder_norm": float(decoder_norm[local_i]),
                "decoder_geometry": float(geom[local_i]),
            }
        )

    summary = {
        "selection_mode": selection_mode,
        "selected_features": int(chosen.size),
        "live_features": int(live.size),
        "live_feature_fraction": float(live.size / var_train.size),
        "selected_variance_fraction": float(selected_var / max(total_var, 1e-12)),
        "selected_mean_firing_rate": float(np.mean(firing_train[chosen])),
        "selected_median_firing_rate": float(np.median(firing_train[chosen])),
        "selected_mean_decoder_norm": float(np.mean(decoder_norm)),
        "selected_mean_decoder_geometry": float(np.mean(geom)),
        "local_isotropy_mean_var": float(np.mean(local_var)),
        "local_isotropy_median_var": float(np.median(local_var)),
        "local_isotropy_max_abs_var_dev": float(np.max(np.abs(local_var - 1.0))),
        "local_isotropy_p95_abs_var_dev": float(np.quantile(np.abs(local_var - 1.0), 0.95)),
        "latent_offdiag_abs_cov_mean": float(pair_abs.mean()),
        "latent_offdiag_abs_cov_median": float(np.median(pair_abs)),
        "latent_offdiag_abs_cov_p95": float(np.quantile(pair_abs, 0.95)),
        "latent_offdiag_abs_cov_max": float(pair_abs.max()),
        "cov_diag_fro": float(np.linalg.norm(cov_diag, ord="fro")),
        "cov_full_fro": float(np.linalg.norm(cov_full, ord="fro")),
        "cov_residual_fro": resid,
        "cov_residual_over_diag_fro": float(resid / denom),
        "cov_residual_over_total_fro": float(resid / total),
        "prop17_bound": bound,
        "prop17_bound_over_diag_fro": float(bound / denom),
        "prop17_bound_over_total_fro": float(bound / total),
    }
    return summary, feature_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-path", required=True)
    parser.add_argument("--sae-dir", default="assets/gpt2_sae_layer6/blocks.6.hook_resid_pre")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--whiten-eps", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument("--min-firing-rate", type=float, default=1e-4)
    parser.add_argument("--feature-csv", default=None)
    parser.add_argument("--feature-column", default="sae_best_feature")
    parser.add_argument("--model-name", default="assets/gpt2")
    parser.add_argument("--text-file", default="assets/wikitext_train.txt")
    parser.add_argument("--nltk-data-dir", default="assets/nltk_data")
    parser.add_argument(
        "--families",
        nargs="+",
        default=["lexical", "linguistic", "pos", "ner"],
        choices=["lexical", "linguistic", "pos", "ner"],
    )
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--max-docs", type=int, default=12000)
    parser.add_argument("--skip-first-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-train-pos", type=int, default=20)
    parser.add_argument("--min-test-pos", type=int, default=20)
    parser.add_argument("--min-train-neg", type=int, default=20)
    parser.add_argument("--min-test-neg", type=int, default=20)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    x = np.load(args.activation_path).astype(np.float32)
    n = x.shape[0]
    perm = rng.permutation(n)
    train_n = int(args.train_frac * n)
    train_idx = perm[:train_n]
    test_idx = perm[train_n:]
    x_train = x[train_idx]
    x_test = x[test_idx]

    mean, whitener = pca_whiten_fit(x_train, args.whiten_eps)
    x_train_w = (x_train - mean) @ whitener
    x_test_w = (x_test - mean) @ whitener
    isotropy_train = covariance_stats(x_train_w.astype(np.float64))
    isotropy_test = covariance_stats(x_test_w.astype(np.float64))

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    weights = load_sae(Path(args.sae_dir), device)
    explicit_feature_ids = None
    if args.feature_csv:
        rows = list(csv.DictReader(Path(args.feature_csv).open("r", encoding="utf-8", newline="")))
        explicit_feature_ids = np.asarray(
            [int(row[args.feature_column]) for row in rows if row.get(args.feature_column) not in {None, "", "nan"}],
            dtype=np.int64,
        )

    second_order, feature_rows = second_order_audit(
        x_train=x_train,
        x_test=x_test,
        x_test_whitened=x_test_w.astype(np.float64),
        whitener=whitener.astype(np.float64),
        weights=weights,
        top_features=args.top_features,
        min_firing_rate=args.min_firing_rate,
        batch_size=args.batch_size,
        device=device,
        feature_ids=explicit_feature_ids,
    )
    write_csv(outdir / "assumption_audit_top_features.csv", feature_rows)

    event_isotropy = None
    try:
        labels = collect_token_labels(args, n)
        event_vars: list[float] = []
        kept_events = 0
        for _, y_all in labels.items():
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]
            if (
                int(y_train.sum()) < args.min_train_pos
                or int(y_test.sum()) < args.min_test_pos
                or int((~y_train).sum()) < args.min_train_neg
                or int((~y_test).sum()) < args.min_test_neg
            ):
                continue
            direction = unit_label_direction(x_train, y_train)
            if direction is None:
                continue
            w_white = direction @ whitener
            norm = float(np.linalg.norm(w_white))
            if norm < 1e-12:
                continue
            score = x_test_w @ (w_white / norm)
            event_vars.append(float(np.var(score)))
            kept_events += 1
        if event_vars:
            event_var = np.asarray(event_vars, dtype=np.float64)
            event_isotropy = {
                "events_evaluated": int(kept_events),
                "mean_var": float(event_var.mean()),
                "median_var": float(np.median(event_var)),
                "max_abs_var_dev": float(np.max(np.abs(event_var - 1.0))),
                "p95_abs_var_dev": float(np.quantile(np.abs(event_var - 1.0), 0.95)),
            }
    except Exception as exc:
        event_isotropy = {"error": str(exc)}

    summary = {
        "mode": "assumption_audit",
        "activation_path": args.activation_path,
        "sae_dir": args.sae_dir,
        "tokens": int(n),
        "train_tokens": int(train_n),
        "test_tokens": int(n - train_n),
        "d_model": int(x.shape[1]),
        "whiten_eps": float(args.whiten_eps),
        "isotropy_train": isotropy_train,
        "isotropy_test": isotropy_test,
        "second_order_audit": second_order,
        "event_direction_isotropy": event_isotropy,
        "top_feature_csv": str(outdir / "assumption_audit_top_features.csv"),
        "device": str(device),
    }
    with (outdir / "assumption_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
