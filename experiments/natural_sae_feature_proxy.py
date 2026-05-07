"""Natural-feature proxy audit for the quartic detection story.

This experiment uses a pretrained GPT-2 SAE as a source of naturally
occurring feature events. For selected SAE latents, the SAE firing event
acts as a proxy label. We then ask whether the corresponding activation
direction is visible to three detector classes:

1. labeled mean evidence along the decoder direction;
2. unrestricted covariance evidence along the decoder direction;
3. feature-independent off-diagonal covariance evidence.

Unlike the injection experiments, this does not create a synthetic target
feature. It audits naturally occurring SAE firing events on real GPT-2
residual-stream activations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_sae(sae_dir: Path, device: torch.device) -> dict[str, torch.Tensor]:
    weights = load_file(str(sae_dir / "sae_weights.safetensors"), device=str(device))
    return {k: weights[k].float().to(device) for k in ["W_enc", "W_dec", "b_enc", "b_dec"]}


def select_features(sae_dir: Path, n_features: int, seed: int, min_log10_fire: float, max_log10_fire: float) -> np.ndarray:
    sparsity_path = sae_dir / "sparsity.safetensors"
    if not sparsity_path.exists():
        raise FileNotFoundError(sparsity_path)
    sparsity = load_file(str(sparsity_path))["sparsity"].cpu().numpy()
    candidates = np.flatnonzero((sparsity >= min_log10_fire) & (sparsity <= max_log10_fire))
    if candidates.size == 0:
        raise ValueError("No SAE features matched the requested sparsity window.")

    # Cover the firing-rate spectrum instead of only taking the most frequent features.
    order = candidates[np.argsort(sparsity[candidates])]
    if order.size <= n_features:
        return np.sort(order.astype(int))
    grid = np.linspace(0, order.size - 1, n_features)
    ids = np.unique(order[np.round(grid).astype(int)])
    if ids.size < n_features:
        rng = np.random.default_rng(seed)
        extra = rng.choice(np.setdiff1d(candidates, ids), size=n_features - ids.size, replace=False)
        ids = np.concatenate([ids, extra])
    return np.sort(ids.astype(int))


def z_two_sample(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.size < 3 or b.size < 3:
        return float("nan"), float("nan")
    diff = float(a.mean() - b.mean())
    se = math.sqrt(float(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))
    return diff, diff / max(se, 1e-12)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    sae_dir = Path(args.sae_dir)
    weights = load_sae(sae_dir, device)

    x_mem = np.load(args.activation_path, mmap_mode="r")
    n_total, d_model = x_mem.shape
    n = min(args.max_tokens, n_total) if args.max_tokens else n_total
    rng = np.random.default_rng(args.seed)
    if args.shuffle and n < n_total:
        token_idx = np.sort(rng.choice(n_total, size=n, replace=False))
    else:
        token_idx = np.arange(n)

    # Use diagonal standardization so the off-diagonal statistic has the same
    # coordinate-wise nuisance interpretation as the feature-independent model.
    x_fit = np.asarray(x_mem[token_idx], dtype=np.float32)
    mean = x_fit.mean(axis=0, keepdims=True)
    std = np.maximum(x_fit.std(axis=0, keepdims=True), args.eps)
    x_std = ((x_fit - mean) / std).astype(np.float32)

    feature_ids = select_features(
        sae_dir=sae_dir,
        n_features=args.n_features,
        seed=args.seed,
        min_log10_fire=args.min_log10_fire,
        max_log10_fire=args.max_log10_fire,
    )
    fid_t = torch.tensor(feature_ids, device=device, dtype=torch.long)
    w_enc = weights["W_enc"].index_select(1, fid_t)
    b_enc = weights["b_enc"].index_select(0, fid_t)
    w_dec_raw = weights["W_dec"].index_select(0, fid_t).detach().cpu().numpy()
    b_dec = weights["b_dec"].detach().cpu().numpy()

    # Decoder directions in standardized coordinates.
    w_std_raw = w_dec_raw / std.reshape(1, -1)
    w_norm = np.linalg.norm(w_std_raw, axis=1, keepdims=True).clip(1e-12)
    w_unit = (w_std_raw / w_norm).astype(np.float32)
    geometry = 1.0 - np.sum(w_unit**4, axis=1)

    acts_all = np.zeros((n, len(feature_ids)), dtype=np.float32)
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        xb_raw = torch.from_numpy(np.asarray(x_mem[token_idx[start:end]], dtype=np.float32)).to(device)
        acts = torch.relu((xb_raw - weights["b_dec"]) @ w_enc + b_enc)
        acts_all[start:end] = acts.detach().cpu().numpy()

    rows: list[dict[str, object]] = []
    for j, fid in enumerate(feature_ids):
        acts = acts_all[:, j]
        active = acts > args.active_threshold
        p = float(active.mean())
        active_count = int(active.sum())
        inactive_count = int((~active).sum())
        if active_count < args.min_active or inactive_count < args.min_inactive:
            continue

        w = w_unit[j]
        x_centered = x_std - x_std.mean(axis=0, keepdims=True)
        score = x_centered @ w
        label_diff, label_z = z_two_sample(score[active], score[~active])

        # Group-centered covariance projections.
        xp = x_std[active] - x_std[active].mean(axis=0, keepdims=True)
        xn = x_std[~active] - x_std[~active].mean(axis=0, keepdims=True)
        proj_p = xp @ w
        proj_n = xn @ w
        diag_p = (xp * xp) @ (w * w)
        diag_n = (xn * xn) @ (w * w)
        full_p = proj_p * proj_p
        full_n = proj_n * proj_n
        off_p = full_p - diag_p
        off_n = full_n - diag_n
        full_diff, full_z = z_two_sample(full_p, full_n)
        off_diff, off_z = z_two_sample(off_p, off_n)

        coeff = acts - acts.mean()
        lambda_eff = float(np.sqrt(np.var(coeff)) * w_norm[j, 0])
        evidence = float(n * (geometry[j] / 4.0) * (lambda_eff**4))
        rows.append(
            {
                "feature": int(fid),
                "tokens": int(n),
                "firing_rate": p,
                "active_count": active_count,
                "inactive_count": inactive_count,
                "geometry": float(geometry[j]),
                "lambda_eff": lambda_eff,
                "evidence_proxy": evidence,
                "log10_evidence_proxy": float(np.log10(max(evidence, 1e-30))),
                "label_mean_diff": label_diff,
                "label_z": label_z,
                "full_cov_diff": full_diff,
                "full_cov_z": full_z,
                "offdiag_cov_diff": off_diff,
                "offdiag_cov_z": off_z,
            }
        )

    write_csv(outdir / "natural_sae_feature_proxy.csv", rows)
    if not rows:
        raise RuntimeError("No features passed the active/inactive count filters.")

    ev = np.array([r["log10_evidence_proxy"] for r in rows], dtype=float)
    label_abs = np.abs(np.array([r["label_z"] for r in rows], dtype=float))
    full_abs = np.abs(np.array([r["full_cov_z"] for r in rows], dtype=float))
    off_abs = np.abs(np.array([r["offdiag_cov_z"] for r in rows], dtype=float))

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            return float("nan")
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    summary = {
        "mode": "natural_sae_feature_proxy",
        "activation_path": args.activation_path,
        "sae_dir": args.sae_dir,
        "tokens": int(n),
        "features_selected": int(len(feature_ids)),
        "features_evaluated": int(len(rows)),
        "median_firing_rate": float(np.median([r["firing_rate"] for r in rows])),
        "median_geometry": float(np.median([r["geometry"] for r in rows])),
        "median_lambda_eff": float(np.median([r["lambda_eff"] for r in rows])),
        "median_abs_label_z": float(np.median(label_abs)),
        "median_abs_full_cov_z": float(np.median(full_abs)),
        "median_abs_offdiag_cov_z": float(np.median(off_abs)),
        "frac_label_z_gt_5": float(np.mean(label_abs > 5.0)),
        "frac_full_cov_z_gt_5": float(np.mean(full_abs > 5.0)),
        "frac_offdiag_z_gt_5": float(np.mean(off_abs > 5.0)),
        "corr_log_evidence_abs_label_z": corr(ev, label_abs),
        "corr_log_evidence_abs_full_cov_z": corr(ev, full_abs),
        "corr_log_evidence_abs_offdiag_z": corr(ev, off_abs),
        "output_csv": str(outdir / "natural_sae_feature_proxy.csv"),
        "output_figure": str(outdir / "natural_sae_feature_proxy.pdf"),
    }
    with (outdir / "natural_sae_feature_proxy_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
        ax[0].scatter(ev, off_abs, s=12, alpha=0.75, label="feature-independent")
        ax[0].scatter(ev, full_abs, s=12, alpha=0.55, label="full covariance")
        ax[0].set_xlabel(r"$\log_{10}(NC_i\lambda_i^4)$ proxy")
        ax[0].set_ylabel("absolute z-statistic")
        ax[0].legend(frameon=False, fontsize=8)
        ax[1].scatter(label_abs, off_abs, s=12, alpha=0.75)
        ax[1].set_xlabel("labeled mean |z|")
        ax[1].set_ylabel("feature-independent |z|")
        fig.tight_layout()
        fig.savefig(outdir / "natural_sae_feature_proxy.png", dpi=220)
        fig.savefig(outdir / "natural_sae_feature_proxy.pdf")
        plt.close(fig)
    except Exception as exc:
        summary["plot_error"] = repr(exc)
        with (outdir / "natural_sae_feature_proxy_summary.json").open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-path", required=True)
    parser.add_argument("--sae-dir", required=True)
    parser.add_argument("--outdir", default="results/real_activation/c9_natural_sae_feature_proxy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--min-log10-fire", type=float, default=-4.0)
    parser.add_argument("--max-log10-fire", type=float, default=-1.0)
    parser.add_argument("--active-threshold", type=float, default=0.0)
    parser.add_argument("--min-active", type=int, default=50)
    parser.add_argument("--min-inactive", type=int, default=1000)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
