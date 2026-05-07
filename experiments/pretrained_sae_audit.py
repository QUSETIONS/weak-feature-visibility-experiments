"""Audit a pretrained GPT-2 SAE on cached real activations.

This is the C2 bridge experiment. It intentionally uses the bare
SAE weight tensors rather than SAELens so it can run in the offline
server environment:

    feature_acts = relu((x - b_dec) @ W_enc + b_enc)
    x_hat = feature_acts @ W_dec + b_dec

The outputs are meant to support paper claims conservatively:
- reconstruction quality on the same activation distribution used downstream;
- empirical sparsity / firing rates;
- decoder geometry terms used by the quartic detection theory;
- an observational feature-direction audit for the most active SAE features.
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
    required = {"W_enc", "W_dec", "b_enc", "b_dec"}
    missing = required.difference(weights)
    if missing:
        raise KeyError(f"Missing SAE tensors: {sorted(missing)}")
    return {k: weights[k].float() for k in sorted(required)}


def batched_indices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


@torch.no_grad()
def encode(x: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.relu((x - weights["b_dec"]) @ weights["W_enc"] + weights["b_enc"])


@torch.no_grad()
def audit(args: argparse.Namespace) -> dict[str, object]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    rng = np.random.default_rng(args.seed)
    x_np = np.load(args.activation_path, mmap_mode="r")
    n_total, d_model = x_np.shape
    n = min(args.max_tokens, n_total) if args.max_tokens else n_total
    if args.shuffle and n < n_total:
        idx = np.sort(rng.choice(n_total, size=n, replace=False))
    else:
        idx = np.arange(n)

    weights = load_sae(Path(args.sae_dir), device)
    w_dec = weights["W_dec"]
    d_sae = int(w_dec.shape[0])
    decoder_norm = torch.linalg.norm(w_dec, dim=1).clamp_min(1e-12)
    w_unit = w_dec / decoder_norm[:, None]
    geometry = (1.0 - torch.sum(w_unit.pow(4), dim=1)).detach().cpu().numpy()
    c_proxy = geometry / 4.0

    count_active = torch.zeros(d_sae, device=device)
    sum_active_value = torch.zeros(d_sae, device=device)
    sum_sq_active_value = torch.zeros(d_sae, device=device)

    sse = 0.0
    l1_sum = 0.0
    active_per_token_sum = 0.0
    x_sum = torch.zeros(d_model, device=device)
    x_sq_sum = torch.zeros(d_model, device=device)

    for start, end in batched_indices(n, args.batch_size):
        rows = idx[start:end]
        xb = torch.from_numpy(np.asarray(x_np[rows])).to(device=device, dtype=torch.float32)
        acts = encode(xb, weights)
        recon = acts @ weights["W_dec"] + weights["b_dec"]
        resid = xb - recon
        sse += float(torch.sum(resid.pow(2)).item())
        active = acts > 0
        count_active += active.sum(dim=0)
        sum_active_value += acts.sum(dim=0)
        sum_sq_active_value += acts.pow(2).sum(dim=0)
        l1_sum += float(torch.sum(torch.abs(acts)).item())
        active_per_token_sum += float(active.sum(dim=1).float().sum().item())
        x_sum += xb.sum(dim=0)
        x_sq_sum += xb.pow(2).sum(dim=0)

    x_mean = x_sum / n
    centered_sse = float(torch.sum(x_sq_sum - n * x_mean.pow(2)).item())
    mse = sse / (n * d_model)
    activation_var = centered_sse / (n * d_model)
    explained_variance = 1.0 - (sse / centered_sse)

    firing_rate = (count_active / n).detach().cpu().numpy()
    mean_act = (sum_active_value / torch.clamp(count_active, min=1.0)).detach().cpu().numpy()
    rms_act = torch.sqrt(sum_sq_active_value / torch.clamp(count_active, min=1.0)).detach().cpu().numpy()
    decoder_norm_np = decoder_norm.detach().cpu().numpy()

    # Pick an interpretable panel: frequent, high-amplitude, high-geometry, and random active features.
    active_mask = firing_rate > args.min_firing_rate
    active_ids = np.flatnonzero(active_mask)
    panels: list[int] = []
    if active_ids.size:
        panels.extend(active_ids[np.argsort(firing_rate[active_ids])[-args.top_k :]].tolist())
        panels.extend(active_ids[np.argsort(mean_act[active_ids])[-args.top_k :]].tolist())
        panels.extend(active_ids[np.argsort(geometry[active_ids])[-args.top_k :]].tolist())
        panels.extend(rng.choice(active_ids, size=min(args.top_k, active_ids.size), replace=False).tolist())
    panel_ids = sorted(set(int(i) for i in panels))

    feature_rows: list[dict[str, object]] = []
    for i in panel_ids:
        feature_rows.append(
            {
                "feature": i,
                "firing_rate": float(firing_rate[i]),
                "mean_active_value": float(mean_act[i]),
                "rms_active_value": float(rms_act[i]),
                "decoder_norm": float(decoder_norm_np[i]),
                "geometry": float(geometry[i]),
                "c_proxy": float(c_proxy[i]),
                "selection": "panel",
            }
        )
    write_csv(outdir / "sae_feature_panel.csv", feature_rows)

    # Direction audit on the selected panel. This is observational: it checks
    # whether SAE-active tokens produce a mean shift along the SAE decoder.
    direction_rows: list[dict[str, object]] = []
    if panel_ids:
        panel = torch.tensor(panel_ids, device=device, dtype=torch.long)
        panel_w = w_unit.index_select(0, panel)
        pos_count = torch.zeros(len(panel_ids), device=device)
        neg_count = torch.zeros(len(panel_ids), device=device)
        pos_sum = torch.zeros(len(panel_ids), device=device)
        neg_sum = torch.zeros(len(panel_ids), device=device)
        pos_sq = torch.zeros(len(panel_ids), device=device)
        neg_sq = torch.zeros(len(panel_ids), device=device)
        for start, end in batched_indices(n, args.batch_size):
            rows = idx[start:end]
            xb = torch.from_numpy(np.asarray(x_np[rows])).to(device=device, dtype=torch.float32)
            acts_panel = encode(xb, weights).index_select(1, panel)
            score = (xb - weights["b_dec"]) @ panel_w.T
            active = acts_panel > 0
            inactive = ~active
            pos_count += active.sum(dim=0)
            neg_count += inactive.sum(dim=0)
            pos_sum += torch.where(active, score, torch.zeros_like(score)).sum(dim=0)
            neg_sum += torch.where(inactive, score, torch.zeros_like(score)).sum(dim=0)
            pos_sq += torch.where(active, score.pow(2), torch.zeros_like(score)).sum(dim=0)
            neg_sq += torch.where(inactive, score.pow(2), torch.zeros_like(score)).sum(dim=0)

        for j, i in enumerate(panel_ids):
            pc = float(pos_count[j].item())
            nc = float(neg_count[j].item())
            if pc < 2 or nc < 2:
                continue
            pm = float(pos_sum[j].item() / pc)
            nm = float(neg_sum[j].item() / nc)
            pv = max(float(pos_sq[j].item() / pc - pm * pm), 1e-12)
            nv = max(float(neg_sq[j].item() / nc - nm * nm), 1e-12)
            pooled = math.sqrt(0.5 * (pv + nv))
            direction_rows.append(
                {
                    "feature": i,
                    "firing_rate": float(firing_rate[i]),
                    "geometry": float(geometry[i]),
                    "mean_shift_along_decoder": pm - nm,
                    "standardized_shift": (pm - nm) / pooled,
                    "active_score_mean": pm,
                    "inactive_score_mean": nm,
                }
            )
    write_csv(outdir / "sae_direction_audit.csv", direction_rows)

    summary = {
        "mode": "pretrained_sae_audit",
        "activation_path": args.activation_path,
        "sae_dir": args.sae_dir,
        "tokens_used": int(n),
        "d_model": int(d_model),
        "d_sae": int(d_sae),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "reconstruction_mse": float(mse),
        "activation_variance": float(activation_var),
        "explained_variance": float(explained_variance),
        "mean_active_features_per_token": float(active_per_token_sum / n),
        "mean_l1_feature_activation": float(l1_sum / n),
        "live_feature_fraction": float(np.mean(firing_rate > args.min_firing_rate)),
        "median_firing_rate": float(np.median(firing_rate)),
        "p95_firing_rate": float(np.quantile(firing_rate, 0.95)),
        "mean_decoder_geometry": float(np.mean(geometry)),
        "median_decoder_geometry": float(np.median(geometry)),
        "mean_c_proxy": float(np.mean(c_proxy)),
        "panel_features": len(panel_ids),
        "output_feature_panel": str(outdir / "sae_feature_panel.csv"),
        "output_direction_audit": str(outdir / "sae_direction_audit.csv"),
    }
    with (outdir / "sae_audit_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-path", required=True)
    parser.add_argument("--sae-dir", required=True)
    parser.add_argument("--outdir", default="results/real_activation/c2_sae_audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--min-firing-rate", type=float, default=1e-4)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
