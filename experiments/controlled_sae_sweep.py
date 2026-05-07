"""Optimized C4 SAE sweep on injected real activations.

This runner is a cleaner replacement for shell-looping
``controlled_sae_training.py``. It loads and centers the activation cache once,
computes the PCA basis once, fixes injected directions across all hyperparameter
configs, and writes aggregate CSV/JSON summaries as it goes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_csv(value: str, cast):
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_pcs(x: np.ndarray, n_pcs: int, max_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xc = x.astype(np.float64) - x.mean(axis=0, keepdims=True)
    if max_rows and xc.shape[0] > max_rows:
        xc = xc[rng.choice(xc.shape[0], size=max_rows, replace=False)]
    _, _, vh = np.linalg.svd(xc, full_matrices=False)
    return vh[:n_pcs]


def make_direction(kind: str, d: int, rng: np.random.Generator, pc_basis: np.ndarray, k: int) -> np.ndarray:
    if kind == "axis":
        w = np.zeros(d)
        w[0] = 1.0
    elif kind == "ksparse":
        w = np.zeros(d)
        idx = rng.choice(d, size=k, replace=False)
        w[idx] = rng.normal(size=k)
    elif kind == "dense":
        w = rng.normal(size=d)
    elif kind == "pc1":
        w = pc_basis[0].copy()
    elif kind == "pc_orth":
        w = rng.normal(size=d)
        for j in range(min(8, pc_basis.shape[0])):
            w = w - np.dot(w, pc_basis[j]) * pc_basis[j]
    else:
        raise ValueError(f"unknown direction kind: {kind}")
    return (w / (np.linalg.norm(w) + 1.0e-12)).astype(np.float32)


class TinySAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, model_type: str = "relu_l1", top_k: int = 32):
        super().__init__()
        self.model_type = model_type
        self.top_k = top_k
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.encoder = nn.Linear(d_in, d_sae)
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.W_dec, std=1.0 / math.sqrt(d_in))
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = F.relu(self.encoder(x - self.b_dec))
        if self.model_type == "topk":
            k = max(1, min(self.top_k, acts.shape[1]))
            values, idx = torch.topk(acts, k=k, dim=1)
            sparse = torch.zeros_like(acts)
            acts = sparse.scatter(1, idx, values)
        elif self.model_type != "relu_l1":
            raise ValueError(f"unknown model type: {self.model_type}")
        recon = acts @ self.W_dec + self.b_dec
        return recon, acts


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy() > 0.5
    n_pos = int(labels_np.sum())
    n_neg = int((~labels_np).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores_np, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores_np) + 1)
    pos_rank_sum = ranks[labels_np].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_auc_np(scores: np.ndarray, labels: np.ndarray) -> float:
    labels_bool = labels > 0.5
    n_pos = int(labels_bool.sum())
    n_neg = int((~labels_bool).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_rank_sum = ranks[labels_bool].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def stable_text_hash(text: str) -> int:
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 1_000_003
    return value


def injection_latent(z: torch.Tensor, p: float, mode: str) -> torch.Tensor:
    if mode == "centered":
        return z - p
    if mode == "positive":
        return z
    raise ValueError(f"unknown injection mode: {mode}")


@torch.no_grad()
def evaluate(
    model: TinySAE,
    x: np.ndarray,
    w_np: np.ndarray,
    amp: float,
    p: float,
    seed: int,
    device: torch.device,
    n_eval: int,
    batch_size: int,
    injection_mode: str,
    auc_null_permutations: int,
) -> dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=min(n_eval, x.shape[0]), replace=False)
    w = torch.tensor(w_np, device=device, dtype=torch.float32)
    decoder = F.normalize(model.W_dec.detach(), dim=1)
    cos = torch.abs(decoder @ w)
    max_cos, argmax = torch.max(cos, dim=0)
    top10_cos = torch.topk(cos, k=min(10, cos.numel())).values.mean()

    signal_true = []
    signal_recon = []
    atom_acts = []
    all_acts = []
    labels = []
    clean_mse = 0.0
    inj_mse = 0.0
    total = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 17)

    for start in range(0, len(idx), batch_size):
        rows = idx[start : start + batch_size]
        xb = torch.from_numpy(np.asarray(x[rows])).to(device=device, dtype=torch.float32)
        z = torch.bernoulli(torch.full((xb.shape[0],), p, device=device), generator=gen)
        zi = injection_latent(z, p, injection_mode)
        xi = xb + amp * zi[:, None] * w[None, :]
        recon_clean, _ = model(xb)
        recon_inj, acts = model(xi)
        delta_recon = recon_inj - recon_clean
        signal_recon.append((delta_recon @ w).detach().cpu())
        signal_true.append((amp * zi).detach().cpu())
        atom_acts.append(acts[:, int(argmax.item())].detach().cpu())
        all_acts.append(acts.detach().cpu())
        labels.append(z.detach().cpu())
        clean_mse += float(F.mse_loss(recon_clean, xb, reduction="sum").item())
        inj_mse += float(F.mse_loss(recon_inj, xi, reduction="sum").item())
        total += xb.numel()

    y_true = torch.cat(signal_true).float()
    y_rec = torch.cat(signal_recon).float()
    atom = torch.cat(atom_acts).float()
    acts_all = torch.cat(all_acts).float()
    lab = torch.cat(labels).float()
    denom = torch.sum(y_true * y_true).clamp_min(1e-9)
    gain = float(torch.sum(y_true * y_rec).item() / denom.item())
    rec_corr = float(torch.corrcoef(torch.stack([y_true, y_rec]))[0, 1].nan_to_num(0.0).item())
    atom_corr = float(torch.corrcoef(torch.stack([lab, atom]))[0, 1].nan_to_num(0.0).item())
    if lab.sum() > 1 and (1 - lab).sum() > 1:
        atom_label_gap = float(atom[lab > 0.5].mean().item() - atom[lab < 0.5].mean().item())
        atom_label_auc = binary_auc(atom, lab)
        labels_np = lab.numpy()
        acts_np = acts_all.numpy()
        atom_aucs = np.array([binary_auc_np(acts_np[:, j], labels_np) for j in range(acts_np.shape[1])])
        atom_abs_aucs = np.maximum(atom_aucs, 1.0 - atom_aucs)
        top_label_atom = int(np.argmax(atom_abs_aucs))
        top_label_auc = float(atom_aucs[top_label_atom])
        top_label_abs_auc = float(atom_abs_aucs[top_label_atom])
        null_abs_aucs = []
        if auc_null_permutations > 0:
            null_rng = np.random.default_rng(seed + 9001)
            for _ in range(auc_null_permutations):
                perm_labels = null_rng.permutation(labels_np)
                perm_aucs = np.array([binary_auc_np(acts_np[:, j], perm_labels) for j in range(acts_np.shape[1])])
                null_abs_aucs.append(float(np.max(np.maximum(perm_aucs, 1.0 - perm_aucs))))
        top_label_abs_auc_null_mean = float(np.mean(null_abs_aucs)) if null_abs_aucs else 0.5
        top_label_abs_auc_excess = top_label_abs_auc - top_label_abs_auc_null_mean
        top_label_gap = float(
            acts_np[labels_np > 0.5, top_label_atom].mean()
            - acts_np[labels_np < 0.5, top_label_atom].mean()
        )
        top_label_decoder_cosine = float(cos[top_label_atom].item())
    else:
        atom_label_gap = 0.0
        atom_label_auc = 0.5
        top_label_atom = int(argmax.item())
        top_label_auc = 0.5
        top_label_abs_auc = 0.5
        top_label_abs_auc_null_mean = 0.5
        top_label_abs_auc_excess = 0.0
        top_label_gap = 0.0
        top_label_decoder_cosine = float(max_cos.item())

    return {
        "max_decoder_cosine": float(max_cos.item()),
        "top10_decoder_cosine": float(top10_cos.item()),
        "best_atom": int(argmax.item()),
        "signal_reconstruction_gain": gain,
        "signal_reconstruction_corr": rec_corr,
        "best_atom_label_gap": atom_label_gap,
        "best_atom_label_corr": atom_corr,
        "best_atom_label_auc": atom_label_auc,
        "top_label_atom": top_label_atom,
        "top_label_atom_auc": top_label_auc,
        "top_label_atom_abs_auc": top_label_abs_auc,
        "top_label_atom_abs_auc_null_mean": top_label_abs_auc_null_mean,
        "top_label_atom_abs_auc_excess": top_label_abs_auc_excess,
        "top_label_atom_gap": top_label_gap,
        "top_label_atom_decoder_cosine": top_label_decoder_cosine,
        "clean_mse": clean_mse / total,
        "injected_mse": inj_mse / total,
    }


def train_one(
    x: np.ndarray,
    w_np: np.ndarray,
    args: argparse.Namespace,
    d_sae: int,
    l1_weight: float,
    amp: float,
    p: float,
    direction: str,
    direction_seed: int,
    config_id: int,
    device: torch.device,
    outdir: Path,
) -> dict[str, object]:
    train_seed = args.seed + 1009 * config_id + 131 * direction_seed + 17 * stable_text_hash(direction)
    rng = np.random.default_rng(train_seed)
    torch.manual_seed(train_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_seed)

    model = TinySAE(x.shape[1], d_sae, model_type=args.model_type, top_k=args.top_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    w = torch.tensor(w_np, device=device, dtype=torch.float32)
    t0 = time.time()
    last_loss = 0.0
    last_mse = 0.0
    last_l1 = 0.0

    for step in range(1, args.steps + 1):
        rows = rng.choice(x.shape[0], size=args.batch_size, replace=False)
        xb = torch.from_numpy(np.asarray(x[rows])).to(device=device, dtype=torch.float32)
        z = torch.bernoulli(torch.full((xb.shape[0],), p, device=device))
        xi = xb + amp * injection_latent(z, p, args.injection_mode)[:, None] * w[None, :]
        recon, acts = model(xi)
        mse = F.mse_loss(recon, xi)
        l1 = acts.abs().mean()
        loss = mse + (0.0 if args.model_type == "topk" else l1_weight) * l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.normalize_decoder()
        last_loss = float(loss.item())
        last_mse = float(mse.item())
        last_l1 = float(l1.item())
        if args.log_every and step % args.log_every == 0:
            print(json.dumps({
                "event": "train",
                "config_id": config_id,
                "direction": direction,
                "direction_seed": direction_seed,
                "step": step,
                "loss": last_loss,
                "mse": last_mse,
                "l1": last_l1,
            }), flush=True)

    metrics = evaluate(
        model=model,
        x=x,
        w_np=w_np,
        amp=amp,
        p=p,
        seed=train_seed + 4049,
        device=device,
        n_eval=args.eval_tokens,
        batch_size=args.eval_batch_size,
        injection_mode=args.injection_mode,
        auc_null_permutations=args.auc_null_permutations,
    )
    row: dict[str, object] = {
        "config_id": config_id,
        "direction": direction,
        "direction_seed": direction_seed,
        "geometry": float(1.0 - np.sum(w_np.astype(np.float64) ** 4)),
        "amp": amp,
        "p": p,
        "injection_mode": args.injection_mode,
        "model_type": args.model_type,
        "top_k": args.top_k,
        "d_sae": d_sae,
        "l1": l1_weight,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "final_loss": last_loss,
        "final_mse": last_mse,
        "final_l1": last_l1,
        "seconds": time.time() - t0,
    }
    row.update(metrics)
    if args.save_checkpoints:
        ckpt_dir = outdir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "direction": w_np, "metrics": row},
            ckpt_dir / f"sae_cfg{config_id}_{direction}_seed{direction_seed}.pt",
        )
    return row


def summarize(rows: list[dict[str, object]], outdir: Path, started_at: float, args: argparse.Namespace) -> None:
    def mean(key: str, group: list[dict[str, object]]) -> float:
        return float(np.nanmean([float(r[key]) for r in group]))

    by_config: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    by_direction: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_config[(row["amp"], row["p"], row["l1"], row["d_sae"])].append(row)
        by_direction[(row["amp"], row["p"], row["l1"], row["d_sae"], row["direction"])].append(row)

    config_rows = []
    for (amp, p, l1, d_sae), group in sorted(by_config.items()):
        config_rows.append({
            "amp": amp,
            "p": p,
            "l1": l1,
            "d_sae": d_sae,
            "rows": len(group),
            "mean_signal_gain": mean("signal_reconstruction_gain", group),
            "mean_max_decoder_cosine": mean("max_decoder_cosine", group),
            "mean_best_atom_label_auc": mean("best_atom_label_auc", group),
            "mean_top_label_atom_abs_auc": mean("top_label_atom_abs_auc", group),
            "mean_top_label_atom_abs_auc_null": mean("top_label_atom_abs_auc_null_mean", group),
            "mean_top_label_atom_abs_auc_excess": mean("top_label_atom_abs_auc_excess", group),
            "mean_top_label_atom_decoder_cosine": mean("top_label_atom_decoder_cosine", group),
            "mean_abs_atom_label_gap": float(np.nanmean([abs(float(r["best_atom_label_gap"])) for r in group])),
            "mean_clean_mse": mean("clean_mse", group),
            "mean_injected_mse": mean("injected_mse", group),
        })

    direction_rows = []
    for (amp, p, l1, d_sae, direction), group in sorted(by_direction.items()):
        direction_rows.append({
            "amp": amp,
            "p": p,
            "l1": l1,
            "d_sae": d_sae,
            "direction": direction,
            "rows": len(group),
            "mean_geometry": mean("geometry", group),
            "mean_signal_gain": mean("signal_reconstruction_gain", group),
            "mean_max_decoder_cosine": mean("max_decoder_cosine", group),
            "mean_best_atom_label_auc": mean("best_atom_label_auc", group),
            "mean_top_label_atom_abs_auc": mean("top_label_atom_abs_auc", group),
            "mean_top_label_atom_abs_auc_null": mean("top_label_atom_abs_auc_null_mean", group),
            "mean_top_label_atom_abs_auc_excess": mean("top_label_atom_abs_auc_excess", group),
            "mean_top_label_atom_decoder_cosine": mean("top_label_atom_decoder_cosine", group),
            "mean_best_atom_label_corr": mean("best_atom_label_corr", group),
            "mean_abs_atom_label_gap": float(np.nanmean([abs(float(r["best_atom_label_gap"])) for r in group])),
        })

    write_csv(outdir / "controlled_sae_sweep_by_config.csv", config_rows)
    write_csv(outdir / "controlled_sae_sweep_by_direction.csv", direction_rows)
    summary = {
        "mode": "controlled_sae_sweep_v2",
        "activation_path": args.activation_path,
        "tokens_used": args.max_tokens,
        "rows": len(rows),
        "configs": len(by_config),
        "directions": parse_csv(args.directions, str),
        "direction_seeds": parse_csv(args.direction_seeds, int),
        "d_sae_grid": parse_csv(args.d_sae_grid, int),
        "l1_grid": parse_csv(args.l1_grid, float),
        "amp_grid": parse_csv(args.amp_grid, float),
        "p_grid": parse_csv(args.p_grid, float),
        "injection_mode": args.injection_mode,
        "model_type": args.model_type,
        "top_k": args.top_k,
        "mean_signal_gain_all": mean("signal_reconstruction_gain", rows),
        "mean_max_decoder_cosine_all": mean("max_decoder_cosine", rows),
        "mean_best_atom_label_auc_all": mean("best_atom_label_auc", rows),
        "mean_top_label_atom_abs_auc_all": mean("top_label_atom_abs_auc", rows),
        "mean_top_label_atom_abs_auc_null_all": mean("top_label_atom_abs_auc_null_mean", rows),
        "mean_top_label_atom_abs_auc_excess_all": mean("top_label_atom_abs_auc_excess", rows),
        "seconds": time.time() - started_at,
        "output_aggregate_csv": str(outdir / "controlled_sae_sweep_aggregate.csv"),
        "output_by_config_csv": str(outdir / "controlled_sae_sweep_by_config.csv"),
        "output_by_direction_csv": str(outdir / "controlled_sae_sweep_by_direction.csv"),
    }
    with (outdir / "controlled_sae_sweep_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps({"event": "summary", **summary}, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-path", required=True)
    parser.add_argument("--outdir", default="results/real_activation/c4_sae_sweep_v2")
    parser.add_argument("--directions", default="axis,dense,ksparse,pc_orth")
    parser.add_argument("--direction-seeds", default="0,1,2")
    parser.add_argument("--d-sae-grid", default="64,128,256,512")
    parser.add_argument("--l1-grid", default="0.01,0.03,0.1")
    parser.add_argument("--amp-grid", default="0.02,0.05,0.10,0.20")
    parser.add_argument("--p-grid", default="0.01,0.05")
    parser.add_argument("--injection-mode", choices=["centered", "positive"], default="centered")
    parser.add_argument("--model-type", choices=["relu_l1", "topk"], default="relu_l1")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--pca-rows", type=int, default=30000)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-tokens", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--k-sparse", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--auc-null-permutations", type=int, default=3)
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, min(args.torch_threads, 4)))

    started_at = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    x_src = np.load(args.activation_path, mmap_mode="r")
    if args.max_tokens and x_src.shape[0] > args.max_tokens:
        idx = np.sort(rng.choice(x_src.shape[0], size=args.max_tokens, replace=False))
        x = np.asarray(x_src[idx]).astype(np.float32)
    else:
        x = np.asarray(x_src).astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    print(json.dumps({"event": "loaded", "shape": list(x.shape), "seconds": time.time() - started_at}), flush=True)

    pcs = top_pcs(x, n_pcs=args.pca_components, max_rows=args.pca_rows, seed=args.seed + 1)
    np.savez_compressed(outdir / "sweep_cache.npz", mean=x.mean(axis=0), pcs=pcs)
    print(json.dumps({"event": "pca_done", "pcs": list(pcs.shape), "seconds": time.time() - started_at}), flush=True)

    directions = {}
    for direction in parse_csv(args.directions, str):
        for direction_seed in parse_csv(args.direction_seeds, int):
            drng = np.random.default_rng(args.seed + 7919 * direction_seed + 101 * len(direction))
            directions[(direction, direction_seed)] = make_direction(direction, x.shape[1], drng, pcs, args.k_sparse)

    np.savez_compressed(
        outdir / "directions.npz",
        **{f"{direction}_seed{seed}": w for (direction, seed), w in directions.items()},
    )

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    rows = []
    aggregate_path = outdir / "controlled_sae_sweep_aggregate.csv"
    config_id = 0
    for amp in parse_csv(args.amp_grid, float):
        for p in parse_csv(args.p_grid, float):
            for l1_weight in parse_csv(args.l1_grid, float):
                for d_sae in parse_csv(args.d_sae_grid, int):
                    config_id += 1
                    print(json.dumps({
                        "event": "config_start",
                        "config_id": config_id,
                        "amp": amp,
                        "p": p,
                        "l1": l1_weight,
                        "d_sae": d_sae,
                    }), flush=True)
                    for (direction, direction_seed), w_np in directions.items():
                        row = train_one(
                            x=x,
                            w_np=w_np,
                            args=args,
                            d_sae=d_sae,
                            l1_weight=l1_weight,
                            amp=amp,
                            p=p,
                            direction=direction,
                            direction_seed=direction_seed,
                            config_id=config_id,
                            device=device,
                            outdir=outdir,
                        )
                        rows.append(row)
                        write_csv(aggregate_path, rows)
                    summarize(rows, outdir, started_at, args)


if __name__ == "__main__":
    main()
