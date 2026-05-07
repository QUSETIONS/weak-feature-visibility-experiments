"""Controlled SAE training on injected real activations.

This is a deliberately small C4 experiment. It asks whether a plain ReLU SAE
trained on a real GPT-2 activation background recovers an injected sparse
feature, and whether recovery depends on the geometry predicted by the quartic
law. The script is not a replacement for the detector experiments; it is a
training-level bridge.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def top_pcs(x: np.ndarray, n_pcs: int = 8) -> np.ndarray:
    xc = x.astype(np.float64) - x.mean(axis=0, keepdims=True)
    if xc.shape[0] > 30000:
        rng = np.random.default_rng(0)
        xc = xc[rng.choice(xc.shape[0], size=30000, replace=False)]
    _, _, vh = np.linalg.svd(xc, full_matrices=False)
    return vh[:n_pcs]


def make_direction(kind: str, d: int, rng: np.random.Generator, pc_basis: np.ndarray | None, k: int) -> np.ndarray:
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
        if pc_basis is None:
            raise ValueError("pc_basis required for pc1")
        w = pc_basis[0].copy()
    elif kind == "pc_orth":
        if pc_basis is None:
            raise ValueError("pc_basis required for pc_orth")
        w = rng.normal(size=d)
        for j in range(min(8, pc_basis.shape[0])):
            w = w - np.dot(w, pc_basis[j]) * pc_basis[j]
    else:
        raise ValueError(kind)
    return w / (np.linalg.norm(w) + 1.0e-12)


class TinySAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
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
        recon = acts @ self.W_dec + self.b_dec
        return recon, acts


@torch.no_grad()
def evaluate(
    model: TinySAE,
    x: np.ndarray,
    w_np: np.ndarray,
    amp: float,
    p: float,
    rng: np.random.Generator,
    device: torch.device,
    n_eval: int,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    idx = rng.choice(x.shape[0], size=min(n_eval, x.shape[0]), replace=False)
    w = torch.tensor(w_np, device=device, dtype=torch.float32)
    decoder = F.normalize(model.W_dec.detach(), dim=1)
    cos = torch.abs(decoder @ w)
    max_cos, argmax = torch.max(cos, dim=0)
    top10_cos = torch.topk(cos, k=min(10, cos.numel())).values.mean()

    signal_true = []
    signal_recon = []
    atom_acts = []
    labels = []
    clean_mse = 0.0
    inj_mse = 0.0
    total = 0

    for start in range(0, len(idx), batch_size):
        rows = idx[start : start + batch_size]
        xb = torch.from_numpy(np.asarray(x[rows])).to(device=device, dtype=torch.float32)
        z = torch.bernoulli(torch.full((xb.shape[0],), p, device=device))
        zi = z - p
        xi = xb + amp * zi[:, None] * w[None, :]
        recon_clean, _ = model(xb)
        recon_inj, acts = model(xi)
        delta_recon = recon_inj - recon_clean
        signal_recon.append((delta_recon @ w).detach().cpu())
        signal_true.append((amp * zi).detach().cpu())
        atom_acts.append(acts[:, int(argmax.item())].detach().cpu())
        labels.append(z.detach().cpu())
        clean_mse += float(F.mse_loss(recon_clean, xb, reduction="sum").item())
        inj_mse += float(F.mse_loss(recon_inj, xi, reduction="sum").item())
        total += xb.numel()

    y_true = torch.cat(signal_true).float()
    y_rec = torch.cat(signal_recon).float()
    atom = torch.cat(atom_acts).float()
    lab = torch.cat(labels).float()
    denom = torch.sum(y_true * y_true).clamp_min(1e-9)
    gain = float(torch.sum(y_true * y_rec).item() / denom.item())
    rec_corr = float(torch.corrcoef(torch.stack([y_true, y_rec]))[0, 1].nan_to_num(0.0).item())
    if lab.sum() > 1 and (1 - lab).sum() > 1:
        atom_label_gap = float(atom[lab > 0.5].mean().item() - atom[lab < 0.5].mean().item())
    else:
        atom_label_gap = 0.0

    return {
        "max_decoder_cosine": float(max_cos.item()),
        "top10_decoder_cosine": float(top10_cos.item()),
        "best_atom": int(argmax.item()),
        "signal_reconstruction_gain": gain,
        "signal_reconstruction_corr": rec_corr,
        "best_atom_label_gap": atom_label_gap,
        "clean_mse": clean_mse / total,
        "injected_mse": inj_mse / total,
    }


def train_one(args: argparse.Namespace, x: np.ndarray, direction: str, seed: int, outdir: Path, pcs: np.ndarray) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    d = x.shape[1]
    w_np = make_direction(direction, d, rng, pcs, args.k_sparse)
    geometry = float(1.0 - np.sum(w_np**4))
    model = TinySAE(d, args.d_sae).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    w = torch.tensor(w_np, device=device, dtype=torch.float32)

    t0 = time.time()
    last_loss = 0.0
    last_mse = 0.0
    last_l1 = 0.0
    for step in range(1, args.steps + 1):
        rows = rng.choice(x.shape[0], size=args.batch_size, replace=False)
        xb = torch.from_numpy(np.asarray(x[rows])).to(device=device, dtype=torch.float32)
        z = torch.bernoulli(torch.full((args.batch_size,), args.p, device=device))
        xi = xb + args.amp * (z - args.p)[:, None] * w[None, :]
        recon, acts = model(xi)
        mse = F.mse_loss(recon, xi)
        l1 = acts.abs().mean()
        loss = mse + args.l1 * l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.normalize_decoder()
        last_loss = float(loss.item())
        last_mse = float(mse.item())
        last_l1 = float(l1.item())
        if args.log_every and step % args.log_every == 0:
            print(json.dumps({"direction": direction, "seed": seed, "step": step, "loss": last_loss, "mse": last_mse, "l1": last_l1}))

    metrics = evaluate(model, x, w_np, args.amp, args.p, rng, device, args.eval_tokens, args.eval_batch_size)
    row: dict[str, object] = {
        "direction": direction,
        "seed": seed,
        "geometry": geometry,
        "amp": args.amp,
        "p": args.p,
        "d_sae": args.d_sae,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "l1": args.l1,
        "final_loss": last_loss,
        "final_mse": last_mse,
        "final_l1": last_l1,
        "seconds": time.time() - t0,
    }
    row.update(metrics)
    torch.save({"state_dict": model.state_dict(), "direction": w_np, "metrics": row}, outdir / f"sae_{direction}_seed{seed}.pt")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-path", required=True)
    parser.add_argument("--outdir", default="results/real_activation/c4_sae_training")
    parser.add_argument("--directions", default="axis,dense,ksparse,pc_orth")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--d-sae", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-tokens", type=int, default=20000)
    parser.add_argument("--amp", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--k-sparse", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    x = np.load(args.activation_path, mmap_mode="r")
    if args.max_tokens and x.shape[0] > args.max_tokens:
        idx = np.sort(rng.choice(x.shape[0], size=args.max_tokens, replace=False))
        x_work = np.asarray(x[idx]).astype(np.float32)
    else:
        x_work = np.asarray(x).astype(np.float32)
    x_work = x_work - x_work.mean(axis=0, keepdims=True)
    pcs = top_pcs(x_work, n_pcs=8)

    rows = []
    for direction in [d.strip() for d in args.directions.split(",") if d.strip()]:
        for seed in [int(s) for s in args.seeds.split(",")]:
            rows.append(train_one(args, x_work, direction, seed, outdir, pcs))
            write_csv(outdir / "controlled_sae_training.csv", rows)

    summary = {
        "mode": "controlled_sae_training",
        "activation_path": args.activation_path,
        "tokens_used": int(x_work.shape[0]),
        "rows": len(rows),
        "mean_signal_gain": float(np.mean([r["signal_reconstruction_gain"] for r in rows])),
        "mean_max_decoder_cosine": float(np.mean([r["max_decoder_cosine"] for r in rows])),
        "output_csv": str(outdir / "controlled_sae_training.csv"),
    }
    with (outdir / "controlled_sae_training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
