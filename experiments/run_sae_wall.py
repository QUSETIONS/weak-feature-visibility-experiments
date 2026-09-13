#!/usr/bin/env python3
"""SAE recovery vs λ for axis and dense. Title-scale figure: the quartic wall.

Frozen before looking at results:
  recovered iff decoder cosine >= 0.80
  lambdas, seeds, N_train, steps locked below
  geometry is not a factor in the SAE threshold; λ is
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P = {
    "m": 8,
    "p": 0.05,
    "lambdas": (0.20, 0.26, 0.32, 0.38, 0.45, 0.55, 0.70),
    "n_seeds": 8,
    "base_seed": 20260824,
    "alpha": 0.05,
    "N_detect": 4096,
    "n_trials": 400,
    "n_null": 800,
    "dict_size": 16,
    "k": 2,
    "N_train": 20000,
    "sae_steps": 4000,
    "sae_batch": 512,
    "sae_lr": 1e-3,
    "cosine_threshold": 0.80,
}


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def axis_dense(m):
    ax = np.zeros(m)
    ax[0] = 1.0
    return unit(ax), unit(np.ones(m))


def sample_cov(h):
    c = h - h.mean(axis=1, keepdims=True)
    return np.einsum("tni,tnj->tij", c, c, optimize=True) / h.shape[1]


def offdiag(S):
    out = S.copy()
    i = np.arange(S.shape[-1])
    out[:, i, i] = 0.0
    return out


def fro(S):
    return np.sqrt(np.sum(S * S, axis=(1, 2)))


def bernoulli_h(rng, T, N, m, a, p, w):
    z = rng.binomial(1, p, size=(T, N, 1)).astype(np.float64)
    h = rng.standard_normal((T, N, m)) + a * (z - p) * w.reshape(1, 1, m)
    return h, z


def label_stat_z(h, z):
    z = z.reshape(h.shape[0], h.shape[1], 1)
    pos_n = np.clip(z.sum(axis=1), 1.0, None)
    neg_n = np.clip((1.0 - z).sum(axis=1), 1.0, None)
    pos = (h * z).sum(axis=1) / pos_n
    neg = (h * (1.0 - z)).sum(axis=1) / neg_n
    d = pos - neg
    return np.sum(d * d, axis=1)


class TopKSAE(nn.Module):
    def __init__(self, m, d, k):
        super().__init__()
        self.enc = nn.Linear(m, d)
        self.dec = nn.Linear(d, m, bias=False)
        self.k = k
        nn.init.orthogonal_(self.dec.weight)

    def forward(self, x):
        pre = F.relu(self.enc(x))
        val, idx = torch.topk(pre, self.k, dim=1)
        code = torch.zeros_like(pre).scatter_(1, idx, val)
        return self.dec(code)


def train_sae(h, m, d, k, steps, batch, lr, seed, device):
    torch.manual_seed(seed)
    model = TopKSAE(m, d, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(h, dtype=torch.float32, device=device)
    n = x.shape[0]
    last = 0.0
    for _ in range(steps):
        b = torch.randint(0, n, (batch,), device=device)
        rec = model(x[b])
        loss = F.mse_loss(rec, x[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            w = model.dec.weight
            model.dec.weight.copy_(w / (w.norm(dim=0, keepdim=True) + 1e-8))
        last = float(loss.item())
    return model, last


def decoder_cosine(model, w):
    D = model.dec.weight.detach().cpu().numpy()
    cols = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    return float(np.max(np.abs(cols.T @ unit(w))))


def boot_mean(vals, rng, n=1000):
    vals = np.asarray(vals, float)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(vals.mean()), "ci95": [float(lo), float(hi)], "seeds": [float(x) for x in vals]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/results_wall"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    m = P["m"]
    ax, de = axis_dense(m)
    p = P["p"]
    rows = []
    t0 = time.time()
    for seed in range(P["base_seed"], P["base_seed"] + P["n_seeds"]):
        rng = np.random.default_rng(seed)
        h0, z0 = bernoulli_h(rng, P["n_null"], P["N_detect"], m, 0.0, p, ax)
        S0 = sample_cov(h0)
        thr = {
            "ind": float(np.quantile(fro(offdiag(S0)), 1 - P["alpha"])),
            "cov": float(np.quantile(fro(S0 - np.eye(m)), 1 - P["alpha"])),
            "label": float(np.quantile(label_stat_z(h0, z0), 1 - P["alpha"])),
        }
        for geo, w in (("axis", ax), ("dense", de)):
            for lam in P["lambdas"]:
                a = lam / np.sqrt(p)
                h1, z1 = bernoulli_h(rng, P["n_trials"], P["N_detect"], m, a, p, w)
                S1 = sample_cov(h1)
                powers = {
                    "label": float(np.mean(label_stat_z(h1, z1) > thr["label"])),
                    "cov": float(np.mean(fro(S1 - np.eye(m)) > thr["cov"])),
                    "ind": float(np.mean(fro(offdiag(S1)) > thr["ind"])),
                }
                h_tr, _ = bernoulli_h(rng, 1, P["N_train"], m, a, p, w)
                model, loss = train_sae(
                    h_tr[0], m, P["dict_size"], P["k"], P["sae_steps"], P["sae_batch"], P["sae_lr"], seed, device
                )
                cos = decoder_cosine(model, w)
                rec = bool(cos >= P["cosine_threshold"])
                row = {
                    "seed": seed,
                    "geometry": geo,
                    "lambda_eff": lam,
                    "C": float(1.0 - np.sum(w**4)),
                    **{f"det_{k}": v for k, v in powers.items()},
                    "decoder_cosine": cos,
                    "recovered": rec,
                    "train_loss": loss,
                }
                rows.append(row)
                print(
                    f"[wall] seed={seed} {geo:5s} lam={lam:.2f} "
                    f"lab={powers['label']:.2f} cov={powers['cov']:.2f} ind={powers['ind']:.2f} "
                    f"cos={cos:.3f} rec={rec}",
                    flush=True,
                )
                (args.out / "sae_wall_raw.json").write_text(json.dumps(rows))
    rng = np.random.default_rng(0)
    summary = {"elapsed_sec": time.time() - t0, "protocol": P, "by": {}}
    for geo in ("axis", "dense"):
        summary["by"][geo] = {}
        for lam in P["lambdas"]:
            sub = [r for r in rows if r["geometry"] == geo and r["lambda_eff"] == lam]
            summary["by"][geo][str(lam)] = {
                "label": boot_mean([r["det_label"] for r in sub], rng),
                "cov": boot_mean([r["det_cov"] for r in sub], rng),
                "ind": boot_mean([r["det_ind"] for r in sub], rng),
                "sae_recovery": boot_mean([float(r["recovered"]) for r in sub], rng),
                "sae_cosine": boot_mean([r["decoder_cosine"] for r in sub], rng),
            }
    (args.out / "sae_wall_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("elapsed_sec",)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
