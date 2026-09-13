#!/usr/bin/env python3
"""SAE sample-complexity N*(λ) on axis vs dense.

Frozen:
  recovered iff decoder cosine >= 0.80
  N_train grid and lambdas locked below
  n_star = smallest N on the grid with cosine >= 0.80
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
    "lambdas": (0.22, 0.28, 0.35, 0.45, 0.55),
    "N_grid": (500, 1000, 2000, 4000, 8000, 16000, 32000, 64000),
    "n_seeds": 6,
    "base_seed": 20260824,
    "dict_size": 16,
    "k": 2,
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


def bernoulli_h(rng, N, m, a, p, w):
    z = rng.binomial(1, p, size=(N, 1)).astype(np.float64)
    h = rng.standard_normal((N, m)) + a * (z - p) * w.reshape(1, m)
    return h


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
    bs = min(batch, n)
    last = 0.0
    for _ in range(steps):
        b = torch.randint(0, n, (bs,), device=device)
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


def slope_loglog(xs, ys):
    x = np.log(np.asarray(xs, float))
    y = np.log(np.asarray(ys, float))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/results_nstar"))
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
        for geo, w in (("axis", ax), ("dense", de)):
            for lam in P["lambdas"]:
                a = lam / np.sqrt(p)
                nstar = None
                cos_star = None
                for N in P["N_grid"]:
                    h = bernoulli_h(rng, N, m, a, p, w)
                    model, loss = train_sae(
                        h, m, P["dict_size"], P["k"], P["sae_steps"], P["sae_batch"], P["sae_lr"], seed + N, device
                    )
                    cos = decoder_cosine(model, w)
                    rec = bool(cos >= P["cosine_threshold"])
                    row = {
                        "seed": seed,
                        "geometry": geo,
                        "lambda_eff": lam,
                        "N_train": N,
                        "decoder_cosine": cos,
                        "recovered": rec,
                        "train_loss": loss,
                    }
                    rows.append(row)
                    print(
                        f"[nstar] seed={seed} {geo:5s} lam={lam:.2f} N={N:5d} cos={cos:.3f} rec={rec}",
                        flush=True,
                    )
                    (args.out / "sae_nstar_raw.json").write_text(json.dumps(rows))
                    if rec and nstar is None:
                        nstar = N
                        cos_star = cos
                if nstar is None:
                    nstar = P["N_grid"][-1]
                    cos_star = cos
                print(f"[nstar]  -> N*={nstar} cos={cos_star:.3f}", flush=True)
    by = {}
    for geo in ("axis", "dense"):
        by[geo] = {}
        ns, lams = [], []
        for lam in P["lambdas"]:
            vals = []
            for seed in range(P["base_seed"], P["base_seed"] + P["n_seeds"]):
                sub = [
                    r
                    for r in rows
                    if r["seed"] == seed and r["geometry"] == geo and r["lambda_eff"] == lam and r["recovered"]
                ]
                nstar = min(r["N_train"] for r in sub) if sub else P["N_grid"][-1]
                vals.append(nstar)
            mean = float(np.mean(vals))
            by[geo][str(lam)] = {"Nstar_mean": mean, "Nstar_seeds": vals}
            ns.append(mean)
            lams.append(lam)
        sl, _ = slope_loglog(lams, ns)
        by[geo]["slope"] = sl
    summary = {"elapsed_sec": time.time() - t0, "protocol": P, "by": by}
    (args.out / "sae_nstar_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
