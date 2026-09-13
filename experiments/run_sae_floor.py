#!/usr/bin/env python3
"""Trained-SAE residual vs predicted detectability floor.

Closes Prop 2: on a known multi-feature mixture, SAE reconstruction
residual should track the quartic (geometry-free) dark-mass, not the
independence-test mass that sends C=0 features to infinity.

Frozen:
  recovered iff max decoder cosine >= 0.80
  two predictors compared to observed missed-variance share
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
    "n_axis": 8,
    "n_rand": 72,
    "p": 0.05,
    "Ns": (1024, 2048, 4096, 8192, 16384, 32768),
    "n_train_extra": 0,
    "dict_size": 192,
    "k": 8,
    "sae_steps": 6000,
    "sae_batch": 512,
    "sae_lr": 1e-3,
    "cosine_threshold": 0.80,
    "n_seeds": 4,
    "base_seed": 20260824,
    "n_eval": 4096,
}


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def geometry_C(w):
    w = unit(w)
    return float(1.0 - np.sum(w**4))


def make_features(rng, m, n_axis, n_rand):
    W = []
    for j in range(n_axis):
        e = np.zeros(m)
        e[j % m] = 1.0
        W.append(e)
    for _ in range(n_rand):
        W.append(unit(rng.standard_normal(m)))
    W = np.stack(W, 0)
    lam = np.exp(rng.normal(-0.85, 0.45, size=len(W)))
    lam = np.clip(lam, 0.15, 1.1)
    return W, lam


def sample_superpos(rng, N, W, lam, p):
    n, m = W.shape
    a = lam / np.sqrt(p)
    z = rng.binomial(1, p, size=(N, n)).astype(np.float64)
    eps = rng.standard_normal((N, m))
    h = eps + ((z - p) * a.reshape(1, n)) @ W
    return h, z


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
        return self.dec(code), code


def train_sae(h, m, d, k, steps, batch, lr, seed, device):
    torch.manual_seed(seed)
    model = TopKSAE(m, d, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(h, dtype=torch.float32, device=device)
    n = x.shape[0]
    last = 0.0
    for _ in range(steps):
        b = torch.randint(0, n, (min(batch, n),), device=device)
        rec, _ = model(x[b])
        loss = F.mse_loss(rec, x[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            w = model.dec.weight
            model.dec.weight.copy_(w / (w.norm(dim=0, keepdim=True) + 1e-8))
        last = float(loss.item())
    return model, last


def decoder_cosines(model, W):
    D = model.dec.weight.detach().cpu().numpy()
    cols = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return np.max(np.abs(W @ cols), axis=1)


def pred_dark(lam, C, N, beta_scale=1.0):
    C = np.maximum(C, 0.0)
    # λ_c = (β / (N C))^{1/4}; C=0 => always dark
    dark = np.zeros(len(lam), dtype=bool)
    pos = C > 1e-12
    dark[~pos] = True
    lam_c = (beta_scale * np.log(N + 1.0) / (N * C[pos])) ** 0.25
    dark[pos] = lam[pos] < lam_c
    return dark


def dark_share(lam, mask):
    return float(np.sum(lam[mask] ** 2) / np.sum(lam**2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/results_sae_floor"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    rows = []
    t0 = time.time()
    for seed in range(P["base_seed"], P["base_seed"] + P["n_seeds"]):
        rng = np.random.default_rng(seed)
        W, lam = make_features(rng, P["m"], P["n_axis"], P["n_rand"])
        C_ind = np.array([geometry_C(w) / 4.0 for w in W])
        C_sae = np.full_like(C_ind, float(np.median(C_ind[C_ind > 1e-6])))
        for N in P["Ns"]:
            h_tr, _ = sample_superpos(rng, N, W, lam, P["p"])
            model, loss = train_sae(
                h_tr, P["m"], P["dict_size"], P["k"], P["sae_steps"], P["sae_batch"], P["sae_lr"], seed, device
            )
            h_ev, _ = sample_superpos(rng, P["n_eval"], W, lam, P["p"])
            with torch.no_grad():
                x = torch.tensor(h_ev, dtype=torch.float32, device=device)
                rec, _ = model(x)
                rec = rec.cpu().numpy()
            mse = float(np.mean((h_ev - rec) ** 2))
            var = float(np.mean(h_ev**2))
            noise = 1.0  # per-coordinate N(0,1)
            excess = max(mse - noise, 0.0)
            cos = decoder_cosines(model, W)
            missed = cos < P["cosine_threshold"]
            f_obs = dark_share(lam, missed)
            f_ind = dark_share(lam, pred_dark(lam, C_ind, N))
            f_sae = dark_share(lam, pred_dark(lam, C_sae, N))
            axis = np.arange(P["n_axis"])
            rand = np.arange(P["n_axis"], len(W))
            row = {
                "seed": seed,
                "N": N,
                "train_loss": loss,
                "mse": mse,
                "var": var,
                "excess": excess,
                "f_obs": f_obs,
                "f_pred_ind": f_ind,
                "f_pred_sae": f_sae,
                "rec_axis": float(np.mean(~missed[axis])),
                "rec_rand": float(np.mean(~missed[rand])),
                "cos_axis": float(np.mean(cos[axis])),
                "cos_rand": float(np.mean(cos[rand])),
                "n_miss": int(missed.sum()),
                "n_axis_miss": int(missed[axis].sum()),
            }
            rows.append(row)
            print(
                f"[floor] seed={seed} N={N:5d} f_obs={f_obs:.3f} f_ind={f_ind:.3f} f_sae={f_sae:.3f} "
                f"rec_ax={row['rec_axis']:.2f} rec_rd={row['rec_rand']:.2f} mse={mse:.3f}",
                flush=True,
            )
            (args.out / "sae_floor_raw.json").write_text(json.dumps(rows))

    def corr(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    f_obs = [r["f_obs"] for r in rows]
    summary = {
        "elapsed_sec": time.time() - t0,
        "protocol": P,
        "corr_ind": corr(f_obs, [r["f_pred_ind"] for r in rows]),
        "corr_sae": corr(f_obs, [r["f_pred_sae"] for r in rows]),
        "mae_ind": float(np.mean(np.abs(np.array(f_obs) - [r["f_pred_ind"] for r in rows]))),
        "mae_sae": float(np.mean(np.abs(np.array(f_obs) - [r["f_pred_sae"] for r in rows]))),
        "mean_rec_axis": float(np.mean([r["rec_axis"] for r in rows])),
        "mean_rec_rand": float(np.mean([r["rec_rand"] for r in rows])),
    }
    (args.out / "sae_floor_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
