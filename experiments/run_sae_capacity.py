#!/usr/bin/env python3
"""Capacity sweep: residual / missed mass vs dictionary size.

Frozen:
  recovered iff max decoder cosine >= 0.80
  k=4 (expected active features ≈ n p = 4)
  N_train fixed; only dict size varies
  quartic predictor uses C=1-1/m for random atoms, C=0 for axis (ind)
  SAE predictor uses a uniform positive C (no axis hole)
"""
from __future__ import annotations

import argparse
import json
import math
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
    "N_train": 16384,
    "n_eval": 4096,
    "dict_sizes": (8, 16, 32, 64, 96, 160, 256),
    "k": 4,
    "n_epochs": 80,
    "sae_batch": 256,
    "sae_lr": 1e-3,
    "cosine_threshold": 0.80,
    "n_seeds": 4,
    "base_seed": 20260824,
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


def n_steps(N, batch, n_epochs):
    return int(n_epochs * max(1, int(math.ceil(N / float(batch)))))


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
    bs = min(batch, n)
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


def decoder_cosines(model, W):
    D = model.dec.weight.detach().cpu().numpy()
    cols = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return np.max(np.abs(Wn @ cols), axis=1)


def pred_dark(lam, C, N, beta=8.0):
    C = np.maximum(np.asarray(C, float), 1e-12)
    lam = np.asarray(lam, float)
    lam_c = (beta / (N * C)) ** 0.25
    dark = lam < lam_c
    num = np.sum((lam**2) * dark)
    den = np.sum(lam**2) + 1e-12
    return float(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/results_capacity"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    proto = dict(P)
    if args.smoke:
        proto["dict_sizes"] = (8, 32)
        proto["n_seeds"] = 1
        proto["n_epochs"] = 5
        proto["N_train"] = 1024
        proto["n_eval"] = 512
    m = proto["m"]
    rows = []
    t0 = time.time()
    n_jobs = proto["n_seeds"] * len(proto["dict_sizes"])
    done = 0
    raw_path = args.out / "sae_capacity_raw.json"
    steps = n_steps(proto["N_train"], proto["sae_batch"], proto["n_epochs"])
    for seed in range(proto["base_seed"], proto["base_seed"] + proto["n_seeds"]):
        rng = np.random.default_rng(seed)
        W, lam = make_features(rng, m, proto["n_axis"], proto["n_rand"])
        C = np.array([geometry_C(w) for w in W])
        h, _ = sample_superpos(rng, proto["N_train"], W, lam, proto["p"])
        h_ev, _ = sample_superpos(rng, proto["n_eval"], W, lam, proto["p"])
        var = float(h_ev.var())
        f_pred_ind = pred_dark(lam, C, proto["N_train"])
        f_pred_sae = pred_dark(lam, np.full_like(C, 1.0 - 1.0 / m), proto["N_train"])
        for d in proto["dict_sizes"]:
            model, loss = train_sae(
                h, m, d, proto["k"], steps, proto["sae_batch"], proto["sae_lr"], seed + 13 * d, device
            )
            cos = decoder_cosines(model, W)
            rec_mask = cos >= proto["cosine_threshold"]
            f_obs = float(np.sum((lam**2) * (~rec_mask)) / (np.sum(lam**2) + 1e-12))
            rec_ax = float(rec_mask[: proto["n_axis"]].mean())
            rec_rd = float(rec_mask[proto["n_axis"] :].mean())
            x = torch.tensor(h_ev, dtype=torch.float32, device=device)
            with torch.no_grad():
                hat = model(x).cpu().numpy()
            mse = float(np.mean((h_ev - hat) ** 2))
            tot_feat = float(np.sum(lam**2) + 1e-12)
            f_resid_miss = float(np.sum((lam**2) * (1.0 - np.clip(cos, 0.0, 1.0) ** 2)) / tot_feat)
            row = {
                "seed": int(seed),
                "dict_size": int(d),
                "k": int(proto["k"]),
                "N_train": int(proto["N_train"]),
                "n_steps": int(steps),
                "train_loss": float(loss),
                "mse": mse,
                "var": var,
                "mse_over_var": float(mse / (var + 1e-12)),
                "f_obs": f_obs,
                "f_pred_ind": f_pred_ind,
                "f_pred_sae": f_pred_sae,
                "f_resid_miss": f_resid_miss,
                "rec_axis": rec_ax,
                "rec_rand": rec_rd,
                "n_miss": int((~rec_mask).sum()),
                "n_axis_miss": int((~rec_mask[: proto["n_axis"]]).sum()),
                "mean_cos_axis": float(cos[: proto["n_axis"]].mean()),
                "mean_cos_rand": float(cos[proto["n_axis"] :].mean()),
            }
            rows.append(row)
            done += 1
            print(
                f"[cap] {done}/{n_jobs} seed={seed} d={d:3d} mse/var={row['mse_over_var']:.3f} "
                f"f_obs={f_obs:.3f} f_sae={f_pred_sae:.3f} f_ind={f_pred_ind:.3f} "
                f"rec_ax={rec_ax:.2f} rec_rd={rec_rd:.2f}",
                flush=True,
            )
            raw_path.write_text(json.dumps(rows))
    by_d = {}
    for d in proto["dict_sizes"]:
        sub = [r for r in rows if r["dict_size"] == d]
        by_d[str(d)] = {
            "mse_over_var": float(np.mean([r["mse_over_var"] for r in sub])),
            "f_obs": float(np.mean([r["f_obs"] for r in sub])),
            "f_pred_sae": float(np.mean([r["f_pred_sae"] for r in sub])),
            "f_pred_ind": float(np.mean([r["f_pred_ind"] for r in sub])),
            "f_resid_miss": float(np.mean([r["f_resid_miss"] for r in sub])),
            "rec_axis": float(np.mean([r["rec_axis"] for r in sub])),
            "rec_rand": float(np.mean([r["rec_rand"] for r in sub])),
        }
    summary = {
        "elapsed_sec": time.time() - t0,
        "protocol": proto,
        "device": device,
        "by_dict": by_d,
    }
    (args.out / "sae_capacity_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(by_d, indent=2), flush=True)


if __name__ == "__main__":
    main()
