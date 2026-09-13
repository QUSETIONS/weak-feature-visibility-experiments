#!/usr/bin/env python3
"""Regime controls for Theorem recon / Proposition oracle.

Channels, frozen before looking:
  pca          — top eigenvector of the sample covariance     (linear global min)
  atom1_lab    — 1-atom SAE, code clamped to z-p              (oracle, no rest)
  atom1_unlab  — 1-atom free TopK                             (trained linear-ish)
  dict16_unlab — free TopK d=16 k=2, report absorption into D_{-best}

Predictions:
  pca and atom1_unlab overlay, slope near -4, axis=dense
  atom1_lab overlays the difference of means, slope near -2
  dict16 oracle floor was extra atoms, not Adam
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

P = {
    "m": 32,
    "p": 0.05,
    "lambdas": (0.30, 0.34, 0.38, 0.42, 0.48, 0.55, 0.65),
    "N_grid": (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
    "n_seeds": 6,
    "base_seed": 20260824,
    "n_epochs": 80,
    "sae_batch": 256,
    "sae_lr": 1e-3,
    "cosine_threshold": 0.80,
    "dict16": 16,
    "k16": 2,
}


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def axis_dense(m):
    ax = np.zeros(m)
    ax[0] = 1.0
    return unit(ax), unit(np.ones(m))


def bernoulli_hz(rng, N, m, a, p, w):
    z = rng.binomial(1, p, size=N).astype(np.float64)
    h = rng.standard_normal((N, m)) + a * (z - p).reshape(-1, 1) * w.reshape(1, m)
    return h, z


def n_steps(N, batch, n_epochs):
    steps_per_epoch = max(1, int(math.ceil(N / float(batch))))
    return int(n_epochs * steps_per_epoch)


class TopKSAE(nn.Module if nn is not None else object):
    def __init__(self, m, d, k):
        super().__init__()
        self.enc = nn.Linear(m, d)
        self.dec = nn.Linear(d, m, bias=False)
        self.k = k
        nn.init.orthogonal_(self.dec.weight)
        self.amp = nn.Parameter(torch.ones(1))

    def forward(self, x, z=None):
        pre = F.relu(self.enc(x))
        if z is None:
            k_use = min(self.k, pre.shape[1])
            val, idx = torch.topk(pre, k_use, dim=1)
            code = torch.zeros_like(pre).scatter_(1, idx, val)
            return self.dec(code)
        code = torch.zeros_like(pre)
        code[:, 0] = (self.amp * z.reshape(-1)).float()
        rest = pre[:, 1:]
        k_rest = max(self.k - 1, 0)
        if k_rest > 0 and rest.shape[1] > 0:
            k_use = min(k_rest, rest.shape[1])
            val, idx = torch.topk(rest, k_use, dim=1)
            code.scatter_(1, idx + 1, val)
        return self.dec(code)


def _renorm_dec(model):
    with torch.no_grad():
        w = model.dec.weight
        model.dec.weight.copy_(w / (w.norm(dim=0, keepdim=True) + 1e-8))


class LinearRank1(nn.Module if nn is not None else object):
    def __init__(self, m):
        super().__init__()
        u = torch.randn(m)
        self.u = nn.Parameter(u / (u.norm() + 1e-8))
        self.amp = nn.Parameter(torch.ones(1))

    def forward(self, x):
        u = self.u / (self.u.norm() + 1e-8)
        coeff = (x @ u) * self.amp
        return coeff.unsqueeze(1) * u.unsqueeze(0)


def train_linear(h, m, steps, batch, lr, seed, device):
    torch.manual_seed(seed)
    model = LinearRank1(m).to(device)
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
        last = float(loss.item())
    u = model.u.detach().cpu().numpy()
    return unit(u), last


def train_sae(h, z, m, d, k, steps, batch, lr, seed, device, labeled):
    torch.manual_seed(seed)
    model = TopKSAE(m, d, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(h, dtype=torch.float32, device=device)
    zt = None if z is None else torch.tensor(z, dtype=torch.float32, device=device)
    n = x.shape[0]
    bs = min(batch, n)
    last = 0.0
    for _ in range(steps):
        b = torch.randint(0, n, (bs,), device=device)
        rec = model(x[b], None if (not labeled) else zt[b])
        loss = F.mse_loss(rec, x[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        _renorm_dec(model)
        last = float(loss.item())
    return model, last


def decoder_geometry(model, w):
    D = model.dec.weight.detach().cpu().numpy()
    cols = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    dots = np.abs(cols.T @ unit(w))
    j = int(np.argmax(dots))
    best = float(dots[j])
    if cols.shape[1] <= 1:
        absorb = 0.0
    else:
        rest = np.delete(cols, j, axis=1)
        # absorbed mass: ||P_rest w||^2
        Q, _ = np.linalg.qr(rest)
        absorb = float(np.sum((Q.T @ unit(w)) ** 2))
    return best, float(dots[0] if dots.size else 0.0), absorb


def pca_cosine(h, w):
    X = np.asarray(h, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    C = (X.T @ X) / max(X.shape[0], 1)
    eig, V = np.linalg.eigh(C)
    u = V[:, int(np.argmax(eig))]
    return float(np.abs(unit(u) @ unit(w)))


def labeled_mean_cosine(h, z, w):
    on = z > 0.5
    off = ~on
    if on.sum() < 1 or off.sum() < 1:
        return 0.0
    what = unit(h[on].mean(0) - h[off].mean(0))
    return float(np.abs(what @ unit(w)))


def slope_loglog(xs, ys):
    x = np.log(np.asarray(xs, float))
    y = np.log(np.asarray(ys, float))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def interpolate_nstar(Ns, cosines, thr):
    Ns = np.asarray(Ns, float)
    cs = np.asarray(cosines, float)
    if cs[0] >= thr:
        return float(Ns[0]), True
    for i in range(1, len(Ns)):
        if cs[i] >= thr:
            c0, c1 = cs[i - 1], cs[i]
            if c1 <= c0:
                return float(Ns[i]), False
            t = (thr - c0) / (c1 - c0)
            logn = np.log(Ns[i - 1]) + t * (np.log(Ns[i]) - np.log(Ns[i - 1]))
            return float(np.exp(logn)), False
    return float(Ns[-1]), True


def summarize(rows, proto, channels):
    by = {}
    nstar_table = {}
    for ch in channels:
        by[ch] = {}
        nstar_table[ch] = {}
        for geo in ("axis", "dense"):
            by[ch][geo] = {}
            lams, nstars, nstars_interp = [], [], []
            absorb_by_lam = {}
            for lam in proto["lambdas"]:
                seed_nstar = []
                seed_cos = {N: [] for N in proto["N_grid"]}
                seed_abs = {N: [] for N in proto["N_grid"]}
                for seed in range(proto["base_seed"], proto["base_seed"] + proto["n_seeds"]):
                    sub = [
                        r
                        for r in rows
                        if r["seed"] == seed
                        and r["geometry"] == geo
                        and abs(r["lambda_eff"] - lam) < 1e-12
                        and r["channel"] == ch
                    ]
                    sub = sorted(sub, key=lambda r: r["N_train"])
                    hit = [r["N_train"] for r in sub if r["recovered"]]
                    nstar = min(hit) if hit else proto["N_grid"][-1]
                    seed_nstar.append(int(nstar))
                    for r in sub:
                        seed_cos[r["N_train"]].append(r["decoder_cosine"])
                        seed_abs[r["N_train"]].append(r.get("absorb", 0.0))
                mean_cos = [float(np.mean(seed_cos[N])) for N in proto["N_grid"]]
                n_interp, cens = interpolate_nstar(proto["N_grid"], mean_cos, proto["cosine_threshold"])
                by[ch][geo][str(lam)] = {
                    "Nstar_mean": float(np.mean(seed_nstar)),
                    "Nstar_seeds": seed_nstar,
                    "mean_cosine_by_N": {str(N): float(np.mean(seed_cos[N])) for N in proto["N_grid"]},
                    "mean_absorb_by_N": {str(N): float(np.mean(seed_abs[N])) for N in proto["N_grid"]},
                    "Nstar_interp": n_interp,
                    "censored_interp": bool(cens),
                }
                absorb_by_lam[str(lam)] = by[ch][geo][str(lam)]["mean_absorb_by_N"]
                lams.append(lam)
                nstars.append(float(np.mean(seed_nstar)))
                nstars_interp.append(n_interp)
            sl, _ = slope_loglog(lams, nstars)
            mask = [
                (by[ch][geo][str(lam)]["Nstar_mean"] > proto["N_grid"][0] * 1.01)
                and (by[ch][geo][str(lam)]["Nstar_mean"] < proto["N_grid"][-1] * 0.99)
                for lam in lams
            ]
            if sum(mask) >= 3:
                sl_unc, _ = slope_loglog(
                    [l for l, k in zip(lams, mask) if k],
                    [n for n, k in zip(nstars, mask) if k],
                )
            else:
                sl_unc = float("nan")
            sl_i, _ = slope_loglog(lams, nstars_interp)
            by[ch][geo]["slope_first_cross"] = sl
            by[ch][geo]["slope_uncensored"] = sl_unc
            by[ch][geo]["slope_interp"] = sl_i
            nstar_table[ch][geo] = {
                "slope_first_cross": sl,
                "slope_uncensored": sl_unc,
                "slope_interp": sl_i,
                "Nstar_interp_by_lam": {str(l): n for l, n in zip(lams, nstars_interp)},
            }
    return by, nstar_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/results_regime"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pca-only", action="store_true")
    ap.add_argument("--linear-only", action="store_true")
    ap.add_argument("--skip-dict16", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    proto = dict(P)
    if args.smoke:
        proto["lambdas"] = (0.38, 0.55)
        proto["N_grid"] = (256, 1024)
        proto["n_seeds"] = 1
        proto["n_epochs"] = 5
    if args.pca_only:
        device = "cpu"
    elif torch is None:
        raise RuntimeError("torch is required unless --pca-only")
    else:
        device = args.device if torch.cuda.is_available() else "cpu"
    m = proto["m"]
    ax, de = axis_dense(m)
    p = proto["p"]
    rows = []
    t0 = time.time()
    channels = ["label_mean", "pca", "atom1_lab", "atom1_unlab"]
    if not args.pca_only and not args.skip_dict16:
        channels.append("dict16_unlab")
    if args.pca_only:
        channels = ["label_mean", "pca"]
    if args.linear_only:
        channels = ["label_mean", "pca", "linear_gd"]
    raw_path = args.out / "sae_regime_raw.json"
    n_jobs = proto["n_seeds"] * 2 * len(proto["lambdas"]) * len(proto["N_grid"])
    done = 0
    for seed in range(proto["base_seed"], proto["base_seed"] + proto["n_seeds"]):
        rng = np.random.default_rng(seed)
        for geo, w in (("axis", ax), ("dense", de)):
            for lam in proto["lambdas"]:
                a = lam / np.sqrt(p)
                for N in proto["N_grid"]:
                    h, z = bernoulli_hz(rng, N, m, a, p, w)
                    zc = z - p
                    recs = []
                    base = {
                        "seed": int(seed),
                        "geometry": geo,
                        "lambda_eff": float(lam),
                        "N_train": int(N),
                    }
                    cos_m = labeled_mean_cosine(h, z, w)
                    recs.append(
                        {
                            **base,
                            "channel": "label_mean",
                            "decoder_cosine": float(cos_m),
                            "absorb": 0.0,
                            "recovered": bool(cos_m >= proto["cosine_threshold"]),
                            "train_loss": 0.0,
                        }
                    )
                    cos_p = pca_cosine(h, w)
                    recs.append(
                        {
                            **base,
                            "channel": "pca",
                            "decoder_cosine": float(cos_p),
                            "absorb": 0.0,
                            "recovered": bool(cos_p >= proto["cosine_threshold"]),
                            "train_loss": 0.0,
                        }
                    )
                    if args.linear_only:
                        steps = n_steps(N, proto["sae_batch"], proto["n_epochs"])
                        u_lin, loss_lin = train_linear(
                            h, m, steps, proto["sae_batch"], proto["sae_lr"], seed + 53 * N, device
                        )
                        cos_lin = float(np.abs(u_lin @ unit(w)))
                        recs.append(
                            {
                                **base,
                                "channel": "linear_gd",
                                "decoder_cosine": cos_lin,
                                "absorb": 0.0,
                                "recovered": bool(cos_lin >= proto["cosine_threshold"]),
                                "train_loss": float(loss_lin),
                            }
                        )
                    elif not args.pca_only:
                        steps = n_steps(N, proto["sae_batch"], proto["n_epochs"])
                        model_lab, loss_lab = train_sae(
                            h, zc, m, 1, 1, steps, proto["sae_batch"], proto["sae_lr"],
                            seed + 17 * N, device, labeled=True,
                        )
                        cos_al, _, abs_al = decoder_geometry(model_lab, w)
                        recs.append(
                            {
                                **base,
                                "channel": "atom1_lab",
                                "decoder_cosine": float(cos_al),
                                "absorb": float(abs_al),
                                "recovered": bool(cos_al >= proto["cosine_threshold"]),
                                "train_loss": float(loss_lab),
                            }
                        )
                        model_un, loss_un = train_sae(
                            h, None, m, 1, 1, steps, proto["sae_batch"], proto["sae_lr"],
                            seed + 31 * N, device, labeled=False,
                        )
                        cos_au, _, abs_au = decoder_geometry(model_un, w)
                        recs.append(
                            {
                                **base,
                                "channel": "atom1_unlab",
                                "decoder_cosine": float(cos_au),
                                "absorb": float(abs_au),
                                "recovered": bool(cos_au >= proto["cosine_threshold"]),
                                "train_loss": float(loss_un),
                            }
                        )
                        if not args.skip_dict16:
                            model16, loss16 = train_sae(
                                h, None, m, proto["dict16"], proto["k16"], steps,
                                proto["sae_batch"], proto["sae_lr"], seed + 47 * N, device, labeled=False,
                            )
                            cos16, _, abs16 = decoder_geometry(model16, w)
                            recs.append(
                                {
                                    **base,
                                    "channel": "dict16_unlab",
                                    "decoder_cosine": float(cos16),
                                    "absorb": float(abs16),
                                    "recovered": bool(cos16 >= proto["cosine_threshold"]),
                                    "train_loss": float(loss16),
                                }
                            )
                    rows.extend(recs)
                    done += 1
                    extra = ""
                    if args.linear_only:
                        extra = f" lin={cos_lin:.3f}"
                    elif not args.pca_only:
                        extra = f" a1lab={cos_al:.3f} a1un={cos_au:.3f}"
                    print(
                        f"[reg] {done}/{n_jobs} seed={seed} {geo:5s} lam={lam:.2f} N={N:5d} "
                        f"mean={cos_m:.3f} pca={cos_p:.3f}{extra}",
                        flush=True,
                    )
                    raw_path.write_text(json.dumps(rows))
    by, nstar_table = summarize(rows, proto, channels)
    summary = {
        "elapsed_sec": time.time() - t0,
        "protocol": proto,
        "device": device,
        "pca_only": bool(args.pca_only),
        "by": by,
        "slopes": nstar_table,
    }
    (args.out / "sae_regime_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"slopes": nstar_table, "elapsed_sec": summary["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
