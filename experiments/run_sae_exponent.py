#!/usr/bin/env python3
"""Labeled vs unlabeled SAE sample complexity, constant epochs.

Causal check for the quartic wall:
  label_mean   — difference of means given z          (quadratic channel)
  label_sae    — TopK decoder, encoder slot 0 = z     (same arch, labels)
  unlab_sae    — free TopK, no labels                 (unlabeled channel)

Frozen before looking at results:
  recovered iff decoder cosine >= 0.80
  steps = n_epochs * ceil(N / batch)  (constant epochs; no extra passes at small N)
  N* = smallest grid N with cosine >= 0.80, else N_grid[-1] (censored)
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
    "p": 0.05,
    "lambdas": (0.16, 0.20, 0.26, 0.32, 0.38, 0.45, 0.55),
    "N_grid": (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
    "n_seeds": 8,
    "base_seed": 20260824,
    "dict_size": 16,
    "k": 2,
    "n_epochs": 80,
    "sae_batch": 256,
    "sae_lr": 1e-3,
    "cosine_threshold": 0.80,
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


class TopKSAE(nn.Module):
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
            val, idx = torch.topk(pre, self.k, dim=1)
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


def decoder_cosines(model, w):
    D = model.dec.weight.detach().cpu().numpy()
    cols = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    dots = np.abs(cols.T @ unit(w))
    return float(np.max(dots)), float(dots[0])


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
        return float(Ns[0]), False
    for i in range(1, len(Ns)):
        if cs[i] >= thr:
            c0, c1 = cs[i - 1], cs[i]
            if c1 <= c0:
                return float(Ns[i]), False
            t = (thr - c0) / (c1 - c0)
            logn = np.log(Ns[i - 1]) + t * (np.log(Ns[i]) - np.log(Ns[i - 1]))
            return float(np.exp(logn)), False
    return float(Ns[-1]), True


def collapse_r(rows, channel, power):
    xs, ys = [], []
    for r in rows:
        if r["channel"] != channel:
            continue
        xs.append(r["N_train"] * (r["lambda_eff"] ** power))
        ys.append(r["decoder_cosine"])
    if len(xs) < 4:
        return float("nan")
    return float(np.corrcoef(np.log(np.clip(xs, 1e-12, None)), ys)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/results_exponent"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--dict-size", type=int, default=None)
    ap.add_argument("--n-epochs", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--skip-label-sae", action="store_true")
    ap.add_argument("--lambdas", type=str, default=None, help="comma-separated λ")
    ap.add_argument("--n-grid", type=str, default=None, help="comma-separated N")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    proto = dict(P)
    if args.m is not None:
        proto["m"] = int(args.m)
    if args.dict_size is not None:
        proto["dict_size"] = int(args.dict_size)
    if args.n_epochs is not None:
        proto["n_epochs"] = int(args.n_epochs)
    if args.n_seeds is not None:
        proto["n_seeds"] = int(args.n_seeds)
    proto["skip_label_sae"] = bool(args.skip_label_sae)
    if args.lambdas:
        proto["lambdas"] = tuple(float(x) for x in args.lambdas.split(",") if x.strip())
    if args.n_grid:
        proto["N_grid"] = tuple(int(x) for x in args.n_grid.split(",") if x.strip())
    if args.smoke:
        proto["lambdas"] = (0.20, 0.38)
        proto["N_grid"] = (128, 512)
        proto["n_seeds"] = 1
        proto["n_epochs"] = 5
    m = proto["m"]
    ax, de = axis_dense(m)
    p = proto["p"]
    rows = []
    t0 = time.time()
    n_jobs = (
        proto["n_seeds"] * 2 * len(proto["lambdas"]) * len(proto["N_grid"])
    )
    done = 0
    raw_path = args.out / "sae_exponent_raw.json"
    for seed in range(proto["base_seed"], proto["base_seed"] + proto["n_seeds"]):
        rng = np.random.default_rng(seed)
        for geo, w in (("axis", ax), ("dense", de)):
            for lam in proto["lambdas"]:
                a = lam / np.sqrt(p)
                for N in proto["N_grid"]:
                    h, z = bernoulli_hz(rng, N, m, a, p, w)
                    zc = z - p
                    steps = n_steps(N, proto["sae_batch"], proto["n_epochs"])
                    cos_mean = labeled_mean_cosine(h, z, w)
                    rec_mean = bool(cos_mean >= proto["cosine_threshold"])
                    if proto.get("skip_label_sae"):
                        cos_lab, cos_lab0, rec_lab, loss_lab = float("nan"), float("nan"), False, 0.0
                    else:
                        model_lab, loss_lab = train_sae(
                            h,
                            zc,
                            m,
                            proto["dict_size"],
                            proto["k"],
                            steps,
                            proto["sae_batch"],
                            proto["sae_lr"],
                            seed + 17 * N,
                            device,
                            labeled=True,
                        )
                        cos_lab, cos_lab0 = decoder_cosines(model_lab, w)
                        rec_lab = bool(cos_lab >= proto["cosine_threshold"])
                    model_un, loss_un = train_sae(
                        h,
                        None,
                        m,
                        proto["dict_size"],
                        proto["k"],
                        steps,
                        proto["sae_batch"],
                        proto["sae_lr"],
                        seed + 31 * N,
                        device,
                        labeled=False,
                    )
                    cos_un, _ = decoder_cosines(model_un, w)
                    rec_un = bool(cos_un >= proto["cosine_threshold"])
                    base = {
                        "seed": int(seed),
                        "geometry": geo,
                        "lambda_eff": float(lam),
                        "N_train": int(N),
                        "n_steps": int(steps),
                    }
                    recs = [
                        {
                            **base,
                            "channel": "label_mean",
                            "decoder_cosine": float(cos_mean),
                            "slot0_cosine": float(cos_mean),
                            "recovered": rec_mean,
                            "train_loss": 0.0,
                        },
                        {
                            **base,
                            "channel": "unlab_sae",
                            "decoder_cosine": float(cos_un),
                            "slot0_cosine": float("nan"),
                            "recovered": rec_un,
                            "train_loss": float(loss_un),
                        },
                    ]
                    if not proto.get("skip_label_sae"):
                        recs.insert(
                            1,
                            {
                                **base,
                                "channel": "label_sae",
                                "decoder_cosine": float(cos_lab),
                                "slot0_cosine": float(cos_lab0),
                                "recovered": rec_lab,
                                "train_loss": float(loss_lab),
                            },
                        )
                    rows.extend(recs)
                    done += 1
                    lab_s = "nan" if proto.get("skip_label_sae") else f"{cos_lab:.3f}"
                    print(
                        f"[exp] {done}/{n_jobs} seed={seed} {geo:5s} lam={lam:.2f} N={N:5d} "
                        f"steps={steps:4d} mean={cos_mean:.3f} lab={lab_s} un={cos_un:.3f}",
                        flush=True,
                    )
                    raw_path.write_text(json.dumps(rows))
    channels = ["label_mean", "unlab_sae"]
    if not proto.get("skip_label_sae"):
        channels = ["label_mean", "label_sae", "unlab_sae"]
    by = {}
    nstar_table = {}
    for ch in channels:
        by[ch] = {}
        nstar_table[ch] = {}
        for geo in ("axis", "dense"):
            by[ch][geo] = {}
            lams, nstars, nstars_interp, n_cens = [], [], [], 0
            for lam in proto["lambdas"]:
                seed_nstar = []
                seed_cos = {N: [] for N in proto["N_grid"]}
                for seed in range(proto["base_seed"], proto["base_seed"] + proto["n_seeds"]):
                    sub = [
                        r
                        for r in rows
                        if r["seed"] == seed
                        and r["geometry"] == geo
                        and r["lambda_eff"] == lam
                        and r["channel"] == ch
                    ]
                    sub = sorted(sub, key=lambda r: r["N_train"])
                    hit = [r["N_train"] for r in sub if r["recovered"]]
                    nstar = min(hit) if hit else proto["N_grid"][-1]
                    seed_nstar.append(int(nstar))
                    for r in sub:
                        seed_cos[r["N_train"]].append(r["decoder_cosine"])
                mean_cos = [float(np.mean(seed_cos[N])) for N in proto["N_grid"]]
                n_interp, cens = interpolate_nstar(proto["N_grid"], mean_cos, proto["cosine_threshold"])
                n_cens += int(cens)
                by[ch][geo][str(lam)] = {
                    "Nstar_mean": float(np.mean(seed_nstar)),
                    "Nstar_seeds": seed_nstar,
                    "mean_cosine_by_N": {
                        str(N): float(np.mean(seed_cos[N])) for N in proto["N_grid"]
                    },
                    "Nstar_interp": n_interp,
                    "censored_interp": bool(cens),
                }
                lams.append(lam)
                nstars.append(float(np.mean(seed_nstar)))
                nstars_interp.append(n_interp)
            sl, _ = slope_loglog(lams, nstars)
            # drop fully censored or floor-censored λ for a secondary slope
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
            }
    collapse = {}
    for ch in channels:
        collapse[ch] = {
            "r_vs_N_lam2": collapse_r(rows, ch, 2),
            "r_vs_N_lam4": collapse_r(rows, ch, 4),
        }
    summary = {
        "elapsed_sec": time.time() - t0,
        "protocol": proto,
        "device": device,
        "by": by,
        "slopes": nstar_table,
        "collapse": collapse,
    }
    (args.out / "sae_exponent_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"slopes": nstar_table, "collapse": collapse, "elapsed_sec": summary["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
