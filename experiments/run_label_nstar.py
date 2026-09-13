#!/usr/bin/env python3
"""Label-only N*(λ) on the same λ grid as the wide SAE run.

Kills the splice: labels previously sat on a complementary small-N / different-λ
grid. This script uses
  λ ∈ {0.30, 0.34, 0.38, 0.42, 0.48, 0.55, 0.65}
  N ∈ {8, 16, 32, 64, 128, 256, 512, 1024}
  m = 32, p = 0.05, cosine ≥ 0.80
and the difference-of-means estimator (no SAE training).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

P = {
    "m": 32,
    "p": 0.05,
    "lambdas": (0.30, 0.34, 0.38, 0.42, 0.48, 0.55, 0.65),
    "N_grid": (8, 16, 32, 64, 128, 256, 512, 1024),
    "n_seeds": 8,
    "base_seed": 20260824,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/results_label_nstar"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    proto = dict(P)
    if args.smoke:
        proto["lambdas"] = (0.38, 0.55)
        proto["N_grid"] = (32, 128)
        proto["n_seeds"] = 1
    m = proto["m"]
    ax, de = axis_dense(m)
    p = proto["p"]
    rows = []
    t0 = time.time()
    n_jobs = proto["n_seeds"] * 2 * len(proto["lambdas"]) * len(proto["N_grid"])
    done = 0
    raw_path = args.out / "label_nstar_raw.json"
    for seed in range(proto["base_seed"], proto["base_seed"] + proto["n_seeds"]):
        rng = np.random.default_rng(seed)
        for geo, w in (("axis", ax), ("dense", de)):
            for lam in proto["lambdas"]:
                a = lam / np.sqrt(p)
                for N in proto["N_grid"]:
                    h, z = bernoulli_hz(rng, N, m, a, p, w)
                    cos_mean = labeled_mean_cosine(h, z, w)
                    rec_mean = bool(cos_mean >= proto["cosine_threshold"])
                    rows.append(
                        {
                            "seed": int(seed),
                            "geometry": geo,
                            "lambda_eff": float(lam),
                            "N_train": int(N),
                            "channel": "label_mean",
                            "decoder_cosine": float(cos_mean),
                            "recovered": rec_mean,
                        }
                    )
                    done += 1
                    if done % 50 == 0 or done == n_jobs:
                        print(
                            f"[lab] {done}/{n_jobs} seed={seed} {geo:5s} "
                            f"lam={lam:.2f} N={N:4d} mean={cos_mean:.3f}",
                            flush=True,
                        )
    raw_path.write_text(json.dumps(rows))
    by = {"label_mean": {}}
    nstar_table = {"label_mean": {}}
    for geo in ("axis", "dense"):
        by["label_mean"][geo] = {}
        lams, nstars, nstars_interp = [], [], []
        for lam in proto["lambdas"]:
            seed_nstar = []
            seed_cos = {N: [] for N in proto["N_grid"]}
            for seed in range(proto["base_seed"], proto["base_seed"] + proto["n_seeds"]):
                sub = [
                    r
                    for r in rows
                    if r["seed"] == seed
                    and r["geometry"] == geo
                    and abs(r["lambda_eff"] - lam) < 1e-12
                    and r["channel"] == "label_mean"
                ]
                sub = sorted(sub, key=lambda r: r["N_train"])
                hit = [r["N_train"] for r in sub if r["recovered"]]
                nstar = min(hit) if hit else proto["N_grid"][-1]
                seed_nstar.append(int(nstar))
                for r in sub:
                    seed_cos[r["N_train"]].append(r["decoder_cosine"])
            mean_cos = [float(np.mean(seed_cos[N])) for N in proto["N_grid"]]
            n_interp, cens = interpolate_nstar(proto["N_grid"], mean_cos, proto["cosine_threshold"])
            by["label_mean"][geo][str(lam)] = {
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
        mask = [
            (by["label_mean"][geo][str(lam)]["Nstar_mean"] > proto["N_grid"][0] * 1.01)
            and (by["label_mean"][geo][str(lam)]["Nstar_mean"] < proto["N_grid"][-1] * 0.99)
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
        by["label_mean"][geo]["slope_first_cross"] = sl
        by["label_mean"][geo]["slope_uncensored"] = sl_unc
        by["label_mean"][geo]["slope_interp"] = sl_i
        nstar_table["label_mean"][geo] = {
            "slope_first_cross": sl,
            "slope_uncensored": sl_unc,
            "slope_interp": sl_i,
            "Nstar_mean_by_lam": {str(l): n for l, n in zip(lams, nstars)},
            "Nstar_interp_by_lam": {str(l): n for l, n in zip(lams, nstars_interp)},
        }
    summary = {
        "elapsed_sec": time.time() - t0,
        "protocol": proto,
        "by": by,
        "slopes": nstar_table,
    }
    (args.out / "label_nstar_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"slopes": nstar_table, "elapsed_sec": summary["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
