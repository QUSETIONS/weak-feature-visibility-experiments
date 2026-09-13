#!/usr/bin/env python3
"""Reproducible extras: scaling slopes, residual floor, anisotropic whitening.

All numbers in the paper that are not GPT-2 should come from this repo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_e1_e2_synthetic import (  # noqa: E402
    axis_and_dense,
    frobenius,
    geometry_coeff,
    labeled_h,
    null_thresholds,
    offdiag,
    power,
    sample_cov,
    stats_label,
    stats_unlabeled,
    unlabeled_h as unlab,
)


def slope_loglog(xs, ys):
    x = np.log(np.asarray(xs, float))
    y = np.log(np.asarray(ys, float))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def n_star(rng, kind, lam, w, m, n_trials, n_null, alpha, target=0.8):
    """Smallest N on a grid with power >= target."""
    grid = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    for N in grid:
        thresh = null_thresholds(rng, N, m, n_null, alpha)
        if kind == "label":
            h = labeled_h(rng, n_trials, N, m, lam, w)
            p = power(stats_label(h), thresh["label"])
        else:
            h = unlab(rng, n_trials, N, m, lam, w)
            _, ind = stats_unlabeled(h, None)
            p = power(ind, thresh["ind"])
        if p >= target:
            return N, p
    return grid[-1], p


def run_scaling(out: Path):
    m = 8
    _, dense = axis_and_dense(m)
    lams = [0.28, 0.36, 0.46, 0.58, 0.74]
    rng = np.random.default_rng(20260824)
    rows = []
    for lam in lams:
        n_lab, p_lab = n_star(rng, "label", lam, dense, m, 300, 400, 0.05)
        n_ind, p_ind = n_star(rng, "ind", lam, dense, m, 300, 400, 0.05)
        rows.append({"lambda": lam, "N_label": n_lab, "pow_label": p_lab, "N_ind": n_ind, "pow_ind": p_ind})
        print(f"[scale] lam={lam:.2f} N_label={n_lab} N_ind={n_ind}", flush=True)
    s_lab, _ = slope_loglog([r["lambda"] for r in rows], [r["N_label"] for r in rows])
    s_ind, _ = slope_loglog([r["lambda"] for r in rows], [r["N_ind"] for r in rows])
    summary = {"rows": rows, "slope_label": s_lab, "slope_unlabeled_ind": s_ind}
    (out / "extra_scaling.json").write_text(json.dumps(summary, indent=2))
    print("[scale] slopes label", s_lab, "ind", s_ind, flush=True)
    return summary


def run_floor(out: Path):
    m, N0, n_feat = 8, 1024, 160
    rng = np.random.default_rng(20260824)
    # heavy-tailed strengths
    lam = np.exp(rng.normal(-1.1, 0.55, size=n_feat))
    lam = np.clip(lam, 0.12, 1.2)
    W = rng.standard_normal((n_feat, m))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    C = np.array([geometry_coeff(w) / 4.0 for w in W])  # σ_eff=1
    Ns = [256, 512, 1024, 2048, 4096]
    # one null per N
    rows = []
    for N in Ns:
        thresh = null_thresholds(rng, N, m, 400, 0.05)
        missed = np.zeros(n_feat, dtype=bool)
        for i in range(n_feat):
            h = unlab(rng, 80, N, m, float(lam[i]), W[i])
            _, ind = stats_unlabeled(h, None)
            missed[i] = power(ind, thresh["ind"]) < 0.5
        pred = (N * C * lam**4) < np.log(N)  # β_N ~ log N, C already /4
        # better: use the same Monte Carlo threshold scale via Ci λ^4 vs 1/N
        pred = lam < (np.log(N) / (N * np.maximum(C, 1e-12))) ** 0.25
        obs = float(np.sum(lam[missed] ** 2) / np.sum(lam**2))
        pr = float(np.sum(lam[pred] ** 2) / np.sum(lam**2))
        rows.append({"N": N, "f_obs": obs, "f_pred": pr, "n_miss": int(missed.sum()), "n_pred": int(pred.sum())})
        print(f"[floor] N={N} f_obs={obs:.3f} f_pred={pr:.3f}", flush=True)
    r = float(np.corrcoef([x["f_obs"] for x in rows], [x["f_pred"] for x in rows])[0, 1])
    mae = float(np.mean(np.abs(np.array([x["f_obs"] for x in rows]) - np.array([x["f_pred"] for x in rows]))))
    summary = {"rows": rows, "corr": r, "mae": mae}
    (out / "extra_floor.json").write_text(json.dumps(summary, indent=2))
    print("[floor] corr", r, "mae", mae, flush=True)
    return summary


def run_aniso(out: Path):
    m, N, T = 8, 1024, 280
    rng = np.random.default_rng(20260824)
    evals = np.array([0.25, 0.4, 0.7, 1.0, 1.3, 2.0, 3.5, 6.0])
    Sigma = np.diag(evals)
    L = np.diag(np.sqrt(evals))
    Linv = np.diag(1.0 / np.sqrt(evals))
    # random directions in ambient space
    n_dir = 24
    W = rng.standard_normal((n_dir, m))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    lam = 0.55
    # null in raw and white
    eps0 = rng.standard_normal((400, N, m))
    raw0 = eps0 @ L.T
    white0 = raw0 @ Linv.T
    def ind_of(h):
        return frobenius(offdiag(sample_cov(h)))

    t_raw = float(np.quantile(ind_of(raw0), 0.95))
    t_white = float(np.quantile(ind_of(white0), 0.95))
    rows = []
    for w in W:
        z = rng.standard_normal((T, N, 1))
        eps = rng.standard_normal((T, N, m))
        h_raw = eps @ L.T + lam * z * w.reshape(1, 1, m)
        h_white = h_raw @ Linv.T
        w_white = Linv @ w
        w_white = w_white / np.linalg.norm(w_white)
        p_raw = power(ind_of(h_raw), t_raw)
        p_white = power(ind_of(h_white), t_white)
        rows.append(
            {
                "C_raw": geometry_coeff(w),
                "C_white": geometry_coeff(w_white),
                "pow_raw": p_raw,
                "pow_white": p_white,
            }
        )
    C_raw = np.array([r["C_raw"] for r in rows])
    C_w = np.array([r["C_white"] for r in rows])
    pr = np.array([r["pow_raw"] for r in rows])
    pw = np.array([r["pow_white"] for r in rows])

    def corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    summary = {
        "r_raw_Craw": corr(C_raw, pr),
        "r_raw_Cwhite": corr(C_w, pr),
        "r_white_Cwhite": corr(C_w, pw),
        "r_white_Craw": corr(C_raw, pw),
        "n_dir": n_dir,
    }
    (out / "extra_aniso.json").write_text(json.dumps({"rows": rows, **summary}, indent=2))
    print("[aniso]", summary, flush=True)
    return summary


def main():
    out = Path("results_extra")
    out.mkdir(exist_ok=True)
    run_scaling(out)
    run_floor(out)
    run_aniso(out)


if __name__ == "__main__":
    main()
