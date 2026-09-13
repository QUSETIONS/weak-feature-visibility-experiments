#!/usr/bin/env python3
"""Faithful E1/E2 synthetic runner for ICLR_experiment_design.md.

Detectors (protocol_v1.json):
  label : H1 = N(lambda w, I), statistic = ||mean||^2
  cov   : unlabeled H1 = eps + lambda z w, statistic = ||S - I||_F
  ind   : same unlabeled samples, statistic = ||offdiag(S_B)||_F in basis B

Null thresholds are Monte Carlo 1-alpha quantiles under H0, never fit on H1.
N is locked on seed base_seed at lambda=0.55 before the 8-seed run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PROTOCOL = {
    "m": 8,
    "alpha": 0.05,
    "n_trials": 700,
    "n_null": 2000,
    "n_seeds": 8,
    "base_seed": 20260824,
    "lambdas": (0.40, 0.55, 0.70),
    "lock_lambda": 0.55,
    "n_grid": (512, 1024, 2048, 4096, 8192, 16384),
    "lock_dense_power": (0.60, 0.90),
    "lock_axis_power_max": 0.10,
    "thetas_deg": (0, 15, 30, 45, 90),
    "n_haar_bases": 12,
}


def geometry_coeff(u: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64)
    u = u / np.linalg.norm(u)
    return float(1.0 - np.sum(u**4))


def predictors(u: np.ndarray) -> dict:
    u = np.asarray(u, dtype=np.float64)
    u = u / np.linalg.norm(u)
    l4 = float(np.sum(u**4))
    return {
        "one_minus_l4": 1.0 - l4,
        "one_minus_linf2": 1.0 - float(np.max(np.abs(u)) ** 2),
        "participation_ratio": 1.0 / l4,
        "l1": float(np.sum(np.abs(u))),
    }


def complete_basis(first: np.ndarray) -> np.ndarray:
    """Orthonormal B with B[:, 0] parallel to `first` (so B.T @ first ~ e_1? No).

    Detector coordinates are y = h @ B, i.e. y = B.T @ h if B is applied as
    B^T h when B's columns are the new axes. We want B.T @ w = u, with w
    given in ambient coordinates. If w = e_1, first column of B equals u.
    In general: build any orthonormal frame whose first column is u, then
    rotate it so B.T @ w = u via a Householder that maps w to e_1 in the
    frame? Simpler path used here: we generate samples already in ambient
    coordinates with a chosen w, then transform to detector coordinates
    with y = h @ Q where Q is orthonormal and Q.T @ w = u, i.e. Q's first
    mix... we construct Q such that Q.T @ w = u.
    """
    u = np.asarray(first, dtype=np.float64)
    u = u / np.linalg.norm(u)
    m = u.size
    A = np.eye(m)
    A[:, 0] = u
    q, _ = np.linalg.qr(A)
    if np.dot(q[:, 0], u) < 0:
        q[:, 0] *= -1
    return q


def basis_sending_w_to_u(w: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Orthogonal B with B.T @ w = u. Then y = h @ B are detector coords."""
    w = np.asarray(w, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    w = w / np.linalg.norm(w)
    u = u / np.linalg.norm(u)
    Bw = complete_basis(w)
    Bu = complete_basis(u)
    # Bw.T @ w = e_1, Bu.T @ u = e_1, so B = Bw @ Bu.T satisfies
    # B.T @ w = Bu @ Bw.T @ w = Bu @ e_1 = u.
    return Bw @ Bu.T


def u_from_theta(theta_rad: float, m: int) -> np.ndarray:
    u = np.zeros(m, dtype=np.float64)
    u[0] = np.cos(theta_rad)
    if m > 1:
        u[1:] = np.sin(theta_rad) / np.sqrt(m - 1)
    n = np.linalg.norm(u)
    return u / n


def sample_cov(h: np.ndarray) -> np.ndarray:
    """h: (T, N, m) -> (T, m, m) MLE covariance (divide by N)."""
    mean = h.mean(axis=1, keepdims=True)
    c = h - mean
    return np.einsum("tni,tnj->tij", c, c, optimize=True) / h.shape[1]


def offdiag(S: np.ndarray) -> np.ndarray:
    out = S.copy()
    idx = np.arange(S.shape[-1])
    out[:, idx, idx] = 0.0
    return out


def frobenius(S: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(S * S, axis=(1, 2)))


def unlabeled_h(rng: np.random.Generator, T: int, N: int, m: int, lam: float, w: np.ndarray) -> np.ndarray:
    eps = rng.standard_normal((T, N, m))
    z = rng.standard_normal((T, N, 1))
    return eps + lam * z * w.reshape(1, 1, m)


def labeled_h(rng: np.random.Generator, T: int, N: int, m: int, lam: float, w: np.ndarray) -> np.ndarray:
    return rng.standard_normal((T, N, m)) + lam * w.reshape(1, 1, m)


def stats_unlabeled(h: np.ndarray, B: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if B is not None:
        h = h @ B
    S = sample_cov(h)
    I = np.eye(S.shape[-1], dtype=np.float64)
    cov = frobenius(S - I)
    ind = frobenius(offdiag(S))
    return cov, ind


def stats_label(h: np.ndarray) -> np.ndarray:
    mean = h.mean(axis=1)
    return np.sum(mean * mean, axis=1)


def null_thresholds(rng: np.random.Generator, N: int, m: int, n_null: int, alpha: float) -> dict:
    h0 = rng.standard_normal((n_null, N, m))
    cov, ind = stats_unlabeled(h0, None)
    lab = stats_label(h0)
    q = 1.0 - alpha
    return {
        "cov": float(np.quantile(cov, q)),
        "ind": float(np.quantile(ind, q)),
        "label": float(np.quantile(lab, q)),
    }


def power(values: np.ndarray, thresh: float) -> float:
    return float(np.mean(values > thresh))


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, rng=None) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = rng or np.random.default_rng(0)
    n = values.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1.0 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def axis_and_dense(m: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.zeros(m)
    axis[0] = 1.0
    dense = np.ones(m) / np.sqrt(m)
    return axis, dense


def run_pair(rng, N, m, lam, n_trials, n_null, alpha, w, B, thresh=None):
    if thresh is None:
        thresh = null_thresholds(rng, N, m, n_null, alpha)
    h_u = unlabeled_h(rng, n_trials, N, m, lam, w)
    h_l = labeled_h(rng, n_trials, N, m, lam, w)
    cov, ind = stats_unlabeled(h_u, B)
    lab = stats_label(h_l)
    return {
        "label": power(lab, thresh["label"]),
        "cov": power(cov, thresh["cov"]),
        "ind": power(ind, thresh["ind"]),
        "C": geometry_coeff((w if B is None else B.T @ w)),
        "thresh": thresh,
    }


def lock_N(out_dir: Path) -> dict:
    m = PROTOCOL["m"]
    lam = PROTOCOL["lock_lambda"]
    axis, dense = axis_and_dense(m)
    rng = np.random.default_rng(PROTOCOL["base_seed"])
    rows = []
    chosen = None
    for N in PROTOCOL["n_grid"]:
        thresh = null_thresholds(rng, N, m, PROTOCOL["n_null"], PROTOCOL["alpha"])
        a = run_pair(rng, N, m, lam, PROTOCOL["n_trials"], PROTOCOL["n_null"], PROTOCOL["alpha"], axis, None, thresh)
        d = run_pair(rng, N, m, lam, PROTOCOL["n_trials"], PROTOCOL["n_null"], PROTOCOL["alpha"], dense, None, thresh)
        row = {
            "N": N,
            "axis_ind": a["ind"],
            "dense_ind": d["ind"],
            "axis_cov": a["cov"],
            "dense_cov": d["cov"],
            "axis_label": a["label"],
            "dense_label": d["label"],
        }
        rows.append(row)
        lo, hi = PROTOCOL["lock_dense_power"]
        ok = (d["ind"] >= lo) and (d["ind"] <= hi) and (a["ind"] <= PROTOCOL["lock_axis_power_max"])
        print(f"[lock] N={N} axis_ind={a['ind']:.3f} dense_ind={d['ind']:.3f} ok={ok}", flush=True)
        if ok and chosen is None:
            chosen = row
    if chosen is None:
        # pick N maximizing dense-axis gap among those with axis_ind <= 0.15
        cand = [r for r in rows if r["axis_ind"] <= 0.15]
        if not cand:
            cand = rows
        chosen = max(cand, key=lambda r: r["dense_ind"] - r["axis_ind"])
        chosen = dict(chosen)
        chosen["lock_note"] = "no grid point met both gates; chose max dense-axis gap with axis_ind<=0.15 if possible"
    else:
        chosen = dict(chosen)
        chosen["lock_note"] = "met both preregistered lock gates"
    payload = {"rows": rows, "locked": chosen}
    (out_dir / "n_lock.json").write_text(json.dumps(payload, indent=2))
    return payload


def run_main(out_dir: Path, locked_N: int) -> dict:
    m = PROTOCOL["m"]
    axis, dense = axis_and_dense(m)
    seeds = [PROTOCOL["base_seed"] + k for k in range(PROTOCOL["n_seeds"])]
    pair_rows = []
    rot_rows = []
    haar_rows = []
    t0 = time.time()

    for seed in seeds:
        rng = np.random.default_rng(seed)
        thresh = null_thresholds(rng, locked_N, m, PROTOCOL["n_null"], PROTOCOL["alpha"])
        for lam in PROTOCOL["lambdas"]:
            for name, w in (("axis", axis), ("dense", dense)):
                res = run_pair(
                    rng, locked_N, m, lam, PROTOCOL["n_trials"], PROTOCOL["n_null"], PROTOCOL["alpha"], w, None, thresh
                )
                pair_rows.append(
                    {
                        "seed": seed,
                        "lambda": lam,
                        "geometry": name,
                        "label": res["label"],
                        "cov": res["cov"],
                        "ind": res["ind"],
                        "C": res["C"],
                    }
                )
                print(
                    f"[pair] seed={seed} lam={lam} {name} "
                    f"label={res['label']:.3f} cov={res['cov']:.3f} ind={res['ind']:.3f}",
                    flush=True,
                )

        # rotation: fix w = e_1, lambda = lock_lambda, vary detector basis
        w = axis
        lam = PROTOCOL["lock_lambda"]
        for deg in PROTOCOL["thetas_deg"]:
            u = u_from_theta(np.deg2rad(deg), m)
            B = basis_sending_w_to_u(w, u)
            res = run_pair(
                rng, locked_N, m, lam, PROTOCOL["n_trials"], PROTOCOL["n_null"], PROTOCOL["alpha"], w, B, thresh
            )
            pred = predictors(u)
            rot_rows.append(
                {
                    "seed": seed,
                    "theta_deg": deg,
                    "label": res["label"],
                    "cov": res["cov"],
                    "ind": res["ind"],
                    **pred,
                }
            )
            print(
                f"[rot] seed={seed} theta={deg} C={pred['one_minus_l4']:.4f} "
                f"label={res['label']:.3f} cov={res['cov']:.3f} ind={res['ind']:.3f}",
                flush=True,
            )

        # E2 Haar vs align
        B_align = basis_sending_w_to_u(w, w)  # u = w = e_1, C=0
        res_a = run_pair(
            rng, locked_N, m, lam, PROTOCOL["n_trials"], PROTOCOL["n_null"], PROTOCOL["alpha"], w, B_align, thresh
        )
        haar_rows.append(
            {
                "seed": seed,
                "basis": "align",
                "ind": res_a["ind"],
                "label": res_a["label"],
                "cov": res_a["cov"],
                **predictors(w),
            }
        )
        for j in range(PROTOCOL["n_haar_bases"]):
            # independent Haar via QR of Gaussian
            G = rng.standard_normal((m, m))
            Q, R = np.linalg.qr(G)
            Q *= np.sign(np.diag(R)).reshape(1, m)
            u = Q.T @ w
            res = run_pair(
                rng, locked_N, m, lam, PROTOCOL["n_trials"], PROTOCOL["n_null"], PROTOCOL["alpha"], w, Q, thresh
            )
            haar_rows.append(
                {
                    "seed": seed,
                    "basis": f"haar_{j}",
                    "ind": res["ind"],
                    "label": res["label"],
                    "cov": res["cov"],
                    **predictors(u),
                }
            )

    # summaries
    rng_boot = np.random.default_rng(0)
    summary = {"locked_N": locked_N, "elapsed_sec": time.time() - t0, "pair": {}, "rotation": {}, "e2": {}}

    for lam in PROTOCOL["lambdas"]:
        summary["pair"][str(lam)] = {}
        for geo in ("axis", "dense"):
            sub = [r for r in pair_rows if r["lambda"] == lam and r["geometry"] == geo]
            entry = {}
            for stat in ("label", "cov", "ind"):
                mean, lo, hi = bootstrap_ci(np.array([r[stat] for r in sub]), rng=rng_boot)
                entry[stat] = {"mean": mean, "ci95": [lo, hi], "seeds": [r[stat] for r in sub]}
            entry["C"] = sub[0]["C"] if sub else None
            summary["pair"][str(lam)][geo] = entry

    rot_ind = np.array([r["ind"] for r in rot_rows])
    rot_lab = np.array([r["label"] for r in rot_rows])
    rot_c = np.array([r["one_minus_l4"] for r in rot_rows])
    summary["rotation"]["r_ind_vs_C"] = pearson(rot_c, rot_ind)
    summary["rotation"]["r_label_vs_C"] = pearson(rot_c, rot_lab)
    summary["rotation"]["r_cov_vs_C"] = pearson(rot_c, np.array([r["cov"] for r in rot_rows]))
    # competitor correlations on rotation + haar (exclude align-only if wanted: include all)
    pred_rows = rot_rows + [r for r in haar_rows if r["basis"] != "align" or True]
    y = np.array([r["ind"] for r in pred_rows])
    names = ["one_minus_l4", "one_minus_linf2", "participation_ratio", "l1"]
    corr = {name: pearson(np.array([r[name] for r in pred_rows]), y) for name in names}
    summary["e2"]["predictor_corr"] = corr
    ranked = sorted(corr.items(), key=lambda kv: abs(kv[1]) if kv[1] == kv[1] else -1, reverse=True)
    summary["e2"]["winner"] = ranked[0][0] if ranked else None
    if len(ranked) >= 2:
        summary["e2"]["winner_margin"] = abs(ranked[0][1]) - abs(ranked[1][1])
    align = [r["ind"] for r in haar_rows if r["basis"] == "align"]
    haar = [r["ind"] for r in haar_rows if r["basis"].startswith("haar")]
    summary["e2"]["align_ind"] = bootstrap_ci(np.array(align), rng=rng_boot)
    summary["e2"]["haar_ind"] = bootstrap_ci(np.array(haar), rng=rng_boot)

    # gates
    p055 = summary["pair"]["0.55"]
    axis_ind = p055["axis"]["ind"]["mean"]
    dense_ind = p055["dense"]["ind"]["mean"]
    def _b(x) -> bool:
        return bool(x) and x is not False

    r_ind = summary["rotation"]["r_ind_vs_C"]
    r_lab = summary["rotation"]["r_label_vs_C"]
    gates = {
        "E1-syn-ind": {
            "pass": _b((axis_ind <= 0.10) and (dense_ind >= 0.80)),
            "axis_ind": axis_ind,
            "dense_ind": dense_ind,
        },
        "E1-rot": {
            "pass": _b((r_ind >= 0.9) and (abs(r_lab) < 0.3 or (isinstance(r_lab, float) and r_lab != r_lab))),
            "r_ind_vs_C": r_ind,
            "r_label_vs_C": r_lab,
        },
        "E2-align": {
            "pass": _b((summary["e2"]["align_ind"][0] <= 0.10) and (summary["e2"]["haar_ind"][0] >= 0.50)),
            "align_mean": summary["e2"]["align_ind"][0],
            "haar_mean": summary["e2"]["haar_ind"][0],
        },
        "E2-coef": {
            "pass": _b(summary["e2"]["winner"] == "one_minus_l4" and summary["e2"].get("winner_margin", 0) >= 0.10),
            "corr": corr,
            "winner": summary["e2"]["winner"],
            "margin": summary["e2"].get("winner_margin"),
        },
    }
    summary["gates"] = gates

    raw = {"pair": pair_rows, "rotation": rot_rows, "haar": haar_rows}
    (out_dir / "e1_e2_raw.json").write_text(json.dumps(raw))
    (out_dir / "e1_e2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"gates": gates, "elapsed_sec": summary["elapsed_sec"]}, indent=2), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--stage", choices=["lock", "main", "all"], default="all")
    p.add_argument("--N", type=int, default=None)
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "protocol_snapshot.json").write_text(json.dumps(PROTOCOL, indent=2))
    locked_N = args.N
    if args.stage in ("lock", "all"):
        lock_payload = lock_N(out_dir)
        locked_N = int(lock_payload["locked"]["N"])
        print(f"[lock] using N={locked_N}", flush=True)
    if args.stage in ("main", "all"):
        if locked_N is None:
            lock_payload = json.loads((out_dir / "n_lock.json").read_text())
            locked_N = int(lock_payload["locked"]["N"])
        run_main(out_dir, locked_N)


if __name__ == "__main__":
    main()
