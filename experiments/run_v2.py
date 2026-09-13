#!/usr/bin/env python3
"""v2 re-runs: E2-sep (4-norm vs l1) and E4 (detector geometry vs free SAE).

Does not re-run E1 pair/rotation. Frozen protocol in ICLR_experiment_design_v2.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None


P = {
    "m": 8,
    "N": 1024,
    "N_detect_e4": 4096,
    "n_trials": 700,
    "n_trials_e4": 400,
    "n_null": 2000,
    "n_null_e4": 800,
    "n_seeds": 8,
    "base_seed": 20260824,
    "alpha": 0.05,
    "lambda_sep": 0.55,
    "p": 0.05,
    "lambdas_e4": (0.35, 0.55, 0.80),
    "cosine_threshold": 0.80,
    "dict_size": 16,
    "k": 2,
    "N_train": 20000,
    "sae_steps": 4000,
    "sae_batch": 512,
    "sae_lr": 1e-3,
}


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n


def axis_dense(m):
    ax = np.zeros(m)
    ax[0] = 1.0
    return unit(ax), unit(np.ones(m))


def geom_C(u):
    u = unit(u)
    return float(1.0 - np.sum(u**4))


def predictors(u):
    u = unit(u)
    l4 = float(np.sum(u**4))
    return {
        "one_minus_l4": 1.0 - l4,
        "one_minus_linf2": 1.0 - float(np.max(np.abs(u)) ** 2),
        "participation_ratio": 1.0 / l4,
        "l1": float(np.sum(np.abs(u))),
    }


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


def gaussian_h(rng, T, N, m, lam, w):
    return rng.standard_normal((T, N, m)) + lam * rng.standard_normal((T, N, 1)) * w.reshape(1, 1, m)


def bernoulli_h(rng, T, N, m, a, p, w):
    z = rng.binomial(1, p, size=(T, N, 1)).astype(np.float64)
    h = rng.standard_normal((T, N, m)) + a * (z - p) * w.reshape(1, 1, m)
    return h, z


def labeled_mean_shift(rng, T, N, m, lam, w):
    return rng.standard_normal((T, N, m)) + lam * w.reshape(1, 1, m)


def ind_stat(h):
    return fro(offdiag(sample_cov(h)))


def cov_stat(h):
    m = h.shape[-1]
    return fro(sample_cov(h) - np.eye(m))


def label_stat_mean(h):
    mu = h.mean(axis=1)
    return np.sum(mu * mu, axis=1)


def label_stat_z(h, z):
    z = z.reshape(h.shape[0], h.shape[1], 1)
    pos_n = np.clip(z.sum(axis=1), 1.0, None)
    neg_n = np.clip((1.0 - z).sum(axis=1), 1.0, None)
    pos = (h * z).sum(axis=1) / pos_n
    neg = (h * (1.0 - z)).sum(axis=1) / neg_n
    d = pos - neg
    return np.sum(d * d, axis=1)


def quantile_thresh(vals, alpha):
    return float(np.quantile(vals, 1.0 - alpha))


def power(vals, thr):
    return float(np.mean(vals > thr))


def boot_mean(vals, rng, n=1000):
    vals = np.asarray(vals, float)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(vals.mean()), "ci95": [float(lo), float(hi)], "seeds": [float(x) for x in vals]}


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def direction_bank(m):
    dirs = []
    ax, de = axis_dense(m)
    dirs.append(("axis", ax))
    dirs.append(("dense", de))
    for k in (1, 2, 4, 8):
        u = np.zeros(m)
        u[:k] = 1.0 / np.sqrt(k)
        dirs.append((f"ksparse_{k}", unit(u)))
    for eps in (0.50, 0.20, 0.05, 0.01):
        u = np.zeros(m)
        u[0] = np.sqrt(1.0 - eps)
        u[1] = np.sqrt(eps)
        dirs.append((f"twosparse_eps{eps:.2f}", unit(u)))
    return dirs


def run_e2_sep(out_dir: Path):
    m, N, T, n_null = P["m"], P["N"], P["n_trials"], P["n_null"]
    lam = P["lambda_sep"]
    dirs = direction_bank(m)
    rows = []
    t0 = time.time()
    for seed in range(P["base_seed"], P["base_seed"] + P["n_seeds"]):
        rng = np.random.default_rng(seed)
        h0 = gaussian_h(rng, n_null, N, m, 0.0, axis_dense(m)[0])
        thr = quantile_thresh(ind_stat(h0), P["alpha"])
        for name, w in dirs:
            h1 = gaussian_h(rng, T, N, m, lam, w)
            pow_i = power(ind_stat(h1), thr)
            pred = predictors(w)
            rows.append({"seed": seed, "name": name, "ind": pow_i, "C": geom_C(w), **pred})
            print(f"[e2sep] seed={seed} {name:22s} C={pred['one_minus_l4']:.3f} l1={pred['l1']:.3f} ind={pow_i:.3f}", flush=True)
    y = np.array([r["ind"] for r in rows])
    corr = {k: pearson([r[k] for r in rows], y) for k in ("one_minus_l4", "one_minus_linf2", "participation_ratio", "l1")}
    ranked = sorted(corr.items(), key=lambda kv: abs(kv[1]) if kv[1] == kv[1] else -1, reverse=True)

    def mean_name(prefix):
        return float(np.mean([r["ind"] for r in rows if r["name"] == prefix]))

    names = sorted({r["name"] for r in rows if r["name"].startswith("twosparse")})
    sep = {n: float(np.mean([r["ind"] for r in rows if r["name"] == n])) for n in names}
    gates = {
        "E2-sep-winner": {
            "pass": bool(ranked[0][0] == "one_minus_l4"),
            "corr": corr,
            "winner": ranked[0][0],
        },
        "E2-sep-unbalanced": {
            "pass": bool(sep.get("twosparse_eps0.01", 1.0) + 0.15 < sep.get("twosparse_eps0.50", 0.0)),
            "by_eps": sep,
        },
    }
    summary = {"elapsed_sec": time.time() - t0, "corr": corr, "gates": gates, "sep": sep}
    (out_dir / "e2sep_raw.json").write_text(json.dumps(rows))
    (out_dir / "e2sep_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if torch is not None:

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

else:  # pragma: no cover
    TopKSAE = None  # type: ignore


def train_sae(h, m, d, k, steps, batch, lr, seed, device):
    torch.manual_seed(seed)
    model = TopKSAE(m, d, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(h, dtype=torch.float32, device=device)
    n = x.shape[0]
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
    return model


def decoder_cosine(model, w):
    D = model.dec.weight.detach().cpu().numpy()
    cols = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    return float(np.max(np.abs(cols.T @ unit(w))))


def diagnostic_class(powers, hit=0.5):
    L, C, I = powers["label"] >= hit, powers["cov"] >= hit, powers["ind"] >= hit
    if not L:
        return "absent"
    if not C:
        return "label_only"
    if not I:
        return "cov_not_ind"
    return "visible"


def run_e4(out_dir: Path, device: str):
    if torch is None:
        raise RuntimeError("torch required for E4")
    m = P["m"]
    ax, de = axis_dense(m)
    rows = []
    t0 = time.time()
    for seed in range(P["base_seed"], P["base_seed"] + P["n_seeds"]):
        rng = np.random.default_rng(seed)
        # shared null at a=0 (stats do not depend on w except label-with-z; use dummy w)
        h0, z0 = bernoulli_h(rng, P["n_null_e4"], P["N_detect_e4"], m, 0.0, P["p"], ax)
        thr = {
            "ind": quantile_thresh(ind_stat(h0), P["alpha"]),
            "cov": quantile_thresh(cov_stat(h0), P["alpha"]),
            "label": quantile_thresh(label_stat_z(h0, z0), P["alpha"]),
        }
        for geo, w in (("axis", ax), ("dense", de)):
            for lam in P["lambdas_e4"]:
                a = lam / np.sqrt(P["p"])
                h1, z1 = bernoulli_h(rng, P["n_trials_e4"], P["N_detect_e4"], m, a, P["p"], w)
                powers = {
                    "label": power(label_stat_z(h1, z1), thr["label"]),
                    "cov": power(cov_stat(h1), thr["cov"]),
                    "ind": power(ind_stat(h1), thr["ind"]),
                }
                dclass = diagnostic_class(powers)
                h_tr, _ = bernoulli_h(rng, 1, P["N_train"], m, a, P["p"], w)
                model = train_sae(
                    h_tr[0], m, P["dict_size"], P["k"], P["sae_steps"], P["sae_batch"], P["sae_lr"], seed, device
                )
                cos = decoder_cosine(model, w)
                rec = bool(cos >= P["cosine_threshold"])
                row = {
                    "seed": seed,
                    "geometry": geo,
                    "lambda_eff": lam,
                    "C": geom_C(w),
                    "class": dclass,
                    **{f"det_{k}": v for k, v in powers.items()},
                    "decoder_cosine": cos,
                    "recovered": rec,
                }
                rows.append(row)
                print(
                    f"[e4v2] seed={seed} {geo:5s} lam={lam:.2f} class={dclass:12s} "
                    f"L={powers['label']:.2f} C={powers['cov']:.2f} I={powers['ind']:.2f} "
                    f"cos={cos:.3f} rec={rec}",
                    flush=True,
                )
    rngb = np.random.default_rng(0)
    summary = {"elapsed_sec": time.time() - t0, "threshold": P["cosine_threshold"]}
    # λ=0.55 geometry contrast
    contrast = {}
    for geo in ("axis", "dense"):
        sub = [r for r in rows if r["geometry"] == geo and abs(r["lambda_eff"] - 0.55) < 1e-9]
        contrast[geo] = {
            "ind": boot_mean([r["det_ind"] for r in sub], rngb),
            "cov": boot_mean([r["det_cov"] for r in sub], rngb),
            "label": boot_mean([r["det_label"] for r in sub], rngb),
            "sae_recovery": boot_mean([float(r["recovered"]) for r in sub], rngb),
            "sae_cosine": boot_mean([r["decoder_cosine"] for r in sub], rngb),
        }
    summary["lambda055"] = contrast
    by_class = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(float(r["recovered"]))
    summary["recovery_by_class"] = {k: float(np.mean(v)) for k, v in by_class.items()}
    axis_i = contrast["axis"]["ind"]["mean"]
    dense_i = contrast["dense"]["ind"]["mean"]
    axis_s = contrast["axis"]["sae_recovery"]["mean"]
    dense_s = contrast["dense"]["sae_recovery"]["mean"]
    order = ["absent", "label_only", "cov_not_ind", "visible"]
    rates = [summary["recovery_by_class"].get(k) for k in order if k in summary["recovery_by_class"]]
    monotone = all(rates[i] <= rates[i + 1] + 1e-9 for i in range(len(rates) - 1)) if len(rates) >= 2 else False
    summary["gates"] = {
        "E4-geom-ind": {"pass": bool(axis_i <= 0.15 and dense_i >= 0.80), "axis_ind": axis_i, "dense_ind": dense_i},
        "E4-geom-SAE": {"pass": bool(abs(axis_s - dense_s) <= 0.25), "axis_rec": axis_s, "dense_rec": dense_s},
        "E4-class": {"pass": bool(monotone), "recovery_by_class": summary["recovery_by_class"]},
    }
    (out_dir / "e4v2_raw.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "e4v2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_v2")
    ap.add_argument("--stage", choices=["e2sep", "e4", "all"], default="all")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol_v2.json").write_text(json.dumps(P, indent=2))
    if args.stage in ("e2sep", "all"):
        run_e2_sep(out)
    if args.stage in ("e4", "all"):
        device = args.device
        if torch is None or (device.startswith("cuda") and not torch.cuda.is_available()):
            device = "cpu"
        run_e4(out, device)


if __name__ == "__main__":
    main()
