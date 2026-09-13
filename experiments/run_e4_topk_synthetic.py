#!/usr/bin/env python3
"""E4 synthetic TopK SAE: diagnostic class vs recovery, plus geometry intervention.

Faithful to ICLR_experiment_design.md §7. Frozen before looking at results:
  recovered iff decoder cosine >= 0.80
  (C16's three-threshold rule is not recoverable here; cosine is the
  specified decoder-cosine component, frozen a priori.)

Data: centered Bernoulli injection h = eps + a (z-p) w, matching the GPT-2
bridge rather than the Gaussian unlabeled model of E1.
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

PROTOCOL = {
    "m": 16,
    "dict_size": 32,
    "k": 2,
    "p": 0.05,
    "a": 0.7,  # lambda_eff = a * sqrt(p) ≈ 0.156 unless overridden
    "lambdas_eff": (0.10, 0.20, 0.35, 0.55),
    "N_train": 20000,
    "N_test": 4000,
    "steps": 4000,
    "batch": 512,
    "lr": 1e-3,
    "cosine_threshold": 0.80,
    "n_seeds": 3,
    "base_seed": 20260824,
    "alpha": 0.05,
    "n_detect_trials": 400,
    "N_detect": 2048,
    "n_null": 800,
}


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def axis_dense(m):
    ax = np.zeros(m)
    ax[0] = 1.0
    de = np.ones(m) / np.sqrt(m)
    return unit(ax), unit(de)


def geometry_C(w):
    w = unit(w)
    return float(1.0 - np.sum(w**4))


def inject(rng, N, m, a, p, w):
    eps = rng.standard_normal((N, m))
    z = rng.binomial(1, p, size=(N, 1)).astype(np.float64)
    return eps + a * (z - p) * w.reshape(1, m), z.ravel()


def sample_cov(h):
    c = h - h.mean(axis=0, keepdims=True)
    return (c.T @ c) / h.shape[0]


def detect_powers(rng, N, m, a, p, w, n_trials, n_null, alpha):
    # null
    null_ind, null_cov, null_lab = [], [], []
    for _ in range(n_null):
        h0, z0 = inject(rng, N, m, 0.0, p, w)
        S = sample_cov(h0)
        od = S.copy()
        np.fill_diagonal(od, 0.0)
        null_ind.append(np.linalg.norm(od))
        null_cov.append(np.linalg.norm(S - np.eye(m)))
        pos = h0[z0 > 0.5].mean(axis=0) if (z0 > 0.5).any() else np.zeros(m)
        neg = h0[z0 <= 0.5].mean(axis=0)
        null_lab.append(np.sum((pos - neg) ** 2))
    t_ind = float(np.quantile(null_ind, 1 - alpha))
    t_cov = float(np.quantile(null_cov, 1 - alpha))
    t_lab = float(np.quantile(null_lab, 1 - alpha))
    ind = cov = lab = 0
    for _ in range(n_trials):
        h, z = inject(rng, N, m, a, p, w)
        S = sample_cov(h)
        od = S.copy()
        np.fill_diagonal(od, 0.0)
        ind += np.linalg.norm(od) > t_ind
        cov += np.linalg.norm(S - np.eye(m)) > t_cov
        pos = h[z > 0.5].mean(axis=0) if (z > 0.5).any() else np.zeros(m)
        neg = h[z <= 0.5].mean(axis=0)
        lab += np.sum((pos - neg) ** 2) > t_lab
    n = float(n_trials)
    return {"label": lab / n, "cov": cov / n, "ind": ind / n}


def diagnostic_class(powers, alpha=0.05, hit=0.5):
    # hit if estimated power >= 0.5 (clearly above alpha)
    L = powers["label"] >= hit
    C = powers["cov"] >= hit
    I = powers["ind"] >= hit
    if not L:
        return "absent"
    if L and not C:
        return "label_only"
    if C and not I:
        return "cov_not_ind"
    return "visible"


class TopKSAE(nn.Module):
    def __init__(self, m, d, k):
        super().__init__()
        self.enc = nn.Linear(m, d, bias=True)
        self.dec = nn.Linear(d, m, bias=False)
        self.k = k
        nn.init.orthogonal_(self.dec.weight)

    def forward(self, x):
        pre = self.enc(x)
        # relu then topk
        pre = F.relu(pre)
        vals, idx = torch.topk(pre, self.k, dim=1)
        code = torch.zeros_like(pre)
        code.scatter_(1, idx, vals)
        rec = self.dec(code)
        return rec, code


def train_sae(h, m, d, k, steps, batch, lr, seed, device):
    torch.manual_seed(seed)
    model = TopKSAE(m, d, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(h, dtype=torch.float32, device=device)
    n = x.shape[0]
    model.train()
    for t in range(steps):
        b = torch.randint(0, n, (batch,), device=device)
        xb = x[b]
        rec, _ = model(xb)
        loss = F.mse_loss(rec, xb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # decoder column normalize
        with torch.no_grad():
            w = model.dec.weight
            model.dec.weight.copy_(w / (w.norm(dim=0, keepdim=True) + 1e-8))
    return model, float(loss.item())


def decoder_cosine(model, w):
    D = model.dec.weight.detach().cpu().numpy()  # m x dict
    w = unit(w)
    cols = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    return float(np.max(np.abs(cols.T @ w)))


def run(out_dir: Path, device: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    P = PROTOCOL
    axis, dense = axis_dense(P["m"])
    rows = []
    t0 = time.time()
    for seed in [P["base_seed"] + i for i in range(P["n_seeds"])]:
        rng = np.random.default_rng(seed)
        for geo_name, w in (("axis", axis), ("dense", dense)):
            for lam in P["lambdas_eff"]:
                a = lam / np.sqrt(P["p"])
                powers = detect_powers(
                    rng, P["N_detect"], P["m"], a, P["p"], w, P["n_detect_trials"], P["n_null"], P["alpha"]
                )
                dclass = diagnostic_class(powers)
                h_tr, _ = inject(rng, P["N_train"], P["m"], a, P["p"], w)
                model, loss = train_sae(
                    h_tr, P["m"], P["dict_size"], P["k"], P["steps"], P["batch"], P["lr"], seed, device
                )
                cos = decoder_cosine(model, w)
                rec = cos >= P["cosine_threshold"]
                row = {
                    "seed": seed,
                    "geometry": geo_name,
                    "lambda_eff": lam,
                    "a": a,
                    "C": geometry_C(w),
                    "class": dclass,
                    **{f"det_{k}": v for k, v in powers.items()},
                    "decoder_cosine": cos,
                    "recovered": rec,
                    "train_loss": loss,
                }
                rows.append(row)
                print(
                    f"[e4] seed={seed} {geo_name} lam={lam:.2f} class={dclass} "
                    f"ind={powers['ind']:.2f} cos={cos:.3f} rec={rec}",
                    flush=True,
                )
    # rotation intervention at lambda_eff=0.35, axis -> dense by changing w only
    for seed in [P["base_seed"] + i for i in range(P["n_seeds"])]:
        rng = np.random.default_rng(seed + 100)
        lam = 0.35
        a = lam / np.sqrt(P["p"])
        for geo_name, w in (("axis", axis), ("dense", dense)):
            h_tr, _ = inject(rng, P["N_train"], P["m"], a, P["p"], w)
            model, loss = train_sae(
                h_tr, P["m"], P["dict_size"], P["k"], P["steps"], P["batch"], P["lr"], seed + 100, device
            )
            cos = decoder_cosine(model, w)
            rows.append(
                {
                    "seed": seed,
                    "geometry": geo_name,
                    "lambda_eff": lam,
                    "subset": "intervention",
                    "C": geometry_C(w),
                    "decoder_cosine": cos,
                    "recovered": cos >= P["cosine_threshold"],
                }
            )
            print(f"[e4-int] seed={seed} {geo_name} cos={cos:.3f}", flush=True)

    summary = {"elapsed_sec": time.time() - t0, "threshold": P["cosine_threshold"]}
    by_class = {}
    for r in rows:
        if "class" not in r:
            continue
        by_class.setdefault(r["class"], []).append(float(r["recovered"]))
    summary["recovery_by_class"] = {k: float(np.mean(v)) for k, v in by_class.items()}
    for geo in ("axis", "dense"):
        sub = [r for r in rows if r.get("geometry") == geo and r.get("lambda_eff") == 0.35 and "class" in r]
        summary[f"recovery_{geo}_lam035"] = float(np.mean([r["recovered"] for r in sub])) if sub else None
        summary[f"cos_{geo}_lam035"] = float(np.mean([r["decoder_cosine"] for r in sub])) if sub else None
    (out_dir / "e4_raw.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "e4_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_e4")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    run(Path(args.out), device)


if __name__ == "__main__":
    main()
