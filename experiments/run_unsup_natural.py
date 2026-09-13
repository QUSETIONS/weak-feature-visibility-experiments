#!/usr/bin/env python3
"""Unsupervised recovery of natural directions (no synthetic injection).

Frozen:
  train TopK on the whitened 32-d residual slice
  token features = difference-of-means of the 40 most frequent tokens
  PC features = e_j in the same slice
  random unit directions as a chance baseline
  recovery = max decoder cosine
This is specified-direction scoring of directions that exist in the data,
not named-concept discovery.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gpt2_bridge import (
    GPT2Model,
    GPT2TokenizerFast,
    apply_whiten,
    collect_residuals,
    decoder_cosine,
    fit_whiten,
    load_texts,
    train_sae,
    unit,
)

P = {
    "layer": 6,
    "seq_len": 128,
    "n_train_seq": 4000,
    "batch": 16,
    "sae_m": 32,
    "dict_size": 128,
    "k": 8,
    "sae_steps": 4000,
    "sae_batch": 512,
    "sae_lr": 1e-3,
    "seed": 20260824,
    "n_natural": 40,
    "n_random": 40,
}


def atom_C(D):
    cols = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    return 1.0 - np.sum(cols**4, axis=0)


def token_dirs(H, tokens, k):
    cnt = Counter(tokens.tolist())
    rows = []
    for tid, _ in cnt.most_common(k):
        mask = tokens == tid
        if mask.sum() < 20 or (~mask).sum() < 20:
            continue
        w = unit(H[mask].mean(0) - H[~mask].mean(0))
        rows.append(
            {
                "kind": "token",
                "token_id": int(tid),
                "n": int(mask.sum()),
                "C": float(1.0 - np.sum(w**4)),
                "w": w,
            }
        )
    return rows


def main():
    out = Path("experiments/results_unsup")
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    local_snap = "/mnt/e2_runs/work/e2_real_experiment_code_v5_scheme_aligned_20260619_0118/.hf_cache/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    print("[unsup] loading", local_snap, flush=True)
    tok = GPT2TokenizerFast.from_pretrained(local_snap)
    model = GPT2Model.from_pretrained(local_snap).to(device)
    texts = load_texts()
    H, tokens = collect_residuals(
        model, tok, texts, P["seq_len"], P["n_train_seq"], P["batch"], P["layer"], device
    )
    mu, W = fit_whiten(H)
    Z = apply_whiten(H, mu, W)
    m = int(P["sae_m"])
    Zs = Z[:, :m]
    print("[unsup] Z", Z.shape, "slice", Zs.shape, flush=True)
    sae, loss = train_sae(
        Zs, m, P["dict_size"], P["k"], P["sae_steps"], P["sae_batch"], P["sae_lr"], P["seed"], device
    )
    D = sae.dec.weight.detach().cpu().numpy()
    c_atoms = atom_C(D)

    nat = token_dirs(Zs, tokens, P["n_natural"])
    for r in nat:
        r["sae_cosine"] = decoder_cosine(sae, r["w"])
        del r["w"]

    pcs = []
    for j in range(m):
        e = np.zeros(m)
        e[j] = 1.0
        pcs.append(
            {
                "kind": "pc",
                "pc": int(j),
                "C": float(1.0 - np.sum(e**4)),
                "sae_cosine": decoder_cosine(sae, e),
            }
        )

    rng = np.random.default_rng(P["seed"])
    rnd = []
    for i in range(P["n_random"]):
        w = unit(rng.standard_normal(m))
        rnd.append(
            {
                "kind": "random",
                "i": int(i),
                "C": float(1.0 - np.sum(w**4)),
                "sae_cosine": decoder_cosine(sae, w),
            }
        )

    def corr_C_cos(rows):
        if len(rows) < 4:
            return float("nan")
        c = np.array([r["C"] for r in rows])
        s = np.array([r["sae_cosine"] for r in rows])
        if np.std(c) < 1e-12 or np.std(s) < 1e-12:
            return float("nan")
        return float(np.corrcoef(c, s)[0, 1])

    summary = {
        "elapsed_sec": time.time() - t0,
        "n_tokens": int(Zs.shape[0]),
        "sae_m": m,
        "dict_size": P["dict_size"],
        "train_loss": loss,
        "atom_C_min": float(c_atoms.min()),
        "atom_C_median": float(np.median(c_atoms)),
        "atom_frac_C_lt_0.2": float(np.mean(c_atoms < 0.2)),
        "token_n": len(nat),
        "token_C_min": min((r["C"] for r in nat), default=None),
        "token_C_median": float(np.median([r["C"] for r in nat])) if nat else None,
        "token_sae_mean": float(np.mean([r["sae_cosine"] for r in nat])) if nat else None,
        "token_corr_C_cos": corr_C_cos(nat),
        "pc_sae": [r["sae_cosine"] for r in pcs],
        "pc0_sae": pcs[0]["sae_cosine"],
        "pc_last_sae": pcs[-1]["sae_cosine"],
        "random_sae_mean": float(np.mean([r["sae_cosine"] for r in rnd])),
        "random_sae_p90": float(np.quantile([r["sae_cosine"] for r in rnd], 0.9)),
        "protocol": P,
    }
    (out / "unsup_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "unsup_tokens.json").write_text(json.dumps(nat))
    (out / "unsup_pcs.json").write_text(json.dumps(pcs))
    print(json.dumps({k: summary[k] for k in summary if k != "pc_sae" and k != "protocol"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
