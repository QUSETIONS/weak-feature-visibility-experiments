#!/usr/bin/env python3
"""Uninjected GPT-2 SAE: decoder geometry in the whitened frame.

Train TopK on WikiText residual (no synthetic injection). Report 1-||w||_4^4
of decoder atoms in the same PCA-whitened coordinates used for the axis hole.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gpt2_bridge import (
    GPT2Model,
    GPT2TokenizerFast,
    TopKSAE,
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
}


def atom_C(D):
    cols = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    c = 1.0 - np.sum(cols**4, axis=0)
    return c


def main():
    out = Path("experiments/results_gpt2")
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    local_snap = "/mnt/e2_runs/work/e2_real_experiment_code_v5_scheme_aligned_20260619_0118/.hf_cache/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    print("[nat] loading", flush=True)
    tok = GPT2TokenizerFast.from_pretrained(local_snap)
    model = GPT2Model.from_pretrained(local_snap).to(device)
    texts = load_texts()
    H, tokens = collect_residuals(
        model, tok, texts, P["seq_len"], P["n_train_seq"], P["batch"], P["layer"], device
    )
    mu, W = fit_whiten(H)
    Z = apply_whiten(H, mu, W)
    Zs = Z[:, : P["sae_m"]]
    print("[nat] Z", Z.shape, "slice", Zs.shape, flush=True)
    sae, loss = train_sae(
        Zs, P["sae_m"], P["dict_size"], P["k"], P["sae_steps"], P["sae_batch"], P["sae_lr"], P["seed"], device
    )
    D = sae.dec.weight.detach().cpu().numpy()
    c = atom_C(D)
    ax = np.zeros(P["sae_m"])
    ax[0] = 1.0
    summary = {
        "elapsed_sec": time.time() - t0,
        "n_tokens": int(Zs.shape[0]),
        "sae_m": P["sae_m"],
        "dict_size": P["dict_size"],
        "train_loss": loss,
        "C_min": float(c.min()),
        "C_median": float(np.median(c)),
        "C_mean": float(c.mean()),
        "C_p10": float(np.quantile(c, 0.10)),
        "frac_C_lt_0.1": float(np.mean(c < 0.1)),
        "frac_C_lt_0.5": float(np.mean(c < 0.5)),
        "max_cosine_to_e1": decoder_cosine(sae, ax),
        "C_atoms": [float(x) for x in c],
        "protocol": P,
    }
    (out / "gpt2_natural_sae.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "C_atoms"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
