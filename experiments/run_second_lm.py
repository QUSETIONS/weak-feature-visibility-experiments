#!/usr/bin/env python3
"""Second-model residual bridge (Pythia-70m), same protocol as GPT-2.

Frozen: PCA-whiten train split, inject in leading 32 PCs along e_0, e_{31}, dense,
known-direction label/cov/ind, TopK SAE cosine >= 0.80.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gpt2_bridge import (
    TopKSAE,
    apply_whiten,
    decoder_cosine,
    detect_powers,
    fit_whiten,
    inject,
    load_texts,
    train_sae,
    unit,
)

P = {
    "layer": 3,
    "seq_len": 128,
    "n_train_seq": 4000,
    "n_test_seq": 1500,
    "batch": 16,
    "p": 0.05,
    "lambdas": (0.35, 0.55, 0.80),
    "n_seeds": 4,
    "base_seed": 20260824,
    "alpha": 0.05,
    "N_detect": 1024,
    "n_trials": 200,
    "n_null": 400,
    "sae_m": 32,
    "dict_size": 64,
    "k": 4,
    "N_train_sae": 24000,
    "sae_steps": 3000,
    "sae_batch": 512,
    "sae_lr": 1e-3,
    "cosine_threshold": 0.80,
}

CANDIDATE_DIRS = (
    Path("experiments/weights/pythia-70m"),
    Path("/mnt/liuzelin/weak_feature_visibility/experiments/weights/pythia-70m"),
)


@torch.no_grad()
def collect_residuals(model, tok, texts, seq_len, n_seq, batch, layer, device):
    ids = []
    for t in texts:
        enc = tok(t, add_special_tokens=False)["input_ids"]
        if len(enc) < 8:
            continue
        ids.extend(enc)
        if len(ids) >= n_seq * seq_len + seq_len:
            break
    need = n_seq * seq_len
    if len(ids) < need:
        ids = (ids * (need // max(len(ids), 1) + 2))[:need]
    arr = np.asarray(ids[:need], dtype=np.int64).reshape(n_seq, seq_len)
    hs = []
    model.eval()
    for i in range(0, n_seq, batch):
        sl = torch.tensor(arr[i : i + batch], device=device)
        out = model(sl, output_hidden_states=True)
        h = out.hidden_states[layer].detach().float().cpu().numpy()
        hs.append(h.reshape(-1, h.shape[-1]))
    return np.concatenate(hs, axis=0)


def find_model():
    for d in CANDIDATE_DIRS:
        if (d / "config.json").is_file():
            return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/results_pythia"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--text-file", type=Path, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model_dir = args.model_dir or find_model()
    if model_dir is None or not (Path(model_dir) / "config.json").is_file():
        report = {"loaded": False, "reason": "pythia-70m weights missing"}
        (args.out / "pythia_summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return
    from transformers import AutoModel, AutoTokenizer

    print("[pythia] from", model_dir, flush=True)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir)).to(device)
    texts = load_texts(args.text_file)
    print("[pythia] n_texts", len(texts), flush=True)
    n_all = P["n_train_seq"] + P["n_test_seq"]
    H_all = collect_residuals(model, tok, texts, P["seq_len"], n_all, P["batch"], P["layer"], device)
    n_tr = P["n_train_seq"] * P["seq_len"]
    H_tr, H_te = H_all[:n_tr], H_all[n_tr:]
    mu, W = fit_whiten(H_tr)
    Z_tr = apply_whiten(H_tr, mu, W)
    Z_te = apply_whiten(H_te, mu, W)
    m_full = Z_tr.shape[1]
    m = int(P["sae_m"])
    Z_tr_s, Z_te_s = Z_tr[:, :m], Z_te[:, :m]
    ax = np.zeros(m)
    ax[0] = 1.0
    last = np.zeros(m)
    last[-1] = 1.0
    de = unit(np.ones(m))
    geos = (("axis", ax), ("last", last), ("dense", de))
    print("[pythia] m_full", m_full, "sae_m", m, "train", Z_tr.shape, flush=True)

    rows = []
    p = P["p"]
    for seed in range(P["base_seed"], P["base_seed"] + P["n_seeds"]):
        rng = np.random.default_rng(seed)
        for geo, w in geos:
            for lam in P["lambdas"]:
                a = lam / np.sqrt(p)
                powers = detect_powers(
                    Z_te_s, a, p, w, rng, P["N_detect"], P["n_trials"], P["n_null"], P["alpha"]
                )
                idx = rng.integers(0, Z_tr_s.shape[0], size=P["N_train_sae"])
                h_tr, _ = inject(Z_tr_s[idx], a, p, w, rng)
                sae, loss = train_sae(
                    h_tr, m, P["dict_size"], P["k"], P["sae_steps"], P["sae_batch"], P["sae_lr"], seed, device
                )
                cos = decoder_cosine(sae, w)
                rec = bool(cos >= P["cosine_threshold"])
                row = {
                    "seed": seed,
                    "geometry": geo,
                    "lambda_eff": lam,
                    "C": float(1.0 - np.sum(w**4)),
                    **{f"det_{k}": v for k, v in powers.items()},
                    "decoder_cosine": cos,
                    "recovered": rec,
                    "train_loss": loss,
                }
                rows.append(row)
                print(
                    f"[pythia] seed={seed} {geo:5s} lam={lam:.2f} "
                    f"lab={powers['label']:.2f} cov={powers['cov']:.2f} ind={powers['ind']:.2f} "
                    f"cos={cos:.3f} rec={rec}",
                    flush=True,
                )
                (args.out / "pythia_raw.json").write_text(json.dumps(rows))

    def mean(xs):
        xs = list(xs)
        return float(np.mean(xs)), (
            [float(np.quantile(xs, 0.025)), float(np.quantile(xs, 0.975))]
            if len(xs) > 1
            else [float(xs[0]), float(xs[0])]
        )

    by = {}
    for geo in ("axis", "last", "dense"):
        by[geo] = {}
        for lam in P["lambdas"]:
            sub = [r for r in rows if r["geometry"] == geo and r["lambda_eff"] == lam]
            by[geo][str(lam)] = {
                k: {"mean": mean([r[k] for r in sub])[0], "ci95": mean([r[k] for r in sub])[1]}
                for k in ("det_label", "det_cov", "det_ind", "decoder_cosine")
            }
            recs = [float(r["recovered"]) for r in sub]
            by[geo][str(lam)]["sae_recovery"] = {"mean": float(np.mean(recs)), "ci95": mean(recs)[1]}
    summary = {
        "loaded": True,
        "model": "EleutherAI/pythia-70m",
        "elapsed_sec": time.time() - t0,
        "m_full": m_full,
        "sae_m": m,
        "by": by,
        "protocol": P,
    }
    (args.out / "pythia_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("elapsed_sec", "m_full", "sae_m")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
