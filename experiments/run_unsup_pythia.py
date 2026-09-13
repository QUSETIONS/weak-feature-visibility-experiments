#!/usr/bin/env python3
"""Unsupervised token/PC recovery on Pythia-70m (no injection)."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gpt2_bridge import apply_whiten, decoder_cosine, fit_whiten, load_texts, train_sae, unit
from run_second_lm import CANDIDATE_DIRS, collect_residuals, find_model

P = {
    "layer": 3,
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


def token_dirs(H, tokens, k):
    cnt = Counter(tokens.tolist() if hasattr(tokens, "tolist") else tokens)
    rows = []
    tok = np.asarray(tokens)
    for tid, _ in cnt.most_common(k):
        mask = tok == tid
        if mask.sum() < 20 or (~mask).sum() < 20:
            continue
        w = unit(H[mask].mean(0) - H[~mask].mean(0))
        rows.append({"kind": "token", "token_id": int(tid), "n": int(mask.sum()), "C": float(1.0 - np.sum(w**4)), "w": w})
    return rows


def corr_C_cos(rows):
    if len(rows) < 4:
        return float("nan")
    c = np.array([r["C"] for r in rows])
    s = np.array([r["sae_cosine"] for r in rows])
    if np.std(c) < 1e-12 or np.std(s) < 1e-12:
        return float("nan")
    return float(np.corrcoef(c, s)[0, 1])


def main():
    out = Path("experiments/results_unsup_pythia")
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model_dir = find_model()
    if model_dir is None:
        (out / "unsup_summary.json").write_text(json.dumps({"loaded": False}))
        print("missing pythia", flush=True)
        return
    from transformers import AutoModel, AutoTokenizer

    print("[unsup-py] from", model_dir, flush=True)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir)).to(device)
    texts = load_texts()
    H = collect_residuals(model, tok, texts, P["seq_len"], P["n_train_seq"], P["batch"], P["layer"], device)
    # token ids for difference-of-means: reuse the same packing as collect
    ids = []
    for t in texts:
        enc = tok(t, add_special_tokens=False)["input_ids"]
        if len(enc) < 8:
            continue
        ids.extend(enc)
        if len(ids) >= P["n_train_seq"] * P["seq_len"]:
            break
    need = P["n_train_seq"] * P["seq_len"]
    if len(ids) < need:
        ids = (ids * (need // max(len(ids), 1) + 2))[:need]
    tokens = np.asarray(ids[:need], dtype=np.int64)
    mu, W = fit_whiten(H)
    Z = apply_whiten(H, mu, W)
    m = int(P["sae_m"])
    Zs = Z[:, :m]
    print("[unsup-py] Z", Z.shape, "slice", Zs.shape, flush=True)
    sae, loss = train_sae(
        Zs, m, P["dict_size"], P["k"], P["sae_steps"], P["sae_batch"], P["sae_lr"], P["seed"], device
    )
    nat = token_dirs(Zs, tokens, P["n_natural"])
    for r in nat:
        r["sae_cosine"] = decoder_cosine(sae, r.pop("w"))
    pcs = []
    for j in range(m):
        e = np.zeros(m)
        e[j] = 1.0
        pcs.append({"kind": "pc", "pc": int(j), "C": float(1.0 - np.sum(e**4)), "sae_cosine": decoder_cosine(sae, e)})
    rng = np.random.default_rng(P["seed"])
    rnd = []
    for i in range(P["n_random"]):
        w = unit(rng.standard_normal(m))
        rnd.append({"kind": "random", "i": int(i), "C": float(1.0 - np.sum(w**4)), "sae_cosine": decoder_cosine(sae, w)})
    summary = {
        "loaded": True,
        "model": "EleutherAI/pythia-70m",
        "elapsed_sec": time.time() - t0,
        "n_tokens": int(Zs.shape[0]),
        "train_loss": loss,
        "token_n": len(nat),
        "token_C_min": min((r["C"] for r in nat), default=None),
        "token_C_median": float(np.median([r["C"] for r in nat])) if nat else None,
        "token_sae_mean": float(np.mean([r["sae_cosine"] for r in nat])) if nat else None,
        "token_corr_C_cos": corr_C_cos(nat),
        "pc0_sae": pcs[0]["sae_cosine"],
        "pc_last_sae": pcs[-1]["sae_cosine"],
        "random_sae_mean": float(np.mean([r["sae_cosine"] for r in rnd])),
        "random_sae_p90": float(np.quantile([r["sae_cosine"] for r in rnd], 0.9)),
        "protocol": P,
    }
    (out / "unsup_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "unsup_tokens.json").write_text(json.dumps(nat))
    print(json.dumps({k: summary[k] for k in summary if k != "protocol"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
