#!/usr/bin/env python3
"""GPT-2 small layer-6 residual-stream bridge.

Frozen protocol:
  model gpt2, hidden_states[6] (after block 6)
  PCA whitening fit on the train split only
  axis = e_0 in the whitened frame; last = e_{m-1}; dense = m^{-1/2} 1
  SAE recovery: decoder cosine >= 0.80
  natural directions: difference-in-means of the 40 most frequent tokens
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
for cache in (
    "/mnt/e1_runs/hf_cache",
    "/mnt/e2_runs/work/e2_real_experiment_code_v5_scheme_aligned_20260619_0118/.hf_cache",
    "/mnt/liuzelin/.cache/huggingface",
):
    hub = os.path.join(cache, "hub", "models--gpt2")
    if os.path.isdir(hub):
        os.environ["HF_HOME"] = cache
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache, "hub")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        break
else:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/mnt/liuzelin/.cache/huggingface")

from transformers import GPT2Model, GPT2TokenizerFast

P = {
    "layer": 6,
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
    "n_natural": 40,
}


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


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def load_texts(text_file=None):
    if text_file is not None:
        path = Path(text_file)
        texts = [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]
        if len(texts) < 200:
            raise RuntimeError(f"real text file has only {len(texts)} nonempty lines; need at least 200")
        print("[gpt2] loaded explicit real text file", path, "n", len(texts), flush=True)
        return texts
    caches = (
        "/mnt/e1_runs/hf_cache",
        "/mnt/e2_runs/work/e2_real_experiment_code_v5_scheme_aligned_20260619_0118/.hf_cache",
    )
    specs = (
        ("Salesforce/wikitext", "wikitext-2-raw-v1"),
        ("wikitext", "wikitext-2-raw-v1"),
        ("wikitext", "wikitext-103-raw-v1"),
    )
    for cache in caches:
        os.environ["HF_HOME"] = cache
        os.environ["HF_DATASETS_CACHE"] = os.path.join(cache, "datasets")
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            from datasets import load_dataset
        except Exception as exc:
            print("[gpt2] datasets import failed:", exc, flush=True)
            break
        for name, config in specs:
            try:
                ds = load_dataset(name, config, split="train")
                texts = [t for t in ds["text"] if t and str(t).strip()]
                if len(texts) >= 200:
                    print("[gpt2] loaded", name, config, "n", len(texts), "from", cache, flush=True)
                    return texts
            except Exception as exc:
                print("[gpt2] skip", name, config, ":", exc, flush=True)
    wiki_dir = "/mnt/e1_runs/work/e1_experiment1_E1_v10_20260616_004802/e1_experiment1_E1_v10_final_runtime_safe/runs_20260620_0008_fastfinish/20260620_000913_e1_llm_fastfinish_seed0_20260620_0008_fastfinish/raw/llm_texts/real_wikitext"
    texts = []
    if os.path.isdir(wiki_dir):
        for root, _, files in os.walk(wiki_dir):
            for fn in files:
                p = os.path.join(root, fn)
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        chunk = f.read()
                    texts.extend([ln for ln in chunk.splitlines() if ln.strip()])
                except OSError:
                    continue
    if len(texts) >= 200:
        print("[gpt2] loaded raw wiki files", len(texts), flush=True)
        return texts
    raise RuntimeError("no approved real WikiText source is available; synthetic text fallback is forbidden")


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
    H = np.concatenate(hs, axis=0)
    return H, arr.reshape(-1)


def fit_whiten(H, eps=1e-5):
    mu = H.mean(axis=0)
    C = np.cov((H - mu).T)
    eig, V = np.linalg.eigh(C)
    order = np.argsort(eig)[::-1]
    eig, V = eig[order], V[:, order]
    eig = np.clip(eig, eps, None)
    W = V * (1.0 / np.sqrt(eig))
    return mu, W


def apply_whiten(H, mu, W):
    return (H - mu) @ W


def inject(base, a, p, w, rng):
    N, m = base.shape
    z = rng.binomial(1, p, size=(N, 1)).astype(np.float64)
    return base + a * (z - p) * w.reshape(1, m), z.ravel()


def label_stat(h, z, w):
    z = z.reshape(-1)
    proj = h @ w
    pos = proj[z > 0.5].mean() if np.any(z > 0.5) else 0.0
    neg = proj[z <= 0.5].mean()
    return float((pos - neg) ** 2)


def cov_stat(h, w):
    p = h @ w
    return float(np.mean(p * p))


def ind_stat(h, w):
    p = h @ w
    quad_diag = (h * h) @ (w * w)
    return float(np.mean(p * p - quad_diag))


def detect_powers(base, a, p, w, rng, N, n_trials, n_null, alpha):
    def draw(n):
        idx = rng.integers(0, base.shape[0], size=n)
        return base[idx]

    null_lab, null_cov, null_ind = [], [], []
    for _ in range(n_null):
        h0, z0 = inject(draw(N), 0.0, p, w, rng)
        null_lab.append(label_stat(h0, z0, w))
        null_cov.append(cov_stat(h0, w))
        null_ind.append(ind_stat(h0, w))
    t_lab = float(np.quantile(null_lab, 1 - alpha))
    t_cov = float(np.quantile(null_cov, 1 - alpha))
    t_ind = float(np.quantile(null_ind, 1 - alpha))
    lab = cov = ind = 0
    for _ in range(n_trials):
        h1, z1 = inject(draw(N), a, p, w, rng)
        lab += label_stat(h1, z1, w) > t_lab
        cov += cov_stat(h1, w) > t_cov
        ind += ind_stat(h1, w) > t_ind
    n = float(n_trials)
    return {"label": lab / n, "cov": cov / n, "ind": ind / n}


def train_sae(h, m, d, k, steps, batch, lr, seed, device):
    torch.manual_seed(seed)
    model = TopKSAE(m, d, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(h, dtype=torch.float32, device=device)
    n = x.shape[0]
    last = 0.0
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
        last = float(loss.item())
    return model, last


def decoder_cosine(model, w):
    D = model.dec.weight.detach().cpu().numpy()
    cols = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    return float(np.max(np.abs(cols.T @ unit(w))))


def natural_dirs(H, tokens, k):
    cnt = Counter(tokens.tolist())
    top = [tid for tid, _ in cnt.most_common(k)]
    rows = []
    for tid in top:
        mask = tokens == tid
        if mask.sum() < 20 or (~mask).sum() < 20:
            continue
        w = unit(H[mask].mean(0) - H[~mask].mean(0))
        c = float(1.0 - np.sum(w**4))
        rows.append({"token_id": int(tid), "n": int(mask.sum()), "one_minus_l4": c})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/results_gpt2"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text-file", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print("[gpt2] loading model", flush=True)
    local_snap = "/mnt/e2_runs/work/e2_real_experiment_code_v5_scheme_aligned_20260619_0118/.hf_cache/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    model_id = str(args.model_dir) if args.model_dir is not None else (local_snap if os.path.isdir(local_snap) else "gpt2")
    print("[gpt2] from", model_id, flush=True)
    tok = GPT2TokenizerFast.from_pretrained(model_id)
    model = GPT2Model.from_pretrained(model_id).to(device)
    texts = load_texts(args.text_file)
    print("[gpt2] n_texts", len(texts), flush=True)
    n_all = P["n_train_seq"] + P["n_test_seq"]
    H_all, tok_all = collect_residuals(
        model, tok, texts, P["seq_len"], n_all, P["batch"], P["layer"], device
    )
    n_tr = P["n_train_seq"] * P["seq_len"]
    H_tr, H_te = H_all[:n_tr], H_all[n_tr:]
    tok_tr = tok_all[:n_tr]
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
    nat = natural_dirs(Z_tr, tok_tr, P["n_natural"])
    nat_min = min((r["one_minus_l4"] for r in nat), default=None)
    print("[gpt2] m_full", m_full, "sae_m", m, "train", Z_tr.shape, "natural_min_C", nat_min, flush=True)

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
                    f"[gpt2] seed={seed} {geo:5s} lam={lam:.2f} "
                    f"lab={powers['label']:.2f} cov={powers['cov']:.2f} ind={powers['ind']:.2f} "
                    f"cos={cos:.3f} rec={rec}",
                    flush=True,
                )
                (args.out / "gpt2_raw.json").write_text(json.dumps(rows))

    def mean(xs):
        xs = list(xs)
        return float(np.mean(xs)), [float(np.quantile(xs, 0.025)), float(np.quantile(xs, 0.975))] if len(xs) > 1 else [float(xs[0]), float(xs[0])]

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
        "elapsed_sec": time.time() - t0,
        "m_full": m_full,
        "sae_m": m,
        "natural_n": len(nat),
        "natural_min_C": nat_min,
        "natural_median_C": float(np.median([r["one_minus_l4"] for r in nat])) if nat else None,
        "by": by,
        "protocol": P,
    }
    (args.out / "gpt2_natural.json").write_text(json.dumps(nat))
    (args.out / "gpt2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("elapsed_sec", "m_full", "sae_m", "natural_min_C", "natural_median_C")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
