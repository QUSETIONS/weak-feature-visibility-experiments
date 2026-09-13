#!/usr/bin/env python3
"""Known-direction scores of published Bloom GPT-2-small layer-6 SAE atoms.

No training, no injection, no bake-off. Encoder labels come from the published
SAE; cov/ind are the same known-direction statistics as the GPT-2 bridge.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gpt2_bridge import (
    GPT2Model,
    GPT2TokenizerFast,
    apply_whiten,
    collect_residuals,
    cov_stat,
    fit_whiten,
    ind_stat,
    load_texts,
    unit,
)


def py(x):
    if isinstance(x, dict):
        return {str(k): py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [py(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if x is None:
        return None
    return x


def atom_C(w):
    w = unit(w)
    return float(1.0 - np.sum(w**4)), w


def label_stat_bin(h, fire, w):
    fire = np.asarray(fire, dtype=bool)
    proj = h @ w
    if fire.any() and (~fire).any():
        return float((proj[fire].mean() - proj[~fire].mean()) ** 2)
    return 0.0


def load_pack(path: Path, atom_ids):
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path)
        ids = np.asarray(z["atoms"], dtype=np.int64)
        if atom_ids is not None:
            want = np.asarray(atom_ids, dtype=np.int64)
            loc = {int(i): k for k, i in enumerate(ids)}
            miss = [int(i) for i in want if int(i) not in loc]
            if miss:
                raise SystemExit("npz missing atoms " + str(miss))
            idx = np.array([loc[int(i)] for i in want], dtype=np.int64)
            ids = want
        else:
            idx = np.arange(len(ids))
        return {
            "atoms": ids,
            "W_dec": np.asarray(z["W_dec"][idx], dtype=np.float64),
            "W_enc": np.asarray(z["W_enc"][:, idx], dtype=np.float64),
            "b_enc": np.asarray(z["b_enc"][idx], dtype=np.float64),
            "b_dec": np.asarray(z["b_dec"], dtype=np.float64),
            "source": str(path),
        }
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        W_dec = np.asarray(f.get_tensor("W_dec"), dtype=np.float64)
        W_enc = np.asarray(f.get_tensor("W_enc"), dtype=np.float64)
        b_dec = np.asarray(f.get_tensor("b_dec"), dtype=np.float64)
        b_enc = np.asarray(f.get_tensor("b_enc"), dtype=np.float64)
    if atom_ids is None:
        raise SystemExit("full safetensors requires --atom-ids or --atoms")
    ids = np.asarray(atom_ids, dtype=np.int64)
    return {
        "atoms": ids,
        "W_dec": W_dec[ids],
        "W_enc": W_enc[:, ids],
        "b_enc": b_enc[ids],
        "b_dec": b_dec,
        "source": str(path),
    }


def encode(H, W_enc, b_enc, b_dec, subtract_bdec=True):
    x = H - b_dec.reshape(1, -1) if subtract_bdec else H
    pre = x @ W_enc + b_enc.reshape(1, -1)
    return np.maximum(pre, 0.0)


def rand_units(rng, n, m):
    u = rng.normal(size=(n, m))
    u /= np.clip(np.linalg.norm(u, axis=1, keepdims=True), 1e-12, None)
    return u


def batch_cov_ind(H, U, H2=None):
    """U is (R, m) unit rows. Returns cov, ind as (R,)."""
    P = H @ U.T
    cov = np.mean(P * P, axis=0)
    if H2 is None:
        H2 = H * H
    quad = H2 @ (U * U).T
    ind = np.mean(P * P - quad, axis=0)
    return cov, ind


def proj_stats(proj, fire=None, quad=None):
    cov = float(np.mean(proj * proj))
    ind = float(np.mean(proj * proj - quad)) if quad is not None else None
    lab = None
    if fire is not None:
        fire = np.asarray(fire, dtype=bool)
        if fire.any() and (~fire).any():
            lab = float((proj[fire].mean() - proj[~fire].mean()) ** 2)
        else:
            lab = 0.0
    return cov, ind, lab


def percentile_ge(null, val):
    null = np.asarray(null, dtype=np.float64)
    return float(np.mean(null >= val))


def subsample_power_proj(proj, N, n_trials, null_q, rng, fire=None, quad=None, kind="cov"):
    n = proj.shape[0]
    hits = 0
    for _ in range(n_trials):
        idx = rng.integers(0, n, size=N)
        if kind == "label":
            stat = proj_stats(proj[idx], fire=fire[idx])[2]
        elif kind == "cov":
            stat = float(np.mean(proj[idx] ** 2))
        else:
            stat = float(np.mean(proj[idx] ** 2 - quad[idx]))
        hits += stat > null_q
    return hits / float(n_trials)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/results_public_atoms"))
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--atoms", type=Path, default=None)
    ap.add_argument("--atom-ids", default=None)
    ap.add_argument("--explanations", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-train-seq", type=int, default=4000)
    ap.add_argument("--n-test-seq", type=int, default=1500)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-null", type=int, default=400)
    ap.add_argument("--n-rand", type=int, default=400)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--N-big", type=int, default=8192)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260824)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    atom_ids = None
    if args.atom_ids:
        atom_ids = [int(x) for x in args.atom_ids.split(",") if x.strip() != ""]
    elif args.atoms is not None:
        raw = json.loads(Path(args.atoms).read_text())
        if isinstance(raw, dict) and "atoms" in raw:
            raw = raw["atoms"]
        if isinstance(raw, dict):
            atom_ids = [int(k) for k in raw.keys()]
        else:
            atom_ids = [int(x) for x in raw]

    expl = {}
    if args.explanations and Path(args.explanations).is_file():
        blob = json.loads(Path(args.explanations).read_text())
        expl = blob.get("atoms", blob)

    pack = load_pack(args.weights, atom_ids)
    ids = [int(i) for i in pack["atoms"]]
    print("[atoms] n", len(ids), "from", pack["source"], "ids", ids, flush=True)

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    local_snap = "/mnt/e2_runs/work/e2_real_experiment_code_v5_scheme_aligned_20260619_0118/.hf_cache/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    model_id = local_snap if os.path.isdir(local_snap) else "gpt2"
    print("[atoms] gpt2", model_id, "device", device, flush=True)
    tok = GPT2TokenizerFast.from_pretrained(model_id)
    model = GPT2Model.from_pretrained(model_id).to(device)
    texts = load_texts()
    n_all = args.n_train_seq + args.n_test_seq
    H_all, _ = collect_residuals(
        model, tok, texts, args.seq_len, n_all, args.batch, args.layer, device
    )
    n_tr = args.n_train_seq * args.seq_len
    H_tr, H_te = H_all[:n_tr], H_all[n_tr:]
    mu, W = fit_whiten(H_tr)
    Z_te = apply_whiten(H_te, mu, W)[:, :32]
    H_te = np.ascontiguousarray(H_te, dtype=np.float64)
    Z_te = np.ascontiguousarray(Z_te, dtype=np.float64)
    print("[atoms] H_te", H_te.shape, "Z_te", Z_te.shape, flush=True)

    b_dec = pack["b_dec"]
    Z = encode(H_te, pack["W_enc"], pack["b_enc"], b_dec, subtract_bdec=True)
    rng = np.random.default_rng(args.seed)
    H2 = H_te * H_te
    Z2 = Z_te * Z_te
    U768 = rand_units(rng, args.n_rand, H_te.shape[1])
    U32 = rand_units(rng, args.n_rand, 32)
    print("[atoms] batch random dirs", args.n_rand, flush=True)
    rand_cov_768, rand_ind_768 = batch_cov_ind(H_te, U768, H2)
    rand_cov_32, rand_ind_32 = batch_cov_ind(Z_te, U32, Z2)
    q_cov_768 = float(np.quantile(rand_cov_768, 1 - args.alpha))
    q_ind_768 = float(np.quantile(rand_ind_768, 1 - args.alpha))
    q_cov_32 = float(np.quantile(rand_cov_32, 1 - args.alpha))
    q_ind_32 = float(np.quantile(rand_ind_32, 1 - args.alpha))

    n_te = H_te.shape[0]
    Un = rand_units(rng, args.n_null, H_te.shape[1])
    Ub = rand_units(rng, args.n_null, H_te.shape[1])
    null_cov_N = np.empty(args.n_null)
    null_ind_N = np.empty(args.n_null)
    null_cov_B = np.empty(args.n_null)
    null_ind_B = np.empty(args.n_null)
    for t in range(args.n_null):
        idx = rng.integers(0, n_te, size=args.N)
        c, i = batch_cov_ind(H_te[idx], Un[t : t + 1], H2[idx])
        null_cov_N[t], null_ind_N[t] = c[0], i[0]
        idxb = rng.integers(0, n_te, size=args.N_big)
        c, i = batch_cov_ind(H_te[idxb], Ub[t : t + 1], H2[idxb])
        null_cov_B[t], null_ind_B[t] = c[0], i[0]
    q_cov_N = float(np.quantile(null_cov_N, 1 - args.alpha))
    q_ind_N = float(np.quantile(null_ind_N, 1 - args.alpha))
    q_cov_B = float(np.quantile(null_cov_B, 1 - args.alpha))
    q_ind_B = float(np.quantile(null_ind_B, 1 - args.alpha))
    print("[atoms] null quantiles ready", flush=True)

    Wp = W[:, :32]
    rows = []
    for k, atom in enumerate(ids):
        C, w = atom_C(pack["W_dec"][k])
        w32 = unit(Wp.T @ w)
        C32 = float(1.0 - np.sum(w32**4))
        z = Z[:, k]
        fire = z > 0
        n_pos = int(fire.sum())
        rate = n_pos / float(len(fire))
        if n_pos >= 5 and (len(fire) - n_pos) >= 5:
            dom = unit(H_te[fire].mean(0) - H_te[~fire].mean(0))
            lab_cos = float(np.abs(dom @ w))
        else:
            lab_cos = 0.0
        proj = H_te @ w
        quad = H2 @ (w * w)
        proj32 = Z_te @ w32
        quad32 = Z2 @ (w32 * w32)
        lab_obs = proj_stats(proj, fire=fire)[2]
        perm = np.empty(args.n_null)
        fire_i = fire.copy()
        for t in range(args.n_null):
            rng.shuffle(fire_i)
            perm[t] = proj_stats(proj, fire=fire_i)[2]
        lab_p = percentile_ge(perm, lab_obs)
        cov_obs = float(np.mean(proj * proj))
        ind_obs = float(np.mean(proj * proj - quad))
        cov32 = float(np.mean(proj32 * proj32))
        ind32 = float(np.mean(proj32 * proj32 - quad32))
        null_lab_N_atom = np.empty(args.n_null)
        null_lab_B_atom = np.empty(args.n_null)
        for t in range(args.n_null):
            idx = rng.integers(0, n_te, size=args.N)
            f = fire[idx].copy()
            rng.shuffle(f)
            null_lab_N_atom[t] = proj_stats(proj[idx], fire=f)[2]
            idxb = rng.integers(0, n_te, size=args.N_big)
            fb = fire[idxb].copy()
            rng.shuffle(fb)
            null_lab_B_atom[t] = proj_stats(proj[idxb], fire=fb)[2]
        q_lab_N = float(np.quantile(null_lab_N_atom, 1 - args.alpha))
        q_lab_B = float(np.quantile(null_lab_B_atom, 1 - args.alpha))
        pow_lab_N = subsample_power_proj(proj, args.N, args.n_trials, q_lab_N, rng, fire=fire, kind="label")
        pow_cov_N = subsample_power_proj(proj, args.N, args.n_trials, q_cov_N, rng, kind="cov")
        pow_ind_N = subsample_power_proj(proj, args.N, args.n_trials, q_ind_N, rng, quad=quad, kind="ind")
        pow_lab_B = subsample_power_proj(proj, args.N_big, args.n_trials, q_lab_B, rng, fire=fire, kind="label")
        pow_cov_B = subsample_power_proj(proj, args.N_big, args.n_trials, q_cov_B, rng, kind="cov")
        pow_ind_B = subsample_power_proj(proj, args.N_big, args.n_trials, q_ind_B, rng, quad=quad, kind="ind")
        st = {"cov": cov_obs, "ind": ind_obs}
        st32 = {"cov": cov32, "ind": ind32}

        meta = expl.get(str(atom), expl.get(atom, {}))
        row = {
            "atom": int(atom),
            "C_resid": C,
            "C_pca32": C32,
            "argmax_dim": int(np.argmax(np.abs(w))),
            "max_abs": float(np.max(np.abs(w))),
            "n_pos": n_pos,
            "fire_rate": rate,
            "label_cosine": lab_cos,
            "label_stat": lab_obs,
            "label_perm_p": lab_p,
            "label_beats_perm": bool(lab_p <= args.alpha),
            "cov": st["cov"],
            "ind": st["ind"],
            "cov_rand_p": percentile_ge(rand_cov_768, st["cov"]),
            "ind_rand_p": percentile_ge(rand_ind_768, st["ind"]),
            "cov_beats_rand": bool(st["cov"] > q_cov_768),
            "ind_beats_rand": bool(st["ind"] > q_ind_768),
            "pca32_cov": st32["cov"],
            "pca32_ind": st32["ind"],
            "pca32_cov_beats_rand": bool(st32["cov"] > q_cov_32),
            "pca32_ind_beats_rand": bool(st32["ind"] > q_ind_32),
            "pca32_cov_rand_p": percentile_ge(rand_cov_32, st32["cov"]),
            "pca32_ind_rand_p": percentile_ge(rand_ind_32, st32["ind"]),
            "power_N1024": {"label": pow_lab_N, "cov": pow_cov_N, "ind": pow_ind_N},
            "power_N8192": {"label": pow_lab_B, "cov": pow_cov_B, "ind": pow_ind_B},
            "description": meta.get("description"),
            "autointerp": meta.get("autointerp") or meta.get("expl"),
        }
        rows.append(row)
        print(
            f"[atoms] {atom:5d} C={C:.3f} C32={C32:.3f} dim={row['argmax_dim']} "
            f"rate={rate:.4f} lab_cos={lab_cos:.3f} perm_p={lab_p:.3f} "
            f"cov_p={row['cov_rand_p']:.3f} ind_p={row['ind_rand_p']:.3f} "
            f"N1k lab/cov/ind={pow_lab_N:.2f}/{pow_cov_N:.2f}/{pow_ind_N:.2f}",
            flush=True,
        )
        (args.out / "atoms.json").write_text(json.dumps(py(rows), indent=2))

    summary = {
        "elapsed_sec": time.time() - t0,
        "n_train_tokens": int(H_tr.shape[0]),
        "n_test_tokens": int(H_te.shape[0]),
        "encoder": "relu((h-b_dec)@W_enc + b_enc)",
        "apply_b_dec_to_input": True,
        "protocol": {
            "N": args.N,
            "N_big": args.N_big,
            "n_null": args.n_null,
            "n_rand": args.n_rand,
            "n_trials": args.n_trials,
            "alpha": args.alpha,
            "layer": args.layer,
            "no_injection": True,
        },
        "rand_cov_768_q": q_cov_768,
        "rand_ind_768_q": q_ind_768,
        "source": pack["source"],
        "repo": "jbloom/GPT2-Small-SAEs-Reformatted",
        "hook": "blocks.6.hook_resid_pre",
        "atoms": py(rows),
    }
    (args.out / "summary.json").write_text(json.dumps(py(summary), indent=2))
    print(json.dumps({k: summary[k] for k in ("elapsed_sec", "n_test_tokens", "encoder")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
