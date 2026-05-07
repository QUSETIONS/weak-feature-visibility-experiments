"""Real-activation bridge experiments for the quartic detection law.

This script deliberately avoids TransformerLens for the first sanity pass and
uses Hugging Face GPT-2 hidden states directly. It supports:

- C1: collect GPT-2 hidden activations from a streaming text dataset;
- C3: inject controlled sparse features into those real activations and compare
  labeled mean-shift detection against unlabeled diagonal-null covariance
  detection.

The injected feature uses h = h_real + amp * z * w with z ~ Bernoulli(p). For
unlabeled covariance detection we center within each sample and test the
off-diagonal covariance component in the known target direction. This is a
target-known statistical detector, not yet SAE training; its job is to validate
the bridge from matched Gaussian data to real activation backgrounds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dataset_text_iter(dataset_name: str, dataset_config: str | None, split: str, max_docs: int, text_file: str | None = None):
    from datasets import load_dataset

    if text_file:
        count = 0
        with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                text = line.strip()
                if len(text) > 20:
                    yield text
                    count += 1
                    if count >= max_docs:
                        break
        return

    try:
        if dataset_config:
            ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as exc:
        print(f"[warn] failed to stream {dataset_name}: {exc}")
        print("[warn] falling back to wikitext-2-raw-v1")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)

    count = 0
    for row in ds:
        text = row.get("text") or row.get("content") or row.get("raw_content") or ""
        if isinstance(text, str) and len(text.strip()) > 20:
            yield text
            count += 1
            if count >= max_docs:
                break


@torch.no_grad()
def collect_activations(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.to(device)
    model.eval()

    chunks: list[np.ndarray] = []
    total_tokens = 0
    t0 = time.time()
    batch_texts: list[str] = []

    def flush_batch(texts: list[str]) -> int:
        if not texts:
            return 0
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.context,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # HF hidden_states[0] is the embedding stream. hidden_states[layer] is
        # the residual stream before block `layer`; hidden_states[layer + 1]
        # is after that block. Pretrained SAEs are hook-specific, so this must
        # match the SAE hook point.
        hs_index = args.layer if args.hook_position == "resid_pre" else args.layer + 1
        acts = out.hidden_states[hs_index].detach().float().cpu()
        mask = attention_mask.detach().cpu().bool()
        if args.skip_first_token and mask.shape[1] > 0:
            mask[:, 0] = False
        flat = acts[mask].numpy().astype(np.float16)
        remaining = args.tokens - sum(c.shape[0] for c in chunks)
        if remaining > 0:
            chunks.append(flat[:remaining])
        return flat.shape[0]

    for text in dataset_text_iter(args.dataset_name, args.dataset_config, args.split, args.max_docs, args.text_file):
        batch_texts.append(text)
        if len(batch_texts) >= args.batch_size:
            total_tokens += flush_batch(batch_texts)
            batch_texts = []
            if sum(c.shape[0] for c in chunks) >= args.tokens:
                break
    if sum(c.shape[0] for c in chunks) < args.tokens:
        total_tokens += flush_batch(batch_texts)

    acts = np.concatenate(chunks, axis=0)[: args.tokens]
    path = outdir / f"gpt2_layer{args.layer}_{args.hook_position}_{acts.shape[0]}tok.npy"
    np.save(path, acts)

    acts32 = acts.astype(np.float32)
    summary = {
        "mode": "collect",
        "model_name": args.model_name,
        "layer": args.layer,
        "hook_position": args.hook_position,
        "skip_first_token": bool(args.skip_first_token),
        "tokens": int(acts.shape[0]),
        "d_model": int(acts.shape[1]),
        "activation_path": str(path),
        "activation_mean": float(acts32.mean()),
        "activation_std": float(acts32.std()),
        "activation_l2_mean": float(np.linalg.norm(acts32, axis=1).mean()),
        "seconds": time.time() - t0,
        "device": str(device),
    }
    with (outdir / "activation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def make_direction(kind: str, d: int, rng: np.random.Generator, pc_basis: np.ndarray | None = None, k: int = 8) -> np.ndarray:
    if kind == "axis":
        w = np.zeros(d, dtype=np.float64)
        w[0] = 1.0
    elif kind == "ksparse":
        w = np.zeros(d, dtype=np.float64)
        idx = rng.choice(d, size=min(k, d), replace=False)
        w[idx] = rng.normal(size=idx.size)
    elif kind == "dense":
        w = rng.normal(size=d)
    elif kind == "pc1":
        if pc_basis is None:
            raise ValueError("pc_basis required for pc1")
        w = pc_basis[0].copy()
    elif kind == "pc_orth":
        if pc_basis is None:
            raise ValueError("pc_basis required for pc_orth")
        w = rng.normal(size=d)
        for j in range(min(8, pc_basis.shape[0])):
            w = w - np.dot(w, pc_basis[j]) * pc_basis[j]
    else:
        raise ValueError(kind)
    return w / (np.linalg.norm(w) + 1.0e-12)


def top_pcs(x: np.ndarray, n_pcs: int = 8) -> np.ndarray:
    xc = x.astype(np.float64) - x.mean(axis=0, keepdims=True)
    # Random subset keeps sanity runs fast.
    if xc.shape[0] > 20000:
        rng = np.random.default_rng(0)
        xc = xc[rng.choice(xc.shape[0], size=20000, replace=False)]
    _, _, vh = np.linalg.svd(xc, full_matrices=False)
    return vh[:n_pcs]


def offdiag_projection_stat(x: np.ndarray, baseline_diag: np.ndarray, baseline_quad: float, w: np.ndarray) -> float:
    """Project off-diagonal covariance onto ww^T without forming d x d cov.

    For centered x, <Cov, ww^T - diag(w^2)> =
    mean((xw)^2) - sum_j w_j^2 Var(x_j).
    """
    xc = x - x.mean(axis=0, keepdims=True)
    var = np.mean(xc * xc, axis=0)
    quad = float(np.mean((xc @ w) ** 2) - np.sum((w * w) * var))
    return quad - baseline_quad


def labeled_stat(x: np.ndarray, z: np.ndarray, w: np.ndarray) -> float:
    pos = z > 0.5
    neg = ~pos
    if pos.sum() < 2 or neg.sum() < 2:
        return 0.0
    diff = x[pos].mean(axis=0) - x[neg].mean(axis=0)
    return float(np.dot(diff, w))


def labeled_stat_1d(y: np.ndarray, z: np.ndarray) -> float:
    pos = z > 0.5
    neg = ~pos
    if pos.sum() < 2 or neg.sum() < 2:
        return 0.0
    return float(y[pos].mean() - y[neg].mean())


def run_injection(args: argparse.Namespace) -> dict[str, object]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    x = np.load(args.activation_path).astype(np.float32)
    if args.max_tokens_for_injection and x.shape[0] > args.max_tokens_for_injection:
        x = x[rng.choice(x.shape[0], size=args.max_tokens_for_injection, replace=False)]
    x = x - x.mean(axis=0, keepdims=True)
    d = x.shape[1]

    baseline_subset = x[rng.choice(x.shape[0], size=min(args.baseline_tokens, x.shape[0]), replace=False)]
    b = baseline_subset - baseline_subset.mean(axis=0, keepdims=True)
    baseline_diag = np.mean(b * b, axis=0)
    pcs = top_pcs(baseline_subset, n_pcs=8)

    rows: list[dict[str, object]] = []
    kinds = args.direction_kinds.split(",")
    amps = [float(v) for v in args.amps.split(",")]
    n_values = [int(v) for v in args.n_values.split(",")]

    for kind in kinds:
        w = make_direction(kind, d, rng, pcs, k=args.k_sparse)
        geom = float(1.0 - np.sum(w**4))
        w2 = w * w
        w3 = w2 * w
        w4_sum = float(np.sum(w2 * w2))
        y_all = (x @ w).astype(np.float64)
        q_all = (x * x) @ w2
        r_all = x @ w3
        baseline_idx = rng.choice(x.shape[0], size=min(args.baseline_tokens, x.shape[0]), replace=False)
        baseline_quad = float(np.mean(y_all[baseline_idx] ** 2 - q_all[baseline_idx]))
        for amp in amps:
            for n in n_values:
                null_labeled = []
                null_unlabeled = []
                alt_labeled = []
                alt_unlabeled = []
                for _ in range(args.trials):
                    idx = rng.choice(x.shape[0], size=n, replace=False)
                    yb = y_all[idx]
                    qb = q_all[idx]
                    rb = r_all[idx]

                    z_null = rng.binomial(1, args.p, size=n).astype(np.float64)
                    null_labeled.append(labeled_stat_1d(yb, z_null))
                    null_unlabeled.append(float(np.mean(yb * yb - qb) - baseline_quad))

                    z = rng.binomial(1, args.p, size=n).astype(np.float64)
                    # Center the injected feature so covariance, not a global mean shift,
                    # is the unlabeled signal; labels still reveal the class difference.
                    zi = z - args.p
                    ya = yb + amp * zi
                    qa = qb + 2.0 * amp * zi * rb + (amp * amp) * (zi * zi) * w4_sum
                    alt_labeled.append(labeled_stat_1d(ya, z))
                    alt_unlabeled.append(float(np.mean(ya * ya - qa) - baseline_quad))

                labeled_thr = float(np.quantile(null_labeled, 1.0 - args.alpha))
                unlabeled_thr = float(np.quantile(null_unlabeled, 1.0 - args.alpha))
                labeled_power = float(np.mean(np.array(alt_labeled) > labeled_thr))
                unlabeled_power = float(np.mean(np.array(alt_unlabeled) > unlabeled_thr)) if geom > 1e-12 else float(np.mean(np.array(alt_unlabeled) > unlabeled_thr))
                lambda_eff = amp * math.sqrt(args.p * (1.0 - args.p))
                rows.append(
                    {
                        "direction": kind,
                        "amp": amp,
                        "p": args.p,
                        "lambda_eff": lambda_eff,
                        "n": n,
                        "geometry": geom,
                        "c_proxy": geom / 4.0,
                        "evidence_proxy": n * (geom / 4.0) * lambda_eff**4,
                        "labeled_threshold": labeled_thr,
                        "unlabeled_threshold": unlabeled_thr,
                        "labeled_power": labeled_power,
                        "unlabeled_power": unlabeled_power,
                    }
                )

    write_csv(outdir / "injection_probe.csv", rows)
    summary = {
        "mode": "inject",
        "activation_path": args.activation_path,
        "tokens_used": int(x.shape[0]),
        "d_model": int(d),
        "trials": args.trials,
        "alpha": args.alpha,
        "p": args.p,
        "rows": len(rows),
        "mean_labeled_power": float(np.mean([r["labeled_power"] for r in rows])),
        "mean_unlabeled_power": float(np.mean([r["unlabeled_power"] for r in rows])),
        "output_csv": str(outdir / "injection_probe.csv"),
    }
    with (outdir / "injection_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--outdir", default="results/real_activation")
    p_collect.add_argument("--model-name", default="gpt2")
    p_collect.add_argument("--dataset-name", default="wikitext")
    p_collect.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p_collect.add_argument("--text-file", default=None)
    p_collect.add_argument("--split", default="train")
    p_collect.add_argument("--layer", type=int, default=6)
    p_collect.add_argument("--hook-position", choices=["resid_pre", "resid_post"], default="resid_post")
    p_collect.add_argument("--skip-first-token", action=argparse.BooleanOptionalAction, default=True)
    p_collect.add_argument("--tokens", type=int, default=10000)
    p_collect.add_argument("--context", type=int, default=128)
    p_collect.add_argument("--batch-size", type=int, default=8)
    p_collect.add_argument("--max-docs", type=int, default=20000)
    p_collect.add_argument("--device", default="cuda:0")

    p_inj = sub.add_parser("inject")
    p_inj.add_argument("--outdir", default="results/real_activation")
    p_inj.add_argument("--activation-path", required=True)
    p_inj.add_argument("--seed", type=int, default=20260502)
    p_inj.add_argument("--max-tokens-for-injection", type=int, default=30000)
    p_inj.add_argument("--baseline-tokens", type=int, default=10000)
    p_inj.add_argument("--direction-kinds", default="axis,ksparse,dense,pc1,pc_orth")
    p_inj.add_argument("--k-sparse", type=int, default=8)
    p_inj.add_argument("--amps", default="0.25,0.5,1.0")
    p_inj.add_argument("--n-values", default="256,1024,4096")
    p_inj.add_argument("--p", type=float, default=0.05)
    p_inj.add_argument("--trials", type=int, default=80)
    p_inj.add_argument("--alpha", type=float, default=0.05)

    args = parser.parse_args()
    if args.cmd == "collect":
        collect_activations(args)
    elif args.cmd == "inject":
        run_injection(args)


if __name__ == "__main__":
    main()
