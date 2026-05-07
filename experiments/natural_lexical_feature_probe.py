"""Natural lexical feature probe for the quartic detection story.

This C9 replacement avoids SAE-defined proxy labels. It uses external token
properties from Wikitext (digits, punctuation, capitalization, common words,
etc.) as naturally occurring feature events. A direction is learned from
labeled train activations, then detector-class evidence is evaluated on a
held-out split. Nulls are built by learning directions from permuted train
labels and evaluating them on the same held-out activations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import string
import time
from pathlib import Path

import numpy as np
import torch


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def iter_text(path: Path, max_docs: int):
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if len(text) > 20:
                yield text
                count += 1
                if count >= max_docs:
                    break


def token_events(token: str) -> dict[str, bool]:
    stripped = token.strip()
    lower = stripped.lower()
    punct = set(string.punctuation)
    return {
        "digit": any(ch.isdigit() for ch in token),
        "punct": bool(stripped) and all(ch in punct for ch in stripped),
        "comma_or_period": stripped in {",", "."},
        "quote": any(ch in token for ch in ['"', "'", "`"]),
        "capitalized": bool(stripped) and stripped[0].isupper(),
        "common_the": lower == "the",
        "common_of": lower == "of",
        "common_and": lower == "and",
        "common_to": lower == "to",
        "common_in": lower == "in",
        "hyphen_piece": "@" in token or "-" in token,
    }


@torch.no_grad()
def collect(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    chunks: list[np.ndarray] = []
    event_chunks: dict[str, list[np.ndarray]] = {}
    total = 0
    batch: list[str] = []
    t0 = time.time()

    def flush(texts: list[str]) -> None:
        nonlocal total
        if not texts or total >= args.tokens:
            return
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.context)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hs_index = args.layer if args.hook_position == "resid_pre" else args.layer + 1
        acts = out.hidden_states[hs_index].detach().float().cpu().numpy()
        ids = input_ids.detach().cpu().numpy()
        mask = attention_mask.detach().cpu().numpy().astype(bool)
        if args.skip_first_token and mask.shape[1] > 0:
            mask[:, 0] = False

        flat_acts = acts[mask].astype(np.float32)
        flat_ids = ids[mask]
        keep = min(args.tokens - total, flat_acts.shape[0])
        if keep <= 0:
            return
        flat_acts = flat_acts[:keep]
        flat_ids = flat_ids[:keep]
        chunks.append(flat_acts)
        total += keep

        labels: dict[str, list[bool]] = {}
        for tok_id in flat_ids:
            tok = tokenizer.decode([int(tok_id)])
            ev = token_events(tok)
            for k, v in ev.items():
                labels.setdefault(k, []).append(v)
        for k, values in labels.items():
            event_chunks.setdefault(k, []).append(np.asarray(values, dtype=bool))

    for text in iter_text(Path(args.text_file), args.max_docs):
        batch.append(text)
        if len(batch) >= args.batch_size:
            flush(batch)
            batch = []
            if total >= args.tokens:
                break
    flush(batch)

    x = np.concatenate(chunks, axis=0)
    events = {k: np.concatenate(v, axis=0) for k, v in event_chunks.items()}
    meta = {
        "tokens": int(x.shape[0]),
        "d_model": int(x.shape[1]),
        "seconds": time.time() - t0,
        "device": str(device),
        "model_name": args.model_name,
        "text_file": args.text_file,
        "layer": args.layer,
        "hook_position": args.hook_position,
    }
    return x, events, meta


def standardize(x_train: np.ndarray, x_test: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = np.maximum(x_train.std(axis=0, keepdims=True), eps)
    return ((x_train - mean) / std).astype(np.float32), ((x_test - mean) / std).astype(np.float32)


def direction_from_labels(x: np.ndarray, z: np.ndarray) -> np.ndarray | None:
    pos = z
    neg = ~z
    if pos.sum() < 2 or neg.sum() < 2:
        return None
    w = x[pos].mean(axis=0) - x[neg].mean(axis=0)
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        return None
    return (w / norm).astype(np.float32)


def full_cov_stat(x: np.ndarray, w: np.ndarray) -> float:
    xc = x - x.mean(axis=0, keepdims=True)
    return float(np.mean((xc @ w) ** 2))


def offdiag_stat(x: np.ndarray, w: np.ndarray) -> float:
    xc = x - x.mean(axis=0, keepdims=True)
    var = np.mean(xc * xc, axis=0)
    return float(np.mean((xc @ w) ** 2) - np.sum((w * w) * var))


def label_stat(x: np.ndarray, z: np.ndarray, w: np.ndarray) -> float:
    pos = z
    neg = ~z
    return float((x[pos] @ w).mean() - (x[neg] @ w).mean())


def zscore(value: float, null: np.ndarray) -> tuple[float, float, float]:
    mu = float(np.mean(null))
    sd = float(np.std(null, ddof=1))
    return mu, sd, float((value - mu) / max(sd, 1e-12))


def analyze(args: argparse.Namespace, x: np.ndarray, events: dict[str, np.ndarray], meta: dict[str, object]) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    n = x.shape[0]
    idx = rng.permutation(n)
    train_n = int(n * args.train_frac)
    train_idx = idx[:train_n]
    test_idx = idx[train_n:]
    x_train_raw = x[train_idx]
    x_test_raw = x[test_idx]
    x_train, x_test = standardize(x_train_raw, x_test_raw, args.eps)

    rows: list[dict[str, object]] = []
    for name, z_all in events.items():
        z_train = z_all[train_idx]
        z_test = z_all[test_idx]
        p = float(z_all.mean())
        if z_train.sum() < args.min_train_pos or z_test.sum() < args.min_test_pos:
            continue
        if (~z_train).sum() < args.min_train_neg or (~z_test).sum() < args.min_test_neg:
            continue

        w = direction_from_labels(x_train, z_train)
        if w is None:
            continue
        geom = float(1.0 - np.sum(w**4))
        label = label_stat(x_test, z_test, w)
        full = full_cov_stat(x_test, w)
        off = offdiag_stat(x_test, w)

        null_label = []
        null_full = []
        null_off = []
        for _ in range(args.null_reps):
            zp = rng.permutation(z_train)
            wp = direction_from_labels(x_train, zp)
            if wp is None:
                continue
            null_label.append(label_stat(x_test, z_test, wp))
            null_full.append(full_cov_stat(x_test, wp))
            null_off.append(offdiag_stat(x_test, wp))
        null_label = np.asarray(null_label, dtype=float)
        null_full = np.asarray(null_full, dtype=float)
        null_off = np.asarray(null_off, dtype=float)
        if null_label.size < 5:
            continue
        _, _, label_z = zscore(label, null_label)
        _, _, full_z = zscore(full, null_full)
        _, _, off_z = zscore(off, null_off)
        lambda_eff = abs(label) * math.sqrt(max(p * (1.0 - p), 1e-12))
        evidence_proxy = x_test.shape[0] * (geom / 4.0) * (lambda_eff**4)
        rows.append(
            {
                "event": name,
                "tokens": int(n),
                "train_tokens": int(x_train.shape[0]),
                "test_tokens": int(x_test.shape[0]),
                "firing_rate": p,
                "train_pos": int(z_train.sum()),
                "test_pos": int(z_test.sum()),
                "geometry": geom,
                "lambda_eff": lambda_eff,
                "evidence_proxy": evidence_proxy,
                "log10_evidence_proxy": float(np.log10(max(evidence_proxy, 1e-30))),
                "label_stat": label,
                "label_z": label_z,
                "full_cov_stat": full,
                "full_cov_z": full_z,
                "offdiag_cov_stat": off,
                "offdiag_cov_z": off_z,
            }
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "natural_lexical_feature_probe.csv", rows)
    if not rows:
        raise RuntimeError("No lexical events passed the filters.")

    label_abs = np.abs(np.array([r["label_z"] for r in rows], dtype=float))
    full_abs = np.abs(np.array([r["full_cov_z"] for r in rows], dtype=float))
    off_abs = np.abs(np.array([r["offdiag_cov_z"] for r in rows], dtype=float))
    ev = np.array([r["log10_evidence_proxy"] for r in rows], dtype=float)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 3:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    summary = {
        "mode": "natural_lexical_feature_probe",
        **meta,
        "events_evaluated": int(len(rows)),
        "median_abs_label_z": float(np.median(label_abs)),
        "median_abs_full_cov_z": float(np.median(full_abs)),
        "median_abs_offdiag_cov_z": float(np.median(off_abs)),
        "frac_label_z_gt_5": float(np.mean(label_abs > 5.0)),
        "frac_full_cov_z_gt_5": float(np.mean(full_abs > 5.0)),
        "frac_offdiag_z_gt_5": float(np.mean(off_abs > 5.0)),
        "corr_log_evidence_abs_label_z": corr(ev, label_abs),
        "corr_log_evidence_abs_full_cov_z": corr(ev, full_abs),
        "corr_log_evidence_abs_offdiag_z": corr(ev, off_abs),
        "rows": rows,
        "output_csv": str(outdir / "natural_lexical_feature_probe.csv"),
        "output_figure": str(outdir / "natural_lexical_feature_probe.pdf"),
    }
    with (outdir / "natural_lexical_feature_probe_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
        ax[0].scatter(label_abs, off_abs, s=24)
        for r in rows:
            ax[0].annotate(r["event"], (abs(r["label_z"]), abs(r["offdiag_cov_z"])), fontsize=6)
        ax[0].set_xlabel("labeled |z|")
        ax[0].set_ylabel("feature-independent |z|")
        ax[1].scatter(ev, off_abs, s=24)
        ax[1].set_xlabel(r"$\log_{10}(NC_i\lambda_i^4)$ proxy")
        ax[1].set_ylabel("feature-independent |z|")
        fig.tight_layout()
        fig.savefig(outdir / "natural_lexical_feature_probe.png", dpi=220)
        fig.savefig(outdir / "natural_lexical_feature_probe.pdf")
        plt.close(fig)
    except Exception as exc:
        summary["plot_error"] = repr(exc)
        with (outdir / "natural_lexical_feature_probe_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="assets/gpt2")
    parser.add_argument("--text-file", default="assets/wikitext_train.txt")
    parser.add_argument("--outdir", default="results/real_activation/c9_natural_lexical_feature_probe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--hook-position", choices=["resid_pre", "resid_post"], default="resid_pre")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-docs", type=int, default=2000)
    parser.add_argument("--tokens", type=int, default=50000)
    parser.add_argument("--skip-first-token", action="store_true", default=True)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--null-reps", type=int, default=200)
    parser.add_argument("--min-train-pos", type=int, default=50)
    parser.add_argument("--min-test-pos", type=int, default=50)
    parser.add_argument("--min-train-neg", type=int, default=500)
    parser.add_argument("--min-test-neg", type=int, default=500)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260503)
    args = parser.parse_args()
    x, events, meta = collect(args)
    analyze(args, x, events, meta)


if __name__ == "__main__":
    main()
