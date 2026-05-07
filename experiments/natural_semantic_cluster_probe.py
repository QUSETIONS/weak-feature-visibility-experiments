"""Semi-open semantic-cluster probe for the quartic detection story.

This C15 probe avoids hand-defining linguistic categories. It builds a
text-only distributional semantic bank from Wikitext: word types are represented
by their local lexical co-occurrence profiles, reduced with SVD, and clustered.
GPT-2 BPE tokens inherit the cluster of their overlapping word type, then the
same held-out detector-class audit used by C9--C14 is applied.

The labels are not learned from GPT-2 activations, so this is a semi-open
semantic-feature stress test rather than a circular activation-discovery probe.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from natural_pos_tagger_feature_probe import (
    direction_from_labels,
    full_cov_stat,
    iter_text,
    label_stat,
    offdiag_stat,
    standardize,
    write_csv,
    zscore,
)


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "this",
    "have",
    "has",
    "had",
    "were",
    "was",
    "are",
    "his",
    "her",
    "its",
    "their",
    "they",
    "them",
    "you",
    "your",
    "but",
    "not",
    "all",
    "can",
    "one",
    "two",
    "who",
    "which",
    "when",
    "where",
    "what",
    "about",
    "also",
    "into",
    "after",
    "before",
    "than",
    "then",
    "there",
    "these",
    "those",
    "been",
    "being",
    "such",
    "more",
    "most",
    "some",
    "over",
    "under",
    "between",
    "during",
    "while",
    "would",
    "could",
    "should",
}


def normalize_word(word: str) -> str | None:
    word = word.lower().strip("'")
    if len(word) < 3 or word in STOPWORDS:
        return None
    if not any(ch.isalpha() for ch in word):
        return None
    return word


def word_sequence(text: str) -> list[str]:
    words: list[str] = []
    for match in WORD_RE.finditer(text):
        word = normalize_word(match.group(0))
        if word is not None:
            words.append(word)
    return words


def word_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in WORD_RE.finditer(text):
        word = normalize_word(match.group(0))
        if word is not None:
            spans.append((match.start(), match.end(), word))
    return spans


def build_semantic_bank(args: argparse.Namespace) -> tuple[dict[str, int], dict[str, object]]:
    docs = [word_sequence(text) for text in iter_text(Path(args.text_file), args.cluster_docs)]
    freq = Counter(word for doc in docs for word in doc)
    target_words = [
        word
        for word, count in freq.most_common(args.target_vocab)
        if count >= args.min_word_freq and not word.isdigit()
    ]
    context_words = [word for word, count in freq.most_common(args.context_vocab) if count >= args.min_context_freq]
    target_index = {word: i for i, word in enumerate(target_words)}
    context_index = {word: i for i, word in enumerate(context_words)}
    if len(target_words) < args.n_clusters * 3:
        raise RuntimeError(f"Too few target words for clustering: {len(target_words)}")

    mat = np.zeros((len(target_words), len(context_words)), dtype=np.float32)
    for doc in docs:
        for i, word in enumerate(doc):
            row = target_index.get(word)
            if row is None:
                continue
            lo = max(0, i - args.context_window)
            hi = min(len(doc), i + args.context_window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                col = context_index.get(doc[j])
                if col is not None:
                    mat[row, col] += 1.0

    total = float(mat.sum())
    if total <= 0:
        raise RuntimeError("Empty co-occurrence matrix.")
    row_sum = mat.sum(axis=1, keepdims=True)
    col_sum = mat.sum(axis=0, keepdims=True)
    expected = np.maximum(row_sum @ col_sum / total, 1e-12)
    ppmi = np.maximum(np.log(np.maximum(mat, 1e-12) / expected), 0.0)
    ppmi[mat == 0] = 0.0

    n_components = min(args.svd_dim, ppmi.shape[0] - 1, ppmi.shape[1] - 1)
    emb = TruncatedSVD(n_components=n_components, random_state=args.seed).fit_transform(ppmi)
    emb = normalize(emb)
    labels = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=20).fit_predict(emb)

    cluster_to_words: dict[int, list[str]] = defaultdict(list)
    for word, label in zip(target_words, labels):
        cluster_to_words[int(label)].append(word)
    word_to_cluster = {word: int(label) for word, label in zip(target_words, labels)}
    cluster_names = {}
    for cluster, words in cluster_to_words.items():
        ranked = sorted(words, key=lambda w: (-freq[w], w))
        cluster_names[str(cluster)] = ranked[: args.top_words_per_cluster]

    meta = {
        "cluster_method": "text_cooccurrence_ppmi_svd_kmeans",
        "cluster_docs": len(docs),
        "target_words": len(target_words),
        "context_words": len(context_words),
        "n_clusters": args.n_clusters,
        "svd_dim": n_components,
        "context_window": args.context_window,
        "cluster_top_words": cluster_names,
    }
    return word_to_cluster, meta


def labels_for_offsets(text: str, offsets: list[tuple[int, int]], word_to_cluster: dict[str, int], n_clusters: int) -> dict[str, np.ndarray]:
    spans = word_spans(text)
    labels = {f"semantic_cluster_{i:02d}": np.zeros(len(offsets), dtype=bool) for i in range(n_clusters)}
    span_i = 0
    for tok_i, (start, end) in enumerate(offsets):
        while span_i < len(spans) and spans[span_i][1] <= start:
            span_i += 1
        if span_i < len(spans):
            s, e, word = spans[span_i]
            if end > s and start < e:
                cluster = word_to_cluster.get(word)
                if cluster is not None:
                    labels[f"semantic_cluster_{cluster:02d}"][tok_i] = True
    return labels


@torch.no_grad()
def collect(args: argparse.Namespace, word_to_cluster: dict[str, int], bank_meta: dict[str, object]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("This probe requires a fast tokenizer for offset mapping.")
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
    t0 = time.time()
    for text in iter_text(Path(args.text_file), args.max_docs):
        if total >= args.tokens:
            break
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.context,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hs_index = args.layer if args.hook_position == "resid_pre" else args.layer + 1
        acts = out.hidden_states[hs_index][0].detach().float().cpu().numpy()
        mask = attention_mask[0].detach().cpu().numpy().astype(bool)
        if args.skip_first_token and mask.shape[0] > 0:
            mask[0] = False

        valid_indices = np.flatnonzero(mask)
        valid_offsets = [tuple(offsets[i]) for i in valid_indices]
        valid_acts = acts[valid_indices].astype(np.float32)
        labels = labels_for_offsets(text, valid_offsets, word_to_cluster, args.n_clusters)
        keep = min(args.tokens - total, valid_acts.shape[0])
        if keep <= 0:
            continue
        chunks.append(valid_acts[:keep])
        for k, values in labels.items():
            event_chunks.setdefault(k, []).append(values[:keep])
        total += keep

    if not chunks:
        raise RuntimeError("No activations collected.")
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
        **bank_meta,
    }
    return x, events, meta


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
    skipped: dict[str, dict[str, int | float]] = {}
    for name, z_all in events.items():
        z_train = z_all[train_idx]
        z_test = z_all[test_idx]
        p = float(z_all.mean())
        if (
            z_train.sum() < args.min_train_pos
            or z_test.sum() < args.min_test_pos
            or (~z_train).sum() < args.min_train_neg
            or (~z_test).sum() < args.min_test_neg
        ):
            skipped[name] = {
                "firing_rate": p,
                "train_pos": int(z_train.sum()),
                "test_pos": int(z_test.sum()),
                "train_neg": int((~z_train).sum()),
                "test_neg": int((~z_test).sum()),
            }
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
            wp = direction_from_labels(x_train, rng.permutation(z_train))
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
    write_csv(outdir / "natural_semantic_cluster_probe.csv", rows)
    if not rows:
        raise RuntimeError(f"No semantic-cluster events passed the filters. Skipped={skipped}")

    label_abs = np.abs(np.array([r["label_z"] for r in rows], dtype=float))
    full_abs = np.abs(np.array([r["full_cov_z"] for r in rows], dtype=float))
    off_abs = np.abs(np.array([r["offdiag_cov_z"] for r in rows], dtype=float))
    ev = np.array([r["log10_evidence_proxy"] for r in rows], dtype=float)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    summary = {
        "mode": "natural_semantic_cluster_probe",
        **meta,
        "events_evaluated": int(len(rows)),
        "events_skipped": skipped,
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
        "output_csv": str(outdir / "natural_semantic_cluster_probe.csv"),
        "output_figure": str(outdir / "natural_semantic_cluster_probe.pdf"),
    }
    with (outdir / "natural_semantic_cluster_probe_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
        ax[0].scatter(label_abs, off_abs, s=24)
        for r in rows:
            ax[0].annotate(r["event"].replace("semantic_", ""), (abs(r["label_z"]), abs(r["offdiag_cov_z"])), fontsize=6)
        ax[0].set_xlabel("labeled |z|")
        ax[0].set_ylabel("feature-independent |z|")
        ax[1].scatter(ev, off_abs, s=24)
        ax[1].set_xlabel(r"$\log_{10}(NC_i\lambda_i^4)$ proxy")
        ax[1].set_ylabel("feature-independent |z|")
        fig.tight_layout()
        fig.savefig(outdir / "natural_semantic_cluster_probe.png", dpi=220)
        fig.savefig(outdir / "natural_semantic_cluster_probe.pdf")
        plt.close(fig)
    except Exception as exc:
        summary["plot_error"] = repr(exc)
        with (outdir / "natural_semantic_cluster_probe_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="assets/gpt2")
    parser.add_argument("--text-file", default="assets/wikitext_train.txt")
    parser.add_argument("--outdir", default="results/real_activation/c15_natural_semantic_cluster_probe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--hook-position", choices=["resid_pre", "resid_post"], default="resid_pre")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--max-docs", type=int, default=2000)
    parser.add_argument("--tokens", type=int, default=10000)
    parser.add_argument("--skip-first-token", action="store_true", default=True)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--null-reps", type=int, default=200)
    parser.add_argument("--min-train-pos", type=int, default=30)
    parser.add_argument("--min-test-pos", type=int, default=30)
    parser.add_argument("--min-train-neg", type=int, default=500)
    parser.add_argument("--min-test-neg", type=int, default=500)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--cluster-docs", type=int, default=2000)
    parser.add_argument("--target-vocab", type=int, default=500)
    parser.add_argument("--context-vocab", type=int, default=800)
    parser.add_argument("--min-word-freq", type=int, default=5)
    parser.add_argument("--min-context-freq", type=int, default=5)
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--svd-dim", type=int, default=32)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--top-words-per-cluster", type=int, default=12)
    args = parser.parse_args()
    word_to_cluster, bank_meta = build_semantic_bank(args)
    x, events, meta = collect(args, word_to_cluster, bank_meta)
    analyze(args, x, events, meta)


if __name__ == "__main__":
    main()
