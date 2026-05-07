"""External NER natural feature probe for the quartic detection story.

This C14 probe uses NLTK's named-entity chunker to define more semantic
external events than lexical/POS tags. GPT-2 BPE tokens inherit the entity type
of the word span they overlap, then the same held-out detector-class audit used
by C9--C11 is applied.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

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


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*")


def ensure_nltk(data_dir: Path) -> None:
    import nltk

    data_dir.mkdir(parents=True, exist_ok=True)
    if str(data_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(data_dir))
    resources = [
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker",
        "maxent_ne_chunker_tab",
        "words",
    ]
    for resource in resources:
        try:
            if resource.startswith("averaged"):
                nltk.data.find(f"taggers/{resource}")
            elif resource == "words":
                nltk.data.find("corpora/words")
            else:
                nltk.data.find(f"chunkers/{resource}")
        except Exception:
            try:
                nltk.download(resource, download_dir=str(data_dir), quiet=True)
            except Exception:
                pass
    try:
        nltk.ne_chunk(nltk.pos_tag(["Barack", "Obama", "visited", "France"]), binary=False)
    except Exception as exc:
        raise RuntimeError(f"Could not load NLTK NER resources into {data_dir}: {exc}") from exc


def ner_events(label: str | None) -> dict[str, bool]:
    label = label or ""
    return {
        "ner_any": bool(label),
        "ner_person": label == "PERSON",
        "ner_organization": label == "ORGANIZATION",
        "ner_gpe": label == "GPE",
        "ner_location": label == "LOCATION",
        "ner_facility": label == "FACILITY",
        "ner_person_or_org": label in {"PERSON", "ORGANIZATION"},
        "ner_place_like": label in {"GPE", "LOCATION", "FACILITY"},
    }


def entity_word_spans(text: str) -> list[tuple[int, int, str]]:
    import nltk

    matches = list(WORD_RE.finditer(text))
    words = [m.group(0) for m in matches]
    if not words:
        return []
    tagged = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tagged, binary=False)
    spans: list[tuple[int, int, str]] = []
    word_i = 0
    for node in tree:
        if hasattr(node, "label"):
            leaves = node.leaves()
            start_i = word_i
            end_i = word_i + len(leaves) - 1
            spans.append((matches[start_i].start(), matches[end_i].end(), node.label()))
            word_i += len(leaves)
        else:
            word_i += 1
    return spans


def labels_for_offsets(text: str, offsets: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    spans = entity_word_spans(text)
    labels: dict[str, list[bool]] = {}
    span_i = 0
    for start, end in offsets:
        while span_i < len(spans) and spans[span_i][1] <= start:
            span_i += 1
        label = None
        if span_i < len(spans):
            s, e, candidate = spans[span_i]
            if end > s and start < e:
                label = candidate
        ev = ner_events(label)
        for k, v in ev.items():
            labels.setdefault(k, []).append(v)
    return {k: np.asarray(v, dtype=bool) for k, v in labels.items()}


@torch.no_grad()
def collect(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    ensure_nltk(Path(args.nltk_data_dir))
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
        labels = labels_for_offsets(text, valid_offsets)
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
        "tagger": "nltk_ne_chunk",
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
    write_csv(outdir / "natural_ner_feature_probe.csv", rows)
    if not rows:
        raise RuntimeError(f"No NER events passed the filters. Skipped={skipped}")

    label_abs = np.abs(np.array([r["label_z"] for r in rows], dtype=float))
    full_abs = np.abs(np.array([r["full_cov_z"] for r in rows], dtype=float))
    off_abs = np.abs(np.array([r["offdiag_cov_z"] for r in rows], dtype=float))
    ev = np.array([r["log10_evidence_proxy"] for r in rows], dtype=float)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    summary = {
        "mode": "natural_ner_feature_probe",
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
        "output_csv": str(outdir / "natural_ner_feature_probe.csv"),
        "output_figure": str(outdir / "natural_ner_feature_probe.pdf"),
    }
    with (outdir / "natural_ner_feature_probe_summary.json").open("w", encoding="utf-8") as f:
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
        fig.savefig(outdir / "natural_ner_feature_probe.png", dpi=220)
        fig.savefig(outdir / "natural_ner_feature_probe.pdf")
        plt.close(fig)
    except Exception as exc:
        summary["plot_error"] = repr(exc)
        with (outdir / "natural_ner_feature_probe_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="assets/gpt2")
    parser.add_argument("--text-file", default="assets/wikitext_train.txt")
    parser.add_argument("--outdir", default="results/real_activation/c14_natural_ner_feature_probe")
    parser.add_argument("--nltk-data-dir", default="assets/nltk_data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--hook-position", choices=["resid_pre", "resid_post"], default="resid_pre")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--max-docs", type=int, default=2000)
    parser.add_argument("--tokens", type=int, default=10000)
    parser.add_argument("--skip-first-token", action="store_true", default=True)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--null-reps", type=int, default=200)
    parser.add_argument("--min-train-pos", type=int, default=20)
    parser.add_argument("--min-test-pos", type=int, default=20)
    parser.add_argument("--min-train-neg", type=int, default=500)
    parser.add_argument("--min-test-neg", type=int, default=500)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260503)
    args = parser.parse_args()
    x, events, meta = collect(args)
    analyze(args, x, events, meta)


if __name__ == "__main__":
    main()
