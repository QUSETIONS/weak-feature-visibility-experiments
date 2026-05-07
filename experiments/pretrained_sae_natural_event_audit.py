"""C17: pretrained SAE recovery audit for externally defined natural events.

This experiment is deliberately SAE-facing but claim-bounded.  It asks whether
natural events that already have detector-law evidence in C9--C14 are more
recoverable by pretrained SAE atoms than by a size-matched random ReLU
dictionary.  It uses cached GPT-2 activations so the expensive model forward is
not repeated; token labels are regenerated from the same Wikitext stream.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import string
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "each", "every", "some", "any"}
PRONOUNS = {
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
}
MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
PREPOSITIONS = {
    "in", "on", "at", "by", "for", "with", "from", "into", "about", "over", "after",
    "before", "between", "under", "through", "during",
}
CONJUNCTIONS = {"and", "or", "but", "if", "because", "while", "although", "though", "since"}
NEGATIONS = {"no", "not", "never", "n't", "nor", "none", "neither"}
MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}
TITLES = {"mr", "mrs", "ms", "dr", "prof", "sir", "lord", "lady", "president"}
DISCOURSE = {"however", "therefore", "moreover", "thus", "hence", "nevertheless", "meanwhile"}
NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "hundred", "thousand", "million",
}
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*")


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_bank(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row["bank_event"]: row for row in csv.DictReader(f)}


def lexical_events(token: str) -> dict[str, bool]:
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


def clean_token(token: str) -> str:
    return token.strip().strip(" \t\r\n.,;:!?()[]{}\"'`")


def linguistic_events(token: str) -> dict[str, bool]:
    word = clean_token(token)
    lower = word.lower()
    alpha = bool(re.fullmatch(r"[A-Za-z][A-Za-z'-]*", word))
    return {
        "determiner": lower in DETERMINERS,
        "pronoun": lower in PRONOUNS,
        "modal_aux": lower in MODALS,
        "preposition": lower in PREPOSITIONS,
        "conjunction": lower in CONJUNCTIONS,
        "negation": lower in NEGATIONS or token.strip().lower() == "n't",
        "month_name": lower in MONTHS,
        "title_word": lower.rstrip(".") in TITLES,
        "discourse_marker": lower in DISCOURSE,
        "number_word": lower in NUMBER_WORDS,
        "verb_ing": alpha and lower.endswith("ing") and len(lower) > 5,
        "verb_ed": alpha and lower.endswith("ed") and len(lower) > 4,
        "noun_suffix": alpha and lower.endswith(("tion", "ment", "ness", "ity", "ship", "ism")),
        "adj_suffix": alpha and lower.endswith(("ous", "ive", "al", "able", "ible", "ful", "less", "ic")),
        "proper_name_proxy": alpha and len(word) > 2 and word[0].isupper() and lower not in MONTHS,
    }


def ensure_nltk(data_dir: Path, need_ner: bool) -> None:
    import nltk

    data_dir.mkdir(parents=True, exist_ok=True)
    if str(data_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(data_dir))
    resources = ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]
    if need_ner:
        resources += ["maxent_ne_chunker", "maxent_ne_chunker_tab", "words"]
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
    nltk.pos_tag(["The", "test", "works", "."])
    if need_ner:
        nltk.ne_chunk(nltk.pos_tag(["Barack", "Obama", "visited", "France"]), binary=False)


def pos_events(tag: str) -> dict[str, bool]:
    return {
        "pos_noun": tag.startswith("NN"),
        "pos_proper_noun": tag in {"NNP", "NNPS"},
        "pos_plural_noun": tag in {"NNS", "NNPS"},
        "pos_verb": tag.startswith("VB"),
        "pos_past_verb": tag in {"VBD", "VBN"},
        "pos_gerund": tag == "VBG",
        "pos_adjective": tag.startswith("JJ"),
        "pos_adverb": tag.startswith("RB"),
        "pos_determiner": tag == "DT",
        "pos_preposition": tag == "IN",
        "pos_pronoun": tag in {"PRP", "PRP$"},
        "pos_modal": tag == "MD",
        "pos_cardinal": tag == "CD",
        "pos_conjunction": tag == "CC",
    }


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


def tagged_word_spans(text: str) -> list[tuple[int, int, str]]:
    import nltk

    matches = list(WORD_RE.finditer(text))
    words = [m.group(0) for m in matches]
    if not words:
        return []
    tags = [tag for _, tag in nltk.pos_tag(words)]
    return [(m.start(), m.end(), tag) for m, tag in zip(matches, tags)]


def entity_word_spans(text: str) -> list[tuple[int, int, str]]:
    import nltk

    matches = list(WORD_RE.finditer(text))
    words = [m.group(0) for m in matches]
    if not words:
        return []
    tree = nltk.ne_chunk(nltk.pos_tag(words), binary=False)
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


def span_labels(offsets: list[tuple[int, int]], spans: list[tuple[int, int, str]], fn) -> dict[str, list[bool]]:
    labels: dict[str, list[bool]] = {}
    span_i = 0
    for start, end in offsets:
        while span_i < len(spans) and spans[span_i][1] <= start:
            span_i += 1
        value = ""
        if span_i < len(spans):
            s, e, candidate = spans[span_i]
            if end > s and start < e:
                value = candidate
        for k, v in fn(value).items():
            labels.setdefault(k, []).append(v)
    return labels


def append_labels(dst: dict[str, list[np.ndarray]], family: str, labels: dict[str, list[bool]], keep: int) -> None:
    for event, values in labels.items():
        dst.setdefault(f"{family}:{event}", []).append(np.asarray(values[:keep], dtype=bool))


def collect_token_labels(args: argparse.Namespace, n_tokens: int) -> dict[str, np.ndarray]:
    from transformers import AutoTokenizer

    families = set(args.families)
    events: dict[str, list[np.ndarray]] = {}

    if families.intersection({"lexical", "linguistic"}):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        total = 0
        batch: list[str] = []

        def flush(texts: list[str]) -> None:
            nonlocal total
            if not texts or total >= n_tokens:
                return
            enc = tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=args.context)
            ids = enc["input_ids"]
            mask = enc["attention_mask"].astype(bool)
            if args.skip_first_token and mask.shape[1] > 0:
                mask[:, 0] = False
            flat_ids = ids[mask]
            keep = min(n_tokens - total, flat_ids.shape[0])
            if keep <= 0:
                return
            labels: dict[str, list[bool]] = {}
            for tok_id in flat_ids[:keep]:
                tok = tokenizer.decode([int(tok_id)])
                if "lexical" in families:
                    for k, v in lexical_events(tok).items():
                        labels.setdefault(f"lexical:{k}", []).append(v)
                if "linguistic" in families:
                    for k, v in linguistic_events(tok).items():
                        labels.setdefault(f"linguistic:{k}", []).append(v)
            for k, values in labels.items():
                events.setdefault(k, []).append(np.asarray(values, dtype=bool))
            total += keep

        for text in iter_text(Path(args.text_file), args.max_docs):
            batch.append(text)
            if len(batch) >= args.batch_size:
                flush(batch)
                batch = []
                if total >= n_tokens:
                    break
        flush(batch)

    if families.intersection({"pos", "ner"}):
        ensure_nltk(Path(args.nltk_data_dir), need_ner="ner" in families)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("POS/NER labels require a fast tokenizer.")
        total = 0
        for text in iter_text(Path(args.text_file), args.max_docs):
            if total >= n_tokens:
                break
            enc = tokenizer(text, return_tensors="np", truncation=True, max_length=args.context, return_offsets_mapping=True)
            offsets_all = [tuple(x) for x in enc["offset_mapping"][0].tolist()]
            mask = enc["attention_mask"][0].astype(bool)
            if args.skip_first_token and mask.shape[0] > 0:
                mask[0] = False
            offsets = [offsets_all[i] for i in np.flatnonzero(mask)]
            keep = min(n_tokens - total, len(offsets))
            if keep <= 0:
                continue
            offsets = offsets[:keep]
            if "pos" in families:
                append_labels(events, "pos_tagger", span_labels(offsets, tagged_word_spans(text), pos_events), keep)
            if "ner" in families:
                append_labels(events, "ner", span_labels(offsets, entity_word_spans(text), ner_events), keep)
            total += keep

    return {k: np.concatenate(v, axis=0)[:n_tokens] for k, v in sorted(events.items())}


def load_sae(sae_dir: Path, device: torch.device) -> dict[str, torch.Tensor]:
    weights = load_file(str(sae_dir / "sae_weights.safetensors"), device=str(device))
    required = {"W_enc", "W_dec", "b_enc", "b_dec"}
    missing = required.difference(weights)
    if missing:
        raise KeyError(f"Missing SAE tensors: {sorted(missing)}")
    return {k: weights[k].float() for k in required}


@torch.no_grad()
def encode_sae_matrix(x: np.ndarray, weights: dict[str, torch.Tensor], device: torch.device, batch_size: int) -> np.ndarray:
    d_sae = int(weights["W_enc"].shape[1])
    out = np.empty((x.shape[0], d_sae), dtype=np.float16)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        xb = torch.from_numpy(np.asarray(x[start:end], dtype=np.float32)).to(device)
        acts = torch.relu((xb - weights["b_dec"]) @ weights["W_enc"] + weights["b_enc"])
        out[start:end] = acts.detach().cpu().numpy().astype(np.float16)
    return out


def random_relu_matrix(
    x: np.ndarray,
    n_features: int,
    seed: int,
    chunk_features: int = 1024,
    center: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((x.shape[0], n_features), dtype=np.float16)
    if center is None:
        center = x.mean(axis=0, keepdims=True)
    xc = x - center
    scale = 1.0 / math.sqrt(x.shape[1])
    for start in range(0, n_features, chunk_features):
        end = min(start + chunk_features, n_features)
        dirs = rng.normal(scale=scale, size=(x.shape[1], end - start)).astype(np.float32)
        acts = np.maximum(xc @ dirs, 0.0)
        out[:, start:end] = acts.astype(np.float16)
    return out


def random_sparse_relu_pair(
    x_train: np.ndarray,
    x_test: np.ndarray,
    n_features: int,
    seed: int,
    target_rates: np.ndarray,
    chunk_features: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_out = np.empty((x_train.shape[0], n_features), dtype=np.float16)
    test_out = np.empty((x_test.shape[0], n_features), dtype=np.float16)
    center = x_train.mean(axis=0, keepdims=True)
    xtr = x_train - center
    xte = x_test - center
    scale = 1.0 / math.sqrt(x_train.shape[1])
    rates = np.asarray(target_rates, dtype=np.float64)
    rates = rates[np.isfinite(rates)]
    rates = rates[(rates >= 0.0) & (rates <= 1.0)]
    if rates.size == 0:
        rates = np.asarray([0.01], dtype=np.float64)
    sampled_rates = rng.choice(rates, size=n_features, replace=True)
    for start in range(0, n_features, chunk_features):
        end = min(start + chunk_features, n_features)
        dirs = rng.normal(scale=scale, size=(x_train.shape[1], end - start)).astype(np.float32)
        train_scores = xtr @ dirs
        test_scores = xte @ dirs
        q = np.clip(1.0 - sampled_rates[start:end], 0.0, 1.0)
        thresholds = np.asarray(
            [np.quantile(train_scores[:, j], q[j], method="linear") for j in range(end - start)],
            dtype=np.float32,
        )
        train_out[:, start:end] = np.maximum(train_scores - thresholds[None, :], 0.0).astype(np.float16)
        test_out[:, start:end] = np.maximum(test_scores - thresholds[None, :], 0.0).astype(np.float16)
    return train_out, test_out


def auc_1d(scores: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    sorted_scores = scores[order]
    rank = 1
    i = 0
    n = scores.shape[0]
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg = 0.5 * (rank + rank + (j - i) - 1)
        ranks[order[i:j]] = avg
        rank += j - i
        i = j
    sum_pos = float(ranks[y].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def select_top_features(acts: np.ndarray, y: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    y = y.astype(bool)
    pos = acts[y].astype(np.float32)
    neg = acts[~y].astype(np.float32)
    pos_mean = pos.mean(axis=0)
    neg_mean = neg.mean(axis=0)
    pooled = np.sqrt(0.5 * (pos.var(axis=0) + neg.var(axis=0)) + 1e-12)
    effect = (pos_mean - neg_mean) / pooled
    order = np.argsort(np.abs(effect))[-min(top_k, acts.shape[1]) :]
    return order.astype(np.int64), effect


def best_recovery_metrics(
    train_acts: np.ndarray,
    test_acts: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    top_k: int,
) -> dict[str, float | int]:
    candidates, train_effect = select_top_features(train_acts, y_train, top_k)
    _, test_effect = select_top_features(test_acts[:, candidates], y_test, min(top_k, candidates.size))
    best: dict[str, float | int] = {
        "best_feature": -1,
        "best_auc": float("nan"),
        "best_abs_auc_excess": -1.0,
        "best_train_effect": float("nan"),
        "best_test_effect": float("nan"),
        "best_pos_mean": float("nan"),
        "best_neg_mean": float("nan"),
        "candidate_features": int(candidates.size),
    }
    for local_i, feature in enumerate(candidates):
        scores = test_acts[:, feature].astype(np.float32)
        auc = auc_1d(scores, y_test)
        excess = abs(auc - 0.5)
        if excess > float(best["best_abs_auc_excess"]):
            best.update(
                {
                    "best_feature": int(feature),
                    "best_auc": float(auc),
                    "best_abs_auc_excess": float(excess),
                    "best_train_effect": float(train_effect[feature]),
                    "best_test_effect": float(test_effect[local_i]),
                    "best_pos_mean": float(scores[y_test].mean()),
                    "best_neg_mean": float(scores[~y_test].mean()),
                }
            )
    return best


def unit_label_direction(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray | None:
    y_train = y_train.astype(bool)
    if y_train.sum() < 2 or (~y_train).sum() < 2:
        return None
    w = x_train[y_train].mean(axis=0) - x_train[~y_train].mean(axis=0)
    norm = float(np.linalg.norm(w))
    if norm < 1e-12:
        return None
    return (w / norm).astype(np.float32)


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def maybe_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def audit(args: argparse.Namespace) -> dict[str, object]:
    t0 = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    x_all = np.load(args.activation_path, mmap_mode="r")
    n = min(int(args.max_tokens), int(x_all.shape[0])) if args.max_tokens else int(x_all.shape[0])
    x = np.asarray(x_all[:n], dtype=np.float32)
    labels = collect_token_labels(args, n)
    bank = load_bank(Path(args.bank_csv))

    idx = rng.permutation(n)
    train_n = int(n * args.train_frac)
    train_idx = idx[:train_n]
    test_idx = idx[train_n:]
    x_train = x[train_idx]
    x_test = x[test_idx]

    weights = load_sae(Path(args.sae_dir), device)
    d_sae = int(weights["W_enc"].shape[1])
    random_features = int(args.random_features or d_sae)
    sae_train_acts = encode_sae_matrix(x_train, weights, device, args.batch_size)
    sae_test_acts = encode_sae_matrix(x_test, weights, device, args.batch_size)
    sae_firing_rates = (sae_train_acts > 0).mean(axis=0)
    if args.random_baseline == "sparse_calibrated":
        random_train_acts, random_test_acts = random_sparse_relu_pair(
            x_train,
            x_test,
            random_features,
            args.seed + 17,
            sae_firing_rates,
            args.random_chunk_features,
        )
    else:
        random_center = x_train.mean(axis=0, keepdims=True)
        random_train_acts = random_relu_matrix(x_train, random_features, args.seed + 17, args.random_chunk_features, random_center)
        random_test_acts = random_relu_matrix(x_test, random_features, args.seed + 17, args.random_chunk_features, random_center)

    w_dec = weights["W_dec"].detach().cpu().numpy().astype(np.float32)
    w_dec_norm = w_dec / np.maximum(np.linalg.norm(w_dec, axis=1, keepdims=True), 1e-12)

    rows: list[dict[str, object]] = []
    skipped: dict[str, dict[str, object]] = {}
    for bank_event, y_all in labels.items():
        family, event = bank_event.split(":", 1)
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        train_pos = int(y_train.sum())
        test_pos = int(y_test.sum())
        if (
            train_pos < args.min_train_pos
            or test_pos < args.min_test_pos
            or int((~y_train).sum()) < args.min_train_neg
            or int((~y_test).sum()) < args.min_test_neg
        ):
            skipped[bank_event] = {"train_pos": train_pos, "test_pos": test_pos, "firing_rate": float(y_all.mean())}
            continue

        sae = best_recovery_metrics(sae_train_acts, sae_test_acts, y_train, y_test, args.top_auc_candidates)
        rnd = best_recovery_metrics(random_train_acts, random_test_acts, y_train, y_test, args.top_auc_candidates)

        direction = unit_label_direction(x_train, y_train)
        best_feature = int(sae["best_feature"])
        best_decoder_cosine = float("nan")
        max_decoder_abs_cosine = float("nan")
        if direction is not None and best_feature >= 0:
            cosines = w_dec_norm @ direction
            best_decoder_cosine = float(cosines[best_feature])
            max_decoder_abs_cosine = float(np.max(np.abs(cosines)))

        bank_row = bank.get(bank_event, {})
        bank_train_pos = maybe_float(bank_row, "train_pos") if bank_row else float("nan")
        bank_test_pos = maybe_float(bank_row, "test_pos") if bank_row else float("nan")
        row = {
            "family": family,
            "event": event,
            "bank_event": bank_event,
            "tokens": int(n),
            "train_pos": train_pos,
            "test_pos": test_pos,
            "firing_rate": float(y_all.mean()),
            "bank_train_pos": bank_train_pos,
            "bank_test_pos": bank_test_pos,
            "count_matches_bank": bool(
                bank_row
                and abs(bank_train_pos - train_pos) < 0.5
                and abs(bank_test_pos - test_pos) < 0.5
            ),
            "bank_log10_evidence_proxy": maybe_float(bank_row, "log10_evidence_proxy") if bank_row else float("nan"),
            "bank_label_z": maybe_float(bank_row, "label_z") if bank_row else float("nan"),
            "bank_full_cov_z": maybe_float(bank_row, "full_cov_z") if bank_row else float("nan"),
            "bank_offdiag_cov_z": maybe_float(bank_row, "offdiag_cov_z") if bank_row else float("nan"),
            "sae_best_feature": int(sae["best_feature"]),
            "sae_best_auc": float(sae["best_auc"]),
            "sae_best_abs_auc_excess": float(sae["best_abs_auc_excess"]),
            "sae_best_train_effect": float(sae["best_train_effect"]),
            "sae_best_test_effect": float(sae["best_test_effect"]),
            "sae_best_pos_mean": float(sae["best_pos_mean"]),
            "sae_best_neg_mean": float(sae["best_neg_mean"]),
            "random_best_feature": int(rnd["best_feature"]),
            "random_best_auc": float(rnd["best_auc"]),
            "random_best_abs_auc_excess": float(rnd["best_abs_auc_excess"]),
            "random_best_train_effect": float(rnd["best_train_effect"]),
            "random_best_test_effect": float(rnd["best_test_effect"]),
            "sae_minus_random_abs_auc_excess": float(sae["best_abs_auc_excess"]) - float(rnd["best_abs_auc_excess"]),
            "sae_beats_random": bool(float(sae["best_abs_auc_excess"]) > float(rnd["best_abs_auc_excess"])),
            "sae_best_decoder_cosine_to_label_direction": best_decoder_cosine,
            "sae_max_abs_decoder_cosine_to_label_direction": max_decoder_abs_cosine,
        }
        rows.append(row)

    write_csv(outdir / "pretrained_sae_natural_event_audit.csv", rows)

    valid = [r for r in rows if np.isfinite(float(r["bank_log10_evidence_proxy"]))]
    evidence = [float(r["bank_log10_evidence_proxy"]) for r in valid]
    sae_excess = [float(r["sae_best_abs_auc_excess"]) for r in valid]
    gap = [float(r["sae_minus_random_abs_auc_excess"]) for r in valid]
    cos = [float(r["sae_max_abs_decoder_cosine_to_label_direction"]) for r in valid]
    cos_pairs = [(e, c) for e, c in zip(evidence, cos) if np.isfinite(c)]

    family_summary: dict[str, dict[str, float | int]] = {}
    for family in sorted({str(r["family"]) for r in rows}):
        sub = [r for r in rows if r["family"] == family]
        family_summary[family] = {
            "events": len(sub),
            "mean_sae_abs_auc_excess": float(np.mean([float(r["sae_best_abs_auc_excess"]) for r in sub])),
            "mean_random_abs_auc_excess": float(np.mean([float(r["random_best_abs_auc_excess"]) for r in sub])),
            "mean_sae_minus_random_abs_auc_excess": float(np.mean([float(r["sae_minus_random_abs_auc_excess"]) for r in sub])),
            "sae_beats_random_fraction": float(np.mean([bool(r["sae_beats_random"]) for r in sub])),
        }

    tertile_summary = {}
    if len(valid) >= 6:
        ordered = sorted(valid, key=lambda r: float(r["bank_log10_evidence_proxy"]))
        k = max(1, len(ordered) // 3)
        low = ordered[:k]
        high = ordered[-k:]
        tertile_summary = {
            "low_evidence_mean_sae_minus_random_abs_auc_excess": float(np.mean([float(r["sae_minus_random_abs_auc_excess"]) for r in low])),
            "high_evidence_mean_sae_minus_random_abs_auc_excess": float(np.mean([float(r["sae_minus_random_abs_auc_excess"]) for r in high])),
            "low_evidence_mean_sae_abs_auc_excess": float(np.mean([float(r["sae_best_abs_auc_excess"]) for r in low])),
            "high_evidence_mean_sae_abs_auc_excess": float(np.mean([float(r["sae_best_abs_auc_excess"]) for r in high])),
        }

    summary = {
        "mode": "pretrained_sae_natural_event_audit",
        "activation_path": args.activation_path,
        "sae_dir": args.sae_dir,
        "bank_csv": args.bank_csv,
        "tokens": int(n),
        "train_tokens": int(train_n),
        "test_tokens": int(n - train_n),
        "d_model": int(x.shape[1]),
        "d_sae": int(d_sae),
        "random_features": int(random_features),
        "random_baseline": args.random_baseline,
        "sae_train_mean_firing_rate": float(np.mean(sae_firing_rates)),
        "sae_train_median_firing_rate": float(np.median(sae_firing_rates)),
        "sae_train_live_feature_fraction": float(np.mean(sae_firing_rates > 0.0)),
        "families_requested": list(args.families),
        "events_generated": int(len(labels)),
        "events_evaluated": int(len(rows)),
        "events_skipped": int(len(skipped)),
        "bank_matched_events": int(sum(bool(r["count_matches_bank"]) for r in rows)),
        "bank_match_fraction_among_evaluated": float(np.mean([bool(r["count_matches_bank"]) for r in rows])) if rows else float("nan"),
        "mean_sae_abs_auc_excess": float(np.mean([float(r["sae_best_abs_auc_excess"]) for r in rows])) if rows else float("nan"),
        "mean_random_abs_auc_excess": float(np.mean([float(r["random_best_abs_auc_excess"]) for r in rows])) if rows else float("nan"),
        "mean_sae_minus_random_abs_auc_excess": float(np.mean([float(r["sae_minus_random_abs_auc_excess"]) for r in rows])) if rows else float("nan"),
        "sae_beats_random_fraction": float(np.mean([bool(r["sae_beats_random"]) for r in rows])) if rows else float("nan"),
        "corr_log10_evidence_vs_sae_abs_auc_excess": corr(evidence, sae_excess),
        "corr_log10_evidence_vs_sae_minus_random_abs_auc_excess": corr(evidence, gap),
        "corr_log10_evidence_vs_max_decoder_abs_cosine": corr([p[0] for p in cos_pairs], [p[1] for p in cos_pairs]),
        "family_summary": family_summary,
        "tertile_summary": tertile_summary,
        "skipped": skipped,
        "csv": str(outdir / "pretrained_sae_natural_event_audit.csv"),
        "seconds": float(time.time() - t0),
        "device": str(device),
    }
    with (outdir / "pretrained_sae_natural_event_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    try:
        import matplotlib

        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt

        if valid:
            fig, ax = plt.subplots(figsize=(4.2, 3.1))
            ax.axhline(0.0, color="0.75", lw=1)
            ax.scatter(evidence, gap, s=22, alpha=0.8)
            ax.set_xlabel("Detector-law evidence (log10 proxy)")
            ax.set_ylabel("SAE - random recovery (AUC excess)")
            ax.set_title("C17 pretrained SAE natural-event audit")
            fig.tight_layout()
            fig.savefig(outdir / "pretrained_sae_natural_event_audit.pdf")
            fig.savefig(outdir / "pretrained_sae_natural_event_audit.png", dpi=180)
            plt.close(fig)
    except Exception as exc:
        summary["plot_error"] = str(exc)
        with (outdir / "pretrained_sae_natural_event_audit_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-path", default="results/real_activation/c1_10k_resid_pre_clean/gpt2_layer6_resid_pre_10000tok.npy")
    parser.add_argument("--sae-dir", default="assets/gpt2_sae_layer6/blocks.6.hook_resid_pre")
    parser.add_argument("--bank-csv", default="results/real_activation/c13_preregistered_feature_bank/preregistered_feature_bank.csv")
    parser.add_argument("--outdir", default="results/real_activation/c17_pretrained_sae_natural_event_audit")
    parser.add_argument("--model-name", default="assets/gpt2")
    parser.add_argument("--text-file", default="assets/wikitext_train.txt")
    parser.add_argument("--nltk-data-dir", default="assets/nltk_data")
    parser.add_argument("--families", nargs="+", default=["lexical", "linguistic", "pos", "ner"], choices=["lexical", "linguistic", "pos", "ner"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--token-label-batch-size", type=int, default=16)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--skip-first-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-train-pos", type=int, default=20)
    parser.add_argument("--min-test-pos", type=int, default=20)
    parser.add_argument("--min-train-neg", type=int, default=20)
    parser.add_argument("--min-test-neg", type=int, default=20)
    parser.add_argument("--random-features", type=int, default=0, help="0 means match the SAE width.")
    parser.add_argument("--random-baseline", choices=["sparse_calibrated", "dense_zero_bias"], default="sparse_calibrated")
    parser.add_argument("--random-chunk-features", type=int, default=1024)
    parser.add_argument("--top-auc-candidates", type=int, default=512)
    args = parser.parse_args()
    args.batch_size = int(args.batch_size)
    args.token_label_batch_size = int(args.token_label_batch_size)
    # Keep the legacy name used by the C1/C9 tokenization code.
    if not hasattr(args, "batch_size"):
        args.batch_size = args.token_label_batch_size
    audit(args)


if __name__ == "__main__":
    main()
