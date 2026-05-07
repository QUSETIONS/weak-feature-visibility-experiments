"""C13 aggregate audit for the external natural-feature bank.

This script treats the C9--C11/C14/C15 external lexical, rule-based linguistic,
POS-tagger, NER, and semantic-cluster probes as one pre-specified event bank. It applies a single
multiple-testing correction across all evaluated events and reports whether the
paper's natural-feature claim survives without cherry-picking individual
events.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_INPUTS = [
    (
        "lexical",
        "results/real_activation/c9_natural_lexical_feature_probe/natural_lexical_feature_probe.csv",
    ),
    (
        "linguistic",
        "results/real_activation/c10_natural_linguistic_feature_probe/natural_linguistic_feature_probe.csv",
    ),
    (
        "pos_tagger",
        "results/real_activation/c11_natural_pos_tagger_feature_probe/natural_pos_tagger_feature_probe.csv",
    ),
    (
        "ner",
        "results/real_activation/c14_natural_ner_feature_probe/natural_ner_feature_probe.csv",
    ),
    (
        "semantic_cluster",
        "results/real_activation/c15_natural_semantic_cluster_probe/natural_semantic_cluster_probe.csv",
    ),
]


def read_rows(inputs: list[tuple[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family, path_str in inputs:
        path = Path(path_str)
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out: dict[str, object] = {
                    "family": family,
                    "event": row["event"],
                    "bank_event": f"{family}:{row['event']}",
                }
                for key, value in row.items():
                    if key == "event":
                        continue
                    try:
                        out[key] = float(value)
                    except ValueError:
                        out[key] = value
                rows.append(out)
    if not rows:
        raise RuntimeError("No input rows found.")
    return rows


def normal_two_sided_p(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q_ranked = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * n / rank)
        q_ranked[i] = running
    qvals = np.empty(n, dtype=float)
    qvals[order] = np.minimum(q_ranked, 1.0)
    return qvals


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize(rows: list[dict[str, object]], outdir: Path) -> dict[str, object]:
    label_z = np.array([float(r["label_z"]) for r in rows])
    full_z = np.array([float(r["full_cov_z"]) for r in rows])
    off_z = np.array([float(r["offdiag_cov_z"]) for r in rows])
    evidence = np.array([float(r["log10_evidence_proxy"]) for r in rows])

    label_p = np.array([normal_two_sided_p(z) for z in label_z])
    full_p = np.array([normal_two_sided_p(z) for z in full_z])
    off_p = np.array([normal_two_sided_p(z) for z in off_z])
    label_q = bh_qvalues(label_p)
    full_q = bh_qvalues(full_p)
    off_q = bh_qvalues(off_p)

    out_rows: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        enriched = dict(row)
        enriched.update(
            {
                "label_p_two_sided_normal": label_p[i],
                "label_bh_q": label_q[i],
                "full_cov_p_two_sided_normal": full_p[i],
                "full_cov_bh_q": full_q[i],
                "offdiag_cov_p_two_sided_normal": off_p[i],
                "offdiag_cov_bh_q": off_q[i],
                "label_fdr_005": bool(label_q[i] < 0.05),
                "full_cov_fdr_005": bool(full_q[i] < 0.05),
                "offdiag_cov_fdr_005": bool(off_q[i] < 0.05),
            }
        )
        out_rows.append(enriched)

    families = sorted({str(r["family"]) for r in rows})
    family_summary = {}
    for family in families:
        idx = np.array([str(r["family"]) == family for r in rows])
        family_summary[family] = {
            "events": int(idx.sum()),
            "frac_label_abs_z_gt_5": float(np.mean(np.abs(label_z[idx]) > 5.0)),
            "frac_offdiag_abs_z_gt_5": float(np.mean(np.abs(off_z[idx]) > 5.0)),
            "frac_label_bh_q_lt_005": float(np.mean(label_q[idx] < 0.05)),
            "frac_offdiag_bh_q_lt_005": float(np.mean(off_q[idx] < 0.05)),
            "median_abs_label_z": float(np.median(np.abs(label_z[idx]))),
            "median_abs_offdiag_z": float(np.median(np.abs(off_z[idx]))),
            "corr_log_evidence_abs_offdiag_z": corr(evidence[idx], np.abs(off_z[idx])),
        }

    summary = {
        "mode": "preregistered_external_feature_bank",
        "selection_rule": "All C9-C11/C14/C15 external events passing the pre-run support filters are included; no event is selected by detector outcome.",
        "families": families,
        "events_evaluated": int(len(rows)),
        "median_abs_label_z": float(np.median(np.abs(label_z))),
        "median_abs_full_cov_z": float(np.median(np.abs(full_z))),
        "median_abs_offdiag_cov_z": float(np.median(np.abs(off_z))),
        "frac_label_abs_z_gt_5": float(np.mean(np.abs(label_z) > 5.0)),
        "frac_full_cov_abs_z_gt_5": float(np.mean(np.abs(full_z) > 5.0)),
        "frac_offdiag_abs_z_gt_5": float(np.mean(np.abs(off_z) > 5.0)),
        "frac_label_bh_q_lt_005": float(np.mean(label_q < 0.05)),
        "frac_full_cov_bh_q_lt_005": float(np.mean(full_q < 0.05)),
        "frac_offdiag_bh_q_lt_005": float(np.mean(off_q < 0.05)),
        "corr_log_evidence_abs_label_z": corr(evidence, np.abs(label_z)),
        "corr_log_evidence_abs_full_cov_z": corr(evidence, np.abs(full_z)),
        "corr_log_evidence_abs_offdiag_z": corr(evidence, np.abs(off_z)),
        "min_family_corr_log_evidence_abs_offdiag_z": float(
            np.nanmin([v["corr_log_evidence_abs_offdiag_z"] for v in family_summary.values()])
        ),
        "family_summary": family_summary,
        "rows": out_rows,
    }

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "preregistered_feature_bank.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    summary["output_csv"] = str(csv_path)

    with (outdir / "preregistered_feature_bank_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {
            "lexical": "#4C78A8",
            "linguistic": "#F58518",
            "pos_tagger": "#54A24B",
            "ner": "#B279A2",
            "semantic_cluster": "#E45756",
        }
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
        for family in families:
            idx = np.array([str(r["family"]) == family for r in rows])
            ax[0].scatter(np.abs(label_z[idx]), np.abs(off_z[idx]), s=28, label=family, color=colors.get(family))
            ax[1].scatter(evidence[idx], np.abs(off_z[idx]), s=28, label=family, color=colors.get(family))
        ax[0].axvline(5.0, color="0.6", lw=0.8, ls="--")
        ax[0].axhline(5.0, color="0.6", lw=0.8, ls="--")
        ax[0].set_xlabel("labeled |z|")
        ax[0].set_ylabel("feature-independent |z|")
        ax[1].axhline(5.0, color="0.6", lw=0.8, ls="--")
        ax[1].set_xlabel(r"$\log_{10}(NC_i\lambda_i^4)$ proxy")
        ax[1].set_ylabel("feature-independent |z|")
        ax[1].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(outdir / "preregistered_feature_bank.png", dpi=220)
        fig.savefig(outdir / "preregistered_feature_bank.pdf")
        plt.close(fig)
    except Exception as exc:
        summary["plot_error"] = repr(exc)
        with (outdir / "preregistered_feature_bank_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/real_activation/c13_preregistered_feature_bank")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input as family=path. Defaults to the formal external-feature CSV files.",
    )
    args = parser.parse_args()

    inputs = DEFAULT_INPUTS
    if args.input:
        parsed = []
        for item in args.input:
            if "=" not in item:
                raise ValueError(f"Expected family=path input, got {item!r}")
            family, path = item.split("=", 1)
            parsed.append((family, path))
        inputs = parsed

    summary = summarize(read_rows(inputs), Path(args.outdir))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
