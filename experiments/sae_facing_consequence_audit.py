"""C16 SAE-facing consequence audit.

This script turns the existing controlled TopK SAE phase diagram into a compact
paper-facing bridge result. It does not claim that SAE optimization is the
theorem. It asks a narrower operational question: when injected features have
larger detector evidence, do SAE-facing recovery diagnostics become stronger?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def state_from_row(row: dict[str, str]) -> str:
    cos = float(row["mean_max_decoder_cosine"])
    auc = float(row["mean_best_atom_label_auc"])
    auc_excess = float(row["mean_top_label_atom_abs_auc_excess"])
    if cos >= 0.8 and auc >= 0.8 and auc_excess >= 0.2:
        return "recovered"
    if auc_excess >= 0.05:
        return "transition"
    return "null"


def summarize(args: argparse.Namespace) -> dict[str, object]:
    rows = read_csv(Path(args.by_config_csv))
    out_rows: list[dict[str, object]] = []
    for row in rows:
        amp = float(row["amp"])
        p = float(row["p"])
        lambda_eff = amp * math.sqrt(p)
        evidence = lambda_eff**4
        state = state_from_row(row)
        out = {
            "amp": amp,
            "p": p,
            "lambda_eff": lambda_eff,
            "lambda_eff_fourth": evidence,
            "log10_lambda_eff_fourth": math.log10(max(evidence, 1e-30)),
            "state": state,
            "recovered": state == "recovered",
            "transition_or_recovered": state in {"transition", "recovered"},
            "mean_signal_gain": float(row["mean_signal_gain"]),
            "mean_max_decoder_cosine": float(row["mean_max_decoder_cosine"]),
            "mean_best_atom_label_auc": float(row["mean_best_atom_label_auc"]),
            "mean_top_label_atom_abs_auc_excess": float(row["mean_top_label_atom_abs_auc_excess"]),
            "mean_top_label_atom_decoder_cosine": float(row["mean_top_label_atom_decoder_cosine"]),
        }
        out_rows.append(out)

    x = np.array([float(r["log10_lambda_eff_fourth"]) for r in out_rows])
    lam = np.array([float(r["lambda_eff"]) for r in out_rows])
    recovered = np.array([1 if r["recovered"] else 0 for r in out_rows], dtype=int)
    transition_or_recovered = np.array([1 if r["transition_or_recovered"] else 0 for r in out_rows], dtype=int)

    metrics = [
        "mean_signal_gain",
        "mean_max_decoder_cosine",
        "mean_best_atom_label_auc",
        "mean_top_label_atom_abs_auc_excess",
        "mean_top_label_atom_decoder_cosine",
    ]
    metric_summary = {}
    for metric in metrics:
        y = np.array([float(r[metric]) for r in out_rows])
        metric_summary[metric] = {
            "pearson_log10_lambda4": pearson(x, y),
            "spearman_log10_lambda4": spearman(x, y),
            "low_third_mean": float(np.mean(y[np.argsort(x)[: len(x) // 3]])),
            "high_third_mean": float(np.mean(y[np.argsort(x)[-len(x) // 3 :]])),
        }

    ordered = sorted(out_rows, key=lambda r: float(r["lambda_eff"]))
    null_lambdas = [float(r["lambda_eff"]) for r in ordered if r["state"] == "null"]
    transition_lambdas = [float(r["lambda_eff"]) for r in ordered if r["state"] == "transition"]
    recovered_lambdas = [float(r["lambda_eff"]) for r in ordered if r["state"] == "recovered"]

    summary = {
        "mode": "sae_facing_consequence_audit",
        "rows": len(out_rows),
        "input_by_config_csv": args.by_config_csv,
        "state_counts": {
            "null": sum(r["state"] == "null" for r in out_rows),
            "transition": sum(r["state"] == "transition" for r in out_rows),
            "recovered": sum(r["state"] == "recovered" for r in out_rows),
        },
        "largest_null_lambda": max(null_lambdas) if null_lambdas else None,
        "smallest_transition_lambda": min(transition_lambdas) if transition_lambdas else None,
        "smallest_recovered_lambda": min(recovered_lambdas) if recovered_lambdas else None,
        "auc_lambda_for_recovered": binary_auc(lam, recovered),
        "auc_log10_lambda4_for_recovered": binary_auc(x, recovered),
        "auc_lambda_for_transition_or_recovered": binary_auc(lam, transition_or_recovered),
        "metric_summary": metric_summary,
        "claim_boundary": (
            "Operational bridge only: TopK SAE recovery diagnostics rise with detector evidence, "
            "but this does not claim a full theory of SAE optimization or open-ended concept discovery."
        ),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "sae_facing_consequence_audit.csv", out_rows)
    with (outdir / "sae_facing_consequence_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"null": "#9AA0A6", "transition": "#F58518", "recovered": "#54A24B"}
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
        for state in ["null", "transition", "recovered"]:
            idx = np.array([r["state"] == state for r in out_rows])
            ax[0].scatter(
                x[idx],
                np.array([float(r["mean_top_label_atom_abs_auc_excess"]) for r in out_rows])[idx],
                s=28,
                label=state,
                color=colors[state],
            )
            ax[1].scatter(
                x[idx],
                np.array([float(r["mean_max_decoder_cosine"]) for r in out_rows])[idx],
                s=28,
                label=state,
                color=colors[state],
            )
        ax[0].axhline(0.05, color="0.6", lw=0.8, ls="--")
        ax[0].set_xlabel(r"$\log_{10}(\lambda_{\mathrm{eff}}^4)$")
        ax[0].set_ylabel("atom AUC excess")
        ax[1].axhline(0.8, color="0.6", lw=0.8, ls="--")
        ax[1].set_xlabel(r"$\log_{10}(\lambda_{\mathrm{eff}}^4)$")
        ax[1].set_ylabel("decoder cosine")
        ax[1].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(outdir / "sae_facing_consequence_audit.png", dpi=220)
        fig.savefig(outdir / "sae_facing_consequence_audit.pdf")
        plt.close(fig)
    except Exception as exc:
        summary["plot_error"] = repr(exc)
        with (outdir / "sae_facing_consequence_audit_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--by-config-csv",
        default="results/real_activation/c4_topk_phase_diagram_whitened_positive/controlled_sae_sweep_by_config.csv",
    )
    parser.add_argument("--outdir", default="results/real_activation/c16_sae_facing_consequence_audit")
    args = parser.parse_args()
    summarize(args)


if __name__ == "__main__":
    main()
