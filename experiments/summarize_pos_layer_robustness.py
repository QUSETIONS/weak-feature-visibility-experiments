"""Summarize C12 POS-tagger cross-layer robustness results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "real_activation" / "c12_pos_tagger_layer_robustness"


LAYER_RUNS = {
    4: ROOT / "results" / "real_activation" / "c12_pos_tagger_layer4_10k" / "natural_pos_tagger_feature_probe_summary.json",
    6: ROOT / "results" / "real_activation" / "c11_natural_pos_tagger_feature_probe" / "natural_pos_tagger_feature_probe_summary.json",
    8: ROOT / "results" / "real_activation" / "c12_pos_tagger_layer8_10k" / "natural_pos_tagger_feature_probe_summary.json",
    10: ROOT / "results" / "real_activation" / "c12_pos_tagger_layer10_10k" / "natural_pos_tagger_feature_probe_summary.json",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer, path in LAYER_RUNS.items():
        if not path.exists():
            continue
        s = read_json(path)
        rows.append(
            {
                "layer": layer,
                "events_evaluated": s["events_evaluated"],
                "tokens": s["tokens"],
                "median_abs_label_z": s["median_abs_label_z"],
                "median_abs_full_cov_z": s["median_abs_full_cov_z"],
                "median_abs_offdiag_cov_z": s["median_abs_offdiag_cov_z"],
                "frac_label_z_gt_5": s["frac_label_z_gt_5"],
                "frac_offdiag_z_gt_5": s["frac_offdiag_z_gt_5"],
                "corr_log_evidence_abs_offdiag_z": s["corr_log_evidence_abs_offdiag_z"],
            }
        )
    if not rows:
        raise RuntimeError("No layer summaries found.")

    with (OUTDIR / "pos_layer_robustness.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {
        "mode": "pos_tagger_layer_robustness",
        "layers": [r["layer"] for r in rows],
        "runs": rows,
        "min_frac_label_z_gt_5": min(r["frac_label_z_gt_5"] for r in rows),
        "min_frac_offdiag_z_gt_5": min(r["frac_offdiag_z_gt_5"] for r in rows),
        "min_median_abs_offdiag_cov_z": min(r["median_abs_offdiag_cov_z"] for r in rows),
        "min_corr_log_evidence_abs_offdiag_z": min(r["corr_log_evidence_abs_offdiag_z"] for r in rows),
    }
    with (OUTDIR / "pos_layer_robustness_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        layers = [r["layer"] for r in rows]
        fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.3))
        ax[0].plot(layers, [r["median_abs_label_z"] for r in rows], marker="o", label="label")
        ax[0].plot(layers, [r["median_abs_offdiag_cov_z"] for r in rows], marker="o", label="offdiag")
        ax[0].set_xlabel("GPT-2 layer")
        ax[0].set_ylabel("median |z|")
        ax[0].legend(frameon=False, fontsize=7)
        ax[1].plot(layers, [r["frac_offdiag_z_gt_5"] for r in rows], marker="o")
        ax[1].set_xlabel("GPT-2 layer")
        ax[1].set_ylabel("offdiag frac |z|>5")
        ax[1].set_ylim(0, 1)
        ax[2].plot(layers, [r["corr_log_evidence_abs_offdiag_z"] for r in rows], marker="o")
        ax[2].set_xlabel("GPT-2 layer")
        ax[2].set_ylabel("evidence corr")
        ax[2].set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(OUTDIR / "pos_layer_robustness.png", dpi=220)
        fig.savefig(OUTDIR / "pos_layer_robustness.pdf")
        plt.close(fig)
    except Exception as exc:
        aggregate["plot_error"] = repr(exc)
        with (OUTDIR / "pos_layer_robustness_summary.json").open("w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2, sort_keys=True)

    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
