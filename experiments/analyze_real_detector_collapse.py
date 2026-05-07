"""Analyze real-activation detector evidence-collapse runs.

Inputs are one or more ``injection_probe.csv`` files from
``real_activation_bridge.py inject``. The output is a compact summary for the
paper mainline: whether GPT-2 controlled injections are organized by the
predicted evidence coordinate N C_i lambda^4.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                out: dict[str, object] = {"source": str(path)}
                for k, v in row.items():
                    if k == "direction":
                        out[k] = v
                    else:
                        out[k] = float(v)
                rows.append(out)
    return rows


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    outdir = Path(args.outdir)
    rows = read_rows(paths)
    if not rows:
        raise SystemExit("no rows")

    nonaxis = [r for r in rows if str(r["direction"]) != "axis" and float(r["geometry"]) > 1e-6]
    evidence = np.array([float(r["evidence_proxy"]) for r in nonaxis])
    log_evidence = np.log10(np.maximum(evidence, 1e-12))
    unlabeled = np.array([float(r["unlabeled_power"]) for r in nonaxis])
    labeled = np.array([float(r["labeled_power"]) for r in nonaxis])
    lam = np.array([float(r["lambda_eff"]) for r in nonaxis])
    nvals = np.array([float(r["n"]) for r in nonaxis])
    geom = np.array([float(r["geometry"]) for r in nonaxis])

    # Equal-frequency bins over the evidence coordinate.
    order = np.argsort(log_evidence)
    bins = np.array_split(order, min(8, len(order)))
    bin_rows = []
    for idx in bins:
        if len(idx) == 0:
            continue
        group = [nonaxis[int(i)] for i in idx]
        bin_rows.append(
            {
                "rows": len(group),
                "log10_evidence_min": float(log_evidence[idx].min()),
                "log10_evidence_max": float(log_evidence[idx].max()),
                "mean_evidence": float(np.mean(evidence[idx])),
                "mean_unlabeled_power": float(np.mean(unlabeled[idx])),
                "mean_labeled_power": float(np.mean(labeled[idx])),
            }
        )

    direction_rows = []
    for direction in sorted({str(r["direction"]) for r in rows}):
        group = [r for r in rows if str(r["direction"]) == direction]
        direction_rows.append(
            {
                "direction": direction,
                "rows": len(group),
                "mean_geometry": float(np.mean([float(r["geometry"]) for r in group])),
                "mean_labeled_power": float(np.mean([float(r["labeled_power"]) for r in group])),
                "mean_unlabeled_power": float(np.mean([float(r["unlabeled_power"]) for r in group])),
                "max_unlabeled_power": float(np.max([float(r["unlabeled_power"]) for r in group])),
            }
        )

    summary = {
        "mode": "real_detector_collapse",
        "inputs": [str(p) for p in paths],
        "rows": len(rows),
        "nonaxis_rows": len(nonaxis),
        "corr_unlabeled_power_log10_evidence": corr(log_evidence, unlabeled),
        "corr_labeled_power_log10_evidence": corr(log_evidence, labeled),
        "corr_unlabeled_power_lambda_eff": corr(lam, unlabeled),
        "corr_unlabeled_power_n": corr(nvals, unlabeled),
        "corr_unlabeled_power_geometry": corr(geom, unlabeled),
        "axis_mean_unlabeled_power": float(np.mean([float(r["unlabeled_power"]) for r in rows if str(r["direction"]) == "axis"])),
        "axis_mean_labeled_power": float(np.mean([float(r["labeled_power"]) for r in rows if str(r["direction"]) == "axis"])),
        "max_unlabeled_power_nonaxis": float(np.max(unlabeled)),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "real_detector_collapse_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_csv(outdir / "real_detector_collapse_bins.csv", bin_rows)
    write_csv(outdir / "real_detector_collapse_by_direction.csv", direction_rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        directions = sorted({str(r["direction"]) for r in nonaxis})
        for direction in directions:
            idx = np.array([str(r["direction"]) == direction for r in nonaxis])
            ax.scatter(
                log_evidence[idx],
                unlabeled[idx],
                s=14,
                alpha=0.45,
                label=direction,
                linewidths=0,
            )
        bin_x = [0.5 * (r["log10_evidence_min"] + r["log10_evidence_max"]) for r in bin_rows]
        bin_y = [r["mean_unlabeled_power"] for r in bin_rows]
        ax.plot(bin_x, bin_y, color="black", marker="o", linewidth=2.0, label="binned mean")
        ax.axhline(0.05, color="0.4", linestyle="--", linewidth=1.0, label="alpha=0.05")
        ax.set_xlabel(r"$\log_{10}(N C_i \lambda^4)$")
        ax.set_ylabel("unlabeled detection power")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("GPT-2 activation bridge: evidence collapse")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(outdir / "real_detector_evidence_collapse.pdf")
        fig.savefig(outdir / "real_detector_evidence_collapse.png", dpi=220)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), gridspec_kw={"width_ratios": [1.35, 1.0]})
        ax = axes[0]
        for direction in directions:
            idx = np.array([str(r["direction"]) == direction for r in nonaxis])
            ax.scatter(
                log_evidence[idx],
                unlabeled[idx],
                s=11,
                alpha=0.38,
                label=direction,
                linewidths=0,
            )
        ax.plot(bin_x, bin_y, color="black", marker="o", markersize=4, linewidth=2.0, label="binned mean")
        ax.axhline(0.05, color="0.4", linestyle="--", linewidth=1.0)
        ax.set_xlabel(r"$\log_{10}(N C_i \lambda^4)$")
        ax.set_ylabel("unlabeled power")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("(a) Evidence collapse")
        ax.legend(frameon=False, fontsize=6, ncol=2, loc="lower right")

        ax = axes[1]
        label_map = {
            "axis": "axis",
            "dense": "dense",
            "ksparse": "8-sparse",
            "pc1": "PC1",
            "pc_orth": "PC-orth",
        }
        ordered = ["axis", "dense", "ksparse", "pc1", "pc_orth"]
        by_dir = {r["direction"]: r for r in direction_rows}
        x_pos = np.arange(len(ordered))
        width = 0.38
        labeled_vals = [by_dir[d]["mean_labeled_power"] for d in ordered]
        unlabeled_vals = [by_dir[d]["mean_unlabeled_power"] for d in ordered]
        ax.bar(x_pos - width / 2, labeled_vals, width=width, color="#4C78A8", label="labeled")
        ax.bar(x_pos + width / 2, unlabeled_vals, width=width, color="#F58518", label="unlabeled")
        ax.axhline(0.05, color="0.4", linestyle="--", linewidth=1.0)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([label_map[d] for d in ordered], rotation=25, ha="right")
        ax.set_ylabel("mean power")
        ax.set_ylim(0, 1.05)
        ax.set_title("(b) Geometry blind spot")
        ax.legend(frameon=False, fontsize=7, loc="lower right")

        fig.tight_layout()
        fig.savefig(outdir / "real_detector_bridge_main.pdf")
        fig.savefig(outdir / "real_detector_bridge_main.png", dpi=240)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is best-effort.
        print(f"[warn] plotting failed: {exc}")

    lines = [
        "# Real-Activation Detector Collapse",
        "",
        f"- Rows: `{summary['rows']}` total, `{summary['nonaxis_rows']}` non-axis.",
        f"- Corr(unlabeled power, log10 evidence): `{summary['corr_unlabeled_power_log10_evidence']:.3f}`.",
        f"- Corr(unlabeled power, lambda_eff): `{summary['corr_unlabeled_power_lambda_eff']:.3f}`.",
        f"- Corr(unlabeled power, N): `{summary['corr_unlabeled_power_n']:.3f}`.",
        f"- Axis mean unlabeled power: `{summary['axis_mean_unlabeled_power']:.3f}`.",
        f"- Axis mean labeled power: `{summary['axis_mean_labeled_power']:.3f}`.",
        f"- Max non-axis unlabeled power: `{summary['max_unlabeled_power_nonaxis']:.3f}`.",
        f"- Figure: `{outdir / 'real_detector_bridge_main.pdf'}`.",
        "",
        "## Interpretation",
        "",
        "This is a bridge result: it tests whether GPT-2 controlled injections are ordered by the theorem's evidence coordinate, not whether SAE training itself is the theorem.",
    ]
    (outdir / "real_detector_collapse_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
