"""Audit geometry coefficients for externally defined natural events.

This script summarizes the geometry term 1 - ||w||_4^4 in the C13
pre-registered natural-feature bank.  Its purpose is to separate two claims:
the axis-aligned controlled injection is a detector-class stress test, while
the natural events occupy the high-geometry regime where visibility is driven
primarily by evidence strength rather than by near-axis degeneracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def audit(args: argparse.Namespace) -> dict[str, object]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with Path(args.bank_csv).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("No rows loaded from feature bank.")

    for row in rows:
        row["geometry_value"] = float(row["geometry"])
        row["l4_fourth_power"] = 1.0 - row["geometry_value"]
        row["abs_offdiag_z"] = abs(float(row["offdiag_cov_z"]))
        row["log10_evidence"] = float(row["log10_evidence_proxy"])

    geometry = np.asarray([r["geometry_value"] for r in rows], dtype=float)
    offdiag = np.asarray([r["abs_offdiag_z"] for r in rows], dtype=float)
    evidence = np.asarray([r["log10_evidence"] for r in rows], dtype=float)

    family_rows: list[dict[str, object]] = []
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(float(row["geometry_value"]))
    for family, vals in sorted(by_family.items()):
        arr = np.asarray(vals, dtype=float)
        stat = summarize(arr)
        family_rows.append(
            {
                "family": family,
                "events": int(arr.size),
                **{f"geometry_{k}": v for k, v in stat.items()},
                "near_axis_geometry_lt_0p9": int(np.sum(arr < 0.9)),
                "near_axis_geometry_lt_0p99": int(np.sum(arr < 0.99)),
            }
        )
    write_csv(outdir / "natural_feature_geometry_by_family.csv", family_rows)

    event_rows = [
        {
            "bank_event": row["bank_event"],
            "family": row["family"],
            "event": row["event"],
            "geometry": float(row["geometry_value"]),
            "l4_fourth_power": float(row["l4_fourth_power"]),
            "abs_offdiag_z": float(row["abs_offdiag_z"]),
            "log10_evidence_proxy": float(row["log10_evidence"]),
        }
        for row in sorted(rows, key=lambda r: float(r["geometry_value"]))
    ]
    write_csv(outdir / "natural_feature_geometry_by_event.csv", event_rows)

    summary = {
        "mode": "natural_feature_geometry_audit",
        "bank_csv": args.bank_csv,
        "events": int(len(rows)),
        "families": sorted(by_family),
        "geometry_summary": summarize(geometry),
        "near_axis_geometry_lt_0p9": int(np.sum(geometry < 0.9)),
        "near_axis_geometry_lt_0p99": int(np.sum(geometry < 0.99)),
        "corr_geometry_abs_offdiag_z": pearson(geometry, offdiag),
        "corr_geometry_log10_evidence_proxy": pearson(geometry, evidence),
        "family_csv": str(outdir / "natural_feature_geometry_by_family.csv"),
        "event_csv": str(outdir / "natural_feature_geometry_by_event.csv"),
    }
    with (outdir / "natural_feature_geometry_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    try:
        import matplotlib

        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.4, 2.8))
        ax.hist(geometry, bins=12, color="#2563eb", alpha=0.82, edgecolor="white")
        ax.axvline(1.0, color="0.25", lw=1, ls="--")
        ax.axvline(float(np.median(geometry)), color="#b91c1c", lw=1.2)
        ax.set_xlabel(r"Geometry coefficient $1-\|w\|_4^4$")
        ax.set_ylabel("Natural events")
        ax.set_title("Natural-event directions are far from the axis blind spot")
        fig.tight_layout()
        fig.savefig(outdir / "natural_feature_geometry_audit.pdf")
        fig.savefig(outdir / "natural_feature_geometry_audit.png", dpi=220)
        plt.close(fig)
    except Exception as exc:
        summary["plot_error"] = str(exc)
        with (outdir / "natural_feature_geometry_audit_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank-csv",
        default="results/real_activation/c13_preregistered_feature_bank/preregistered_feature_bank.csv",
    )
    parser.add_argument("--outdir", default="results/real_activation/c18_natural_feature_geometry_audit")
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
