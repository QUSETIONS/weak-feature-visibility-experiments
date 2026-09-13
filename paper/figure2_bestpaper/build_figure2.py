#!/usr/bin/env python3
"""Build the raw-data Figure 2 package from the frozen E4-v2 artifact."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "source_data" / "e4v2_raw.json"
REFERENCE_SUMMARY_PATH = ROOT / "source_data" / "e4v2_summary.json"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

PRIMARY_STRENGTH = 0.55
EXPECTED_STRENGTHS = (0.35, 0.55, 0.80)
EXPECTED_SEEDS = tuple(range(20260824, 20260832))
RECOVERY_THRESHOLD = 0.80
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0

AXIS = "#2C5AA0"
DENSE = "#C45C26"
NEUTRAL = "#6F7378"
LIGHT_NEUTRAL = "#D7D9DC"
AGREE = "#B9C9E5"
MISMATCH = "#E9B38F"
STRENGTH_COLORS = {0.35: "#B9C4DF", 0.55: "#7566A8", 0.80: "#3D315F"}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.7,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_mean(values: list[float], rng: np.random.Generator) -> dict[str, object]:
    """Reproduce run_v2.py's seed-level percentile bootstrap exactly."""
    arr = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(arr), size=(BOOTSTRAP_RESAMPLES, len(arr)))
    means = arr[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(arr.mean()),
        "ci95": [float(low), float(high)],
        "n": int(len(arr)),
        "values": [float(x) for x in arr],
    }


def load_and_pair() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, dict[str, object]]]]:
    raw = json.loads(RAW_PATH.read_text())
    if len(raw) != 48:
        raise ValueError(f"Expected 48 raw rows, found {len(raw)}")

    required = {
        "seed",
        "geometry",
        "lambda_eff",
        "C",
        "class",
        "det_label",
        "det_cov",
        "det_ind",
        "decoder_cosine",
        "recovered",
    }
    for index, row in enumerate(raw):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Row {index} is missing {sorted(missing)}")

    seeds = tuple(sorted({int(row["seed"]) for row in raw}))
    strengths = tuple(sorted({float(row["lambda_eff"]) for row in raw}))
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"Unexpected seeds: {seeds}")
    if strengths != EXPECTED_STRENGTHS:
        raise ValueError(f"Unexpected strengths: {strengths}")

    indexed: dict[tuple[int, float], dict[str, dict[str, object]]] = {}
    for row in raw:
        key = (int(row["seed"]), float(row["lambda_eff"]))
        geometry = str(row["geometry"])
        if geometry in indexed.setdefault(key, {}):
            raise ValueError(f"Duplicate row for {key} / {geometry}")
        indexed[key][geometry] = row

    pairs: list[dict[str, object]] = []
    for seed in EXPECTED_SEEDS:
        for strength in EXPECTED_STRENGTHS:
            group = indexed[(seed, strength)]
            if set(group) != {"axis", "dense"}:
                raise ValueError(f"Incomplete pair for seed={seed}, strength={strength}")
            axis = group["axis"]
            dense = group["dense"]
            pair = {
                "seed": seed,
                "lambda_eff": strength,
                "axis_C": float(axis["C"]),
                "dense_C": float(dense["C"]),
                "axis_class": str(axis["class"]),
                "dense_class": str(dense["class"]),
                "axis_det_label": float(axis["det_label"]),
                "dense_det_label": float(dense["det_label"]),
                "axis_det_cov": float(axis["det_cov"]),
                "dense_det_cov": float(dense["det_cov"]),
                "axis_det_ind": float(axis["det_ind"]),
                "dense_det_ind": float(dense["det_ind"]),
                "axis_decoder_cosine": float(axis["decoder_cosine"]),
                "dense_decoder_cosine": float(dense["decoder_cosine"]),
                "axis_recovered": int(bool(axis["recovered"])),
                "dense_recovered": int(bool(dense["recovered"])),
                "delta_ind_dense_minus_axis": float(dense["det_ind"] - axis["det_ind"]),
                "delta_decoder_cosine_dense_minus_axis": float(
                    dense["decoder_cosine"] - axis["decoder_cosine"]
                ),
                "detector_class_flip": int(axis["class"] != dense["class"]),
                "recovery_agrees": int(bool(axis["recovered"]) == bool(dense["recovered"])),
            }
            pairs.append(pair)

    endpoint = [pair for pair in pairs if abs(float(pair["lambda_eff"]) - PRIMARY_STRENGTH) < 1e-12]
    if len(endpoint) != 8:
        raise ValueError(f"Expected 8 primary endpoint pairs, found {len(endpoint)}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    field_map = {
        "ind": "det_ind",
        "cov": "det_cov",
        "label": "det_label",
        "sae_recovery": "recovered",
        "sae_cosine": "decoder_cosine",
    }
    bootstrap: dict[str, dict[str, dict[str, object]]] = {"axis": {}, "dense": {}}
    for geometry in ("axis", "dense"):
        subset = [
            row
            for row in raw
            if row["geometry"] == geometry and abs(float(row["lambda_eff"]) - PRIMARY_STRENGTH) < 1e-12
        ]
        subset.sort(key=lambda row: int(row["seed"]))
        for metric, source_field in field_map.items():
            values = [float(row[source_field]) for row in subset]
            bootstrap[geometry][metric] = bootstrap_mean(values, rng)

    reference = json.loads(REFERENCE_SUMMARY_PATH.read_text())["lambda055"]
    for geometry in ("axis", "dense"):
        for metric in field_map:
            got = bootstrap[geometry][metric]
            expected = reference[geometry][metric]
            if not np.isclose(got["mean"], expected["mean"], atol=1e-15):
                raise ValueError(f"Mean mismatch for {geometry}/{metric}")
            if not np.allclose(got["ci95"], expected["ci95"], atol=1e-15):
                raise ValueError(f"Bootstrap CI mismatch for {geometry}/{metric}")

    return raw, pairs, bootstrap


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_data_files(
    raw: list[dict[str, object]],
    pairs: list[dict[str, object]],
    bootstrap: dict[str, dict[str, dict[str, object]]],
) -> dict[str, object]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(DATA_DIR / "figure2_raw_long_48.csv", raw)
    write_csv(DATA_DIR / "figure2_paired_24.csv", pairs)
    write_csv(
        DATA_DIR / "figure2_primary_endpoint_8.csv",
        [pair for pair in pairs if abs(float(pair["lambda_eff"]) - PRIMARY_STRENGTH) < 1e-12],
    )

    bootstrap_rows: list[dict[str, object]] = []
    for geometry in ("axis", "dense"):
        for metric in ("ind", "cov", "label", "sae_recovery", "sae_cosine"):
            result = bootstrap[geometry][metric]
            bootstrap_rows.append(
                {
                    "geometry": geometry,
                    "metric": metric,
                    "lambda_eff": PRIMARY_STRENGTH,
                    "n_seeds": result["n"],
                    "mean": result["mean"],
                    "ci95_low": result["ci95"][0],
                    "ci95_high": result["ci95"][1],
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_unit": "seed",
                    "interval": "percentile_2.5_97.5",
                }
            )
    write_csv(DATA_DIR / "figure2_bootstrap_summary.csv", bootstrap_rows)

    detector_flips = sum(int(pair["detector_class_flip"]) for pair in pairs)
    recovery_agreements = sum(int(pair["recovery_agrees"]) for pair in pairs)
    primary = [pair for pair in pairs if abs(float(pair["lambda_eff"]) - PRIMARY_STRENGTH) < 1e-12]
    audit = {
        "source_sha256": sha256(RAW_PATH),
        "raw_rows": len(raw),
        "paired_rows": len(pairs),
        "seeds": list(EXPECTED_SEEDS),
        "strengths": list(EXPECTED_STRENGTHS),
        "primary_strength": PRIMARY_STRENGTH,
        "primary_mean_delta_ind_dense_minus_axis": float(
            np.mean([float(pair["delta_ind_dense_minus_axis"]) for pair in primary])
        ),
        "primary_mean_delta_decoder_cosine_dense_minus_axis": float(
            np.mean([float(pair["delta_decoder_cosine_dense_minus_axis"]) for pair in primary])
        ),
        "detector_class_flips": detector_flips,
        "detector_class_pairs": len(pairs),
        "recovery_agreements": recovery_agreements,
        "recovery_pairs": len(pairs),
        "missing_pairs": 0,
        "excluded_rows": 0,
        "bootstrap_reproduces_frozen_summary": True,
    }
    (DATA_DIR / "figure2_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def paired_endpoint_panel(
    ax: plt.Axes,
    endpoint: list[dict[str, object]],
    bootstrap: dict[str, dict[str, dict[str, object]]],
    axis_field: str,
    dense_field: str,
    metric: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    threshold: float | None = None,
) -> None:
    jitter = np.linspace(-0.035, 0.035, len(endpoint))
    for offset, pair in zip(jitter, endpoint):
        y0 = float(pair[axis_field])
        y1 = float(pair[dense_field])
        ax.plot([offset, 1 + offset], [y0, y1], color=LIGHT_NEUTRAL, lw=0.8, zorder=1)
        ax.scatter(offset, y0, s=19, color=AXIS, edgecolor="white", linewidth=0.35, zorder=3)
        ax.scatter(1 + offset, y1, s=19, color=DENSE, edgecolor="white", linewidth=0.35, zorder=3)

    for x, geometry, color in ((0, "axis", AXIS), (1, "dense", DENSE)):
        result = bootstrap[geometry][metric]
        mean = float(result["mean"])
        low, high = map(float, result["ci95"])
        ax.errorbar(
            x,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.55,
            markersize=5.2,
            elinewidth=1.2,
            capsize=3,
            zorder=5,
        )

    if threshold is not None:
        ax.axhline(threshold, color=NEUTRAL, lw=0.8, ls="--", zorder=0)
        ax.text(1.16, threshold + 0.008, "frozen threshold", color=NEUTRAL, fontsize=6.2, ha="right")
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 1], ["axis", "dense"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=5, fontweight="bold")
    ax.grid(axis="y", color="#ECEDEF", lw=0.55, zorder=0)


def build_figure(
    pairs: list[dict[str, object]],
    bootstrap: dict[str, dict[str, dict[str, object]]],
    audit: dict[str, object],
) -> None:
    endpoint = [pair for pair in pairs if abs(float(pair["lambda_eff"]) - PRIMARY_STRENGTH) < 1e-12]
    endpoint.sort(key=lambda pair: int(pair["seed"]))

    fig = plt.figure(figsize=(7.0, 6.0))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.02, 1.44, 0.88],
        left=0.085,
        right=0.985,
        top=0.965,
        bottom=0.075,
        hspace=0.54,
        wspace=0.34,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    paired_endpoint_panel(
        ax_a,
        endpoint,
        bootstrap,
        "axis_det_ind",
        "dense_det_ind",
        "ind",
        r"Pre-specified endpoint: detector visibility ($\lambda_{\rm eff}=0.55$)",
        "independence power",
        (0.0, 1.04),
    )
    panel_label(ax_a, "a")
    ax_a.text(
        0.5,
        0.52,
        "0.081  →  1.000",
        transform=ax_a.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#25282B",
    )

    ax_b = fig.add_subplot(grid[0, 1])
    paired_endpoint_panel(
        ax_b,
        endpoint,
        bootstrap,
        "axis_decoder_cosine",
        "dense_decoder_cosine",
        "sae_cosine",
        "Same runs: free TopK decoder",
        "decoder cosine",
        (0.78, 1.005),
        threshold=RECOVERY_THRESHOLD,
    )
    panel_label(ax_b, "b")
    ax_b.text(
        0.5,
        0.17,
        "8/8 recovered       8/8 recovered",
        transform=ax_b.transAxes,
        ha="center",
        color="#25282B",
        fontsize=6.6,
    )

    ax_c = fig.add_subplot(grid[1, :])
    for seed in EXPECTED_SEEDS:
        seed_pairs = sorted(
            [pair for pair in pairs if int(pair["seed"]) == seed],
            key=lambda pair: float(pair["lambda_eff"]),
        )
        ax_c.plot(
            [float(pair["delta_ind_dense_minus_axis"]) for pair in seed_pairs],
            [float(pair["delta_decoder_cosine_dense_minus_axis"]) for pair in seed_pairs],
            color=LIGHT_NEUTRAL,
            lw=0.75,
            zorder=1,
        )
    for strength in EXPECTED_STRENGTHS:
        subset = [pair for pair in pairs if abs(float(pair["lambda_eff"]) - strength) < 1e-12]
        ax_c.scatter(
            [float(pair["delta_ind_dense_minus_axis"]) for pair in subset],
            [float(pair["delta_decoder_cosine_dense_minus_axis"]) for pair in subset],
            s=28,
            color=STRENGTH_COLORS[strength],
            edgecolor="white",
            linewidth=0.45,
            label=rf"$\lambda_{{\rm eff}}={strength:.2f}$",
            zorder=3,
        )

    primary_x = float(audit["primary_mean_delta_ind_dense_minus_axis"])
    primary_y = float(audit["primary_mean_delta_decoder_cosine_dense_minus_axis"])
    ax_c.scatter(primary_x, primary_y, marker="*", s=92, color="#111111", zorder=5)
    primary_annotation = (
        "pre-specified mean\n"
        + rf"$\Delta_{{\rm ind}}={primary_x:.3f}$, $\Delta_{{\rm cos}}={primary_y:.3f}$"
    )
    ax_c.annotate(
        primary_annotation,
        xy=(primary_x, primary_y),
        xytext=(0.785, 0.175),
        textcoords="data",
        arrowprops={"arrowstyle": "-", "lw": 0.75, "color": NEUTRAL},
        fontsize=6.7,
        ha="left",
        va="bottom",
    )
    ax_c.axhline(0.0, color=NEUTRAL, lw=0.8, ls="--", zorder=0)
    ax_c.set_xlabel(r"detector split, $\Delta_{\rm ind}={\rm dense}-{\rm axis}$")
    ax_c.set_ylabel(r"decoder split, $\Delta_{\rm cos}={\rm dense}-{\rm axis}$")
    ax_c.set_title(
        "Detector–decoder decoupling across 24 matched seed × strength pairs",
        loc="left",
        pad=5,
        fontweight="bold",
    )
    ax_c.grid(color="#ECEDEF", lw=0.5, zorder=0)
    ax_c.legend(loc="upper left", ncol=3, handletextpad=0.35, columnspacing=1.0)
    panel_label(ax_c, "c", x=-0.06, y=1.07)

    ax_d = fig.add_subplot(grid[2, :])
    ordered_pairs = sorted(pairs, key=lambda pair: (int(pair["seed"]), float(pair["lambda_eff"])))
    for row_index, seed in enumerate(EXPECTED_SEEDS):
        for col_index, strength in enumerate(EXPECTED_STRENGTHS):
            pair = next(
                pair
                for pair in ordered_pairs
                if int(pair["seed"]) == seed and abs(float(pair["lambda_eff"]) - strength) < 1e-12
            )
            agrees = bool(pair["recovery_agrees"])
            ax_d.add_patch(
                Rectangle(
                    (col_index - 0.38, row_index - 0.36),
                    0.76,
                    0.72,
                    facecolor=AGREE if agrees else MISMATCH,
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
            ax_d.text(
                col_index,
                row_index,
                "same" if agrees else "diff",
                ha="center",
                va="center",
                fontsize=5.8,
                color="#223047" if agrees else "#7B321D",
                fontweight="bold",
            )
    ax_d.set_xlim(-0.55, 4.15)
    ax_d.set_ylim(7.55, -0.55)
    ax_d.set_xticks(range(3), [rf"$\lambda_{{\rm eff}}={value:.2f}$" for value in EXPECTED_STRENGTHS])
    ax_d.set_yticks(range(8), [str(seed) for seed in EXPECTED_SEEDS])
    ax_d.set_ylabel("seed")
    ax_d.set_title(
        "Paired descriptive audit across all 24 seed × strength pairs",
        loc="left",
        pad=5,
        fontweight="bold",
    )
    ax_d.spines[:].set_visible(False)
    ax_d.tick_params(length=0)
    ax_d.legend(
        handles=[
            Line2D([0], [0], marker="s", color="none", markerfacecolor=AGREE, markeredgecolor="none", label="recovery agrees"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=MISMATCH, markeredgecolor="none", label="recovery differs"),
        ],
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
    )
    ax_d.text(2.72, 2.2, "detector class\n24/24 flips", fontsize=8.0, fontweight="bold", color="#25282B")
    ax_d.text(2.72, 4.8, "binary recovery\n22/24 same", fontsize=8.0, fontweight="bold", color="#25282B")
    panel_label(ax_d, "d", x=-0.06, y=1.10)

    pdf_metadata = {
        "Title": "A detector-class split does not transfer to matched TopK recovery",
        "Subject": "Raw-data E4-v2 Figure 2",
        "Creator": "Python/matplotlib; source data frozen in package",
    }
    svg_metadata = {
        "Title": "A detector-class split does not transfer to matched TopK recovery",
        "Description": "Raw-data E4-v2 Figure 2",
        "Creator": "Python/matplotlib; source data frozen in package",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "figure2_detector_decoder_decoupling"
    fig.savefig(stem.with_suffix(".svg"), metadata=svg_metadata)
    fig.savefig(stem.with_suffix(".pdf"), metadata=pdf_metadata)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    raw, pairs, bootstrap = load_and_pair()
    audit = write_data_files(raw, pairs, bootstrap)
    build_figure(pairs, bootstrap, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
