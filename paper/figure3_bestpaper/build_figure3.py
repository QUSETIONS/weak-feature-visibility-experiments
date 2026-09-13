#!/usr/bin/env python3
"""Build the matched-compute spectral warm-start figure from raw observations."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
WARM_RAW = ROOT / "experiments/results_spectral_warm_start_matched/warm_start_raw.json"
WARM_SUMMARY = ROOT / "experiments/results_spectral_warm_start_matched/warm_start_summary.json"
READOUT_SUMMARY = ROOT / "experiments/results_spectral_anchor_readout_audit/readout_summary.json"
OUT = HERE / "output"
DATA = HERE / "data"

OBJECTIVES = ["mse", "mse_l1", "mse_l2"]
CAPACITIES = ["d16-k2", "d32-k4"]
VARIANTS = ["vanilla-random", "spectral-warm-start"]
OBJECTIVE_LABELS = {"mse": "MSE", "mse_l1": r"MSE+$\ell_1$", "mse_l2": r"MSE+$\ell_2$"}
CAPACITY_LABELS = {"d16-k2": "$d=16, k=2$", "d32-k4": "$d=32, k=4$"}

COLORS = {
    "vanilla": "#8A9199",
    "warm": "#2B7A78",
    "d16": "#2F6690",
    "d32": "#D17A22",
    "recovery": "#35618A",
    "certificate": "#2B7A78",
    "grid": "#D8DDE3",
}


def load_and_validate():
    rows = json.loads(WARM_RAW.read_text())
    summary = json.loads(WARM_SUMMARY.read_text())
    readout = json.loads(READOUT_SUMMARY.read_text())

    assert len(rows) == 96
    assert summary["complete"] and summary["pair_integrity"] and summary["evidence_eligible"]
    assert summary["pairs"] == 48
    assert readout["complete"] and readout["shared_model_integrity"] and readout["evidence_eligible"]

    paired = defaultdict(dict)
    for row in rows:
        paired[row["pair_id"]][row["variant_id"]] = row
    assert len(paired) == 48
    for pair_id, variants in paired.items():
        assert set(variants) == set(VARIANTS), pair_id
        a, b = variants[VARIANTS[0]], variants[VARIANTS[1]]
        for field in ("objective_id", "capacity_id", "geometry", "seed", "train_seed", "data_seed", "validation_seed"):
            assert a[field] == b[field], (pair_id, field)

    for variant in VARIANTS:
        subset = [r for r in rows if r["variant_id"] == variant]
        reported = summary["variant_summary"][variant]
        assert np.isclose(np.mean([r["recovered"] for r in subset]), reported["recovery_rate"])
        assert np.isclose(np.mean([r["certificate"]["certified"] for r in subset]), reported["certificate_pass_rate"])

    return rows, summary, readout, paired


def export_source_data(rows, readout):
    DATA.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id",
        "variant_id",
        "objective_id",
        "capacity_id",
        "geometry",
        "seed",
        "train_seed",
        "data_seed",
        "validation_seed",
        "target_cosine",
        "recovered",
        "certificate_pass",
        "certificate_cosine_lower_bound",
        "objective_loss",
    ]
    with (DATA / "figure3_matched_96.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "pair_id": row["pair_id"],
                    "variant_id": row["variant_id"],
                    "objective_id": row["objective_id"],
                    "capacity_id": row["capacity_id"],
                    "geometry": row["geometry"],
                    "seed": row["seed"],
                    "train_seed": row["train_seed"],
                    "data_seed": row["data_seed"],
                    "validation_seed": row["validation_seed"],
                    "target_cosine": row["target_cosine"],
                    "recovered": int(row["recovered"]),
                    "certificate_pass": int(row["certificate"]["certified"]),
                    "certificate_cosine_lower_bound": row["certificate"]["cosine_lower_bound"],
                    "objective_loss": row["objective_loss"],
                }
            )

    with (DATA / "figure3_readout_3.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["readout", "kind", "rows", "recovered", "certified", "mean_target_cosine", "minimum_target_cosine"])
        for name, values in readout["readout_summary"].items():
            writer.writerow(
                [
                    name,
                    values["kind"],
                    values["rows"],
                    round(values["recovery_rate"] * values["rows"]),
                    round(values["certificate_pass_rate"] * values["rows"]),
                    values["mean_target_cosine"],
                    values["minimum_target_cosine"],
                ]
            )


def panel_label(ax, label):
    ax.text(-0.27, 1.10, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top")


def build_figure(rows, summary, readout, paired):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.0, 2.15))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.20, 0.92], wspace=0.58)

    # a: all 48 paired observations, with no aggregation or exclusions.
    ax = fig.add_subplot(grid[0, 0])
    for capacity, color, marker, label in (
        ("d16-k2", COLORS["d16"], "o", CAPACITY_LABELS["d16-k2"]),
        ("d32-k4", COLORS["d32"], "s", CAPACITY_LABELS["d32-k4"]),
    ):
        x, y = [], []
        for variants in paired.values():
            vanilla = variants["vanilla-random"]
            warm = variants["spectral-warm-start"]
            if vanilla["capacity_id"] == capacity:
                x.append(vanilla["target_cosine"])
                y.append(warm["target_cosine"])
        ax.scatter(x, y, s=22, marker=marker, facecolor=color, edgecolor="white", linewidth=0.45, alpha=0.88, label=label, zorder=3)
    ax.plot([0, 1], [0, 1], color="#6F7780", lw=0.8, ls="--", zorder=1)
    ax.axhline(0.8, color=COLORS["grid"], lw=0.8, zorder=0)
    ax.axvline(0.8, color=COLORS["grid"], lw=0.8, zorder=0)
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xlabel="vanilla target cosine", ylabel="warm-start target cosine")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Paired target cosine (48 conditions)", pad=4)
    ax.text(0.05, 0.71, "44 improve\n4 decline", transform=ax.transAxes, va="top", fontsize=6.3)
    ax.legend(loc="lower right", handletextpad=0.35, borderpad=0.15)
    panel_label(ax, "a")

    # b: exact certificate counts for each objective x capacity cell (8 rows/cell).
    ax = fig.add_subplot(grid[0, 1])
    labels, vanilla_rates, warm_rates, vanilla_counts, warm_counts = [], [], [], [], []
    for objective in OBJECTIVES:
        for capacity in CAPACITIES:
            labels.append(f"{OBJECTIVE_LABELS[objective]}  {CAPACITY_LABELS[capacity]}")
            cell = [r for r in rows if r["objective_id"] == objective and r["capacity_id"] == capacity]
            vc = sum(r["certificate"]["certified"] for r in cell if r["variant_id"] == "vanilla-random")
            wc = sum(r["certificate"]["certified"] for r in cell if r["variant_id"] == "spectral-warm-start")
            vanilla_counts.append(vc)
            warm_counts.append(wc)
            vanilla_rates.append(vc / 8)
            warm_rates.append(wc / 8)
    y = np.arange(len(labels))[::-1]
    for yi, v, w in zip(y, vanilla_rates, warm_rates):
        ax.plot([v, w], [yi, yi], color="#C6CBD1", lw=2.2, solid_capstyle="round", zorder=1)
    ax.scatter(vanilla_rates, y, s=28, color=COLORS["vanilla"], edgecolor="white", linewidth=0.45, label="vanilla", zorder=3)
    ax.scatter(warm_rates, y, s=28, color=COLORS["warm"], edgecolor="white", linewidth=0.45, label="spectral warm start", zorder=3)
    for yi, v, w, vc, wc in zip(y, vanilla_rates, warm_rates, vanilla_counts, warm_counts):
        ax.text(v, yi + 0.20, f"{vc}/8", ha="center", va="bottom", fontsize=5.6, color="#626A73")
        ax.text(w, yi - 0.20, f"{wc}/8", ha="center", va="top", fontsize=5.6, color="#1E625F")
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("held-out certificate pass rate")
    ax.set_title(r"Certificate passes (gray $\rightarrow$ teal): 19/48 $\rightarrow$ 44/48", pad=4)
    ax.grid(axis="x", color="#E6E9ED", lw=0.6)
    panel_label(ax, "b")

    # c: target-free readouts from the same 48 warm-start models.
    ax = fig.add_subplot(grid[0, 2])
    order = ["usage-atom", "spectral-linked-atom", "signed-pair-span"]
    labels = ["usage\natom", "linked\natom", "pair\nspan"]
    recovered, certified = [], []
    for key in order:
        item = readout["readout_summary"][key]
        recovered.append(round(item["recovery_rate"] * item["rows"]))
        certified.append(round(item["certificate_pass_rate"] * item["rows"]))
    x = np.arange(len(order))
    width = 0.34
    b1 = ax.bar(x - width / 2, recovered, width, color=COLORS["recovery"], label="recovered")
    b2 = ax.bar(x + width / 2, certified, width, color=COLORS["certificate"], hatch="///", edgecolor="white", linewidth=0.4, label="certified")
    for xi, value in zip(x - width / 2, recovered):
        ax.text(xi, value - 2.1, f"{value}/48", ha="center", va="top", fontsize=5.7, color="white", fontweight="bold")
    for xi, value in zip(x + width / 2, certified):
        ax.text(xi, value + 0.8, f"{value}/48", ha="center", va="bottom", fontsize=5.7, color="#1E625F")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 52)
    ax.set_yticks([0, 24, 48])
    ax.set_ylabel("conditions passing")
    ax.set_title("Readouts (blue: rec.; hatched: cert.)", pad=4)
    ax.axhline(48, color=COLORS["grid"], lw=0.8, zorder=0)
    panel_label(ax, "c")

    fig.subplots_adjust(left=0.065, right=0.995, top=0.86, bottom=0.30)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure3_spectral_warm_start.svg", bbox_inches="tight")
    fig.savefig(OUT / "figure3_spectral_warm_start.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure3_spectral_warm_start.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "figure3_spectral_warm_start.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    rows, summary, readout, paired = load_and_validate()
    export_source_data(rows, readout)
    build_figure(rows, summary, readout, paired)


if __name__ == "__main__":
    main()
