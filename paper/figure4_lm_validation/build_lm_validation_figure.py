#!/usr/bin/env python3
"""Build the real-residual detector and matched warm-start validation figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

BRIDGE_PATHS = {
    "GPT-2 L6": ROOT / "experiments/results_gpt2_corrected_20260910/gpt2_summary.json",
    "Pythia L3": ROOT / "experiments/results_pythia_corrected_20260910/pythia_summary.json",
}
MATCHED_ROOT = ROOT / "experiments/results_lm_spectral_warm_start_matched"
MATCHED_PATHS = {
    "GPT-2 L6": MATCHED_ROOT / "gpt2-small-l6/matched_raw.json",
    "Pythia L3": MATCHED_ROOT / "pythia-70m-l3/matched_raw.json",
}
MATCHED_SUMMARY = MATCHED_ROOT / "matched_summary_all_models.json"
MODEL_IDS = {"GPT-2 L6": "gpt2-small-l6", "Pythia L3": "pythia-70m-l3"}

VANILLA = "#8B97A6"
WARM = "#007C91"
GPT2 = "#3E6FB0"
PYTHIA = "#C66A3D"
INK = "#243142"
GRID = "#D9E0E6"
PALE = LinearSegmentedColormap.from_list("pale_power", ["#F5F7F9", "#CFE0E9", "#4C89A1"])

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.6,
    "axes.labelsize": 6.7,
    "axes.titlesize": 7.4,
    "legend.fontsize": 6.0,
    "xtick.labelsize": 6.1,
    "ytick.labelsize": 6.1,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def load_json(path: Path):
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty source data: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(values: list[float], seed: int, resamples: int = 10000) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.10, label, transform=ax.transAxes, fontsize=8.2, fontweight="bold", color=INK, va="top")


def annotate_matrix(ax, values: np.ndarray, formatter) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = float(values[row, col])
            color = "white" if value >= 0.66 else INK
            ax.text(col, row, formatter(value, row, col), ha="center", va="center", fontsize=5.6, color=color)


def build_source_data():
    bridge_summaries = {name: load_json(path) for name, path in BRIDGE_PATHS.items()}
    matched_rows = {name: load_json(path) for name, path in MATCHED_PATHS.items()}
    summary = load_json(MATCHED_SUMMARY)

    bridge_csv = []
    for model_name, result in bridge_summaries.items():
        for geometry in ("axis", "last", "dense"):
            for strength, values in result["by"][geometry].items():
                bridge_csv.append({
                    "model": model_name,
                    "geometry": geometry,
                    "strength": float(strength),
                    "det_label_mean": values["det_label"]["mean"],
                    "det_cov_mean": values["det_cov"]["mean"],
                    "det_ind_mean": values["det_ind"]["mean"],
                    "decoder_cosine_mean": values["decoder_cosine"]["mean"],
                    "sae_recovery_rate": values["sae_recovery"]["mean"],
                    "bridge_seeds": 4,
                })
    write_csv(HERE / "source_data_bridge.csv", bridge_csv)

    raw_csv = []
    cluster_csv = []
    for model_name, rows in matched_rows.items():
        for row in rows:
            raw_csv.append({
                "model": model_name,
                "direction": row["direction"],
                "strength": row["strength"],
                "seed": row["seed"],
                "pair_id": row["pair_id"],
                "variant": row["variant_id"],
                "oracle_target_cosine": row["oracle_target_cosine"],
                "spectral_readout_target_cosine": row["spectral_readout_target_cosine"],
                "recovered": int(row["recovered"]),
                "heldout_reconstruction_mse": row["heldout_reconstruction_mse"],
                "heldout_active_l0": row["heldout_active_l0"],
                "heldout_dead_atom_fraction": row["heldout_dead_atom_fraction"],
            })
        for seed in sorted({int(row["seed"]) for row in rows}):
            for variant in ("vanilla-random", "spectral-warm-start"):
                subset = [row for row in rows if int(row["seed"]) == seed and row["variant_id"] == variant]
                if len(subset) != 6:
                    raise ValueError(f"incomplete seed cluster: {model_name} {seed} {variant}")
                cluster_csv.append({
                    "model": model_name,
                    "seed": seed,
                    "variant": variant,
                    "conditions": len(subset),
                    "recovery_rate": float(np.mean([row["recovered"] for row in subset])),
                    "mean_oracle_target_cosine": float(np.mean([row["oracle_target_cosine"] for row in subset])),
                    "mean_heldout_reconstruction_mse": float(np.mean([row["heldout_reconstruction_mse"] for row in subset])),
                })
    write_csv(HERE / "source_data_matched.csv", raw_csv)
    write_csv(HERE / "source_data_seed_clusters.csv", cluster_csv)
    return bridge_summaries, matched_rows, cluster_csv, summary


def make_figure() -> None:
    bridge, matched, clusters, summary = build_source_data()
    fig = plt.figure(figsize=(5.5, 3.65), facecolor="white")
    grid = fig.add_gridspec(2, 3, width_ratios=[1.06, 1.08, 0.92], height_ratios=[0.90, 1.10], left=0.075, right=0.985, bottom=0.12, top=0.94, hspace=0.57, wspace=0.42)

    # a: detector transfer at the weakest tested strength.
    ax_a = fig.add_subplot(grid[0, 0])
    geometries = ["axis", "last", "dense"]
    detector = np.asarray([
        [bridge[model]["by"][geometry]["0.35"][f"det_{stat}"]["mean"] for model in ("GPT-2 L6", "Pythia L3") for stat in ("cov", "ind")]
        for geometry in geometries
    ])
    ax_a.imshow(detector, vmin=0, vmax=1, cmap=PALE, aspect="auto")
    annotate_matrix(ax_a, detector, lambda value, _r, _c: f"{value:.2f}")
    ax_a.set_yticks(range(3), ["PC1", "PC32", "dense"])
    ax_a.set_xticks(range(4), ["cov", "ind", "cov", "ind"])
    ax_a.tick_params(length=0, pad=2)
    ax_a.text(0.5, -0.36, "GPT-2", transform=ax_a.get_xaxis_transform(), ha="center", color=GPT2, fontweight="bold")
    ax_a.text(2.5, -0.36, "Pythia", transform=ax_a.get_xaxis_transform(), ha="center", color=PYTHIA, fontweight="bold")
    ax_a.axvline(1.5, color="white", lw=1.4)
    ax_a.set_title(r"Detector power at $\lambda=.35$", loc="left", pad=4, fontweight="bold", color=INK)
    panel_label(ax_a, "a")
    for spine in ax_a.spines.values():
        spine.set_visible(False)

    # b: corrected vanilla SAE recovery over geometry and strength.
    ax_b = fig.add_subplot(grid[0, 1:])
    columns = [(geometry, strength) for geometry in geometries for strength in ("0.35", "0.55", "0.8")]
    recovery = np.asarray([[bridge[model]["by"][geometry][strength]["sae_recovery"]["mean"] for geometry, strength in columns] for model in ("GPT-2 L6", "Pythia L3")])
    ax_b.imshow(recovery, vmin=0, vmax=1, cmap=PALE, aspect="auto")
    annotate_matrix(ax_b, recovery, lambda value, _r, _c: f"{int(round(value * 4))}/4")
    ax_b.set_yticks([0, 1], ["GPT-2", "Pythia"])
    ax_b.get_yticklabels()[0].set_color(GPT2)
    ax_b.get_yticklabels()[1].set_color(PYTHIA)
    ax_b.set_xticks(range(9), [".35", ".55", ".80"] * 3)
    ax_b.tick_params(length=0, pad=2)
    for start, label in zip((1, 4, 7), ("PC1", "PC32", "dense")):
        ax_b.text(start, -0.43, label, transform=ax_b.get_xaxis_transform(), ha="center", color=INK)
    ax_b.axvline(2.5, color="white", lw=1.4)
    ax_b.axvline(5.5, color="white", lw=1.4)
    ax_b.set_title("Vanilla SAE recovery (real WikiText rerun)", loc="left", pad=4, fontweight="bold", color=INK)
    panel_label(ax_b, "b")
    for spine in ax_b.spines.values():
        spine.set_visible(False)

    # c: hero paired seed-cluster recovery panel.
    ax_c = fig.add_subplot(grid[1, :2])
    positions = {"GPT-2 L6": (0.0, 0.82), "Pythia L3": (1.75, 2.57)}
    model_colors = {"GPT-2 L6": GPT2, "Pythia L3": PYTHIA}
    for model_name, (x0, x1) in positions.items():
        model_rows = [row for row in clusters if row["model"] == model_name]
        seeds = sorted({int(row["seed"]) for row in model_rows})
        vanilla_values, warm_values = [], []
        for seed_index, seed in enumerate(seeds):
            vanilla_value = next(float(row["recovery_rate"]) for row in model_rows if int(row["seed"]) == seed and row["variant"] == "vanilla-random")
            warm_value = next(float(row["recovery_rate"]) for row in model_rows if int(row["seed"]) == seed and row["variant"] == "spectral-warm-start")
            vanilla_values.append(vanilla_value)
            warm_values.append(warm_value)
            jitter = (seed_index - (len(seeds) - 1) / 2) * 0.012
            ax_c.plot([x0 + jitter, x1 + jitter], [vanilla_value, warm_value], color=model_colors[model_name], alpha=0.25, lw=0.75, zorder=1)
            ax_c.scatter([x0 + jitter, x1 + jitter], [vanilla_value, warm_value], s=8, facecolor="white", edgecolor=model_colors[model_name], linewidth=0.55, zorder=2)
        for xpos, values, color in ((x0, vanilla_values, VANILLA), (x1, warm_values, WARM)):
            mean, low, high = bootstrap_mean_ci(values, 20260910 + int(xpos * 100))
            ax_c.errorbar(xpos, mean, yerr=[[mean - low], [high - mean]], fmt="o", ms=5.0, color=color, mec="white", mew=0.55, capsize=2.2, lw=1.1, zorder=4)
        raw_rows = matched[model_name]
        vanilla_count = sum(row["recovered"] for row in raw_rows if row["variant_id"] == "vanilla-random")
        warm_count = sum(row["recovered"] for row in raw_rows if row["variant_id"] == "spectral-warm-start")
        ax_c.text((x0 + x1) / 2, 1.035, f"{model_name}: {vanilla_count}/48 → {warm_count}/48", ha="center", va="bottom", fontsize=6.5, color=model_colors[model_name], fontweight="bold")
    ax_c.axhline(0.80, color="#8D98A5", lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax_c.text(2.70, 0.807, "0.80", ha="left", va="bottom", fontsize=5.4, color="#6F7B87")
    ax_c.set_xlim(-0.35, 2.92)
    ax_c.set_ylim(-0.03, 1.13)
    ax_c.set_xticks([0.0, 0.82, 1.75, 2.57], ["vanilla", "warm", "vanilla", "warm"])
    ax_c.set_ylabel("recovery rate within seed cluster")
    ax_c.set_title("Matched initialization-only intervention", loc="left", pad=5, fontweight="bold", color=INK)
    ax_c.grid(axis="y", color=GRID, lw=0.45, alpha=0.8)
    panel_label(ax_c, "c")

    # d: model-specific clustered effect and held-out cost.
    dgrid = grid[1, 2].subgridspec(2, 1, hspace=0.78)
    ax_d1 = fig.add_subplot(dgrid[0, 0])
    ax_d2 = fig.add_subplot(dgrid[1, 0])
    model_order = ["GPT-2 L6", "Pythia L3"]
    colors = [GPT2, PYTHIA]
    for axis, key, scale, title, limits, seed_offset in (
        (ax_d1, "mean_delta_cosine", 1.0, r"$\Delta$ target cosine", (-0.015, 0.16), 0),
        (ax_d2, "mean_delta_heldout_mse", 1000.0, r"$\Delta$ held-out MSE $\times10^3$", (-2.2, 2.2), 50),
    ):
        for y, (model_name, color) in enumerate(zip(model_order, colors)):
            model_id = MODEL_IDS[model_name]
            analysis = summary["models"][model_id]["clustered_paired_analysis"]
            mean = float(analysis[key]) * scale
            low, high = [float(value) * scale for value in analysis[key + "_bootstrap_ci95"]]
            axis.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="o", color=color, ms=4.0, capsize=2.0, lw=1.15, mec="white", mew=0.45)
        axis.axvline(0, color="#8D98A5", lw=0.7, ls=(0, (3, 2)))
        axis.set_xlim(*limits)
        axis.set_yticks([0, 1], ["GPT-2", "Pythia"])
        axis.invert_yaxis()
        axis.set_title(title, loc="left", pad=2.5, fontweight="bold", color=INK, fontsize=6.8)
        axis.grid(axis="x", color=GRID, lw=0.4, alpha=0.8)
        axis.tick_params(axis="both", pad=1.5)
    ax_d1.set_xlabel("warm − vanilla", labelpad=1.5)
    ax_d2.set_xlabel("warm − vanilla", labelpad=1.5)
    ax_d1.text(-0.20, 1.22, "d", transform=ax_d1.transAxes, fontsize=8.2, fontweight="bold", color=INK, va="top")
    ax_d1.text(1.0, 1.31, "n = 8 seed clusters/model", transform=ax_d1.transAxes, ha="right", va="bottom", fontsize=5.3, color="#6F7B87")

    fig.text(0.985, 0.017, "Known injected directions; oracle atom matching is evaluation-only. No Gaussian certificate is asserted.", ha="right", va="bottom", fontsize=5.25, color="#65717E")
    prefix = OUT / "figure_lm_validation"
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
