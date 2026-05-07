"""Re-export server formal figures from existing CSVs.

The full formal experiment is intentionally expensive. This utility only
recreates the PDF/PNG figures from already generated CSV files, which is useful
for submission hygiene changes such as font embedding.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def bootstrap_slope(
    rng: np.random.Generator,
    rows: list[dict[str, str]],
    mode: str,
    bootstrap: int,
) -> tuple[float, float, float]:
    by_lambda: dict[float, list[float]] = {}
    for row in rows:
        if row["mode"] == mode:
            by_lambda.setdefault(float(row["lambda"]), []).append(float(row["n_boundary"]))
    lambdas = np.array(sorted(by_lambda))
    means = np.array([np.mean(by_lambda[lam]) for lam in lambdas])
    slope = float(np.polyfit(np.log(lambdas), np.log(means), 1)[0])
    slopes = []
    for _ in range(bootstrap):
        sampled = np.array([rng.choice(by_lambda[lam]) for lam in lambdas])
        slopes.append(float(np.polyfit(np.log(lambdas), np.log(sampled), 1)[0]))
    lo, hi = np.quantile(slopes, [0.025, 0.975])
    return slope, float(lo), float(hi)


def plot_scaling(outdir: Path, bootstrap: int) -> None:
    rows = read_csv(outdir / "formal_scaling_boundaries.csv")
    slope_rng = np.random.default_rng(12345)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    colors = {"unlabeled": "#2f6f9f", "labeled": "#b85c38"}
    names = {"unlabeled": "unlabeled covariance", "labeled": "labeled mean"}
    for mode in ("unlabeled", "labeled"):
        slope, lo, hi = bootstrap_slope(slope_rng, rows, mode, bootstrap)
        xs = np.array(sorted({float(r["lambda"]) for r in rows if r["mode"] == mode}))
        means = []
        sds = []
        for lam in xs:
            vals = np.array(
                [
                    float(r["n_boundary"])
                    for r in rows
                    if r["mode"] == mode and float(r["lambda"]) == lam
                ]
            )
            means.append(float(np.mean(vals)))
            sds.append(float(np.std(vals, ddof=1)))
        means_arr = np.array(means)
        ax.errorbar(xs, means_arr, yerr=sds, fmt="o", capsize=2.5, color=colors[mode])
        fit_slope, fit_intercept = np.polyfit(np.log(xs), np.log(means_arr), 1)
        grid = np.linspace(xs.min(), xs.max(), 120)
        ax.plot(
            grid,
            np.exp(fit_intercept) * grid**fit_slope,
            color=colors[mode],
            label=f"{names[mode]}: {slope:.2f} [{lo:.2f}, {hi:.2f}]",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"feature strength $\lambda$")
    ax.set_ylabel(r"samples for 80\% detection")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "formal_scaling_law.pdf")
    fig.savefig(outdir / "formal_scaling_law.png", dpi=240)
    plt.close(fig)


def plot_geometry(outdir: Path) -> None:
    rows = read_csv(outdir / "formal_geometry_power.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    colors = ["#2f6f9f", "#4b7f52", "#b85c38"]
    lambdas = sorted({float(r["lambda"]) for r in rows})
    for lam, color in zip(lambdas, colors):
        xs = np.array(sorted({float(r["geometry"]) for r in rows if float(r["lambda"]) == lam}))
        means = []
        sds = []
        for x in xs:
            vals = np.array(
                [
                    float(r["power"])
                    for r in rows
                    if float(r["lambda"]) == lam and abs(float(r["geometry"]) - x) < 1.0e-12
                ]
            )
            means.append(float(np.mean(vals)))
            sds.append(float(np.std(vals, ddof=1)))
        ax.errorbar(xs, means, yerr=sds, marker="o", capsize=2.5, color=color, label=fr"$\lambda={lam}$")
    ax.set_xlabel(r"geometric factor $1-\|w\|_4^4$")
    ax.set_ylabel("detection power")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "formal_geometry_penalty.pdf")
    fig.savefig(outdir / "formal_geometry_penalty.png", dpi=240)
    plt.close(fig)


def plot_evidence(outdir: Path) -> None:
    rows = read_csv(outdir / "formal_evidence_collapse.csv")
    evidence = np.array([float(r["evidence"]) for r in rows])
    power = np.array([float(r["power"]) for r in rows])
    bins = np.array([0.6, 0.9, 1.3, 1.9, 2.8, 4.2, 6.5])
    bin_centers = []
    bin_means = []
    bin_sds = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (evidence >= lo) & (evidence < hi)
        if np.any(mask):
            bin_centers.append(float(np.sqrt(lo * hi)))
            bin_means.append(float(np.mean(power[mask])))
            bin_sds.append(float(np.std(power[mask], ddof=1)))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.scatter(evidence, power, s=10, alpha=0.18, color="#4f6272", label="individual settings")
    ax.errorbar(bin_centers, bin_means, yerr=bin_sds, marker="o", capsize=2.5, color="#b85c38", label="binned mean")
    ax.set_xscale("log")
    ax.set_xlabel(r"accumulated evidence $N C_i\lambda^4$")
    ax.set_ylabel("detection power")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "formal_evidence_collapse.pdf")
    fig.savefig(outdir / "formal_evidence_collapse.png", dpi=240)
    plt.close(fig)


def plot_dark_floor(outdir: Path) -> None:
    rows = read_csv(outdir / "formal_dark_floor.csv")
    xs = np.array(sorted({int(r["n_samples"]) for r in rows}))
    pred_mean = []
    pred_sd = []
    obs_mean = []
    obs_sd = []
    for n_samples in xs:
        pred_vals = np.array([float(r["predicted_f_dark"]) for r in rows if int(r["n_samples"]) == int(n_samples)])
        obs_vals = np.array([float(r["observed_floor"]) for r in rows if int(r["n_samples"]) == int(n_samples)])
        pred_mean.append(float(np.mean(pred_vals)))
        pred_sd.append(float(np.std(pred_vals, ddof=1)))
        obs_mean.append(float(np.mean(obs_vals)))
        obs_sd.append(float(np.std(obs_vals, ddof=1)))
    pred_mean_arr = np.array(pred_mean)
    obs_mean_arr = np.array(obs_mean)
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    ax.errorbar(xs, pred_mean, yerr=pred_sd, marker="o", capsize=2.5, color="#2f6f9f", label=r"predicted $f_{\mathrm{dark}}$")
    ax.errorbar(xs, obs_mean, yerr=obs_sd, marker="s", capsize=2.5, color="#b85c38", label="observed floor")
    ax.set_xscale("log")
    ax.set_xlabel("samples")
    ax.set_ylabel("fraction of feature variance")
    ax.set_ylim(0.0, max(0.8, float(max(np.max(pred_mean_arr), np.max(obs_mean_arr))) + 0.08))
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "formal_dark_floor.pdf")
    fig.savefig(outdir / "formal_dark_floor.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results/server_formal_v2"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    plot_scaling(args.outdir, args.bootstrap)
    plot_geometry(args.outdir)
    plot_evidence(args.outdir)
    plot_dark_floor(args.outdir)


if __name__ == "__main__":
    main()
