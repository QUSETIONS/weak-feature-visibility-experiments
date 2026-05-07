"""Synthetic validations for the quartic detection law paper.

The script produces three artifacts used by the paper:

1. a scaling experiment comparing unlabeled covariance detection against a
   labeled mean-shift benchmark;
2. a geometry experiment varying the loading concentration;
3. a population experiment comparing the predicted dark matter fraction with
   the residual of a recover-or-miss decoder.

All experiments are deliberately small and CPU-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def unit_spread(m: int) -> np.ndarray:
    return np.ones(m, dtype=float) / math.sqrt(m)


def offdiag_stat(samples: np.ndarray) -> np.ndarray:
    """Return ||offdiag(sample covariance)||_F^2 for each trial."""
    cov = np.einsum("tnm,tnk->tmk", samples, samples) / samples.shape[1]
    off = cov.copy()
    idx = np.arange(off.shape[1])
    off[:, idx, idx] = 0.0
    return np.sum(off * off, axis=(1, 2))


def simulate_unlabeled_power(
    lam: float,
    w: np.ndarray,
    n_samples: int,
    trials: int,
    alpha: float,
    rng: np.random.Generator,
) -> float:
    m = w.size
    null = rng.normal(size=(trials, n_samples, m))
    threshold = np.quantile(offdiag_stat(null), 1.0 - alpha)

    eps = rng.normal(size=(trials, n_samples, m))
    z = rng.normal(size=(trials, n_samples, 1))
    alt = eps + lam * z * w.reshape(1, 1, m)
    return float(np.mean(offdiag_stat(alt) > threshold))


def mean_stat(samples: np.ndarray) -> np.ndarray:
    mean = np.mean(samples, axis=1)
    return np.sum(mean * mean, axis=1)


def simulate_labeled_power(
    lam: float,
    w: np.ndarray,
    n_samples: int,
    trials: int,
    alpha: float,
    rng: np.random.Generator,
) -> float:
    m = w.size
    null = rng.normal(size=(trials, n_samples, m))
    threshold = np.quantile(mean_stat(null), 1.0 - alpha)

    alt = rng.normal(size=(trials, n_samples, m)) + lam * w.reshape(1, 1, m)
    return float(np.mean(mean_stat(alt) > threshold))


def find_boundary(
    lam: float,
    w: np.ndarray,
    mode: str,
    trials: int,
    alpha: float,
    target_power: float,
    rng: np.random.Generator,
) -> tuple[int, float]:
    geom = max(1.0 - float(np.sum(w**4)), 1.0e-8)
    if mode == "unlabeled":
        c_i = geom / 4.0
        center = 2.8 / (c_i * lam**4)
        power_fn = simulate_unlabeled_power
    elif mode == "labeled":
        center = 7.5 / (lam**2)
        power_fn = simulate_labeled_power
    else:
        raise ValueError(f"unknown mode: {mode}")

    multipliers = np.array([0.25, 0.38, 0.57, 0.85, 1.25, 1.85, 2.75, 4.1])
    candidates = np.unique(np.maximum(40, np.round(center * multipliers)).astype(int))

    last_power = 0.0
    for n_samples in candidates:
        power = power_fn(lam, w, int(n_samples), trials, alpha, rng)
        last_power = power
        if power >= target_power:
            return int(n_samples), power
    return int(candidates[-1]), last_power


def run_scaling(
    outdir: Path,
    rng: np.random.Generator,
    m: int,
    trials: int,
    alpha: float,
    target_power: float,
) -> dict[str, float]:
    w = unit_spread(m)
    lambdas = np.array([0.33, 0.38, 0.44, 0.51, 0.59, 0.68])
    rows = []
    for mode in ("unlabeled", "labeled"):
        for lam in lambdas:
            n_boundary, power = find_boundary(
                float(lam), w, mode, trials, alpha, target_power, rng
            )
            rows.append(
                {
                    "mode": mode,
                    "lambda": float(lam),
                    "n_boundary": n_boundary,
                    "power": power,
                }
            )

    csv_path = outdir / "scaling_boundaries.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    slopes = {}
    fig, ax = plt.subplots(figsize=(5.3, 3.5))
    colors = {"unlabeled": "#2f6f9f", "labeled": "#b85c38"}
    labels = {"unlabeled": "unlabeled covariance", "labeled": "labeled mean"}
    for mode in ("unlabeled", "labeled"):
        xs = np.array([r["lambda"] for r in rows if r["mode"] == mode])
        ys = np.array([r["n_boundary"] for r in rows if r["mode"] == mode])
        slope, intercept = np.polyfit(np.log(xs), np.log(ys), deg=1)
        slopes[f"{mode}_slope"] = float(slope)
        ax.scatter(xs, ys, color=colors[mode], s=30)
        grid = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(grid, np.exp(intercept) * grid**slope, color=colors[mode], label=f"{labels[mode]} ({slope:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"feature strength $\lambda$")
    ax.set_ylabel(r"samples for 80\% detection")
    ax.legend(frameon=False)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(outdir / "scaling_law.pdf")
    fig.savefig(outdir / "scaling_law.png", dpi=220)
    plt.close(fig)
    return slopes


def run_geometry(
    outdir: Path,
    rng: np.random.Generator,
    m: int,
    trials: int,
    alpha: float,
) -> dict[str, float]:
    lam = 0.48
    n_samples = 4200
    rows = []
    for k in range(1, m + 1):
        w = np.zeros(m)
        w[:k] = 1.0 / math.sqrt(k)
        geom = 1.0 - float(np.sum(w**4))
        power = simulate_unlabeled_power(lam, w, n_samples, trials, alpha, rng)
        rows.append({"active_coordinates": k, "geometry": geom, "power": power})

    csv_path = outdir / "geometry_power.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(5.3, 3.2))
    geom = np.array([r["geometry"] for r in rows])
    power = np.array([r["power"] for r in rows])
    ax.plot(geom, power, marker="o", color="#4b7f52")
    for r in rows:
        ax.annotate(str(r["active_coordinates"]), (r["geometry"], r["power"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(r"geometric factor $1-\|w\|_4^4$")
    ax.set_ylabel("detection power")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(outdir / "geometry_penalty.pdf")
    fig.savefig(outdir / "geometry_penalty.png", dpi=220)
    plt.close(fig)
    return {
        "axis_power": float(rows[0]["power"]),
        "spread_power": float(rows[-1]["power"]),
    }


def random_unit_vectors(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    w = rng.normal(size=(n, m))
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def dark_fraction(lambdas: np.ndarray, c_values: np.ndarray, n_samples: int) -> float:
    beta = math.log(n_samples)
    boundary = (beta / (n_samples * c_values)) ** 0.25
    dark = lambdas < boundary
    return float(np.sum(lambdas[dark] ** 2) / np.sum(lambdas**2))


def observed_floor(
    rng: np.random.Generator,
    lambdas: np.ndarray,
    c_values: np.ndarray,
    n_samples: int,
    replicates: int,
) -> tuple[float, float]:
    beta = math.log(n_samples)
    evidence = n_samples * c_values * lambdas**4
    floors = []
    for _ in range(replicates):
        noise = rng.normal(scale=0.35, size=lambdas.size)
        recovered = np.log(evidence + 1.0e-12) + noise >= math.log(beta)
        floors.append(float(np.sum(lambdas[~recovered] ** 2) / np.sum(lambdas**2)))
    return float(np.mean(floors)), float(np.std(floors, ddof=1))


def run_dark_floor(outdir: Path, rng: np.random.Generator, m: int) -> dict[str, float]:
    n_features = 280
    ranks = np.arange(1, n_features + 1)
    lambdas = 0.86 * ranks ** (-0.34)
    lambdas *= rng.lognormal(mean=0.0, sigma=0.12, size=n_features)
    lambdas = np.clip(lambdas, 0.055, 0.82)

    w = random_unit_vectors(rng, n_features, m)
    geom = 1.0 - np.sum(w**4, axis=1)
    c_values = np.maximum(geom / 4.0, 1.0e-8)
    sample_sizes = np.array([500, 800, 1300, 2100, 3400, 5500, 8900, 14400, 23300])

    rows = []
    for n_samples in sample_sizes:
        pred = dark_fraction(lambdas, c_values, int(n_samples))
        obs, sd = observed_floor(rng, lambdas, c_values, int(n_samples), replicates=80)
        rows.append(
            {
                "n_samples": int(n_samples),
                "predicted_f_dark": pred,
                "observed_floor": obs,
                "observed_floor_sd": sd,
            }
        )

    csv_path = outdir / "dark_floor.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([r["n_samples"] for r in rows])
    pred = np.array([r["predicted_f_dark"] for r in rows])
    obs = np.array([r["observed_floor"] for r in rows])
    sd = np.array([r["observed_floor_sd"] for r in rows])
    corr = float(np.corrcoef(pred, obs)[0, 1])
    mae = float(np.mean(np.abs(pred - obs)))

    fig, ax = plt.subplots(figsize=(5.3, 3.3))
    ax.plot(x, pred, marker="o", color="#2f6f9f", label=r"predicted $f_{\mathrm{dark}}$")
    ax.errorbar(x, obs, yerr=sd, marker="s", color="#b85c38", linewidth=1.4, capsize=2.5, label="observed floor")
    ax.set_xscale("log")
    ax.set_xlabel("samples")
    ax.set_ylabel("fraction of feature variance")
    ax.set_ylim(0.0, max(0.75, float(np.max(obs + sd)) + 0.05))
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "dark_floor.pdf")
    fig.savefig(outdir / "dark_floor.png", dpi=220)
    plt.close(fig)
    return {"dark_floor_corr": corr, "dark_floor_mae": mae}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results/synthetic"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=220)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.80)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summary = {
        "seed": args.seed,
        "m": args.m,
        "trials": args.trials,
        "alpha": args.alpha,
        "target_power": args.target_power,
    }
    summary.update(run_scaling(args.outdir, rng, args.m, args.trials, args.alpha, args.target_power))
    summary.update(run_geometry(args.outdir, rng, args.m, args.trials, args.alpha))
    summary.update(run_dark_floor(args.outdir, rng, args.m))

    with (args.outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
