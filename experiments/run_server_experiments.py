"""Server-scale synthetic experiments for the quartic detection law.

This script is the formal version of the lightweight local smoke test. It runs
multiple independent seeds, reports bootstrap uncertainty, and produces the
figures/tables used for the paper's experimental section.

The experiments are CPU-only by design: the model is Gaussian and all sufficient
statistics are covariance or mean statistics, so GPU training would add queueing
and memory contention without changing the statistical validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Config:
    m: int
    trials: int
    seeds: int
    alpha: float
    target_power: float
    bootstrap: int


def unit_k(m: int, k: int) -> np.ndarray:
    w = np.zeros(m, dtype=float)
    w[:k] = 1.0 / math.sqrt(k)
    return w


def unit_spread(m: int) -> np.ndarray:
    return unit_k(m, m)


def geom_factor(w: np.ndarray) -> float:
    return float(1.0 - np.sum(w**4))


def offdiag_stat(samples: np.ndarray) -> np.ndarray:
    cov = np.einsum("tnm,tnk->tmk", samples, samples, optimize=True) / samples.shape[1]
    idx = np.arange(cov.shape[1])
    cov[:, idx, idx] = 0.0
    return np.sum(cov * cov, axis=(1, 2))


def mean_stat(samples: np.ndarray) -> np.ndarray:
    mean = np.mean(samples, axis=1)
    return np.sum(mean * mean, axis=1)


def unlabeled_power(
    rng: np.random.Generator,
    lam: float,
    w: np.ndarray,
    n_samples: int,
    trials: int,
    alpha: float,
) -> float:
    m = w.size
    null = rng.normal(size=(trials, n_samples, m))
    threshold = np.quantile(offdiag_stat(null), 1.0 - alpha)
    eps = rng.normal(size=(trials, n_samples, m))
    z = rng.normal(size=(trials, n_samples, 1))
    alt = eps + lam * z * w.reshape(1, 1, m)
    return float(np.mean(offdiag_stat(alt) > threshold))


def labeled_power(
    rng: np.random.Generator,
    lam: float,
    w: np.ndarray,
    n_samples: int,
    trials: int,
    alpha: float,
) -> float:
    m = w.size
    null = rng.normal(size=(trials, n_samples, m))
    threshold = np.quantile(mean_stat(null), 1.0 - alpha)
    alt = rng.normal(size=(trials, n_samples, m)) + lam * w.reshape(1, 1, m)
    return float(np.mean(mean_stat(alt) > threshold))


def boundary_candidates(lam: float, w: np.ndarray, mode: str) -> np.ndarray:
    if mode == "unlabeled":
        c_i = max(geom_factor(w), 1.0e-8) / 4.0
        center = 2.8 / (c_i * lam**4)
    elif mode == "labeled":
        center = 7.0 / lam**2
    else:
        raise ValueError(mode)
    multipliers = np.array([0.22, 0.32, 0.46, 0.68, 1.0, 1.47, 2.15, 3.15, 4.6])
    return np.unique(np.maximum(32, np.round(center * multipliers)).astype(int))


def estimate_boundary(
    rng: np.random.Generator,
    lam: float,
    w: np.ndarray,
    mode: str,
    cfg: Config,
) -> tuple[float, float]:
    fn = unlabeled_power if mode == "unlabeled" else labeled_power
    prev_n = None
    prev_power = None
    last_power = 0.0
    candidates = boundary_candidates(lam, w, mode)
    for n_samples in boundary_candidates(lam, w, mode):
        power = fn(rng, lam, w, int(n_samples), cfg.trials, cfg.alpha)
        last_power = power
        if power >= cfg.target_power:
            if prev_n is not None and prev_power is not None and power > prev_power:
                frac = (cfg.target_power - prev_power) / (power - prev_power)
                log_boundary = math.log(prev_n) + float(np.clip(frac, 0.0, 1.0)) * (math.log(int(n_samples)) - math.log(prev_n))
                return float(math.exp(log_boundary)), power
            return float(n_samples), power
        prev_n = int(n_samples)
        prev_power = power
    return float(candidates[-1]), last_power


def bootstrap_slope(
    rng: np.random.Generator,
    rows: list[dict[str, float | int | str]],
    mode: str,
    bootstrap: int,
) -> tuple[float, float, float]:
    data = [(float(r["lambda"]), float(r["n_boundary"])) for r in rows if r["mode"] == mode]
    xs = np.array([x for x, _ in data])
    ys = np.array([y for _, y in data])
    slope = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
    by_lambda = {}
    for r in rows:
        if r["mode"] == mode:
            by_lambda.setdefault(float(r["lambda"]), []).append(float(r["n_boundary"]))
    slopes = []
    lambdas = np.array(sorted(by_lambda))
    for _ in range(bootstrap):
        sampled = np.array([rng.choice(by_lambda[lam]) for lam in lambdas])
        slopes.append(float(np.polyfit(np.log(lambdas), np.log(sampled), 1)[0]))
    lo, hi = np.quantile(slopes, [0.025, 0.975])
    return slope, float(lo), float(hi)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_scaling(outdir: Path, cfg: Config, master_rng: np.random.Generator) -> dict[str, float]:
    # Wider lambda window for the reviewer-facing quartic-fit stress test.
    lambdas = np.array([0.24, 0.28, 0.32, 0.37, 0.43, 0.50, 0.58, 0.68, 0.80])
    w = unit_spread(cfg.m)
    rows: list[dict[str, object]] = []
    for seed_id in range(cfg.seeds):
        rng = np.random.default_rng(int(master_rng.integers(0, 2**31 - 1)))
        for mode in ("unlabeled", "labeled"):
            for lam in lambdas:
                n_boundary, power = estimate_boundary(rng, float(lam), w, mode, cfg)
                rows.append(
                    {
                        "seed_id": seed_id,
                        "mode": mode,
                        "lambda": float(lam),
                        "n_boundary": n_boundary,
                        "power_at_boundary": power,
                    }
                )
    write_csv(outdir / "formal_scaling_boundaries.csv", rows)

    slope_rng = np.random.default_rng(12345)
    summary: dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    colors = {"unlabeled": "#2f6f9f", "labeled": "#b85c38"}
    names = {"unlabeled": "unlabeled covariance", "labeled": "labeled mean"}
    for mode in ("unlabeled", "labeled"):
        slope, lo, hi = bootstrap_slope(slope_rng, rows, mode, cfg.bootstrap)
        summary[f"{mode}_slope"] = slope
        summary[f"{mode}_slope_ci_low"] = lo
        summary[f"{mode}_slope_ci_high"] = hi

        xs = np.array(sorted(set(float(r["lambda"]) for r in rows if r["mode"] == mode)))
        means = []
        sds = []
        for lam in xs:
            vals = np.array([float(r["n_boundary"]) for r in rows if r["mode"] == mode and float(r["lambda"]) == lam])
            means.append(float(np.mean(vals)))
            sds.append(float(np.std(vals, ddof=1)))
        means_arr = np.array(means)
        ax.errorbar(xs, means_arr, yerr=sds, fmt="o", capsize=2.5, color=colors[mode])
        fit_slope, fit_intercept = np.polyfit(np.log(xs), np.log(means_arr), 1)
        grid = np.linspace(xs.min(), xs.max(), 120)
        ax.plot(grid, np.exp(fit_intercept) * grid**fit_slope, color=colors[mode], label=f"{names[mode]}: {slope:.2f} [{lo:.2f}, {hi:.2f}]")
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
    return summary


def run_geometry(outdir: Path, cfg: Config, master_rng: np.random.Generator) -> dict[str, float]:
    lambdas = [0.34, 0.42, 0.50]
    n_samples = 3600
    rows: list[dict[str, object]] = []
    for seed_id in range(cfg.seeds):
        rng = np.random.default_rng(int(master_rng.integers(0, 2**31 - 1)))
        for lam in lambdas:
            for k in range(1, cfg.m + 1):
                w = unit_k(cfg.m, k)
                g = geom_factor(w)
                power = unlabeled_power(rng, lam, w, n_samples, cfg.trials, cfg.alpha)
                rows.append(
                    {
                        "seed_id": seed_id,
                        "lambda": lam,
                        "active_coordinates": k,
                        "geometry": g,
                        "power": power,
                        "predicted_coefficient": g / 4.0,
                    }
                )
    write_csv(outdir / "formal_geometry_power.csv", rows)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    colors = ["#2f6f9f", "#4b7f52", "#b85c38"]
    for lam, color in zip(lambdas, colors):
        xs = np.array(sorted(set(float(r["geometry"]) for r in rows if float(r["lambda"]) == lam)))
        means = []
        sds = []
        for x in xs:
            vals = np.array([float(r["power"]) for r in rows if float(r["lambda"]) == lam and abs(float(r["geometry"]) - x) < 1.0e-12])
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

    axis_vals = np.array([float(r["power"]) for r in rows if int(r["active_coordinates"]) == 1])
    spread_vals = np.array([float(r["power"]) for r in rows if int(r["active_coordinates"]) == cfg.m and abs(float(r["lambda"]) - 0.50) < 1.0e-12])
    return {
        "axis_power_mean": float(np.mean(axis_vals)),
        "axis_power_sd": float(np.std(axis_vals, ddof=1)),
        "spread_power_mean_at_lambda_0p50": float(np.mean(spread_vals)),
    }


def run_evidence_collapse(outdir: Path, cfg: Config, master_rng: np.random.Generator) -> dict[str, float]:
    rows: list[dict[str, object]] = []
    lambdas = [0.34, 0.40, 0.48, 0.57]
    ks = [2, 3, 5, 8]
    evidence_targets = [0.7, 1.1, 1.7, 2.6, 4.0, 6.2]
    for seed_id in range(cfg.seeds):
        rng = np.random.default_rng(int(master_rng.integers(0, 2**31 - 1)))
        for lam in lambdas:
            for k in ks:
                w = unit_k(cfg.m, k)
                c_i = max(geom_factor(w), 1.0e-8) / 4.0
                for target in evidence_targets:
                    n_samples = int(max(80, round(target / (c_i * lam**4))))
                    power = unlabeled_power(rng, lam, w, n_samples, max(160, cfg.trials // 2), cfg.alpha)
                    rows.append(
                        {
                            "seed_id": seed_id,
                            "lambda": lam,
                            "active_coordinates": k,
                            "c_i": c_i,
                            "n_samples": n_samples,
                            "evidence": n_samples * c_i * lam**4,
                            "power": power,
                        }
                    )
    write_csv(outdir / "formal_evidence_collapse.csv", rows)

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

    corr = float(np.corrcoef(np.log(evidence), power)[0, 1])
    return {"evidence_power_corr": corr}


def random_unit_vectors(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    w = rng.normal(size=(n, m))
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def dark_fraction(lambdas: np.ndarray, c_values: np.ndarray, n_samples: int) -> float:
    beta = math.log(n_samples)
    boundary = (beta / (n_samples * c_values)) ** 0.25
    dark = lambdas < boundary
    return float(np.sum(lambdas[dark] ** 2) / np.sum(lambdas**2))


def noisy_floor(
    rng: np.random.Generator,
    lambdas: np.ndarray,
    c_values: np.ndarray,
    n_samples: int,
    repeats: int,
) -> tuple[float, float]:
    beta = math.log(n_samples)
    evidence = n_samples * c_values * lambdas**4
    floors = []
    for _ in range(repeats):
        noise = rng.normal(scale=0.32, size=lambdas.size)
        recovered = np.log(evidence + 1.0e-12) + noise >= math.log(beta)
        floors.append(float(np.sum(lambdas[~recovered] ** 2) / np.sum(lambdas**2)))
    return float(np.mean(floors)), float(np.std(floors, ddof=1))


def run_dark_floor(outdir: Path, cfg: Config, master_rng: np.random.Generator) -> dict[str, float]:
    sample_sizes = np.array([500, 750, 1100, 1650, 2500, 3800, 5700, 8600, 13000, 19500, 29200])
    rows: list[dict[str, object]] = []
    for seed_id in range(cfg.seeds * 3):
        rng = np.random.default_rng(int(master_rng.integers(0, 2**31 - 1)))
        n_features = 360
        ranks = np.arange(1, n_features + 1)
        lambdas = 0.92 * ranks ** (-0.35)
        lambdas *= rng.lognormal(mean=0.0, sigma=0.14, size=n_features)
        lambdas = np.clip(lambdas, 0.045, 0.88)
        w = random_unit_vectors(rng, n_features, cfg.m)
        c_values = np.maximum((1.0 - np.sum(w**4, axis=1)) / 4.0, 1.0e-8)
        for n_samples in sample_sizes:
            pred = dark_fraction(lambdas, c_values, int(n_samples))
            obs, sd = noisy_floor(rng, lambdas, c_values, int(n_samples), repeats=80)
            rows.append(
                {
                    "seed_id": seed_id,
                    "n_samples": int(n_samples),
                    "predicted_f_dark": pred,
                    "observed_floor": obs,
                    "within_population_sd": sd,
                }
            )
    write_csv(outdir / "formal_dark_floor.csv", rows)

    xs = sample_sizes
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

    pred_all = np.array([float(r["predicted_f_dark"]) for r in rows])
    obs_all = np.array([float(r["observed_floor"]) for r in rows])
    return {
        "dark_floor_corr_all": float(np.corrcoef(pred_all, obs_all)[0, 1]),
        "dark_floor_mae_all": float(np.mean(np.abs(pred_all - obs_all))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results/server_formal"))
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--trials", type=int, default=700)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    cfg = Config(
        m=args.m,
        trials=args.trials,
        seeds=args.seeds,
        alpha=args.alpha,
        target_power=args.target_power,
        bootstrap=args.bootstrap,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(args.seed)

    start = time.time()
    summary: dict[str, object] = {
        "seed": args.seed,
        "m": cfg.m,
        "trials": cfg.trials,
        "seeds": cfg.seeds,
        "alpha": cfg.alpha,
        "target_power": cfg.target_power,
        "bootstrap": cfg.bootstrap,
        "host": socket.gethostname(),
        "platform": platform.platform(),
    }
    summary.update(run_scaling(args.outdir, cfg, master_rng))
    summary.update(run_geometry(args.outdir, cfg, master_rng))
    summary.update(run_evidence_collapse(args.outdir, cfg, master_rng))
    summary.update(run_dark_floor(args.outdir, cfg, master_rng))
    summary["wall_time_seconds"] = time.time() - start

    with (args.outdir / "formal_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
