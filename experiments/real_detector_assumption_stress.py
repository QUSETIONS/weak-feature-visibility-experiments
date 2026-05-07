"""Assumption stress test for the feature visibility principle.

This real-activation bridge compares three detector classes on the same
controlled injections:

1. labeled mean detector: uses activation labels and should see every feature;
2. unrestricted covariance detector: uses variance along the known direction;
3. feature-independent detector: removes diagonal covariance evidence and uses
   only the off-diagonal residual predicted by tangent-space absorption.

The key reviewer-facing test is the axis direction. If the injected feature is
present, labeled and unrestricted covariance detectors should recover it. If the
feature-independent null is the source of the blind spot, the off-diagonal
detector should remain at the calibrated null level for the same axis feature.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def top_pcs(x: np.ndarray, n_pcs: int = 8) -> np.ndarray:
    xc = x.astype(np.float64) - x.mean(axis=0, keepdims=True)
    if xc.shape[0] > 20000:
        rng = np.random.default_rng(0)
        xc = xc[rng.choice(xc.shape[0], size=20000, replace=False)]
    _, _, vh = np.linalg.svd(xc, full_matrices=False)
    return vh[:n_pcs]


def make_direction(kind: str, d: int, rng: np.random.Generator, pcs: np.ndarray) -> tuple[np.ndarray, str]:
    if kind == "axis":
        w = np.zeros(d, dtype=np.float64)
        w[0] = 1.0
        return w, "axis"
    if kind.startswith("k"):
        k = int(kind[1:])
        w = np.zeros(d, dtype=np.float64)
        idx = rng.choice(d, size=min(k, d), replace=False)
        signs = rng.choice([-1.0, 1.0], size=idx.size)
        w[idx] = signs / np.sqrt(idx.size)
        return w, f"{idx.size}-sparse"
    if kind == "dense":
        w = rng.normal(size=d)
        w /= np.linalg.norm(w) + 1e-12
        return w, "dense"
    if kind == "pc_orth":
        w = rng.normal(size=d)
        for j in range(min(8, pcs.shape[0])):
            w = w - np.dot(w, pcs[j]) * pcs[j]
        w /= np.linalg.norm(w) + 1e-12
        return w, "PC-orth"
    raise ValueError(f"unknown direction kind: {kind}")


def summarize_by_label(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    labels = sorted({str(r["direction_label"]) for r in rows}, key=lambda s: (s != "axis", s))
    out: list[dict[str, object]] = []
    for label in labels:
        group = [r for r in rows if str(r["direction_label"]) == label]
        out.append(
            {
                "direction_label": label,
                "rows": len(group),
                "mean_geometry": float(np.mean([float(r["geometry"]) for r in group])),
                "mean_labeled_power": float(np.mean([float(r["labeled_power"]) for r in group])),
                "mean_unrestricted_cov_power": float(
                    np.mean([float(r["unrestricted_cov_power"]) for r in group])
                ),
                "mean_feature_independent_power": float(
                    np.mean([float(r["feature_independent_power"]) for r in group])
                ),
                "sd_feature_independent_power": float(
                    np.std([float(r["feature_independent_power"]) for r in group])
                ),
            }
        )
    return out


def run(args: argparse.Namespace) -> dict[str, object]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    x = np.load(args.activation_path).astype(np.float32)
    if x.shape[0] > args.max_tokens:
        x = x[rng.choice(x.shape[0], size=args.max_tokens, replace=False)]
    x = x - x.mean(axis=0, keepdims=True)
    d = x.shape[1]
    pcs = top_pcs(x, n_pcs=8)

    baseline_idx = rng.choice(x.shape[0], size=min(args.baseline_tokens, x.shape[0]), replace=False)
    kinds = [v.strip() for v in args.direction_kinds.split(",") if v.strip()]
    direction_seeds = [int(v) for v in args.direction_seeds.split(",") if v.strip()]

    rows: list[dict[str, object]] = []
    for kind in kinds:
        seeds = [0] if kind == "axis" else direction_seeds
        for direction_seed in seeds:
            drng = np.random.default_rng(args.seed + 7919 * direction_seed + 101 * len(kind))
            w, label = make_direction(kind, d, drng, pcs)
            geom = float(1.0 - np.sum(w**4))
            w2 = w * w
            w3 = w2 * w
            w4_sum = float(np.sum(w2 * w2))

            y_all = (x @ w).astype(np.float64)
            q_all = x @ w2
            r_all = x @ w3
            baseline_var = float(np.mean(y_all[baseline_idx] ** 2))
            baseline_offdiag = float(np.mean(y_all[baseline_idx] ** 2 - q_all[baseline_idx]))

            null_labeled: list[float] = []
            null_cov: list[float] = []
            null_ind: list[float] = []
            alt_labeled: list[float] = []
            alt_cov: list[float] = []
            alt_ind: list[float] = []

            for _ in range(args.trials):
                idx = rng.choice(x.shape[0], size=args.n, replace=False)
                yb = y_all[idx]
                qb = q_all[idx]
                rb = r_all[idx]

                z_null = rng.binomial(1, args.p, size=args.n).astype(np.float64)
                pos = z_null > 0.5
                neg = ~pos
                null_labeled.append(float(yb[pos].mean() - yb[neg].mean()) if pos.sum() > 1 and neg.sum() > 1 else 0.0)
                null_cov.append(float(np.mean(yb * yb) - baseline_var))
                null_ind.append(float(np.mean(yb * yb - qb) - baseline_offdiag))

                z = rng.binomial(1, args.p, size=args.n).astype(np.float64)
                zi = z - args.p
                ya = yb + args.amp * zi
                qa = qb + 2.0 * args.amp * zi * rb + (args.amp * args.amp) * (zi * zi) * w4_sum
                pos = z > 0.5
                neg = ~pos
                alt_labeled.append(float(ya[pos].mean() - ya[neg].mean()) if pos.sum() > 1 and neg.sum() > 1 else 0.0)
                alt_cov.append(float(np.mean(ya * ya) - baseline_var))
                alt_ind.append(float(np.mean(ya * ya - qa) - baseline_offdiag))

            labeled_thr = float(np.quantile(null_labeled, 1.0 - args.alpha))
            cov_thr = float(np.quantile(null_cov, 1.0 - args.alpha))
            ind_thr = float(np.quantile(null_ind, 1.0 - args.alpha))
            lambda_eff = float(args.amp * np.sqrt(args.p * (1.0 - args.p)))
            rows.append(
                {
                    "direction_kind": kind,
                    "direction_label": label,
                    "direction_seed": direction_seed,
                    "geometry": geom,
                    "amp": args.amp,
                    "p": args.p,
                    "n": args.n,
                    "lambda_eff": lambda_eff,
                    "labeled_power": float(np.mean(np.array(alt_labeled) > labeled_thr)),
                    "unrestricted_cov_power": float(np.mean(np.array(alt_cov) > cov_thr)),
                    "feature_independent_power": float(np.mean(np.array(alt_ind) > ind_thr)),
                    "labeled_threshold": labeled_thr,
                    "unrestricted_cov_threshold": cov_thr,
                    "feature_independent_threshold": ind_thr,
                    "trials": args.trials,
                }
            )

    write_csv(outdir / "real_detector_assumption_stress.csv", rows)
    by_label = summarize_by_label(rows)
    write_csv(outdir / "real_detector_assumption_stress_by_direction.csv", by_label)

    axis = [r for r in rows if r["direction_label"] == "axis"][0]
    geom = np.array([float(r["geometry"]) for r in rows])
    ind = np.array([float(r["feature_independent_power"]) for r in rows])
    cov = np.array([float(r["unrestricted_cov_power"]) for r in rows])
    labeled = np.array([float(r["labeled_power"]) for r in rows])
    summary = {
        "mode": "real_detector_assumption_stress",
        "activation_path": args.activation_path,
        "rows": len(rows),
        "amp": args.amp,
        "p": args.p,
        "n": args.n,
        "trials": args.trials,
        "lambda_eff": float(args.amp * np.sqrt(args.p * (1.0 - args.p))),
        "axis_labeled_power": float(axis["labeled_power"]),
        "axis_unrestricted_cov_power": float(axis["unrestricted_cov_power"]),
        "axis_feature_independent_power": float(axis["feature_independent_power"]),
        "mean_labeled_power": float(np.mean(labeled)),
        "mean_unrestricted_cov_power": float(np.mean(cov)),
        "mean_feature_independent_power": float(np.mean(ind)),
        "corr_feature_independent_power_geometry": float(np.corrcoef(geom, ind)[0, 1]),
        "corr_unrestricted_cov_power_geometry": float(np.corrcoef(geom, cov)[0, 1]),
    }
    with (outdir / "real_detector_assumption_stress_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    try:
        import matplotlib.pyplot as plt

        labels = [str(r["direction_label"]) for r in by_label]
        xs = np.arange(len(labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        ax.bar(xs - width, [float(r["mean_labeled_power"]) for r in by_label], width, label="labeled")
        ax.bar(xs, [float(r["mean_unrestricted_cov_power"]) for r in by_label], width, label="full covariance")
        ax.bar(
            xs + width,
            [float(r["mean_feature_independent_power"]) for r in by_label],
            width,
            label="feature-independent",
        )
        ax.axhline(args.alpha, color="0.4", linestyle="--", linewidth=1.0)
        ax.set_ylabel("detection power")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title("Detector-class stress test")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(outdir / "real_detector_assumption_stress.pdf")
        fig.savefig(outdir / "real_detector_assumption_stress.png", dpi=240)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] plotting failed: {exc}")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-path", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--direction-kinds", default="axis,k2,k4,k8,k16,k32,dense,pc_orth")
    parser.add_argument("--direction-seeds", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--baseline-tokens", type=int, default=20000)
    parser.add_argument("--amp", type=float, default=0.7)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260503)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
