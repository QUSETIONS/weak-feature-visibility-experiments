"""Geometry-only intervention on real activations.

This detector-only bridge fixes the activation background, sample size,
activation probability, and injected strength. It varies only the loading
geometry to test the feature visibility principle:

    E_ind ~ N (1 - ||w||_4^4) lambda^4.

The script intentionally does not train an SAE. It asks whether the known
off-diagonal statistic predicted by the theorem becomes more powerful as the
same-strength feature is spread across more coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        # Equal magnitude makes the geometry intervention especially clean:
        # 1 - ||w||_4^4 = 1 - 1/k.
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
    kinds = args.direction_kinds.split(",")
    direction_seeds = [int(v) for v in args.direction_seeds.split(",")]

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
            baseline_quad = float(np.mean(y_all[baseline_idx] ** 2 - q_all[baseline_idx]))

            null_labeled: list[float] = []
            null_unlabeled: list[float] = []
            alt_labeled: list[float] = []
            alt_unlabeled: list[float] = []

            for _ in range(args.trials):
                idx = rng.choice(x.shape[0], size=args.n, replace=False)
                yb = y_all[idx]
                qb = q_all[idx]
                rb = r_all[idx]

                z_null = rng.binomial(1, args.p, size=args.n).astype(np.float64)
                pos = z_null > 0.5
                neg = ~pos
                null_labeled.append(float(yb[pos].mean() - yb[neg].mean()) if pos.sum() > 1 and neg.sum() > 1 else 0.0)
                null_unlabeled.append(float(np.mean(yb * yb - qb) - baseline_quad))

                z = rng.binomial(1, args.p, size=args.n).astype(np.float64)
                zi = z - args.p
                ya = yb + args.amp * zi
                qa = qb + 2.0 * args.amp * zi * rb + (args.amp * args.amp) * (zi * zi) * w4_sum
                pos = z > 0.5
                neg = ~pos
                alt_labeled.append(float(ya[pos].mean() - ya[neg].mean()) if pos.sum() > 1 and neg.sum() > 1 else 0.0)
                alt_unlabeled.append(float(np.mean(ya * ya - qa) - baseline_quad))

            labeled_thr = float(np.quantile(null_labeled, 1.0 - args.alpha))
            unlabeled_thr = float(np.quantile(null_unlabeled, 1.0 - args.alpha))
            labeled_power = float(np.mean(np.array(alt_labeled) > labeled_thr))
            unlabeled_power = float(np.mean(np.array(alt_unlabeled) > unlabeled_thr))
            lambda_eff = float(args.amp * np.sqrt(args.p * (1.0 - args.p)))
            evidence_proxy = float(args.n * (geom / 4.0) * lambda_eff**4)
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
                    "evidence_proxy": evidence_proxy,
                    "labeled_power": labeled_power,
                    "unlabeled_power": unlabeled_power,
                    "labeled_threshold": labeled_thr,
                    "unlabeled_threshold": unlabeled_thr,
                    "trials": args.trials,
                }
            )

    write_csv(outdir / "real_geometry_intervention.csv", rows)

    by_label = []
    for label in sorted({str(r["direction_label"]) for r in rows}, key=lambda s: (s != "axis", s)):
        group = [r for r in rows if str(r["direction_label"]) == label]
        by_label.append(
            {
                "direction_label": label,
                "rows": len(group),
                "mean_geometry": float(np.mean([float(r["geometry"]) for r in group])),
                "mean_labeled_power": float(np.mean([float(r["labeled_power"]) for r in group])),
                "mean_unlabeled_power": float(np.mean([float(r["unlabeled_power"]) for r in group])),
                "sd_unlabeled_power": float(np.std([float(r["unlabeled_power"]) for r in group])),
                "mean_evidence_proxy": float(np.mean([float(r["evidence_proxy"]) for r in group])),
            }
        )
    write_csv(outdir / "real_geometry_intervention_by_direction.csv", by_label)

    geom = np.array([float(r["geometry"]) for r in rows])
    unlab = np.array([float(r["unlabeled_power"]) for r in rows])
    lab = np.array([float(r["labeled_power"]) for r in rows])
    summary = {
        "mode": "real_geometry_intervention",
        "activation_path": args.activation_path,
        "rows": len(rows),
        "amp": args.amp,
        "p": args.p,
        "n": args.n,
        "trials": args.trials,
        "lambda_eff": float(args.amp * np.sqrt(args.p * (1.0 - args.p))),
        "corr_unlabeled_power_geometry": float(np.corrcoef(geom, unlab)[0, 1]),
        "corr_labeled_power_geometry": float(np.corrcoef(geom, lab)[0, 1]),
        "axis_unlabeled_power": float([r for r in rows if r["direction_label"] == "axis"][0]["unlabeled_power"]),
        "axis_labeled_power": float([r for r in rows if r["direction_label"] == "axis"][0]["labeled_power"]),
        "max_unlabeled_power": float(np.max(unlab)),
    }
    with (outdir / "real_geometry_intervention_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        ax.scatter(geom, unlab, s=22, alpha=0.65, label="unlabeled")
        ax.scatter(geom, lab, s=18, alpha=0.35, label="labeled")
        xs = [r["mean_geometry"] for r in by_label]
        ys = [r["mean_unlabeled_power"] for r in by_label]
        labels = [r["direction_label"] for r in by_label]
        ax.plot(xs, ys, color="black", linewidth=1.8, marker="o", label="unlabeled mean")
        for x0, y0, label in zip(xs, ys, labels):
            ax.text(x0 + 0.01, y0 + 0.015, label, fontsize=7)
        ax.axhline(args.alpha, color="0.45", linestyle="--", linewidth=1.0)
        ax.set_xlabel(r"geometry $1-\|w\|_4^4$")
        ax.set_ylabel("detection power")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(-0.03, 1.03)
        ax.set_title("GPT-2 geometry intervention")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(outdir / "real_geometry_intervention.pdf")
        fig.savefig(outdir / "real_geometry_intervention.png", dpi=240)
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
