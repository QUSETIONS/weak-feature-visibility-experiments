#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    e4 = json.loads(Path("results_v2/e4v2_summary.json").read_text())
    e2 = json.loads(Path("results_v2/e2sep_summary.json").read_text())
    raw2 = json.loads(Path("results_v2/e2sep_raw.json").read_text())
    c = e4["lambda055"]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    ax = axes[0]
    stats = ["label", "cov", "ind", "sae_recovery"]
    labels = ["label", "cov", "ind", "SAE"]
    x = np.arange(len(stats))
    w = 0.36
    for i, geo in enumerate(("axis", "dense")):
        means = [c[geo][s]["mean"] for s in stats]
        yerr = np.array(
            [
                [c[geo][s]["mean"] - c[geo][s]["ci95"][0] for s in stats],
                [c[geo][s]["ci95"][1] - c[geo][s]["mean"] for s in stats],
            ]
        )
        ax.bar(x + (i - 0.5) * w, means, w, yerr=yerr, capsize=3, label=geo)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("power / recovery")
    ax.set_title(r"$\lambda=0.55$: only ind cares about geometry")
    ax.legend(frameon=False, loc="center right")

    ax = axes[1]
    names = ["one_minus_l4", "l1", "one_minus_linf2", "participation_ratio"]
    pretty = [r"$1-\|u\|_4^4$", r"$\|u\|_1$", r"$1-\|u\|_\infty^2$", "PR"]
    vals = [abs(e2["corr"][k]) if e2["corr"][k] == e2["corr"][k] else 0 for k in names]
    ax.bar(np.arange(len(names)), vals)
    ax.set_xticks(np.arange(len(names)), pretty, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("|r| with ind power")
    ax.set_title("E2-sep: 4-norm vs competitors")

    fig.tight_layout()
    out = Path("results_v2/fig_v2.pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print("wrote", out)

    # small eps panel
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    by = {}
    for r in raw2:
        by.setdefault(r["name"], []).append(r["ind"])
    order = [k for k in ["twosparse_eps0.01", "twosparse_eps0.05", "twosparse_eps0.20", "twosparse_eps0.50"] if k in by]
    ax.bar(range(len(order)), [np.mean(by[k]) for k in order])
    ax.set_xticks(range(len(order)), [k.replace("twosparse_", "") for k in order], rotation=20)
    ax.set_ylabel("ind power")
    ax.set_title("2-sparse imbalance: 4-norm, not l1")
    fig.tight_layout()
    fig.savefig(Path("results_v2/fig_e2sep_eps.png"), dpi=160)


if __name__ == "__main__":
    main()
