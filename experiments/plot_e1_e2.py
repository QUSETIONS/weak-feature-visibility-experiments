#!/usr/bin/env python3
"""Main-text figure for E1/E2 synthetic: same-lambda bars + rotation curve."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    src = Path("results/e1_e2_summary.json")
    raw = json.loads(Path("results/e1_e2_raw.json").read_text())
    summary = json.loads(src.read_text())
    pair = summary["pair"]["0.55"]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

    ax = axes[0]
    stats = ["label", "cov", "ind"]
    x = np.arange(len(stats))
    width = 0.36
    for i, geo in enumerate(("axis", "dense")):
        means = [pair[geo][s]["mean"] for s in stats]
        yerr = np.array(
            [
                [pair[geo][s]["mean"] - pair[geo][s]["ci95"][0] for s in stats],
                [pair[geo][s]["ci95"][1] - pair[geo][s]["mean"] for s in stats],
            ]
        )
        ax.bar(x + (i - 0.5) * width, means, width, yerr=yerr, capsize=3, label=geo)
    ax.set_xticks(x, stats)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("power")
    ax.set_title(r"same $\lambda$, opposite sides of the boundary")
    ax.axhline(0.05, color="0.5", ls="--", lw=0.8)
    ax.legend(frameon=False)

    ax = axes[1]
    rot = raw["rotation"]
    by = {}
    for r in rot:
        by.setdefault(r["theta_deg"], {"C": r["one_minus_l4"], "ind": [], "label": []})
        by[r["theta_deg"]]["ind"].append(r["ind"])
        by[r["theta_deg"]]["label"].append(r["label"])
    thetas = sorted(by)
    C = [by[t]["C"] for t in thetas]
    ind_m = [np.mean(by[t]["ind"]) for t in thetas]
    ind_s = [np.std(by[t]["ind"], ddof=1) / np.sqrt(len(by[t]["ind"])) for t in thetas]
    lab_m = [np.mean(by[t]["label"]) for t in thetas]
    ax.errorbar(C, ind_m, yerr=ind_s, fmt="o-", label="ind")
    ax.plot(C, lab_m, "s--", label="label")
    ax.set_xlabel(r"$1-\|B^\top w\|_4^4$")
    ax.set_ylabel("power")
    ax.set_ylim(0, 1.05)
    ax.set_title("rotate the basis, not $\\lambda$")
    ax.legend(frameon=False)

    fig.tight_layout()
    out = Path("results/fig_e1_e2_main.pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
