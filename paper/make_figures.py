#!/usr/bin/env python3
"""Publication figures for the ICLR rewrite."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

C_AXIS = "#2c5aa0"
C_DENSE = "#c45c26"
C_LAST = "#2e7d4f"


def chance_max_cosine(m, d, n=4000, seed=0):
    """Expected max |<atom, e_1>| for a random unit dictionary."""
    rng = np.random.default_rng(seed)
    w = np.zeros(m)
    w[0] = 1.0
    u = rng.standard_normal((m, d, n))
    u /= np.linalg.norm(u, axis=0, keepdims=True) + 1e-12
    dots = np.abs(np.einsum("m,mdn->dn", w, u))
    return float(dots.max(axis=0).mean())


def err_from_ci(mean, ci):
    return np.array([mean - ci[0], ci[1] - mean])


def fig_e1():
    raw = json.loads((OLD / "experiments/results/e1_e2_raw.json").read_text())
    summary = json.loads((OLD / "experiments/results/e1_e2_summary.json").read_text())
    pair = summary["pair"]["0.55"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15))

    ax = axes[0]
    stats = ["label", "cov", "ind"]
    x = np.arange(len(stats))
    w = 0.36
    for i, (geo, color, hatch) in enumerate((("axis", C_AXIS, None), ("dense", C_DENSE, "///"))):
        means = [pair[geo][s]["mean"] for s in stats]
        yerr = np.vstack([err_from_ci(pair[geo][s]["mean"], pair[geo][s]["ci95"]) for s in stats]).T
        ax.bar(
            x + (i - 0.5) * w,
            means,
            w,
            yerr=yerr,
            capsize=2.5,
            color=color,
            hatch=hatch,
            label=geo,
            error_kw={"elinewidth": 0.8},
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x, stats)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("detection power")
    ax.set_title(r"Same $\lambda$, opposite sides of the boundary")
    ax.axhline(0.05, color="0.55", ls="--", lw=0.7)
    ax.legend(frameon=False, loc="center left")

    ax = axes[1]
    by = {}
    for r in raw["rotation"]:
        by.setdefault(r["theta_deg"], {"C": r["one_minus_l4"], "ind": [], "label": []})
        by[r["theta_deg"]]["ind"].append(r["ind"])
        by[r["theta_deg"]]["label"].append(r["label"])
    thetas = sorted(by)
    C = [by[t]["C"] for t in thetas]
    ind_m = [np.mean(by[t]["ind"]) for t in thetas]
    ind_s = [np.std(by[t]["ind"], ddof=1) / np.sqrt(len(by[t]["ind"])) for t in thetas]
    lab_m = [np.mean(by[t]["label"]) for t in thetas]
    ax.errorbar(C, ind_m, yerr=ind_s, fmt="o-", color=C_AXIS, lw=1.2, ms=4.5, capsize=2, label="ind")
    ax.plot(C, lab_m, "s--", color=C_DENSE, lw=1.2, ms=4.5, label="label")
    ax.set_xlabel(r"$1-\|B^{\top}w\|_{4}^{4}$")
    ax.set_ylabel("detection power")
    ax.set_ylim(0, 1.08)
    ax.set_title(r"Rotate the basis, not $\lambda$")
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT / "fig_same_lambda.pdf")
    fig.savefig(OUT / "fig_same_lambda.png", dpi=200)
    plt.close()


def fig_sae_wall():
    """Title-scale plot: SAE tracks λ, ind tracks geometry."""
    rows = json.loads((OLD / "experiments/results_v2/e4v2_raw.json").read_text())
    wall_path = OLD / "experiments/results_wall/sae_wall_summary.json"
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15))

    by = {}
    for r in rows:
        by.setdefault((r["geometry"], r["lambda_eff"]), {"cos": [], "rec": [], "ind": []})
        by[(r["geometry"], r["lambda_eff"])]["cos"].append(r["decoder_cosine"])
        by[(r["geometry"], r["lambda_eff"])]["rec"].append(float(r["recovered"]))
        by[(r["geometry"], r["lambda_eff"])]["ind"].append(r["det_ind"])

    if wall_path.exists():
        wall = json.loads(wall_path.read_text())
        lams = [float(x) for x in wall["by"]["axis"]]
        lams = sorted(lams)
        ax = axes[0]
        for geo, color, mk in (("axis", C_AXIS, "o"), ("dense", C_DENSE, "s")):
            ys = [wall["by"][geo][str(l)]["sae_cosine"]["mean"] for l in lams]
            yerr = np.vstack(
                [
                    err_from_ci(
                        wall["by"][geo][str(l)]["sae_cosine"]["mean"],
                        wall["by"][geo][str(l)]["sae_cosine"]["ci95"],
                    )
                    for l in lams
                ]
            ).T
            ax.errorbar(lams, ys, yerr=yerr, fmt=mk + "-", color=color, lw=1.2, ms=5, capsize=2, label=f"SAE {geo}")
        chance = chance_max_cosine(8, 16)
        ax.axhline(chance, color="0.55", ls=":", lw=0.8)
        ax.axhline(0.80, color="0.55", ls="--", lw=0.7)
        ax.set_xlabel(r"$\lambda_{\mathrm{eff}}$")
        ax.set_ylabel("decoder cosine")
        ax.set_ylim(0, 1.08)
        ax.set_title("SAE cosine")
        ax.legend(frameon=False, loc="lower right", fontsize=7, borderpad=0.2)
        ax = axes[1]
        for geo, color, mk in (("axis", C_AXIS, "o"), ("dense", C_DENSE, "s")):
            ys = [wall["by"][geo][str(l)]["ind"]["mean"] for l in lams]
            yerr = np.vstack(
                [
                    err_from_ci(
                        wall["by"][geo][str(l)]["ind"]["mean"],
                        wall["by"][geo][str(l)]["ind"]["ci95"],
                    )
                    for l in lams
                ]
            ).T
            ax.errorbar(lams, ys, yerr=yerr, fmt=mk + "-", color=color, lw=1.2, ms=5, capsize=2, label=geo)
        ax.set_xlabel(r"$\lambda_{\mathrm{eff}}$")
        ax.set_ylabel("ind detection power")
        ax.set_ylim(0, 1.08)
        ax.set_title("Ind power")
        ax.legend(frameon=False, loc="center right", fontsize=7)
    else:
        lams = sorted({r["lambda_eff"] for r in rows})
        ax = axes[0]
        for geo, color, mk in (("axis", C_AXIS, "o"), ("dense", C_DENSE, "s")):
            xs, ys, ses = [], [], []
            for lam in lams:
                v = np.array(by[(geo, lam)]["cos"])
                xs.append(lam)
                ys.append(v.mean())
                ses.append(v.std(ddof=1) / np.sqrt(len(v)))
            ax.errorbar(xs, ys, yerr=ses, fmt=mk + "-", color=color, lw=1.2, ms=5, capsize=2, label=geo)
        ax.axhline(0.80, color="0.55", ls="--", lw=0.7)
        ax.set_xlabel(r"$\lambda_{\mathrm{eff}}$")
        ax.set_ylabel("decoder cosine")
        ax.set_ylim(0, 1.08)
        ax.set_title("SAE cosine vs strength")
        ax.legend(frameon=False, loc="lower right")

        ax = axes[1]
        width = 0.12
        x = np.arange(len(lams))
        for i, (geo, color, hatch) in enumerate((("axis", C_AXIS, None), ("dense", C_DENSE, "///"))):
            rec = [np.mean(by[(geo, lam)]["rec"]) for lam in lams]
            ind = [np.mean(by[(geo, lam)]["ind"]) for lam in lams]
            ax.plot(x + (i - 0.5) * 0.08, rec, "o-", color=color, lw=1.3, ms=6, label=f"SAE {geo}")
            ax.plot(x + (i - 0.5) * 0.08, ind, "s--", color=color, lw=1.0, ms=5, alpha=0.85, label=f"ind {geo}")
        ax.set_xticks(x, [str(l) for l in lams])
        ax.set_xlabel(r"$\lambda_{\mathrm{eff}}$")
        ax.set_ylabel("recovery / ind power")
        ax.set_ylim(0, 1.08)
        ax.set_title("Ind splits geometries; SAE does not")
        ax.legend(frameon=False, loc="center right", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig_sae_wall.pdf")
    fig.savefig(OUT / "fig_sae_wall.png", dpi=200)
    plt.close()


def fig_sae_floor():
    rows = json.loads((OLD / "experiments/results_sae_floor/sae_floor_raw.json").read_text())
    by = {}
    for r in rows:
        by.setdefault(r["N"], {"ax": [], "rd": [], "obs": [], "ind": [], "sae": []})
        by[r["N"]]["ax"].append(r["rec_axis"])
        by[r["N"]]["rd"].append(r["rec_rand"])
        by[r["N"]]["obs"].append(r["f_obs"])
        by[r["N"]]["ind"].append(r["f_pred_ind"])
        by[r["N"]]["sae"].append(r["f_pred_sae"])
    Ns = sorted(by)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15))

    ax = axes[0]
    ax.errorbar(
        Ns,
        [np.mean(by[n]["ax"]) for n in Ns],
        yerr=[np.std(by[n]["ax"], ddof=1) / np.sqrt(len(by[n]["ax"])) for n in Ns],
        fmt="o-",
        color=C_AXIS,
        lw=1.2,
        ms=5,
        capsize=2,
        label="axis",
    )
    ax.errorbar(
        Ns,
        [np.mean(by[n]["rd"]) for n in Ns],
        yerr=[np.std(by[n]["rd"], ddof=1) / np.sqrt(len(by[n]["rd"])) for n in Ns],
        fmt="s-",
        color=C_DENSE,
        lw=1.2,
        ms=5,
        capsize=2,
        label="random",
    )
    ax.set_xscale("log")
    ax.set_ylim(0.5, 1.05)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("SAE recovery")
    ax.set_title("Known mix: axis is not harder")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    cap_path = OLD / "experiments/results_capacity/sae_capacity_summary.json"
    if cap_path.exists():
        cap = json.loads(cap_path.read_text())
        ds = [int(x) for x in cap["by_dict"]]
        ds = sorted(ds)
        ax.plot(ds, [cap["by_dict"][str(d)]["f_obs"] for d in ds], "o-", color="0.2", lw=1.2, ms=5, label="SAE missed mass")
        ax.plot(ds, [cap["by_dict"][str(d)]["f_pred_ind"] for d in ds], "s--", color=C_AXIS, lw=1.2, ms=5, label="ind predictor")
        ax.plot(ds, [cap["by_dict"][str(d)]["f_pred_sae"] for d in ds], "^:", color=C_DENSE, lw=1.2, ms=5, label="quartic floor")
        ax.set_xlabel("dictionary size")
        ax.set_ylabel("variance share")
        ax.set_ylim(0, 0.95)
        ax.set_title("Leftover tracks capacity")
        ax.legend(frameon=False, loc="upper right", fontsize=7)
    else:
        ax.plot(Ns, [np.mean(by[n]["obs"]) for n in Ns], "o-", color="0.2", lw=1.2, ms=5, label="SAE missed mass")
        ax.plot(Ns, [np.mean(by[n]["ind"]) for n in Ns], "s--", color=C_AXIS, lw=1.2, ms=5, label="ind predictor")
        ax.plot(Ns, [np.mean(by[n]["sae"]) for n in Ns], "^:", color=C_DENSE, lw=1.2, ms=5, label="quartic predictor")
        ax.set_xscale("log")
        ax.set_ylim(0, 0.4)
        ax.set_xlabel(r"$N$")
        ax.set_ylabel("variance share")
        ax.set_title("Missed mass vs two floors")
        ax.legend(frameon=False, loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "fig_sae_floor.pdf")
    fig.savefig(OUT / "fig_sae_floor.png", dpi=200)
    plt.close()


def _bar_cov_ind(ax, wall, title):
    keys = sorted(wall["by"]["axis"], key=float)
    lam_k = "0.35" if "0.35" in wall["by"]["axis"] else keys[0]
    stats = ["det_cov", "det_ind"]
    x = np.arange(len(stats))
    w = 0.24
    geos = (("axis", C_AXIS, None), ("last", C_LAST, ".."), ("dense", C_DENSE, "///"))
    if "last" not in wall["by"]:
        geos = (("axis", C_AXIS, None), ("dense", C_DENSE, "///"))
        w = 0.36
    for i, (geo, color, hatch) in enumerate(geos):
        off = (i - (len(geos) - 1) / 2.0) * w
        means = [wall["by"][geo][lam_k][s]["mean"] for s in stats]
        yerr = np.vstack(
            [err_from_ci(wall["by"][geo][lam_k][s]["mean"], wall["by"][geo][lam_k][s]["ci95"]) for s in stats]
        ).T
        ax.bar(
            x + off,
            means,
            w,
            yerr=yerr,
            capsize=2.0,
            color=color,
            hatch=hatch,
            label=geo,
            error_kw={"elinewidth": 0.8},
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x, ["cov", "ind"])
    ax.set_ylabel("detection power")
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.axhline(0.05, color="0.55", ls="--", lw=0.7)
    ax.legend(frameon=False, loc="upper right", fontsize=6.5)


def fig_gpt2():
    wall = json.loads((OLD / "experiments/results_gpt2_corrected_20260910/gpt2_summary.json").read_text())
    keys = sorted(wall["by"]["axis"], key=float)
    lams = [float(k) for k in keys]
    geos = [("axis", C_AXIS, "o"), ("dense", C_DENSE, "s")]
    if "last" in wall["by"]:
        geos = [("axis", C_AXIS, "o"), ("last", C_LAST, "D"), ("dense", C_DENSE, "s")]
    py_path = OLD / "experiments/results_pythia_corrected_20260910/pythia_summary.json"
    un_path = OLD / "experiments/results_unsup/unsup_summary.json"
    unp_path = OLD / "experiments/results_unsup_pythia/unsup_summary.json"
    two_row = py_path.exists() or un_path.exists()
    if two_row:
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.55))
        ax00, ax01 = axes[0]
        ax10, ax11 = axes[1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15))
        ax00, ax01 = axes[0], axes[1]
        ax10 = ax11 = None

    ax = ax00
    for geo, color, mk in geos:
        ys = [wall["by"][geo][k]["decoder_cosine"]["mean"] for k in keys]
        yerr = np.vstack(
            [
                err_from_ci(
                    wall["by"][geo][k]["decoder_cosine"]["mean"],
                    wall["by"][geo][k]["decoder_cosine"]["ci95"],
                )
                for k in keys
            ]
        ).T
        ax.errorbar(lams, ys, yerr=yerr, fmt=mk + "-", color=color, lw=1.2, ms=5, capsize=2, label=geo)
    ax.axhline(0.80, color="0.55", ls="--", lw=0.7)
    ax.set_xlabel(r"$\lambda_{\mathrm{eff}}$")
    ax.set_ylabel("decoder cosine")
    ax.set_ylim(0, 1.08)
    ax.set_title("GPT-2 SAE cosine")
    ax.legend(frameon=False, loc="lower right")

    _bar_cov_ind(ax01, wall, r"GPT-2 $\lambda{=}0.35$")
    if ax10 is not None:
        if py_path.exists():
            py = json.loads(py_path.read_text())
            _bar_cov_ind(ax10, py, r"Pythia-70m $\lambda{=}0.35$")
        else:
            ax10.axis("off")
        labs, vals = [], []
        if un_path.exists():
            u = json.loads(un_path.read_text())
            labs += ["tok", "rnd"]
            vals += [u["token_sae_mean"], u["random_sae_mean"]]
        if unp_path.exists():
            up = json.loads(unp_path.read_text())
            if up.get("token_sae_mean") is not None:
                labs += ["tok-P", "rnd-P"]
                vals += [up["token_sae_mean"], up["random_sae_mean"]]
        if labs:
            cols = [C_AXIS, "0.65", C_LAST, "0.75"][: len(labs)]
            ax11.bar(np.arange(len(vals)), vals, color=cols, edgecolor="white", linewidth=0.4)
            ax11.set_xticks(np.arange(len(vals)), labs)
            ax11.set_ylim(0, 1.08)
            ax11.set_ylabel("SAE cosine")
            ax11.set_title("Uninjected dirs")
            ax11.axhline(0.80, color="0.55", ls="--", lw=0.7)
        else:
            ax11.axis("off")

    fig.tight_layout()
    fig.savefig(OUT / "fig_gpt2.pdf")
    fig.savefig(OUT / "fig_gpt2.png", dpi=200)
    plt.close()


def fig_e4():
    e4 = json.loads((OLD / "experiments/results_v2/e4v2_summary.json").read_text())
    e2 = json.loads((OLD / "experiments/results_v2/e2sep_summary.json").read_text())
    raw2 = json.loads((OLD / "experiments/results_v2/e2sep_raw.json").read_text())
    c = e4["lambda055"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15))

    ax = axes[0]
    stats = ["label", "cov", "ind", "sae_recovery"]
    labels = ["label", "cov", "ind", "SAE"]
    x = np.arange(len(stats))
    w = 0.36
    for i, (geo, color, hatch) in enumerate((("axis", C_AXIS, None), ("dense", C_DENSE, "///"))):
        means = [c[geo][s]["mean"] for s in stats]
        yerr = np.vstack([err_from_ci(c[geo][s]["mean"], c[geo][s]["ci95"]) for s in stats]).T
        ax.bar(
            x + (i - 0.5) * w,
            means,
            w,
            yerr=yerr,
            capsize=2.5,
            color=color,
            hatch=hatch,
            label=geo,
            error_kw={"elinewidth": 0.8},
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("power / recovery")
    ax.set_title(r"Matched strength: detector split; TopK matched", fontsize=9, pad=3)
    legend_handles, legend_labels = ax.get_legend_handles_labels()

    ax = axes[1]
    by = {}
    for r in raw2:
        by.setdefault(r["name"], []).append(r["ind"])
    order = ["twosparse_eps0.01", "twosparse_eps0.05", "twosparse_eps0.20", "twosparse_eps0.50"]
    xs = np.arange(len(order))
    means = [np.mean(by[k]) for k in order]
    ses = [np.std(by[k], ddof=1) / np.sqrt(len(by[k])) for k in order]
    ax.bar(xs, means, color=C_AXIS, yerr=ses, capsize=2.5, error_kw={"elinewidth": 0.8})
    ax.set_xticks(xs, [r"$\varepsilon$=0.01", "0.05", "0.20", "0.50"])
    ax.set_ylabel("ind power")
    ax.set_ylim(0, 1.08)
    ax.set_title(r"2-sparse follows $1-\|u\|_4^4$")

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        fontsize=7,
        borderpad=0.1,
        handlelength=1.4,
    )
    fig.savefig(OUT / "fig_sae_vs_ind.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "fig_sae_vs_ind.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print("e2 corr", e2["corr"])


def fig_schematic():
    m = 4
    w_axis = np.zeros(m)
    w_axis[0] = 1.0
    w_dense = np.full(m, 0.5)
    spike_axis = np.outer(w_axis, w_axis)
    spike_dense = np.outer(w_dense, w_dense)
    off_axis = spike_axis.copy()
    off_dense = spike_dense.copy()
    np.fill_diagonal(off_axis, np.nan)
    np.fill_diagonal(off_dense, np.nan)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "spike", [(0.0, "#f4f6f8"), (0.22, "#8fb0d6"), (1.0, "#1f4e8c")]
    )
    cmap = cmap.copy()
    cmap.set_bad("white")

    fig = plt.figure(figsize=(5.5, 3.05))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.28, 1.0, 1.0],
        left=0.05,
        right=0.99,
        top=0.90,
        bottom=0.10,
        wspace=0.38,
        hspace=0.62,
    )
    ax_lab = fig.add_subplot(gs[:, 0])
    ax_ca = fig.add_subplot(gs[0, 1])
    ax_ia = fig.add_subplot(gs[0, 2])
    ax_cd = fig.add_subplot(gs[1, 1])
    ax_id = fig.add_subplot(gs[1, 2])

    xs = np.linspace(-3.1, 4.1, 400)
    h0 = np.exp(-0.5 * xs**2)
    h1 = np.exp(-0.5 * (xs - 1.45) ** 2)
    ax_lab.fill_between(xs, h0, color="0.82", alpha=0.95)
    ax_lab.plot(xs, h0, color="0.45", lw=1.15, label=r"$H_0$")
    ax_lab.fill_between(xs, h1, color=C_AXIS, alpha=0.20)
    ax_lab.plot(xs, h1, color=C_AXIS, lw=1.55, label=r"$H_1$")
    ax_lab.annotate(
        "",
        xy=(1.45, 0.10),
        xytext=(0.0, 0.10),
        arrowprops=dict(arrowstyle="<->", color=C_DENSE, lw=1.15),
    )
    ax_lab.text(0.72, 0.17, r"$\lambda$", color=C_DENSE, ha="center", fontsize=9)
    ax_lab.set_xlim(-3.1, 4.1)
    ax_lab.set_ylim(0, 1.28)
    ax_lab.set_xticks([])
    ax_lab.set_yticks([])
    ax_lab.legend(frameon=False, loc="upper right", fontsize=7, borderpad=0.1)
    ax_lab.set_title("label: mean shift", fontsize=9, pad=4)
    ax_lab.set_xlabel(r"$w^{\top}h$", fontsize=8, labelpad=1)

    def heat(ax, M, title, note):
        ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1.0, interpolation="nearest")
        for i in range(m):
            for j in range(m):
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="0.88", lw=0.45)
                )
        ax.set_xticks(range(m), [str(i + 1) for i in range(m)])
        ax.set_yticks(range(m), [str(i + 1) for i in range(m)])
        ax.tick_params(length=0, labelsize=7, pad=1)
        ax.set_title(title, fontsize=9, pad=2)
        ax.set_xlabel(note, fontsize=8, labelpad=1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    heat(ax_ca, spike_axis, "cov, axis", r"$ww^{\top}$")
    heat(ax_ia, off_axis, "ind, axis", r"$C=0$")
    heat(ax_cd, spike_dense, "cov, dense", r"$ww^{\top}$")
    heat(ax_id, off_dense, "ind, dense", r"$C=1-1/m$")
    ax_ia.text(1.5, 1.5, "absorbed", ha="center", va="center", color="0.55", fontsize=8)

    fig.savefig(OUT / "fig_schematic.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "fig_schematic.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close()


def _interp_nstar(Ns, cs, thr):
    Ns = np.asarray(Ns, float)
    cs = np.asarray(cs, float)
    if cs[0] >= thr:
        return float(Ns[0]), True
    for i in range(1, len(Ns)):
        if cs[i] >= thr:
            c0, c1 = cs[i - 1], cs[i]
            if c1 <= c0:
                return float(Ns[i]), False
            t = (thr - c0) / (c1 - c0)
            logn = np.log(Ns[i - 1]) + t * (np.log(Ns[i]) - np.log(Ns[i - 1]))
            return float(np.exp(logn)), False
    return float(Ns[-1]), True


def _load_cos_table(path):
    raw = json.loads(path.read_text())
    by = {}
    lams, Ns = set(), set()
    for r in raw:
        by.setdefault((r["channel"], r["geometry"], r["lambda_eff"], r["N_train"]), []).append(
            r["decoder_cosine"]
        )
        lams.add(r["lambda_eff"])
        Ns.add(r["N_train"])
    return by, sorted(lams), sorted(Ns)


def fig_public_atoms():
    # Compact diagnostic for the nine published Bloom SAE atoms.
    path = OLD / "experiments/results_public_atoms/atoms.json"
    if not path.exists():
        print("skip fig_public_atoms (no public atom scores)")
        return
    rows = json.loads(path.read_text())
    rows = sorted(rows, key=lambda r: r["C_resid"])
    atoms = [str(r["atom"]) for r in rows]
    C = np.array([r["C_resid"] for r in rows], dtype=float)
    label_cos = np.array([r["label_cosine"] for r in rows], dtype=float)
    powers = np.array([[r["power_N1024"][k] for k in ("label", "cov", "ind")] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.05), gridspec_kw={"width_ratios": [1.05, 1.35]})
    ax = axes[0]
    colors = [C_AXIS if c < 0.2 else C_DENSE for c in C]
    ax.scatter(C, label_cos, c=colors, s=22, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(0.2, color="0.6", ls=":", lw=0.7)
    ax.set_xlabel(r"$C_{\mathrm{resid}}$")
    ax.set_ylabel("encoder-label cosine")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Labels survive low $C$")
    ax.annotate("23123", (C[0], label_cos[0]), xytext=(7, -10), textcoords="offset points", fontsize=6.5, color=C_AXIS)

    ax = axes[1]
    x = np.arange(len(rows))
    width = 0.24
    for j, (name, color) in enumerate((("label", C_AXIS), ("cov", C_LAST), ("ind", C_DENSE))):
        ax.bar(x + (j - 1) * width, powers[:, j], width, color=color, label=name, edgecolor="white", linewidth=0.35)
    ax.set_xticks(x, atoms, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("power, $N=1024$")
    ax.set_ylim(0, 1.08)
    ax.set_title("Same atoms, different detectors")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=3, borderpad=0.1, handlelength=1.2)

    fig.tight_layout(pad=0.45)
    fig.savefig(OUT / "fig_public_atoms.pdf")
    fig.savefig(OUT / "fig_public_atoms.png", dpi=220)
    plt.close()
    print("wrote fig_public_atoms")


def fig_sae_exponent():
    """Labeled vs unlabeled vs oracle SAE, m=32, constant epochs."""
    wide_path = OLD / "experiments/results_exponent_wide/sae_exponent_raw.json"
    lab_path = OLD / "experiments/results_label_nstar/label_nstar_raw.json"
    if not lab_path.exists():
        lab_path = OLD / "experiments/results_exponent_m32/sae_exponent_raw.json"
    if not wide_path.exists():
        print("skip fig_sae_exponent (no wide results)")
        return
    by_w, lams_w, Ns_w = _load_cos_table(wide_path)
    by_l, lams_l, Ns_l = _load_cos_table(lab_path) if lab_path.exists() else (by_w, lams_w, Ns_w)
    thr = 0.80
    chance = chance_max_cosine(32, 16)

    def mean_curve(by, ch, geo, lam, Ns):
        return [float(np.mean(by[(ch, geo, lam, N)])) for N in Ns]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15))
    ax = axes[0]
    # labels from the small-N m=32 grid; SAE/oracle from the wide grid
    for ch, by, lams, Ns, fmt, color, lab, use_cens in (
        ("label_mean", by_l, lams_l, Ns_l, "o-", C_AXIS, "label, axis", False),
        ("label_mean", by_l, lams_l, Ns_l, "s--", C_DENSE, "label, dense", False),
        ("unlab_sae", by_w, lams_w, Ns_w, "^:", C_AXIS, "SAE, axis", True),
        ("unlab_sae", by_w, lams_w, Ns_w, "v:", C_DENSE, "SAE, dense", True),
        ("label_sae", by_w, lams_w, Ns_w, "D-.", "0.35", "oracle, axis", False),
    ):
        if "dense" in lab:
            geo = "dense"
        else:
            geo = "axis"
        pts = [_interp_nstar(Ns, mean_curve(by, ch, geo, lam, Ns), thr) for lam in lams]
        ys = np.array([p[0] for p in pts])
        cens = np.array([p[1] for p in pts])
        mk, ls = fmt[0], fmt[1:]
        if use_cens:
            ax.plot(np.array(lams)[~cens], ys[~cens], mk + ls, color=color, lw=1.2, ms=5, label=lab)
            if cens.any():
                ax.plot(np.array(lams)[cens], ys[cens], mk, color=color, ms=5, mfc="white", mew=1.0)
        else:
            ax.plot(lams, ys, fmt, color=color, lw=1.2, ms=5, label=lab)
    xref = np.linspace(0.28, 0.67, 60)
    m_th, p_th, c2 = 32, 0.05, 0.36

    def n_label_th(lam):
        # The conditional difference-of-means stage has variance 1/[p(1-p)N].
        return (16.0 / 9.0) * (m_th - 1) / ((1.0 - p_th) * lam**2)

    def n_pca_th(lam):
        beta = (lam**2) * (1.0 - p_th)
        return (m_th - 1) * (1.0 + beta) / (c2 * beta**2)

    ax.plot(xref, [n_label_th(x) for x in xref], color="0.55", lw=0.7, ls="-")
    ax.plot(xref, [n_pca_th(x) for x in xref], color="0.55", lw=0.7, ls="--")
    pca_path = OLD / "experiments/results_regime_pca/sae_regime_raw.json"
    if pca_path.exists():
        by_p, lams_p, Ns_p = _load_cos_table(pca_path)
        ys = [
            _interp_nstar(Ns_p, mean_curve(by_p, "pca", "axis", lam, Ns_p), thr)[0] for lam in lams_p
        ]
        ax.plot(lams_p, ys, "x", color="0.25", ms=4.5, mew=0.9, label="PCA emp.")
    reg_path = OLD / "experiments/results_regime/sae_regime_raw.json"
    if reg_path.exists():
        by_r, lams_r, Ns_r = _load_cos_table(reg_path)
        if any(k[0] == "atom1_lab" for k in by_r):
            ys = [
                _interp_nstar(Ns_r, mean_curve(by_r, "atom1_lab", "axis", lam, Ns_r), thr)[0]
                for lam in lams_r
            ]
            ax.plot(lams_r, ys, "P--", color="0.35", lw=1.1, ms=4.5, label="1-atom oracle")
    ax.text(0.50, 95, r"$\lambda^{-2}$", color="0.4", fontsize=7)
    ax.text(0.52, 900, r"PCA $\lambda^{-4}$", color="0.4", fontsize=7)
    ax.set_yscale("log")
    ax.set_xlim(0.27, 0.68)
    ax.set_ylim(80, 2e5)
    ax.set_xticks([0.30, 0.38, 0.48, 0.65])
    ax.set_xlabel(r"$\lambda_{\mathrm{eff}}$")
    ax.set_ylabel(r"$N^\star$")
    ax.set_title(r"Labels vs free SAE $N^\star$")
    ax.legend(frameon=False, loc="upper left", fontsize=6, borderpad=0.2)

    ax = axes[1]
    lam = 0.38
    for ch, geo, fmt, color, lab in (
        ("label_mean", "axis", "o-", C_AXIS, "label"),
        ("label_sae", "axis", "D--", "0.35", "oracle SAE"),
        ("unlab_sae", "axis", "^:", C_AXIS, "free SAE, axis"),
        ("unlab_sae", "dense", "v:", C_DENSE, "free SAE, dense"),
    ):
        ys = mean_curve(by_w, ch, geo, lam, Ns_w)
        yerr = [
            float(np.std(by_w[(ch, geo, lam, N)], ddof=1) / np.sqrt(len(by_w[(ch, geo, lam, N)])))
            for N in Ns_w
        ]
        ax.errorbar(Ns_w, ys, yerr=yerr, fmt=fmt, color=color, lw=1.2, ms=4.5, capsize=2, label=lab)
    ax.axhline(chance, color="0.55", ls=":", lw=0.8)
    ax.axhline(thr, color="0.55", ls="--", lw=0.7)
    ax.set_xscale("log")
    ax.set_ylim(0.15, 1.05)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("decoder cosine")
    ax.set_title(r"Same $\lambda=0.38$, three channels")
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig_sae_exponent.pdf")
    fig.savefig(OUT / "fig_sae_exponent.png", dpi=200)
    plt.close()
    print("wrote fig_sae_exponent wide")


if __name__ == "__main__":
    fig_e1()
    fig_e4()
    fig_sae_wall()
    fig_sae_floor()
    fig_gpt2()
    fig_schematic()
    fig_sae_exponent()
    fig_public_atoms()
    print("wrote", OUT)
