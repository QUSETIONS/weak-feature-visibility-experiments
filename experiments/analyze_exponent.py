#!/usr/bin/env python3
"""Post-process constant-epoch labeled vs unlabeled SAE JSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "results_exponent"
RAW = ROOT / "sae_exponent_raw.json"
SUM = ROOT / "sae_exponent_summary.json"


def slope_loglog(xs, ys):
    x = np.log(np.asarray(xs, float))
    y = np.log(np.asarray(ys, float))
    A = np.vstack([x, np.ones_like(x)]).T
    sl, ic = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(sl), float(ic)


def interp_nstar(Ns, cs, thr):
    Ns = np.asarray(Ns, float)
    cs = np.asarray(cs, float)
    if cs[0] >= thr:
        return float(Ns[0]), "floor"
    for i in range(1, len(Ns)):
        if cs[i] >= thr:
            c0, c1 = cs[i - 1], cs[i]
            if c1 <= c0:
                return float(Ns[i]), "ok"
            t = (thr - c0) / (c1 - c0)
            logn = np.log(Ns[i - 1]) + t * (np.log(Ns[i]) - np.log(Ns[i - 1]))
            return float(np.exp(logn)), "ok"
    return float(Ns[-1]), "ceil"


def main():
    raw = json.loads(RAW.read_text())
    summary = json.loads(SUM.read_text()) if SUM.exists() else {}
    proto = summary.get("protocol") or {}
    lams = proto.get("lambdas") or sorted({r["lambda_eff"] for r in raw})
    Ns = proto.get("N_grid") or sorted({r["N_train"] for r in raw})
    print("n_rows", len(raw), "elapsed", summary.get("elapsed_sec"))
    print("protocol", {k: proto[k] for k in ("lambdas", "N_grid", "n_epochs", "n_seeds") if k in proto})
    if "collapse" in summary:
        print("collapse", json.dumps(summary["collapse"], indent=2))
    if "slopes" in summary:
        print("slopes", json.dumps(summary["slopes"], indent=2))

    by = {}
    for r in raw:
        key = (r["channel"], r["geometry"], r["lambda_eff"], r["N_train"])
        by.setdefault(key, {"cos": [], "slot0": []})
        by[key]["cos"].append(r["decoder_cosine"])
        s0 = r.get("slot0_cosine")
        if s0 is not None and s0 == s0:
            by[key]["slot0"].append(s0)

    print("\nmean cosine")
    for ch in ("label_mean", "label_sae", "unlab_sae"):
        print(f"\n== {ch} ==")
        for geo in ("axis", "dense"):
            print(f"  {geo}")
            for lam in lams:
                xs = []
                for N in Ns:
                    vals = by.get((ch, geo, lam, N), {}).get("cos", [])
                    if vals:
                        xs.append(f"{N}:{np.mean(vals):.2f}")
                print(f"    λ={lam:.2f}  " + "  ".join(xs))

    print("\nN* from mean cosine")
    for thr in (0.80, 0.90, 0.95):
        print(f"\n-- threshold {thr} --")
        for ch in ("label_mean", "unlab_sae"):
            for geo in ("axis", "dense"):
                l_used, n_used = [], []
                flags = []
                for lam in lams:
                    cs = []
                    ok = True
                    for N in Ns:
                        vals = by.get((ch, geo, lam, N), {}).get("cos", [])
                        if not vals:
                            ok = False
                            break
                        cs.append(float(np.mean(vals)))
                    if not ok:
                        continue
                    nstar, flag = interp_nstar(Ns, cs, thr)
                    l_used.append(lam)
                    n_used.append(nstar)
                    flags.append(flag)
                if len(l_used) >= 2:
                    sl, _ = slope_loglog(l_used, n_used)
                    unc = [(l, n) for l, n, f in zip(l_used, n_used, flags) if f == "ok"]
                    sl_u = slope_loglog(*zip(*unc))[0] if len(unc) >= 3 else float("nan")
                    print(
                        f"  {ch:12s} {geo:5s} slope_all={sl:6.2f}  slope_unc={sl_u:6.2f}  "
                        + " ".join(f"{l:.2f}:{n:.0f}{f[0]}" for l, n, f in zip(l_used, n_used, flags))
                    )


if __name__ == "__main__":
    main()
