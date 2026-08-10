"""U5: measure the melting from both sides.

Left: correlation dimension D2 of tanh states vs rho (Grassberger-
Procaccia on 2k-state subsamples: slope of log C(r) vs log r over the
10th-40th distance-percentile window), against the Moran/IFS prediction
D = ln(27)/ln(1/rho). Right: the tropical pond's distinct-state count vs
1/|lambda| (from t7), diverging toward the 20k sample ceiling — the
same approach to criticality, seen from the discrete side.

    .venv/bin/python experiments/track_u/u5_moran_dimension.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rcllm.dump_states import DUMP_DIR, dump_name  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results", "track_u")
TRACK_T = os.path.join(ROOT, "results", "track_t")
RHOS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.95]
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#eceae6"


def corr_dim(states: np.ndarray, n_sub: int = 2000, seed: int = 7):
    rng = np.random.default_rng(seed)
    ds = []
    for rep in range(2):
        idx = rng.choice(states.shape[0], n_sub, replace=False)
        dist = pdist(states[idx].astype(np.float64))
        lo, hi = np.quantile(dist, [0.10, 0.40])
        rs = np.geomspace(lo, hi, 12)
        C = np.array([(dist <= r).mean() for r in rs])
        slope = np.polyfit(np.log(rs), np.log(C), 1)[0]
        ds.append(slope)
    return float(np.mean(ds)), float(abs(ds[0] - ds[1]) / 2)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    meas = {}
    for rho in RHOS:
        d = np.load(os.path.join(DUMP_DIR, dump_name(5000, rho, 1.0, 1.0, 0,
                                                     200_000)))
        meas[rho] = corr_dim(d["states"])
        print(f"rho={rho}: D2 = {meas[rho][0]:.2f} ± {meas[rho][1]:.2f} "
              f"(Moran: {np.log(27) / np.log(1 / rho):.2f})", flush=True)

    with open(os.path.join(TRACK_T, "t7_kstar_sweep.json")) as f:
        t7 = json.load(f)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=SURFACE)
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(MUTED)
        a.tick_params(colors=MUTED, labelcolor=INK2)
        a.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)

    xs = np.linspace(0.15, 0.99, 200)
    ax.plot(xs, np.log(27) / np.log(1 / xs), color=ORANGE, lw=2,
            label="Moran: ln 27 / ln(1/ρ)")
    ax.errorbar(RHOS, [meas[r][0] for r in RHOS],
                yerr=[max(meas[r][1], 0.05) for r in RHOS], color=BLUE,
                fmt="o", markersize=6, lw=2, capsize=3,
                label="measured D₂ (2k states)")
    ax.axhspan(6, 8, color="#eceae6", alpha=0.6, lw=0)
    ax.annotate("finite-sample ceiling", (0.17, 6.9), color=INK2, fontsize=9)
    ax.set_ylim(0, 12)
    ax.set_xlabel("spectral radius ρ", color=INK2)
    ax.set_ylabel("correlation dimension D₂", color=INK2)
    ax.set_title("Smooth pond: fractal dimension vs. ρ (leak=1, N=5000)",
                 color=INK, fontsize=11, loc="left", pad=10)
    ax.legend(frameon=False, labelcolor=INK2)

    lams = sorted((abs(float(k)), v["n_distinct_states"]) for k, v in t7.items())
    inv = [1 / l for l, _ in lams]
    ns = [n for _, n in lams]
    ax2.loglog(inv, ns, color=BLUE, lw=2, marker="o", markersize=6)
    ax2.axhline(20_000, color=MUTED, lw=1, ls="--")
    ax2.annotate("sample ceiling (20k)", (inv[0], 20_000),
                 textcoords="offset points", xytext=(4, 6), color=INK2,
                 fontsize=9)
    for (l, n), x in zip(lams, inv):
        ax2.annotate(f"λ=−{l}", (x, n), textcoords="offset points",
                     xytext=(6, -10), color=INK2, fontsize=8)
    ax2.set_xlabel("1 / |λ|  (distance from criticality)", color=INK2)
    ax2.set_ylabel("distinct states (of 20k samples)", color=INK2)
    ax2.set_title("Tropical pond: the automaton melts toward criticality",
                  color=INK, fontsize=11, loc="left", pad=10)

    fig.tight_layout()
    out = os.path.join(RESULTS, "u5_moran_melting.png")
    fig.savefig(out, dpi=150)
    with open(os.path.join(RESULTS, "u5_moran_dimension.json"), "w") as f:
        json.dump({"D2": {str(r): {"measured": meas[r][0], "spread": meas[r][1],
                                   "moran": float(np.log(27) / np.log(1 / r))}
                          for r in RHOS}}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
