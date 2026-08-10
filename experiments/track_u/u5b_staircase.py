"""U5b: the correlation-integral staircase — reading Moran off the stairs.

For a near-ultrametric attractor, C(r) is a staircase: each stair is one
suffix-tree level. Amended pre-registration: horizontal spacing
~= log2(1/rho) per stair, vertical drop ~= log2(27) ~= 4.75 bits, so the
across-stairs slope equals Moran's ln27/ln(1/rho); stairs blur as
rho -> 1.

    .venv/bin/python experiments/track_u/u5b_staircase.py
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
RHOS = [0.2, 0.4, 0.6, 0.95]
RAMP4 = ["#86b6ef", "#5598e7", "#2a78d6", "#104281"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#eceae6"


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(8.5, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    out = {}
    for rho, color in zip(RHOS, RAMP4):
        d = np.load(os.path.join(DUMP_DIR, dump_name(5000, rho, 1.0, 1.0, 0,
                                                     200_000)))
        idx = rng.choice(d["states"].shape[0], 3000, replace=False)
        dist = pdist(d["states"][idx].astype(np.float64))
        dist = dist[dist > 0]
        rmax = dist.max()
        rs = np.geomspace(dist.min(), rmax, 400)
        C = np.searchsorted(np.sort(dist), rs) / len(dist)
        keep = C > 0
        x, y = np.log2(rs[keep] / rmax), np.log2(C[keep])
        ax.plot(x, y, color=color, lw=2, label=f"ρ={rho}")
        # wide-window slope across levels: mass between 1e-4 and 0.5
        m = (C[keep] > 1e-4) & (C[keep] < 0.5)
        slope = np.polyfit(x[m], y[m], 1)[0] if m.sum() > 10 else float("nan")
        moran = np.log(27) / np.log(1 / rho)
        out[str(rho)] = {"wide_window_slope": round(float(slope), 2),
                         "moran": round(float(moran), 2)}
        print(f"rho={rho}: wide-window slope {slope:.2f} vs Moran {moran:.2f}",
              flush=True)

    # guide: Moran slope for rho=0.2 anchored lower-left
    ax.plot([-9.5, -5.5], [-17, -17 + 4 * np.log(27) / np.log(5)],
            color="#eb6834", lw=1.5, ls="--")
    ax.annotate("Moran slope for ρ=0.2 (2.05)", (-9.4, -16.4),
                color="#eb6834", fontsize=9)
    ax.annotate("↔ one stair = one suffix-tree level", (-4.6, -13.5),
                color=INK2, fontsize=9)
    ax.set_xlabel("log₂ (r / r_max)", color=INK2)
    ax.set_ylabel("log₂ C(r)   (fraction of pairs within r)", color=INK2)
    ax.set_title("The correlation staircase: quantized scales at small ρ, "
                 "melting smooth as ρ → 1", color=INK, fontsize=11,
                 loc="left", pad=10)
    ax.legend(frameon=False, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    path = os.path.join(RESULTS, "u5b_staircase.png")
    fig.savefig(path, dpi=150)
    with open(os.path.join(RESULTS, "u5b_staircase.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
