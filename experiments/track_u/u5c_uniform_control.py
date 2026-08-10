"""U5c: uniform-drive staircase control.

Drive the same tanh pond with i.i.d. UNIFORM symbols instead of text8.
Pre-registered: stair heights measure the drive's collision entropy, so
the wide-window slope at matched rho exceeds the text8 slope by
~ log2(27)/H2(text8). Overlay figure + slope table + Renyi-2 reference.

    .venv/bin/python experiments/track_u/u5c_uniform_control.py
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
from rcllm.data import V, load_text8  # noqa: E402
from rcllm.dump_states import DUMP_DIR, dump_name, dump_config  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results", "track_u")
RHOS = [0.2, 0.6]
BLUE, LBLUE, ORANGE, LORANGE = "#2a78d6", "#86b6ef", "#eb6834", "#f0a382"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#eceae6"


def staircase(states: np.ndarray, seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = rng.choice(states.shape[0], 3000, replace=False)
    dist = pdist(states[idx].astype(np.float64))
    dist = dist[dist > 0]
    rmax = dist.max()
    rs = np.geomspace(dist.min(), rmax, 400)
    C = np.searchsorted(np.sort(dist), rs) / len(dist)
    keep = C > 0
    x, y = np.log2(rs[keep] / rmax), np.log2(C[keep])
    m = (C[keep] > 1e-4) & (C[keep] < 0.5)
    slope = float(np.polyfit(x[m], y[m], 1)[0])
    return x, y, slope


def main():
    os.makedirs(RESULTS, exist_ok=True)
    train, _, _ = load_text8()
    p = np.bincount(train[:5_000_000], minlength=V) / 5_000_000
    h2_text8 = float(-np.log2((p**2).sum()))
    print(f"unigram Renyi-2 of text8: {h2_text8:.3f} bits "
          f"(uniform would be {np.log2(V):.3f})", flush=True)

    rng = np.random.default_rng(99)
    uids = rng.integers(0, V, 200_000).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    out = {"h2_text8_unigram_bits": round(h2_text8, 3),
           "log2_V": round(float(np.log2(V)), 3)}
    colors = {(0.2, "text8"): LBLUE, (0.2, "uniform"): LORANGE,
              (0.6, "text8"): BLUE, (0.6, "uniform"): ORANGE}
    for rho in RHOS:
        d8 = np.load(os.path.join(DUMP_DIR, dump_name(5000, rho, 1.0, 1.0, 0,
                                                      200_000)))
        du = dump_config(uids, 5000, rho, 1.0)
        for drive, st in (("text8", d8["states"]), ("uniform", du["states"])):
            x, y, slope = staircase(st)
            ax.plot(x, y, color=colors[(rho, drive)], lw=2,
                    label=f"ρ={rho}, {drive} (slope {slope:.2f})")
            out[f"rho{rho}_{drive}_slope"] = round(slope, 3)
        r = out[f"rho{rho}_uniform_slope"] / out[f"rho{rho}_text8_slope"]
        out[f"rho{rho}_slope_ratio"] = round(r, 3)
        print(f"rho={rho}: uniform/text8 slope ratio {r:.2f} "
              f"(predicted ~{np.log2(V) / h2_text8:.2f} from unigram H2)",
              flush=True)

    ax.set_xlabel("log₂ (r / r_max)", color=INK2)
    ax.set_ylabel("log₂ C(r)", color=INK2)
    ax.set_title("Staircase control: what the drive's entropy does to the "
                 "stairs", color=INK, fontsize=11, loc="left", pad=10)
    ax.legend(frameon=False, labelcolor=INK2, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "u5c_uniform_control.png"), dpi=150)
    with open(os.path.join(RESULTS, "u5c_uniform_control.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS}/u5c_uniform_control.png / .json")


if __name__ == "__main__":
    main()
