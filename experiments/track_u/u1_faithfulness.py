"""U1: does state geometry mirror the suffix tree, and does that explain bpc?

For each sweep cell: sample state pairs from its dump (I1), compute the
Spearman correlation between Euclidean state distance and the suffix
ultrametric d = 2^-(shared trailing chars). That's the cell's
"faithfulness". Then across all 40 cells, correlate faithfulness with the
cell's val bpc from the 2M-char sweep. PUNCHLIST accept criterion:
|Spearman(faithfulness, bpc)| >= 0.8 means the heatmap is explained by
one number; residual cells are otherwise the finding.

    .venv/bin/python experiments/track_u/u1_faithfulness.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rcllm.dump_states import DUMP_DIR, dump_name  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results", "track_u")
EXP02 = os.path.join(ROOT, "results", "exp02")

RHOS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.95, 1.1, 1.25, 1.4]
LEAKS = [0.1, 0.3, 0.6, 1.0]
# ordinal blue ramp steps (250/350/450/600) for the 4 ordered leak values
LEAK_COLOR = {0.1: "#86b6ef", 0.3: "#5598e7", 0.6: "#2a78d6", 1.0: "#184f95"}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#eceae6"


def shared_suffix_len(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A, B: (n, L) uint8 suffixes (last char = current). Trailing match len."""
    eq_rev = A[:, ::-1] == B[:, ::-1]
    return np.where(eq_rev.all(axis=1), A.shape[1], np.argmin(eq_rev, axis=1))


def faithfulness(states: np.ndarray, suffixes: np.ndarray, n_pairs: int,
                 rng: np.random.Generator) -> float:
    n = states.shape[0]
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    euclid = np.linalg.norm(states[i] - states[j], axis=1)
    ultra = 2.0 ** -shared_suffix_len(suffixes[i], suffixes[j])
    return float(spearmanr(euclid, ultra).statistic)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(42)
    cells = []
    for leak in LEAKS:
        for rho in RHOS:
            d = np.load(os.path.join(DUMP_DIR, dump_name(5000, rho, leak,
                                                         1.0, 0, 200_000)))
            f = faithfulness(d["states"], d["suffixes"], 20_000, rng)
            bpc_file = os.path.join(EXP02, f"esn_text8_N5000_r{rho}_a{leak}"
                                           "_is1.0_lam0.01_seed0_T2000000.json")
            with open(bpc_file) as fh:
                bpc = json.load(fh)["val_bpc"]
            cells.append({"rho": rho, "leak": leak, "faithfulness": round(f, 4),
                          "val_bpc": round(bpc, 4)})
            print(f"rho={rho} leak={leak}: faithfulness {f:+.3f} bpc {bpc:.3f}",
                  flush=True)

    fs = np.array([c["faithfulness"] for c in cells])
    bs = np.array([c["val_bpc"] for c in cells])
    overall = spearmanr(fs, bs)

    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for leak in LEAKS:
        m = [c for c in cells if c["leak"] == leak]
        ax.scatter([c["faithfulness"] for c in m], [c["val_bpc"] for c in m],
                   s=42, color=LEAK_COLOR[leak], label=f"leak {leak}", zorder=3)
    best = min(cells, key=lambda c: c["val_bpc"])
    ax.annotate(f"champion (ρ={best['rho']}, a={best['leak']})",
                (best["faithfulness"], best["val_bpc"]),
                textcoords="offset points", xytext=(10, -4), color=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Ultrametric faithfulness vs. language performance, 40 cells  "
                 f"(Spearman {overall.statistic:+.2f})",
                 color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel("Spearman(state distance, suffix ultrametric distance)",
                  color=INK2)
    ax.set_ylabel("val bpc (lower = better)", color=INK2)
    ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "u1_faithfulness.png"), dpi=150)

    out = {"cells": cells, "spearman_faithfulness_vs_bpc":
           {"statistic": float(overall.statistic), "pvalue": float(overall.pvalue)},
           "accept_criterion": "|rho| >= 0.8"}
    with open(os.path.join(RESULTS, "u1_faithfulness.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\noverall Spearman(faithfulness, bpc) = {overall.statistic:+.3f} "
          f"(p={overall.pvalue:.2e})")
    print(f"wrote {RESULTS}/u1_faithfulness.png / .json")


if __name__ == "__main__":
    main()
