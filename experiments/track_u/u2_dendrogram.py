"""U2: the tree picture. Single-linkage clustering of champion-cell states.

In a true ultrametric space, single-linkage recovers the underlying tree
exactly. How tree-like is the pond? Dendrogram over 2k sampled states
with a strip below showing each leaf's last two characters — if geometry
mirrors the suffix tree, same-last-char states form contiguous blocks.
Quantitative one-number summary: cophenetic correlation.

    .venv/bin/python experiments/track_u/u2_dendrogram.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rcllm.dump_states import DUMP_DIR, dump_name  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results", "track_u")

RHO, LEAK, N_SAMPLE = 0.6, 1.0, 2000
VOCAB = " abcdefghijklmnopqrstuvwxyz"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"


def main():
    os.makedirs(RESULTS, exist_ok=True)
    d = np.load(os.path.join(DUMP_DIR, dump_name(5000, RHO, LEAK, 1.0, 0,
                                                 200_000)))
    rng = np.random.default_rng(7)
    idx = rng.choice(d["states"].shape[0], N_SAMPLE, replace=False)
    states, sfx = d["states"][idx], d["suffixes"][idx]

    dist = pdist(states)
    Z = linkage(dist, method="single")
    coph = cophenet(Z, dist)[0]
    print(f"cophenetic correlation (tree-likeness): {coph:.3f}")

    fig, (ax_d, ax_s) = plt.subplots(
        2, 1, figsize=(11, 5.6), facecolor=SURFACE,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.04})
    dn = dendrogram(Z, ax=ax_d, no_labels=True, color_threshold=0,
                    link_color_func=lambda _: "#2a78d6")
    order = np.array(dn["leaves"])
    ax_d.set_facecolor(SURFACE)
    for side in ("top", "right", "bottom"):
        ax_d.spines[side].set_visible(False)
    ax_d.spines["left"].set_color(MUTED)
    ax_d.tick_params(colors=MUTED, labelcolor=INK2)
    ax_d.set_title(
        f"Single-linkage dendrogram of {N_SAMPLE} champion-cell states "
        f"(ρ={RHO}, leak={LEAK}) — cophenetic corr {coph:.2f}",
        color=INK, fontsize=11, loc="left", pad=10)
    ax_d.set_ylabel("merge distance", color=INK2)

    # label strip: row 0 = last char, row 1 = previous char, in leaf order.
    # 27 identity classes need 27 hues — beyond any safe categorical palette,
    # so blocks are ALSO annotated with their letter (the readable channel).
    strip = np.stack([sfx[order, -1], sfx[order, -2]])
    ax_s.imshow(strip, aspect="auto", cmap="hsv", interpolation="nearest",
                vmin=0, vmax=26)
    runs = np.flatnonzero(np.diff(strip[0])) + 1
    for a, b in zip(np.r_[0, runs], np.r_[runs, strip.shape[1]]):
        if b - a >= 25:
            ax_s.text((a + b) / 2, 0, VOCAB[strip[0, a]].replace(" ", "␣"),
                      ha="center", va="center", color="#0b0b0b", fontsize=9,
                      fontweight="bold")
    ax_s.set_yticks([0, 1], ["last char", "prev char"], color=INK2, fontsize=9)
    ax_s.set_xticks([])
    for side in ax_s.spines.values():
        side.set_visible(False)

    fig.savefig(os.path.join(RESULTS, "u2_dendrogram.png"), dpi=150,
                bbox_inches="tight")
    with open(os.path.join(RESULTS, "u2_dendrogram.json"), "w") as f:
        json.dump({"rho": RHO, "leak": LEAK, "n_sample": N_SAMPLE,
                   "cophenetic_correlation": float(coph)}, f, indent=2)
    print(f"wrote {RESULTS}/u2_dendrogram.png / .json")


if __name__ == "__main__":
    main()
