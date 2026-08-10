"""U4: see the fractal. PCA of champion-cell states colored by last
character; then zoom INTO one letter's cloud and recolor by the
previous character — the same 27-fold structure reappears one level
down. Self-similarity is the visual signature of the IFS.

    .venv/bin/python experiments/track_u/u4_fractal_zoom.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rcllm.dump_states import DUMP_DIR, dump_name  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results", "track_u")
VOCAB = " abcdefghijklmnopqrstuvwxyz"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"


def pca2(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0)
    C = (Xc.T @ Xc) / len(Xc)
    w, v = np.linalg.eigh(C.astype(np.float64))
    return Xc @ v[:, -2:][:, ::-1]


def panel(ax, P, labels, title):
    ax.set_facecolor(SURFACE)
    ax.scatter(P[:, 0], P[:, 1], c=labels, cmap="hsv", vmin=0, vmax=26,
               s=3, linewidths=0, alpha=0.7, rasterized=True)
    counts = np.bincount(labels, minlength=27)
    for c in np.argsort(counts)[-10:]:
        m = labels == c
        ax.text(np.median(P[m, 0]), np.median(P[m, 1]),
                VOCAB[c].replace(" ", "␣"), ha="center", va="center",
                color=INK, fontsize=11, fontweight="bold")
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    d = np.load(os.path.join(DUMP_DIR, dump_name(5000, 0.6, 1.0, 1.0, 0,
                                                 200_000)))
    st, sfx = d["states"].astype(np.float64), d["suffixes"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.2), facecolor=SURFACE)
    panel(a1, pca2(st), sfx[:, -1],
          "All 20k states (ρ=0.6, a=1), colored by LAST char — PCA")
    e = sfx[:, -1] == VOCAB.index("e")
    panel(a2, pca2(st[e]), sfx[e, -2],
          f"Zoom: only states whose last char is 'e' ({e.sum()} pts), "
          "recolored by PREVIOUS char")
    fig.suptitle("The IFS attractor: same structure, one level down",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(RESULTS, "u4_fractal_zoom.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
