"""T3 (chat-Fable's insertion): did the max-plus pond grow the same suffix
tree as the tanh pond?

Dump states of the tropical champion (N=1k, lambda=-0.1) and the matched
tanh control (N=1k, rho=0.6, leak=1) over the same val slice, then run
Track U's instruments on both: U1 faithfulness (state distance vs suffix
ultrametric) and U2 tree-likeness (single-linkage cophenetic correlation
+ dendrogram with last/prev-char strips). Same bpc with the same
near-perfect suffix tree would fuse Findings 2 and 4: two incompatible
algebras converging on one representation.

    .venv/bin/python experiments/track_t/t3_geometry.py
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "track_u"))
from rcllm.data import V, load_text8  # noqa: E402
from rcllm.dump_states import DUMP_DIR, dump_config  # noqa: E402
from rcllm.esn import ESN  # noqa: E402
from rcllm.tropical import TropicalESN  # noqa: E402
from u1_faithfulness import faithfulness  # noqa: E402

RESULTS = os.path.join(HERE, "..", "..", "results", "track_t")
VOCAB = " abcdefghijklmnopqrstuvwxyz"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"


def tree_panel(fig, gs_col, states, sfx, title, n_sample=2000, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.choice(states.shape[0], n_sample, replace=False)
    st, sf = states[idx], sfx[idx]
    dist = pdist(st)
    Z = linkage(dist, method="single")
    coph = cophenet(Z, dist)[0]
    ax_d = fig.add_subplot(gs_col[0])
    ax_s = fig.add_subplot(gs_col[1])
    dn = dendrogram(Z, ax=ax_d, no_labels=True, color_threshold=0,
                    link_color_func=lambda _: "#2a78d6")
    order = np.array(dn["leaves"])
    ax_d.set_facecolor(SURFACE)
    for side in ("top", "right", "bottom"):
        ax_d.spines[side].set_visible(False)
    ax_d.spines["left"].set_color(MUTED)
    ax_d.tick_params(colors=MUTED, labelcolor=INK2)
    ax_d.set_title(f"{title} — cophenetic {coph:.3f}", color=INK,
                   fontsize=10, loc="left", pad=8)
    strip = np.stack([sf[order, -1], sf[order, -2]])
    ax_s.imshow(strip, aspect="auto", cmap="hsv", interpolation="nearest",
                vmin=0, vmax=26)
    runs = np.flatnonzero(np.diff(strip[0])) + 1
    for a, b in zip(np.r_[0, runs], np.r_[runs, strip.shape[1]]):
        if b - a >= 25:
            ax_s.text((a + b) / 2, 0, VOCAB[strip[0, a]].replace(" ", "␣"),
                      ha="center", va="center", color="#0b0b0b", fontsize=8,
                      fontweight="bold")
    ax_s.set_yticks([0, 1], ["last", "prev"], color=INK2, fontsize=8)
    ax_s.set_xticks([])
    for side in ax_s.spines.values():
        side.set_visible(False)
    return float(coph)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    _, val, _ = load_text8()
    ids = val[:200_000]

    systems = {
        "tanh (N=1k, ρ=0.6, a=1)": (
            ESN(V, 1000, spectral_radius=0.6, leak_rate=1.0, seed=0), None),
        "tropical (N=1k, λ=−0.1)": (
            TropicalESN(V, 1000, cycle_mean=-0.1, input_scale=1.0, seed=0),
            lambda s: s),
    }
    dumps, out = {}, {"cells": {}}
    rng = np.random.default_rng(42)
    for name, (esn, enc) in systems.items():
        kw = {"esn": esn} if enc is None else {"esn": esn, "encode": enc}
        d = dump_config(ids, **kw)                    # alignment gate inside
        dumps[name] = d
        tag = "tanh_N1000" if "tanh" in name else "tropical_N1000"
        np.savez_compressed(os.path.join(DUMP_DIR, f"states_{tag}_geo.npz"),
                            **d, config=np.array([name]))
        f = faithfulness(d["states"], d["suffixes"], 20_000, rng)
        out["cells"][name] = {"faithfulness": round(f, 4)}
        print(f"{name}: faithfulness {f:+.3f}", flush=True)

    fig = plt.figure(figsize=(13, 5.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=0.05, wspace=0.08)
    for col, (name, d) in enumerate(dumps.items()):
        coph = tree_panel(fig, gs[:, col].subgridspec(2, 1,
                          height_ratios=[4, 1], hspace=0.04),
                          d["states"], d["suffixes"], name)
        out["cells"][name]["cophenetic"] = coph
        print(f"{name}: cophenetic {coph:.3f}", flush=True)

    fig.savefig(os.path.join(RESULTS, "t3_geometry.png"), dpi=150,
                bbox_inches="tight")
    with open(os.path.join(RESULTS, "t3_geometry.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS}/t3_geometry.png / .json")


if __name__ == "__main__":
    main()
