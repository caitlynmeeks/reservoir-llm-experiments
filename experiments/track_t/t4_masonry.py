"""T4 (chat-Fable's wedge): same tree, different masonry?

tanh stores the past in SUPERPOSITION (everything attenuated, graded);
max-plus stores it COLUMNARLY (a character survives in a coordinate only
while it is still the max there, then is erased outright, not faded).
Two testable predictions:
  (a) boring: tropical's cophenetic deficit (0.970 vs 0.985) is partly
      tied distances from exact state collisions — count them.
  (b) fun: tropical's digit-decay curve is a STAIRCASE where tanh's is a
      smooth geometric cliff.

    .venv/bin/python experiments/track_t/t4_masonry.py
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from rcllm import ChunkedRidge  # noqa: E402
from rcllm.data import V, load_text8  # noqa: E402
from rcllm.dump_states import dump_config  # noqa: E402
from rcllm.esn import ESN  # noqa: E402
from rcllm.tropical import TropicalESN  # noqa: E402

RESULTS = os.path.join(HERE, "..", "..", "results", "track_t")
L = 20
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#eceae6"


def decode_curve(states, suffixes, train_frac=0.7, lam=10.0):
    n = states.shape[0]
    n_tr = int(train_frac * n)
    Y = np.zeros((n, L * V), dtype=np.float32)
    for k in range(L):
        Y[np.arange(n), k * V + suffixes[:, L - 1 - k].astype(np.int64)] = 1.0
    reg = ChunkedRidge(states.shape[1], L * V).partial_fit(states[:n_tr], Y[:n_tr])
    reg.solve(lam)
    pred = reg.predict(states[n_tr:])
    return np.array([
        (pred[:, k * V : (k + 1) * V].argmax(1)
         == suffixes[n_tr:, L - 1 - k].astype(np.int64)).mean()
        for k in range(L)])


def tie_stats(states, n_sample=2000, seed=7):
    rng = np.random.default_rng(seed)
    st = states[rng.choice(states.shape[0], n_sample, replace=False)]
    d = pdist(st)
    uniq = np.unique(st, axis=0).shape[0]
    return {"zero_dist_frac": float((d == 0).mean()),
            "tied_dist_frac": float(1 - np.unique(d).size / d.size),
            "duplicate_states": int(n_sample - uniq)}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    _, val, _ = load_text8()
    ids = val[:200_000]
    systems = {
        "tanh": (ESN(V, 1000, spectral_radius=0.6, leak_rate=1.0, seed=0), None),
        "tropical": (TropicalESN(V, 1000, cycle_mean=-0.1, input_scale=1.0,
                                 seed=0), lambda s: s),
    }
    out, curves = {}, {}
    for name, (esn, enc) in systems.items():
        kw = {"esn": esn, "suffix_len": L}
        if enc is not None:
            kw["encode"] = enc
        d = dump_config(ids, **kw)
        curves[name] = decode_curve(d["states"], d["suffixes"])
        out[name] = {"decode_curve": curves[name].round(4).tolist(),
                     "ties": tie_stats(d["states"])}
        print(f"{name}: ties {out[name]['ties']}", flush=True)
        print(f"{name}: decode {curves[name].round(3).tolist()}", flush=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ks = np.arange(L)
    ax.plot(ks, curves["tanh"], color=BLUE, lw=2, marker="o", markersize=5,
            label="tanh (superposed memory)")
    ax.plot(ks, curves["tropical"], color=ORANGE, lw=2, marker="o",
            markersize=5, label="tropical (columnar memory)")
    ax.axhline(1 / V, color=MUTED, lw=1, ls="--")
    ax.annotate("chance", (L - 1, 1 / V), textcoords="offset points",
                xytext=(-2, 6), color=INK2, ha="right", fontsize=9)
    ax.set_title("Same tree, different masonry? Digit decay at N=1k, "
                 "matched-bpc champions", color=INK, fontsize=11,
                 loc="left", pad=10)
    ax.set_xlabel("lag k (chars into the past)", color=INK2)
    ax.set_ylabel("held-out decode accuracy", color=INK2)
    ax.set_xticks(ks[::2])
    ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "t4_masonry.png"), dpi=150)
    with open(os.path.join(RESULTS, "t4_masonry.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS}/t4_masonry.png / .json")


if __name__ == "__main__":
    main()
