"""E1-T3: figures for the token-recall probes.

Left: (system × lag) accuracy heatmap — 8 transformer layers, then the
two ESN controls. Right: recall curves, transformer (shallow + mid
layer) vs. both ESN controls. Reads the lamsel probe JSON.

    .venv/bin/python experiments/01_llm_as_reservoir/make_figures.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "results", "exp01")
JSON = os.path.join(RESULTS,
                    "probe_Llama-3.2-1B-Instruct-4bit_B32_T512_M64_seed0_lamsel.json")

RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE, LBLUE, ORANGE = "#2a78d6", "#86b6ef", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#eceae6"


def main():
    with open(JSON) as f:
        d = json.load(f)
    layers = sorted(int(k.split("_")[1]) for k in d if k.startswith("layer_"))
    rows = [(f"L{i}", np.array(d[f"layer_{i}"]["raw"])) for i in layers]
    rows += [("esn_D", np.array(d["esn_D"]["raw"])),
             ("esn_4D", np.array(d["esn_4D"]["raw"]))]
    mat = np.stack([r for _, r in rows])
    k_max = mat.shape[1]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURFACE,
                                  gridspec_kw={"width_ratios": [5, 4]})
    ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("blues", RAMP)
    mesh = ax.pcolormesh(np.arange(0.5, k_max + 0.5 + 1), np.arange(len(rows) + 1),
                         mat, cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(rows)) + 0.5, [n for n, _ in rows], fontsize=8)
    ax.axhline(len(layers), color=SURFACE, lw=3)
    ax.set_xlabel("lag k (tokens into the past)", color=INK2)
    ax.invert_yaxis()
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Token-recall accuracy: frozen Llama residual stream vs. ESN "
                 "controls", color=INK, fontsize=11, loc="left", pad=10)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("held-out accuracy", color=INK2)
    cbar.ax.tick_params(colors=MUTED, labelcolor=INK2)
    cbar.outline.set_visible(False)

    ax2.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax2.spines[s].set_color(MUTED)
    ax2.tick_params(colors=MUTED, labelcolor=INK2)
    ax2.grid(True, color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    ks = np.arange(1, k_max + 1)
    series = [("transformer L1", np.array(d["layer_1"]["raw"]), BLUE, "-"),
              ("transformer L9", np.array(d["layer_9"]["raw"]), LBLUE, "--"),
              ("ESN N=2048", np.array(d["esn_D"]["raw"]), ORANGE, "--"),
              ("ESN N=8192", np.array(d["esn_4D"]["raw"]), ORANGE, "-")]
    for name, y, c, ls in series:
        ax2.semilogx(ks, y, color=c, ls=ls, lw=2, base=2)
        i = {"transformer L1": 1, "transformer L9": 2,
             "ESN N=2048": 4, "ESN N=8192": 7}[name]
        ax2.annotate(name, (ks[i], y[i]), textcoords="offset points",
                     xytext=(6, 4), color=c, fontsize=9)
    ax2.axhline(1 / 64, color=MUTED, lw=1, ls=":")
    ax2.annotate("chance", (k_max, 1 / 64), textcoords="offset points",
                 xytext=(-2, 5), color=INK2, ha="right", fontsize=9)
    ax2.set_xlabel("lag k (log scale)", color=INK2)
    ax2.set_ylabel("held-out accuracy", color=INK2)
    ax2.set_title("A working set vs. a tape", color=INK, fontsize=11,
                  loc="left", pad=10)
    ax2.set_xticks([1, 2, 4, 8, 16, 32, 64],
                   ["1", "2", "4", "8", "16", "32", "64"])

    fig.tight_layout()
    out = os.path.join(RESULTS, "probe_recall_figures.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
