"""E2-T3: (spectral radius × leak) grid at fixed N — the edge-of-chaos map
for language, mirroring the memory-capacity curve in lesson 01 demo 3.

Runs run_esn_lm.py once per cell, sequentially (each run already saturates
the machine through Accelerate), at a reduced train budget so the whole
grid stays under an hour; rank cells here, then rerun winners at full
budget. Resumable: existing per-cell JSONs are reused, so a killed sweep
picks up where it left off.

    .venv/bin/python experiments/02_esn_vs_transformer/sweep_rho_leak.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "results", "exp02")

# validated blue sequential ramp (steps 100..700), reversed so darker = lower bpc
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, MUTED, ACCENT = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#eb6834"


def cell_path(a, rho: float, leak: float) -> str:
    cfg = (f"N{a.n_reservoir}_r{rho}_a{leak}_is{a.input_scale}"
           f"_lam{a.lam}_seed{a.seed}_T{a.train_chars}")
    return os.path.join(RESULTS, f"esn_text8_{cfg}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reservoir", type=int, default=5000)
    ap.add_argument("--rhos", type=float, nargs="*",
                    default=[0.6, 0.8, 0.95, 1.1, 1.25, 1.4])
    ap.add_argument("--leaks", type=float, nargs="*", default=[0.1, 0.3, 0.6, 1.0])
    ap.add_argument("--train-chars", type=int, default=2_000_000)
    ap.add_argument("--eval-chars", type=int, default=200_000)
    ap.add_argument("--input-scale", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    grid = np.full((len(a.leaks), len(a.rhos)), np.nan)
    n_cells = grid.size
    done = 0
    for i, leak in enumerate(a.leaks):
        for j, rho in enumerate(a.rhos):
            path = cell_path(a, rho, leak)
            if not os.path.exists(path):
                t0 = time.time()
                subprocess.run(
                    [sys.executable, os.path.join(HERE, "run_esn_lm.py"),
                     "--n-reservoir", str(a.n_reservoir),
                     "--spectral-radius", str(rho), "--leak", str(leak),
                     "--input-scale", str(a.input_scale), "--lam", str(a.lam),
                     "--seed", str(a.seed), "--train-chars", str(a.train_chars),
                     "--eval-chars", str(a.eval_chars)],
                    check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
                print(f"cell rho={rho} leak={leak}: {time.time() - t0:.0f}s", flush=True)
            with open(path) as f:
                grid[i, j] = json.load(f)["val_bpc"]
            done += 1
            print(f"[{done}/{n_cells}] rho={rho} leak={leak} -> "
                  f"val_bpc {grid[i, j]:.3f}", flush=True)

    # ---- heatmap ----
    cmap = LinearSegmentedColormap.from_list("blues_rev", RAMP[::-1])
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    mesh = ax.pcolormesh(grid, cmap=cmap, edgecolors=SURFACE, linewidth=2)
    ax.set_xticks(np.arange(len(a.rhos)) + 0.5, [str(r) for r in a.rhos])
    ax.set_yticks(np.arange(len(a.leaks)) + 0.5, [str(l) for l in a.leaks])
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("spectral radius ρ", color=INK2)
    ax.set_ylabel("leak rate a", color=INK2)
    ax.set_title(f"val bpc over (ρ × leak), N={a.n_reservoir}, "
                 f"{a.train_chars // 1_000_000}M train chars",
                 color=INK, fontsize=11, loc="left", pad=10)
    vmid = (np.nanmin(grid) + np.nanmax(grid)) / 2
    for i in range(len(a.leaks)):
        for j in range(len(a.rhos)):
            dark_cell = grid[i, j] < vmid  # darker end of reversed ramp = lower bpc
            ax.text(j + 0.5, i + 0.5, f"{grid[i, j]:.2f}", ha="center",
                    va="center", fontsize=9,
                    color="#ffffff" if dark_cell else INK)
    bi, bj = np.unravel_index(np.nanargmin(grid), grid.shape)
    ax.add_patch(plt.Rectangle((bj + 0.03, bi + 0.03), 0.94, 0.94, fill=False,
                               edgecolor=ACCENT, lw=2))
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("val bpc (darker = better)", color=INK2)
    cbar.ax.tick_params(colors=MUTED, labelcolor=INK2)
    cbar.outline.set_visible(False)
    fig.tight_layout()

    stem = os.path.join(RESULTS, f"sweep_rho_leak_N{a.n_reservoir}_T{a.train_chars}"
                        f"_{len(a.rhos)}x{len(a.leaks)}")
    fig.savefig(stem + ".png", dpi=150)
    out = {"config": vars(a), "val_bpc_grid": grid.tolist(),
           "best": {"rho": a.rhos[bj], "leak": a.leaks[bi],
                    "val_bpc": float(grid[bi, bj])}}
    with open(stem + ".json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["best"], indent=2))
    print(f"wrote {stem}.png / .json")


if __name__ == "__main__":
    main()
