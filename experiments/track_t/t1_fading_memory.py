"""T1 verification: does the tropical reservoir have fading memory when
the max cycle mean lambda < 0? Same design as lesson 01 demo 1: one
network, one input stream, two different initial states; watch the
distance. Max-plus contraction is EXACT — trajectories don't just
converge, they collide in finite time (distance hits literal zero).

    .venv/bin/python experiments/track_t/t1_fading_memory.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rcllm.tropical import TropicalESN  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "track_t")
os.makedirs(RESULTS, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#eceae6"

N, T = 500, 400
rng = np.random.default_rng(11)
chars = rng.integers(0, 27, size=T)

fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
summary = {}
for lam, color in [(-0.5, BLUE), (-0.1, ORANGE), (0.0, AQUA)]:
    esn = TropicalESN(27, N, cycle_mean=lam, seed=1)
    ids = np.tile(chars[None, :], (2, 1))
    x0 = np.stack([np.zeros(N, dtype=np.float32),
                   rng.uniform(-5.0, 0.0, N).astype(np.float32)])
    x0 -= x0.max(axis=1, keepdims=True)
    states, _ = esn.run_batch(ids, washout=0, state=x0)
    dist = np.abs(states[0] - states[1]).max(axis=-1)
    first_zero = int(np.argmax(dist == 0)) if (dist == 0).any() else None
    summary[lam] = {"final_dist": float(dist[-1]), "first_exact_zero": first_zero}
    ax.semilogy(np.maximum(dist, 1e-9), color=color, lw=2,
                label=f"λ = {lam}" + (f"  (exact collision at t={first_zero})"
                                      if first_zero else "  (never collides)"))

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
ax.tick_params(colors=MUTED, labelcolor=INK2)
ax.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_title("Tropical fading memory: two initial states, same input (N=500)",
             color=INK, fontsize=11, loc="left", pad=10)
ax.set_xlabel("timestep", color=INK2)
ax.set_ylabel("L∞ distance between trajectories", color=INK2)
ax.set_ylim(1e-9, 1e1)
ax.legend(frameon=False, labelcolor=INK2)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "tropical_esp.png"), dpi=150)
print(json.dumps(summary, indent=2))
print(f"wrote {RESULTS}/tropical_esp.png")
