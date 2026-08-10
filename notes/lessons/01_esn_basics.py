"""Lesson 01: four tiny experiments that show what an echo state network is.

Generates the figures for notes/lessons/01_esn_basics.md. Rerun any time:
    .venv/bin/python notes/lessons/01_esn_basics.py
Everything is small (N=200-400, seconds per demo) — tweak the constants
marked TRY below and rerun to build intuition.
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
from rcllm import ESN  # noqa: E402
from rcllm.tasks import esn_memory_capacity, parity_accuracy_from_states  # noqa: E402

IMG = os.path.join(os.path.dirname(__file__), "img")
os.makedirs(IMG, exist_ok=True)

# categorical slots 1-3 of the validated palette; muted axis / secondary ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#eceae6"
summary: dict = {}


def new_fig(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    return fig, ax


def finish(fig, ax, name: str, legend: bool = True):
    if legend:
        ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout()
    path = os.path.join(IMG, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


# ----------------------------------------------------------------------
# Demo 1 — the echo state property (why "echo", and why washout exists).
# Same network, same input, two very different initial states. If the
# dynamics are contracting, the two trajectories converge: the state
# becomes a function of the input HISTORY, not of where it started.
# ----------------------------------------------------------------------
rng = np.random.default_rng(7)
T1, N1 = 400, 200
u1 = rng.uniform(-1, 1, T1)
fig, ax = new_fig(
    "Echo state property: same input, two different initial states",
    "timestep", "distance between the two trajectories",
)
summary["esp"] = {}
for rho, color in [(0.8, BLUE), (1.4, ORANGE), (2.5, AQUA)]:  # TRY other radii
    esn = ESN(1, N1, spectral_radius=rho, leak_rate=1.0, input_scale=0.5,
              seed=7, dtype=np.float64)
    U = np.tile(u1[None, :, None], (2, 1, 1))
    x0 = np.stack([np.zeros(N1), rng.uniform(-1, 1, N1)])
    states, _ = esn.run_batch(U, washout=0, state=x0)
    dist = np.linalg.norm(states[0] - states[1], axis=-1)
    ax.semilogy(np.maximum(dist, 1e-16), color=color, lw=2,
                label=f"spectral radius {rho}")
    summary["esp"][rho] = float(dist[-1])
ax.set_ylim(1e-16, 1e2)
finish(fig, ax, "esp_convergence.png")

# ----------------------------------------------------------------------
# Demo 2 — fading memory, measured. Drive with white noise, train one
# ridge readout per lag k to reconstruct input from k steps ago; r^2 per
# lag is the memory curve, its sum is Jaeger's memory capacity (MC <= N).
# ----------------------------------------------------------------------
fig, ax = new_fig(
    "Fading memory: how well can a linear readout recover u(t−k)?",
    "lag k (steps into the past)", "reconstruction r²",
)
summary["mc"] = {}
configs = [  # TRY: leak_rate < 1 to stretch memory for slow signals
    ("N=100, gentle drive", 100, 0.2, BLUE),
    ("N=400, gentle drive", 400, 0.2, ORANGE),
    ("N=400, hard drive (saturating)", 400, 3.0, AQUA),
]
for label, n, in_scale, color in configs:
    esn = ESN(1, n, spectral_radius=0.9, leak_rate=1.0, input_scale=in_scale, seed=1)
    mc, r2 = esn_memory_capacity(esn, T=6000, k_max=60)
    ax.plot(np.arange(1, 61), r2, color=color, lw=2, label=f"{label} — MC≈{mc:.0f}")
    summary["mc"][label] = round(mc, 1)
finish(fig, ax, "memory_curves.png")

# ----------------------------------------------------------------------
# Demo 3 — the edge of chaos. Total memory capacity as the spectral
# radius sweeps through 1.0: too small = the echo dies instantly, too
# large = chaos scrambles it. Best is just below the edge.
# ----------------------------------------------------------------------
rhos = np.linspace(0.2, 1.8, 17)
mcs = []
for rho in rhos:
    esn = ESN(1, 200, spectral_radius=float(rho), leak_rate=1.0,
              input_scale=0.2, seed=1)
    mc, _ = esn_memory_capacity(esn, T=4000, k_max=60)
    mcs.append(mc)
fig, ax = new_fig(
    "Edge of chaos: memory capacity vs. spectral radius (N=200)",
    "spectral radius ρ", "total memory capacity",
)
ax.plot(rhos, mcs, color=BLUE, lw=2, marker="o", markersize=5)
best = int(np.argmax(mcs))
ax.annotate(f"peak at ρ={rhos[best]:.1f}", (rhos[best], mcs[best]),
            textcoords="offset points", xytext=(8, 4), color=INK2)
ax.axvline(1.0, color=MUTED, lw=1, ls="--")
summary["edge"] = {"rho": float(rhos[best]), "mc": round(float(mcs[best]), 1)}
finish(fig, ax, "edge_of_chaos.png", legend=False)

# ----------------------------------------------------------------------
# Demo 4 — nonlinear computation for free. k-parity (XOR of the last k
# bits) is impossible for ANY linear readout on the raw bits: every bit
# window is uncorrelated with the answer. But a linear readout on the
# reservoir STATE solves it — the recurrent tanh mixing has already
# computed the nonlinear features. This is the whole reservoir bargain:
# training stays linear, nonlinearity comes from the fixed dynamics.
# ----------------------------------------------------------------------
T4 = 8000
bits = rng.integers(0, 2, T4)
esn = ESN(1, 400, spectral_radius=0.9, leak_rate=1.0, input_scale=2.0, seed=3)
states, _ = esn.run_batch((bits * 2.0 - 1.0)[None, :, None].astype(np.float32),
                          washout=100)
windows = np.lib.stride_tricks.sliding_window_view(bits, 8).astype(np.float32)
ks = list(range(2, 7))
acc_esn = [parity_accuracy_from_states(states[0], bits[100:], k) for k in ks]
acc_raw = [parity_accuracy_from_states(windows, bits[7:], k) for k in ks]
summary["parity"] = {"esn": [round(a, 3) for a in acc_esn],
                     "raw": [round(a, 3) for a in acc_raw]}
fig, ax = new_fig(
    "k-parity (XOR of last k bits): linear readout accuracy",
    "k", "held-out accuracy",
)
ax.plot(ks, acc_esn, color=BLUE, lw=2, marker="o", markersize=5,
        label="readout on reservoir state (N=400)")
ax.plot(ks, acc_raw, color=ORANGE, lw=2, marker="o", markersize=5,
        label="readout on the raw bits themselves")
ax.axhline(0.5, color=MUTED, lw=1, ls="--")
ax.annotate("chance", (ks[-1], 0.5), textcoords="offset points",
            xytext=(-2, 6), color=INK2, ha="right")
ax.set_xticks(ks)
ax.set_ylim(0.4, 1.02)
finish(fig, ax, "parity.png")

print(json.dumps(summary, indent=2))
