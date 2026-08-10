"""U3: the state as a numeral in base 1/rho whose digits are characters.

At leak=1 the linearized state is sum_k rho^k Win e_{c(t-k)} — old
characters are low-order digits, fading at rate rho. Test the picture:
linear-decode the char at lag k from sampled states (dump I1), per rho.
Predictions if the numeral story holds: (a) decode accuracy falls with k,
(b) decode depth grows like -1/ln(rho), (c) at high rho depth stops
growing (digit pile-up / tanh smear) — which is where the story breaks
and the heatmap's interior optimum comes from.

    .venv/bin/python experiments/track_u/u3_digit_decay.py
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
from rcllm import ChunkedRidge  # noqa: E402
from rcllm.dump_states import DUMP_DIR, dump_name  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results", "track_u")

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--suffix-len", type=int, default=8)
_ap.add_argument("--rhos", type=float, nargs="*",
                 default=[0.2, 0.4, 0.6, 0.8, 0.95, 1.25])
_args = _ap.parse_args()

RHOS = _args.rhos
LEAK = 1.0
V, L = 27, _args.suffix_len
TAG = "" if L == 8 else f"_L{L}"
# 6-step ordinal blue ramp (light -> dark with rho), all >= step 250 on light
RAMP6 = ["#86b6ef", "#6da7ec", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#eceae6"


def decode_curve(states: np.ndarray, suffixes: np.ndarray,
                 train_frac: float = 0.7, lam: float = 10.0) -> np.ndarray:
    """Accuracy of ridge-decoding the char at lag k (k=0..L-1), one joint fit."""
    n = states.shape[0]
    n_tr = int(train_frac * n)
    Y = np.zeros((n, L * V), dtype=np.float32)
    for k in range(L):
        Y[np.arange(n), k * V + suffixes[:, L - 1 - k].astype(np.int64)] = 1.0
    reg = ChunkedRidge(states.shape[1], L * V).partial_fit(states[:n_tr], Y[:n_tr])
    reg.solve(lam)
    pred = reg.predict(states[n_tr:])
    acc = np.empty(L)
    for k in range(L):
        got = pred[:, k * V : (k + 1) * V].argmax(axis=1)
        acc[k] = (got == suffixes[n_tr:, L - 1 - k]).mean()
    return acc


def depth(acc: np.ndarray, thresh: float = 0.5) -> float:
    """Interpolated largest lag with accuracy >= thresh."""
    if acc[0] < thresh:
        return 0.0
    for k in range(1, len(acc)):
        if acc[k] < thresh:
            return (k - 1) + (acc[k - 1] - thresh) / (acc[k - 1] - acc[k])
    return float(len(acc) - 1)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    curves, depths = {}, {}
    for rho in RHOS:
        d = np.load(os.path.join(DUMP_DIR, dump_name(5000, rho, LEAK, 1.0, 0,
                                                     200_000, L)))
        acc = decode_curve(d["states"], d["suffixes"])
        curves[rho], depths[rho] = acc, depth(acc)
        print(f"rho={rho}: acc by lag {np.round(acc, 3).tolist()} "
              f"depth {depths[rho]:.2f}", flush=True)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE,
                                  gridspec_kw={"width_ratios": [3, 2]})
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(MUTED)
        a.tick_params(colors=MUTED, labelcolor=INK2)
        a.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)

    for rho, color in zip(RHOS, RAMP6):
        ax.plot(np.arange(L), curves[rho], color=color, lw=2, marker="o",
                markersize=5)
        ax.annotate(f"ρ={rho}", (L - 1, curves[rho][-1]),
                    textcoords="offset points", xytext=(6, -2),
                    color=color, fontsize=9)
    ax.axhline(1 / V, color=MUTED, lw=1, ls="--")
    ax.annotate("chance", (0, 1 / V), textcoords="offset points",
                xytext=(2, 5), color=INK2, fontsize=9)
    ax.set_title("Decoding the char at lag k from the state (leak=1)",
                 color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel("lag k (chars into the past)", color=INK2)
    ax.set_ylabel("held-out decode accuracy", color=INK2)
    ax.set_xlim(-0.3, L + 0.6)

    xs = np.array([-1.0 / np.log(r) if r < 1 else np.nan for r in RHOS])
    ys = np.array([depths[r] for r in RHOS])
    ok = ~np.isnan(xs) & (ys < L - 1 - 1e-9)  # exclude window-capped depths
    if ok.sum() < 2:
        ok = ~np.isnan(xs)
    coef = np.polyfit(xs[ok], ys[ok], 1)
    ax2.scatter(xs[ok], ys[ok], s=42, color="#2a78d6", zorder=3)
    xf = np.linspace(0, np.nanmax(xs) * 1.1, 50)
    ax2.plot(xf, np.polyval(coef, xf), color="#eb6834", lw=2)
    for rho, x, y in zip(RHOS, xs, ys):
        if not np.isnan(x):
            ax2.annotate(f"{rho}", (x, y), textcoords="offset points",
                         xytext=(6, -3), color=INK2, fontsize=9)
    ax2.set_title("Decode depth vs. −1/ln ρ (numeral prediction: linear)",
                  color=INK, fontsize=11, loc="left", pad=10)
    ax2.set_xlabel("−1 / ln ρ", color=INK2)
    ax2.set_ylabel("depth (lags at ≥50% accuracy)", color=INK2)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"u3_digit_decay{TAG}.png"), dpi=150)
    out = {"leak": LEAK, "curves": {str(r): curves[r].tolist() for r in RHOS},
           "depths": {str(r): depths[r] for r in RHOS},
           "fit_slope_intercept": coef.tolist()}
    with open(os.path.join(RESULTS, f"u3_digit_decay{TAG}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS}/u3_digit_decay{TAG}.png / .json")


if __name__ == "__main__":
    main()
