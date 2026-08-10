"""P1 + Finding 3 acid test: logistic readout over frozen ESN states,
mini rho-sweep, compared against ridge at the SAME 2M-char budget.

The question bolted on by chat-Fable: does the optimal rho migrate
rightward under a direction-native readout? Migration => part of the
superposition tax was ridge's poverty. Pinned at ~0.5-0.6 => the tax is
deeper than the readout.

    .venv/bin/python experiments/track_p/p1_logistic.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "02_esn_vs_transformer"))
from rcllm import ESN  # noqa: E402
from rcllm.data import V, load_text8, as_parallel_segments  # noqa: E402
from rcllm.readouts import LogisticReadout  # noqa: E402
from run_esn_lm import stream_states  # noqa: E402

RESULTS = os.path.join(HERE, "..", "..", "results", "track_p")
EXP02 = os.path.join(HERE, "..", "..", "results", "exp02")

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#eceae6"


def collect(esn: ESN, ids: np.ndarray, segments: int, washout: int,
            chunk: int = 256, dtype=np.float32):
    """Stream all post-washout (state, next-char) rows into RAM."""
    seg = as_parallel_segments(ids, segments)
    S, T = seg.shape
    n_rows = S * (T - 1 - washout)
    X = np.empty((n_rows, esn.N), dtype=dtype)
    y = np.empty(n_rows, dtype=np.uint8)
    cur = 0

    def consume(st, tg):
        nonlocal cur
        k = st.shape[0] * st.shape[1]
        X[cur : cur + k] = st.reshape(-1, esn.N)
        y[cur : cur + k] = tg.reshape(-1)
        cur += k

    stream_states(esn, seg, chunk, washout, consume)
    return X[:cur], y[:cur]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reservoir", type=int, default=5000)
    ap.add_argument("--rhos", type=float, nargs="*",
                    default=[0.4, 0.5, 0.6, 0.8, 0.95, 1.1])
    ap.add_argument("--leak", type=float, default=1.0)
    ap.add_argument("--train-chars", type=int, default=2_000_000)
    ap.add_argument("--eval-chars", type=int, default=200_000)
    ap.add_argument("--segments", type=int, default=64)
    ap.add_argument("--washout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp16-buffer", action="store_true",
                    help="store train states float16 (halves RAM; math stays f32)")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    train, val, test = load_text8()
    train = train[: a.train_chars]
    val, test = val[: a.eval_chars], test[: a.eval_chars]

    rows = []
    for rho in a.rhos:
        t0 = time.time()
        esn = ESN(V, a.n_reservoir, spectral_radius=rho, leak_rate=a.leak,
                  seed=a.seed)
        Xtr, ytr = collect(esn, train, a.segments, a.washout,
                           dtype=np.float16 if a.fp16_buffer else np.float32)
        Xva, yva = collect(esn, val, 8, a.washout)
        Xte, yte = collect(esn, test, 8, a.washout)
        print(f"rho={rho}: {Xtr.shape[0]} train rows "
              f"({Xtr.nbytes / 1e9:.1f}GB), collect {time.time() - t0:.0f}s",
              flush=True)
        lr = LogisticReadout(a.n_reservoir, V, lr=a.lr, max_epochs=20,
                             seed=a.seed)
        lr.fit(Xtr, ytr, Xva, yva)
        val_bpc, test_bpc = lr.bpc(Xva, yva), lr.bpc(Xte, yte)

        ridge_file = os.path.join(
            EXP02, f"esn_text8_N{a.n_reservoir}_r{rho}_a{a.leak}"
                   f"_is1.0_lam0.01_seed{a.seed}_T{a.train_chars}.json")
        try:
            with open(ridge_file) as f:
                ridge_bpc = json.load(f)["val_bpc"]
        except FileNotFoundError:
            ridge_bpc = float("nan")
        rows.append({"rho": rho, "logistic_val_bpc": round(val_bpc, 4),
                     "logistic_test_bpc": round(test_bpc, 4),
                     "ridge_val_bpc": round(ridge_bpc, 4),
                     "wall_s": round(time.time() - t0, 1)})
        print(f"rho={rho}: logistic val {val_bpc:.4f} vs ridge {ridge_bpc:.4f}"
              f"  (delta {ridge_bpc - val_bpc:+.4f})", flush=True)
        del Xtr, ytr, Xva, yva, Xte, yte

    best_l = min(rows, key=lambda r: r["logistic_val_bpc"])
    ridge_rows = [r for r in rows if not np.isnan(r["ridge_val_bpc"])]
    best_r = min(ridge_rows or rows, key=lambda r: r["ridge_val_bpc"])

    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    rhos = [r["rho"] for r in rows]
    ax.plot(rhos, [r["ridge_val_bpc"] for r in rows], color=BLUE, lw=2,
            marker="o", markersize=5, label="ridge + temperature")
    ax.plot(rhos, [r["logistic_val_bpc"] for r in rows], color=ORANGE, lw=2,
            marker="o", markersize=5, label="logistic (direction-native)")
    for best, color in ((best_r, BLUE), (best_l, ORANGE)):
        key = "ridge_val_bpc" if color == BLUE else "logistic_val_bpc"
        ax.annotate(f"best ρ={best['rho']}", (best["rho"], best[key]),
                    textcoords="offset points", xytext=(8, 6), color=color,
                    fontsize=9)
    ax.set_title(f"Acid test: does the optimal ρ move under a better readout? "
                 f"(N={a.n_reservoir}, leak={a.leak}, 2M chars)",
                 color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel("spectral radius ρ", color=INK2)
    ax.set_ylabel("val bpc", color=INK2)
    ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"p1_acid_test{a.tag}.png"), dpi=150)

    out = {"config": vars(a), "rows": rows,
           "best_rho": {"ridge": best_r["rho"], "logistic": best_l["rho"]}}
    with open(os.path.join(RESULTS, f"p1_acid_test{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["best_rho"], indent=2))
    print(f"wrote {RESULTS}/p1_acid_test{a.tag}.png / .json")


if __name__ == "__main__":
    main()
