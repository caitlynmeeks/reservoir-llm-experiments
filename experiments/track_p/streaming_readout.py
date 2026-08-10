"""Fence Assault III: the streaming logistic readout.

Trains on EVERY post-washout row of the corpus by re-driving the
reservoir each epoch — one Adam step per chunk as states are produced,
nothing materialized. S=256 segments × 32-step chunks = exactly one
8,192-row batch per chunk. Gradient clipping (global norm 1.0) and the
auto-void rule (val worsens >0.5 bpc in one epoch => verdict unread)
per house law. Early stopping on val bpc, patience 2.

    .venv/bin/python experiments/track_p/streaming_readout.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
from rcllm import ESN  # noqa: E402
from rcllm.data import V, load_text8, one_hot, as_parallel_segments  # noqa: E402
from p1_logistic import collect  # noqa: E402

RESULTS = os.path.join(HERE, "..", "..", "results", "track_p")


def bpc(W: np.ndarray, X: np.ndarray, y: np.ndarray, chunk: int = 65536) -> float:
    tot, n = 0.0, 0
    for i in range(0, X.shape[0], chunk):
        xb = np.asarray(X[i : i + chunk], dtype=np.float32)
        z = xb @ W[:-1] + W[-1]
        z -= z.max(axis=1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        tot += -lp[np.arange(len(xb)), y[i : i + chunk]].sum()
        n += len(xb)
    return float(tot / n / np.log(2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reservoir", type=int, default=20000)
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--leak", type=float, default=1.0)
    ap.add_argument("--train-chars", type=int, default=80_000_000)
    ap.add_argument("--eval-chars", type=int, default=200_000)
    ap.add_argument("--segments", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--washout", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    train, val, test = load_text8()
    train = train[: a.train_chars]
    val, test = val[: a.eval_chars], test[: a.eval_chars]

    esn = ESN(V, a.n_reservoir, spectral_radius=a.rho, leak_rate=a.leak,
              seed=a.seed)
    print("collecting eval buffers...", flush=True)
    Xva, yva = collect(esn, val, 8, a.washout, dtype=np.float16)
    Xte, yte = collect(esn, test, 8, a.washout, dtype=np.float16)

    N = a.n_reservoir
    W = np.zeros((N + 1, V), dtype=np.float32)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t_adam = 0

    def adam_step(xb: np.ndarray, yb: np.ndarray):
        nonlocal t_adam, m, v
        z = xb @ W[:-1] + W[-1]
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        p[np.arange(len(yb)), yb] -= 1.0
        p /= len(yb)
        g = np.empty_like(W)
        g[:-1] = xb.T @ p
        g[-1] = p.sum(axis=0)
        gn = float(np.linalg.norm(g))
        if gn > 1.0:
            g *= 1.0 / gn
        t_adam += 1
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mh = m / (1 - beta1**t_adam)
        vh = v / (1 - beta2**t_adam)
        W[...] -= a.lr * mh / (np.sqrt(vh) + eps)

    seg = as_parallel_segments(train, a.segments)
    S, T = seg.shape
    best_bpc, best_W, bad, prev = np.inf, W.copy(), 0, None
    voided = False
    for epoch in range(1, a.max_epochs + 1):
        t0 = time.time()
        X = None
        for c0 in range(0, T - 1, a.chunk):
            c1 = min(c0 + a.chunk, T - 1)
            st, X = esn.run_batch(one_hot(seg[:, c0:c1]), washout=0, state=X)
            tg = seg[:, c0 + 1 : c1 + 1]
            lo = min(max(a.washout - c0, 0), c1 - c0)
            st, tg = st[:, lo:], tg[:, lo:]
            if st.shape[1]:
                adam_step(st.reshape(-1, N), tg.reshape(-1))
        val_bpc = bpc(W, Xva, yva)
        print(f"epoch {epoch}: val {val_bpc:.4f} bpc "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if prev is not None and (val_bpc - prev) > 0.5:
            print("AUTO-VOID: val worsened >0.5 bpc in one epoch", flush=True)
            voided = True
            break
        prev = val_bpc
        if val_bpc < best_bpc - 1e-5:
            best_bpc, best_W, bad = val_bpc, W.copy(), 0
        else:
            bad += 1
            if bad > a.patience:
                break

    test_bpc = bpc(best_W, Xte, yte)
    out = {"config": vars(a), "voided": voided,
           "val_bpc": round(best_bpc, 4), "test_bpc": round(test_bpc, 4),
           "trainable_params": int((N + 1) * V)}
    print(json.dumps({k: out[k] for k in ("val_bpc", "test_bpc", "voided")},
                     indent=2))
    with open(os.path.join(RESULTS, f"streaming{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS}/streaming{a.tag}.json")


if __name__ == "__main__":
    main()
