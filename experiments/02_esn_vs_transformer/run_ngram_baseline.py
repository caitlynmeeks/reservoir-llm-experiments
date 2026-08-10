"""Char-level n-gram baselines for text8 — the low anchors of the frontier.

Add-alpha smoothed n-grams (alpha tuned on val per order), computed on the
SAME train/val/test slices the ESN runs use, so bpc numbers are directly
comparable. Honest caveat: add-alpha is weak smoothing — interpolated
Kneser-Ney would sit a bit lower — but it's the standard cheap anchor.

    .venv/bin/python experiments/02_esn_vs_transformer/run_ngram_baseline.py \
        --train-chars 5000000 --eval-chars 500000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rcllm.data import V, load_text8  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp02")


def context_index(ids: np.ndarray, order: int) -> np.ndarray:
    """idx[i] base-27 encodes the `order` chars before position i+order.
    Aligned with targets ids[order:]."""
    T = len(ids)
    idx = np.zeros(T - order, dtype=np.int64)
    for j in range(order):
        idx += ids[j : T - order + j].astype(np.int64) * (V ** (order - 1 - j))
    return idx


def ngram_bpc(train: np.ndarray, evals: dict[str, np.ndarray], order: int,
              alphas=(0.005, 0.02, 0.1, 0.5, 1.0)):
    """Fit counts on train, tune alpha on evals['val'], return bpc per split."""
    idx = context_index(train, order)
    tgt = train[order:].astype(np.int64)
    counts = np.bincount(idx * V + tgt, minlength=(V**order) * V).astype(np.float64)
    counts = counts.reshape(-1, V)
    ctx_tot = counts.sum(axis=1, keepdims=True)

    eval_idx = {s: (context_index(e, order), e[order:].astype(np.int64))
                for s, e in evals.items()}

    def bpc(split: str, alpha: float) -> float:
        i, t = eval_idx[split]
        p = (counts[i, t] + alpha) / (ctx_tot[i, 0] + alpha * V)
        return float(-np.log2(p).mean())

    best_alpha = min(alphas, key=lambda a: bpc("val", a))
    return {"alpha": best_alpha,
            **{s: round(bpc(s, best_alpha), 4) for s in evals}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-chars", type=int, default=5_000_000)
    ap.add_argument("--eval-chars", type=int, default=500_000)
    ap.add_argument("--max-n", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    train, val, test = load_text8()
    train = train[: args.train_chars]
    evals = {"val": val[: args.eval_chars], "test": test[: args.eval_chars]}

    out = {"config": vars(args), "uniform_bpc": float(np.log2(V)), "models": {}}
    for n in range(1, args.max_n + 1):
        res = ngram_bpc(train, evals, order=n - 1)
        out["models"][f"{n}-gram"] = res
        print(f"{n}-gram: val {res['val']:.3f}  test {res['test']:.3f}  "
              f"(alpha={res['alpha']})")

    path = os.path.join(RESULTS, f"ngram_baseline_train{args.train_chars}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
