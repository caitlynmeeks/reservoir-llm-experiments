"""Experiment 01 runner: probe a frozen LLM's residual stream like a reservoir.

For each kept layer (and for ESN controls at N=D and N=4D fed the model's
own dequantized embeddings), fits ONE joint ridge from states to the
one-hot token identity at every lag k=1..k_max (targets stacked — ridge
is separable across targets, so this equals the per-lag fit at ~k_max×
the speed), and reports argmax accuracy per lag on held-out timesteps.
Each system is probed twice: raw states and unit-L2-normalized states
(the projectivity ablation, PUNCHLIST P2 / journal E1-PR3).

Usage (Mac):
    python run_probe.py --model mlx-community/Llama-3.2-1B-Instruct-4bit
    python run_probe.py --esn-only          # control path, runs anywhere
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rcllm import ESN, ChunkedRidge  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp01")


LAM_GRID = (1e-2, 1.0, 1e2, 1e4)


def token_recall_probe(states: np.ndarray, ids: np.ndarray, m_vocab: int,
                       k_max: int = 64) -> np.ndarray:
    """states: (B, T, N); ids: (B, T) in [0, m_vocab).

    One normal-equation accumulation, one Cholesky per lambda in LAM_GRID;
    lambda chosen PER LAG on a validation slice, accuracy reported on a
    disjoint test slice (60/20/20 split of timestep rows). Returns test
    accuracy per lag (k_max,)."""
    B, T, N = states.shape
    X = states[:, k_max:, :].reshape(-1, N)
    n = X.shape[0]
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    Y = np.zeros((n, k_max * m_vocab), dtype=np.float32)
    targets = {}
    for k in range(1, k_max + 1):
        tgt = ids[:, k_max - k : T - k].reshape(-1)
        targets[k] = tgt
        Y[np.arange(n), (k - 1) * m_vocab + tgt] = 1.0
    reg = ChunkedRidge(N, k_max * m_vocab).partial_fit(X[:n_tr], Y[:n_tr])
    acc = np.zeros(k_max)
    best_va = np.full(k_max, -1.0)
    for lam in LAM_GRID:
        beta = reg.solve(lam)
        pred_va = reg.predict(X[n_tr : n_tr + n_va], beta)
        pred_te = reg.predict(X[n_tr + n_va :], beta)
        for k in range(1, k_max + 1):
            sl = slice((k - 1) * m_vocab, k * m_vocab)
            va = float((pred_va[:, sl].argmax(1)
                        == targets[k][n_tr : n_tr + n_va]).mean())
            if va > best_va[k - 1]:
                best_va[k - 1] = va
                acc[k - 1] = float((pred_te[:, sl].argmax(1)
                                    == targets[k][n_tr + n_va :]).mean())
    return acc


def unit_norm(states: np.ndarray) -> np.ndarray:
    return states / (np.linalg.norm(states, axis=-1, keepdims=True) + 1e-8)


def probe_both(states: np.ndarray, ids: np.ndarray, m: int, k_max: int) -> dict:
    return {"raw": token_recall_probe(states, ids, m, k_max=k_max).tolist(),
            "norm": token_recall_probe(unit_norm(states), ids, m,
                                       k_max=k_max).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="default: every 2nd layer")
    ap.add_argument("--esn-only", action="store_true")
    ap.add_argument("--m-vocab", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--washout", type=int, default=64)
    ap.add_argument("--k-max", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results: dict = {"config": vars(args)}

    ids = rng.integers(0, args.m_vocab, size=(args.batch, args.seq_len))
    kept = ids[:, args.washout :]

    if not args.esn_only:
        from rcllm.activations import (load_model, collect_hidden_states,
                                       embedding_table, build_single_token_subset)
        model, tok = load_model(args.model)
        n_layers = len(model.model.layers)
        layers = args.layers or list(range(1, n_layers, 2))
        subset = build_single_token_subset(tok, args.m_vocab)
        results["subset_token_ids"] = subset.tolist()
        real_ids = subset[ids]
        hs = collect_hidden_states(model, real_ids, layers=layers)
        emb = embedding_table(model)[subset].astype(np.float32)  # (M, D)
        D = emb.shape[1]
        for layer in layers:
            h = hs.pop(layer)[:, args.washout :, :]
            results[f"layer_{layer}"] = probe_both(h, kept, args.m_vocab,
                                                   args.k_max)
            r = results[f"layer_{layer}"]
            print(f"layer {layer:2d}: raw acc@lag8={r['raw'][7]:.3f} "
                  f"@lag32={r['raw'][31]:.3f}  norm @lag32={r['norm'][31]:.3f}",
                  flush=True)
    else:
        D = 512
        emb = (rng.standard_normal((args.m_vocab, D)) / np.sqrt(D)).astype(np.float32)

    for label, N in [("esn_D", D), ("esn_4D", 4 * D)]:
        esn = ESN(n_inputs=D, n_reservoir=N, seed=args.seed,
                  spectral_radius=0.95, leak_rate=0.5)
        states, _ = esn.run_batch(emb[ids], washout=args.washout)
        results[label] = probe_both(states, kept, args.m_vocab, args.k_max)
        r = results[label]
        print(f"{label} (N={N}): raw acc@lag8={r['raw'][7]:.3f} "
              f"@lag32={r['raw'][31]:.3f}  norm @lag32={r['norm'][31]:.3f}",
              flush=True)

    tag = ("esnonly" if args.esn_only else args.model.split("/")[-1])
    path = os.path.join(RESULTS, f"probe_{tag}_B{args.batch}_T{args.seq_len}"
                                 f"_M{args.m_vocab}_seed{args.seed}_lamsel.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
