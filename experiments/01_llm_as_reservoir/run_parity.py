"""E1-T5: k-parity probes — frozen transformer layers vs. ESN controls.

Binary sequences over two single-token words; target is XOR of the last
k bits (k=2..8), a purely nonlinear memory probe (chance 0.5 for any
linear readout on raw bits). Ridge probe with per-k lambda selection on
a validation slice (60/20/20 timestep split), sign readout, test
accuracy reported.

    python run_parity.py            (Mac; --esn-only runs anywhere)
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
K_MIN, K_MAX = 2, 8


def parity_probe(states: np.ndarray, bits: np.ndarray) -> dict:
    """states: (B, T', N) aligned with bits (B, T'). Test accuracy per k."""
    B, T, N = states.shape
    X = states[:, K_MAX - 1 :, :].reshape(-1, N)
    n = X.shape[0]
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    ks = list(range(K_MIN, K_MAX + 1))
    Y = np.zeros((n, len(ks)), dtype=np.float32)
    targets = {}
    for j, k in enumerate(ks):
        win = np.lib.stride_tricks.sliding_window_view(bits, k, axis=1)
        par = win.sum(axis=2) % 2                      # (B, T-k+1)
        tgt = par[:, K_MAX - k :].reshape(-1)
        targets[k] = tgt
        Y[:, j] = tgt * 2.0 - 1.0
    reg = ChunkedRidge(N, len(ks)).partial_fit(X[:n_tr], Y[:n_tr])
    acc, best_va = {}, {k: -1.0 for k in ks}
    for lam in LAM_GRID:
        beta = reg.solve(lam)
        pred_va = np.sign(reg.predict(X[n_tr : n_tr + n_va], beta))
        pred_te = np.sign(reg.predict(X[n_tr + n_va :], beta))
        for j, k in enumerate(ks):
            va = float((pred_va[:, j] == targets[k][n_tr : n_tr + n_va] * 2 - 1).mean())
            if va > best_va[k]:
                best_va[k] = va
                acc[k] = float((pred_te[:, j]
                                == targets[k][n_tr + n_va :] * 2 - 1).mean())
    return {str(k): round(acc[k], 4) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--esn-only", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--washout", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results: dict = {"config": vars(args), "chance": 0.5}

    bits = rng.integers(0, 2, size=(args.batch, args.seq_len))
    kept_bits = bits[:, args.washout :]

    if not args.esn_only:
        from rcllm.activations import (load_model, collect_hidden_states,
                                       embedding_table, build_single_token_subset)
        model, tok = load_model(args.model)
        n_layers = len(model.model.layers)
        layers = args.layers or list(range(1, n_layers, 2))
        two = build_single_token_subset(tok, 2)
        results["token_ids"] = two.tolist()
        hs = collect_hidden_states(model, two[bits], layers=layers)
        emb = embedding_table(model)[two].astype(np.float32)     # (2, D)
        D = emb.shape[1]
        for layer in layers:
            h = hs.pop(layer)[:, args.washout :, :]
            results[f"layer_{layer}"] = parity_probe(h, kept_bits)
            print(f"layer {layer:2d}: {results[f'layer_{layer}']}", flush=True)
    else:
        D = 512
        emb = (rng.standard_normal((2, D)) / np.sqrt(D)).astype(np.float32)

    for label, N in [("esn_D", D), ("esn_4D", 4 * D)]:
        esn = ESN(n_inputs=D, n_reservoir=N, seed=args.seed,
                  spectral_radius=0.95, leak_rate=0.5)
        states, _ = esn.run_batch(emb[bits], washout=args.washout)
        results[label] = parity_probe(states, kept_bits)
        print(f"{label} (N={N}): {results[label]}", flush=True)

    tag = ("esnonly" if args.esn_only else args.model.split("/")[-1])
    path = os.path.join(RESULTS, f"parity_{tag}_B{args.batch}_T{args.seq_len}"
                                 f"_seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
