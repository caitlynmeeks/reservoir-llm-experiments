"""Experiment 01 runner: probe a frozen LLM's residual stream like a reservoir.

Runs end-to-end for the ESN control on any machine; the transformer path
needs mlx-lm on the Mac (see SPEC.md and CLAUDE.md task list).

Usage (on the Mac):
    python run_probe.py --model <mlx-community-repo-id> --layers 0 4 8 12 ...
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
from rcllm.tasks import parity_accuracy_from_states  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp01")


def token_recall_probe(states: np.ndarray, ids: np.ndarray, m_vocab: int,
                       k_max: int = 64, train_frac: float = 0.7, lam: float = 1e-2):
    """states: (B, T, N); ids: (B, T) values in [0, m_vocab). Flattens batch,
    probes one-hot identity of token t-k. Returns accuracy per lag (k_max,)."""
    B, T, N = states.shape
    acc = np.zeros(k_max)
    X = states[:, k_max:, :].reshape(-1, N)
    n_train = int(train_frac * X.shape[0])
    for k in range(1, k_max + 1):
        target_ids = ids[:, k_max - k : T - k].reshape(-1)
        Y = np.eye(m_vocab, dtype=np.float32)[target_ids]
        reg = ChunkedRidge(N, m_vocab).partial_fit(X[:n_train], Y[:n_train])
        reg.solve(lam)
        pred = reg.predict(X[n_train:]).argmax(axis=1)
        acc[k - 1] = float((pred == target_ids[n_train:]).mean())
    return acc


def run_esn_control(ids: np.ndarray, emb: np.ndarray, n_reservoir: int,
                    washout: int, seed: int = 0, **esn_kw):
    """ids: (B, T) subset-relative ids; emb: (M, D_in) input vectors per id."""
    esn = ESN(n_inputs=emb.shape[1], n_reservoir=n_reservoir, seed=seed, **esn_kw)
    U = emb[ids]                                   # (B, T, D_in)
    states, _ = esn.run_batch(U, washout=washout)
    return states, ids[:, washout:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="mlx-community repo id (Mac only)")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
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

    # --- inputs: random ids in [0, M). Mapping to real token ids happens
    # only on the transformer path (TODO E1-T2: build single-token subset).
    ids = rng.integers(0, args.m_vocab, size=(args.batch, args.seq_len))

    if not args.esn_only:
        from rcllm.activations import load_model, collect_hidden_states, embedding_table
        model, tok = load_model(args.model)
        # TODO(E1-T2): subset_token_ids = build_single_token_subset(tok, args.m_vocab)
        subset_token_ids = np.arange(1000, 1000 + args.m_vocab)  # placeholder
        real_ids = subset_token_ids[ids]
        hs = collect_hidden_states(model, real_ids, layers=args.layers)
        emb = embedding_table(model)[subset_token_ids]           # (M, D)
        for layer, h in hs.items():
            acc = token_recall_probe(h[:, args.washout:, :], ids[:, args.washout:],
                                     args.m_vocab, k_max=args.k_max)
            results[f"layer_{layer}"] = acc.tolist()
            print(f"layer {layer}: acc@lag1={acc[0]:.3f} acc@lag16={acc[15]:.3f}")
        D = emb.shape[1]
    else:
        # standalone control: random Gaussian "embeddings"
        D = 512
        emb = rng.standard_normal((args.m_vocab, D)).astype(np.float32) / np.sqrt(D)

    for label, N in [("esn_D", D), ("esn_4D", 4 * D)]:
        states, kept_ids = run_esn_control(ids, emb.astype(np.float32), N,
                                           args.washout, seed=args.seed,
                                           spectral_radius=0.95, leak_rate=0.5)
        acc = token_recall_probe(states, kept_ids, args.m_vocab, k_max=args.k_max)
        results[label] = acc.tolist()
        print(f"{label} (N={N}): acc@lag1={acc[0]:.3f} acc@lag16={acc[15]:.3f}")

    # TODO(E1-T5): parity probes for both systems via parity_accuracy_from_states
    # TODO(E1-T3): heatmap + curves (matplotlib) into results/exp01/
    with open(os.path.join(RESULTS, "probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {RESULTS}/probe_results.json")


if __name__ == "__main__":
    main()
