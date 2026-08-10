"""Experiment 02: echo state network as a character-level language model.

Streams the corpus through the reservoir in time-chunks (states are never
all in RAM), accumulates the ridge normal equations, solves once, then
evaluates bits-per-character with a temperature-calibrated softmax over
the ridge scores.

Smoke test (any machine):   python run_esn_lm.py --synthetic
Real run (Mac, downloads):  python run_esn_lm.py --n-reservoir 5000 \
                                --segments 256 --train-chars 20000000

TODO(E2-T3): sweep driver for (N, spectral_radius, leak) grids.
TODO(E2-T2b): optional logistic readout on frozen states (fairer bpc).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rcllm import ESN, ChunkedRidge  # noqa: E402
from rcllm.data import V, load_text8, one_hot, as_parallel_segments  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp02")


def stream_states(esn: ESN, ids: np.ndarray, chunk: int, washout: int,
                  consume) -> None:
    """Drive esn over ids (S, T); call consume(states, target_ids) per chunk.

    states: (S, C, N) for the chunk; target = next char, so the final
    timestep of each chunk is dropped from training (no target).
    """
    S, T = ids.shape
    X = None
    for t0 in range(0, T - 1, chunk):
        t1 = min(t0 + chunk, T - 1)
        U = one_hot(ids[:, t0:t1])
        states, X = esn.run_batch(U, washout=0, state=X)
        targets = ids[:, t0 + 1 : t1 + 1]
        if t0 < washout:  # drop pre-washout timesteps of the whole stream
            cut = min(washout - t0, t1 - t0)
            states, targets = states[:, cut:], targets[:, cut:]
        if states.shape[1]:
            consume(states, targets)


def bpc_from_scores(scores: np.ndarray, targets: np.ndarray, temp: float) -> float:
    z = scores / temp
    z -= z.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    nll_nats = -logp[np.arange(len(targets)), targets].mean()
    return float(nll_nats / np.log(2.0))


def fit_temperature(scores: np.ndarray, targets: np.ndarray) -> float:
    """1-D golden-section search on val NLL over log-temperature."""
    lo, hi = np.log(0.05), np.log(20.0)
    phi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    f = lambda lt: bpc_from_scores(scores, targets, np.exp(lt))
    fc, fd = f(c), f(d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return float(np.exp((a + b) / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reservoir", type=int, default=2000)
    ap.add_argument("--spectral-radius", type=float, default=0.95)
    ap.add_argument("--leak", type=float, default=0.6)
    ap.add_argument("--input-scale", type=float, default=1.0)
    ap.add_argument("--segments", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=256, help="timesteps per streamed chunk")
    ap.add_argument("--washout", type=int, default=200)
    ap.add_argument("--train-chars", type=int, default=5_000_000)
    ap.add_argument("--eval-chars", type=int, default=500_000)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true",
                    help="random Markov corpus instead of text8 (smoke test)")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    if args.synthetic:
        rng = np.random.default_rng(1)
        P = rng.dirichlet(np.ones(V) * 0.3, size=V)          # order-1 Markov
        ids = np.empty(args.train_chars + 2 * args.eval_chars, dtype=np.uint8)
        ids[0] = 0
        for t in range(1, len(ids)):                          # slow but fine for smoke
            ids[t] = rng.choice(V, p=P[ids[t - 1]])
        train = ids[: args.train_chars]
        val = ids[args.train_chars : args.train_chars + args.eval_chars]
        test = ids[args.train_chars + args.eval_chars :]
    else:
        train, val, test = load_text8()
        train, val, test = (train[: args.train_chars], val[: args.eval_chars],
                            test[: args.eval_chars])

    esn = ESN(V, args.n_reservoir, spectral_radius=args.spectral_radius,
              leak_rate=args.leak, input_scale=args.input_scale, seed=args.seed)
    reg = ChunkedRidge(args.n_reservoir, V)

    t0 = time.time()
    seg = as_parallel_segments(train, args.segments)
    stream_states(esn, seg, args.chunk, args.washout,
                  lambda st, tg: reg.partial_fit(st.reshape(-1, esn.N),
                                                 one_hot(tg).reshape(-1, V)))
    t_states = time.time() - t0
    beta = reg.solve(args.lam)
    t_solve = time.time() - t0 - t_states

    def eval_split(split_ids: np.ndarray):
        sc, tg = [], []
        stream_states(esn, as_parallel_segments(split_ids, max(4, args.segments // 8)),
                      args.chunk, args.washout,
                      lambda st, t: (sc.append(reg.predict(st.reshape(-1, esn.N), beta)),
                                     tg.append(t.reshape(-1))))
        return np.concatenate(sc), np.concatenate(tg)

    v_scores, v_tg = eval_split(val)
    temp = fit_temperature(v_scores, v_tg)
    val_bpc = bpc_from_scores(v_scores, v_tg, temp)
    s_scores, s_tg = eval_split(test)
    test_bpc = bpc_from_scores(s_scores, s_tg, temp)

    out = {"config": vars(args), "temperature": temp, "val_bpc": val_bpc,
           "test_bpc": test_bpc, "uniform_bpc": float(np.log2(V)),
           "trainable_params": int((esn.N + 1) * V + 1),
           "wall_clock": {"state_pass_s": t_states, "solve_s": t_solve}}
    print(json.dumps({k: out[k] for k in ("val_bpc", "test_bpc", "temperature",
                                          "trainable_params", "wall_clock")}, indent=2))
    tag = "synth" if args.synthetic else "text8"
    cfg = (f"N{args.n_reservoir}_r{args.spectral_radius}_a{args.leak}"
           f"_is{args.input_scale}_lam{args.lam}_seed{args.seed}"
           f"_T{args.train_chars}")
    with open(os.path.join(RESULTS, f"esn_{tag}_{cfg}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
