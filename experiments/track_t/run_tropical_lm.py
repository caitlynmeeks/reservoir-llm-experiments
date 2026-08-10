"""T2: tropical (max-plus) reservoir as a char-level LM on text8.

Same protocol as run_esn_lm.py (ridge readout + temperature-calibrated
softmax, bpc), but states come from TropicalESN and the input is symbol
ids directly (tropical "one-hot" = selecting a Win column). Sanity target
from the punchlist: beat the 1-gram (4.13). Report honestly wherever it
lands — nobody has run a soft-Viterbi pond on text8 before.

Pilot grid:
    .venv/bin/python experiments/track_t/run_tropical_lm.py --grid
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "02_esn_vs_transformer"))
from rcllm import ChunkedRidge  # noqa: E402
from rcllm.data import V, load_text8, one_hot, as_parallel_segments  # noqa: E402
from rcllm.tropical import TropicalESN  # noqa: E402
from run_esn_lm import bpc_from_scores, fit_temperature  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "track_t")


def stream_states(esn: TropicalESN, ids: np.ndarray, chunk: int, washout: int,
                  consume) -> None:
    S, T = ids.shape
    X = None
    for t0 in range(0, T - 1, chunk):
        t1 = min(t0 + chunk, T - 1)
        states, X = esn.run_batch(ids[:, t0:t1], washout=0, state=X)
        targets = ids[:, t0 + 1 : t1 + 1]
        if t0 < washout:
            cut = min(washout - t0, t1 - t0)
            states, targets = states[:, cut:], targets[:, cut:]
        if states.shape[1]:
            consume(states, targets)


def run_one(train, val, test, a, cycle_mean: float, input_scale: float) -> dict:
    esn = TropicalESN(V, a.n_reservoir, cycle_mean=cycle_mean,
                      input_scale=input_scale, seed=a.seed)
    reg = ChunkedRidge(a.n_reservoir, V)
    t0 = time.time()
    seg = as_parallel_segments(train, a.segments)
    stream_states(esn, seg, a.chunk, a.washout,
                  lambda st, tg: reg.partial_fit(st.reshape(-1, esn.N),
                                                 one_hot(tg).reshape(-1, V)))
    t_states = time.time() - t0
    beta = reg.solve(a.lam)

    def eval_split(split_ids):
        sc, tg = [], []
        stream_states(esn, as_parallel_segments(split_ids, max(4, a.segments // 8)),
                      a.chunk, a.washout,
                      lambda st, t: (sc.append(reg.predict(st.reshape(-1, esn.N), beta)),
                                     tg.append(t.reshape(-1))))
        return np.concatenate(sc), np.concatenate(tg)

    v_scores, v_tg = eval_split(val)
    temp = fit_temperature(v_scores, v_tg)
    s_scores, s_tg = eval_split(test)
    out = {"cycle_mean": cycle_mean, "input_scale": input_scale,
           "val_bpc": bpc_from_scores(v_scores, v_tg, temp),
           "test_bpc": bpc_from_scores(s_scores, s_tg, temp),
           "temperature": temp, "state_pass_s": round(t_states, 1)}
    print(json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reservoir", type=int, default=1000)
    ap.add_argument("--cycle-means", type=float, nargs="*",
                    default=[-0.3, -0.1, -0.02])
    ap.add_argument("--input-scales", type=float, nargs="*", default=[1.0, 3.0])
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--segments", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--washout", type=int, default=200)
    ap.add_argument("--train-chars", type=int, default=1_000_000)
    ap.add_argument("--eval-chars", type=int, default=200_000)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    train, val, test = load_text8()
    train, val, test = (train[: a.train_chars], val[: a.eval_chars],
                        test[: a.eval_chars])
    cells = (list(itertools.product(a.cycle_means, a.input_scales))
             if a.grid else [(a.cycle_means[0], a.input_scales[0])])
    results = {"config": vars(a), "uniform_bpc": float(np.log2(V)),
               "trainable_params": int((a.n_reservoir + 1) * V + 1),
               "runs": [run_one(train, val, test, a, cm, isc)
                        for cm, isc in cells]}
    path = os.path.join(RESULTS,
                        f"tropical_N{a.n_reservoir}_T{a.train_chars}"
                        f"_seed{a.seed}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
