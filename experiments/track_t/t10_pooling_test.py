"""T10: is the tropical rare-context advantage backoff-by-merging?

Per the registered operationalization: reproduce the raw-frequency gap
(tropical minus tanh accuracy, rare deciles), then split the same
rare-raw rows at the median POOLED class count (sum of train counts
over all suffixes sharing the row's FSM state) and see whether the
advantage lives in the pooled-HIGH half (backoff) or is uniform
(something stranger).

    .venv/bin/python experiments/track_t/t10_pooling_test.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
from rcllm import ESN, ChunkedRidge  # noqa: E402
from rcllm.data import V, load_text8, one_hot, as_parallel_segments  # noqa: E402
from rcllm.tropical import TropicalESN  # noqa: E402
from t8_concat_duel import collect_states, ridge_eval  # noqa: E402

RESULTS = os.path.join(HERE, "..", "..", "results", "track_t")
K, SEGMENTS, WASHOUT, CHUNK = 7, 64, 200, 256


def main():
    os.makedirs(RESULTS, exist_ok=True)
    train, val, test = load_text8()
    train, val, test = train[:1_000_000], val[:200_000], test[:200_000]

    tanh1 = ESN(V, 1000, spectral_radius=0.6, leak_rate=1.0, seed=0)
    trop1 = TropicalESN(V, 1000, cycle_mean=-0.1, input_scale=1.0, seed=0)

    correct, val_pack = {}, None
    for name, esn, enc in [("tanh", tanh1, one_hot), ("trop", trop1, lambda s: s)]:
        tr = collect_states(esn, train, enc)
        va = collect_states(esn, val, enc)
        te = collect_states(esn, test, enc)
        r = ridge_eval(tr[0], tr[1], va[0], va[1], te[0], te[1])
        correct[name] = r.pop("val_correct")
        print(f"{name}: {r}", flush=True)
        if name == "trop":
            val_pack = va
    Xva_trop, yva, cva = val_pack

    # pooled class counts: stream tropical over train, state-bytes -> count
    class_count: dict[bytes, int] = defaultdict(int)
    seg = as_parallel_segments(train, SEGMENTS)
    S, T = seg.shape
    X = None
    for t0 in range(0, T - 1, CHUNK):
        t1 = min(t0 + CHUNK, T - 1)
        st, X = trop1.run_batch(seg[:, t0:t1], washout=0, state=X)
        for k in range(st.shape[1]):
            if t0 + k >= WASHOUT:
                for s in range(S):
                    class_count[st[s, k].tobytes()] += 1

    pooled = np.array([class_count.get(Xva_trop[i].tobytes(), 0)
                       for i in range(Xva_trop.shape[0])])
    seen = pooled > 0
    print(f"val rows with train-seen FSM class: {seen.mean():.3f}", flush=True)

    # raw 4-char context counts on train
    tr_ctx = np.zeros(27**4, dtype=np.int64)
    trn = train.astype(np.int64)
    keys = np.zeros(len(trn) - 4, dtype=np.int64)
    for j in range(4):
        keys += trn[j : len(trn) - 4 + j] * 27 ** (3 - j)
    np.add.at(tr_ctx, keys, 1)
    raw = tr_ctx[cva]

    # rare-raw rows (deciles 1-4 by raw count), among class-seen rows
    q = np.quantile(raw[seen], np.linspace(0.1, 0.9, 9))
    dec = np.searchsorted(q, raw)
    rare = seen & (dec <= 3)
    gap_all = float(correct["trop"][rare].mean() - correct["tanh"][rare].mean())
    n_rare = int(rare.sum())
    se = float(np.sqrt(0.5 / n_rare))

    med = np.median(pooled[rare])
    hi = rare & (pooled > med)
    lo = rare & (pooled <= med)
    gap_hi = float(correct["trop"][hi].mean() - correct["tanh"][hi].mean())
    gap_lo = float(correct["trop"][lo].mean() - correct["tanh"][lo].mean())
    se_half = float(np.sqrt(0.5 / hi.sum()))

    out = {"seen_frac": round(float(seen.mean()), 4),
           "n_rare_rows": n_rare,
           "gap_rare_raw": round(gap_all, 4), "se": round(se, 4),
           "gap_pooled_high": round(gap_hi, 4),
           "gap_pooled_low": round(gap_lo, 4), "se_half": round(se_half, 4),
           "registered": "hi >= 2x lo and |lo| < 2*se_half => backoff"}
    verdict = ("BACKOFF" if (gap_hi >= 2 * gap_lo and abs(gap_lo) < 2 * se_half)
               else "NOT_POOLING")
    out["verdict"] = verdict
    print(json.dumps(out, indent=2))
    with open(os.path.join(RESULTS, "t10_pooling_test.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS}/t10_pooling_test.json")


if __name__ == "__main__":
    main()
