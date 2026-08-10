"""T3b: does algebraic diversity buy anything a bigger pond doesn't?

Systems: tanh-1k, tropical-1k, their concatenation (2k features), and
the size-matched null: a lone tanh-2k. Ridge + temperature at 1M chars.
Pre-registered: concat within ±0.01 of tanh-2k (no mean gain); parents'
error sets Jaccard < 0.8 with concat recovering >50% of
exactly-one-parent-correct positions. Slices by context frequency.

    .venv/bin/python experiments/track_t/t8_concat_duel.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from rcllm import ESN, ChunkedRidge  # noqa: E402
from rcllm.data import V, load_text8, one_hot, as_parallel_segments  # noqa: E402
from rcllm.readouts import fit_temperature, scores_bpc  # noqa: E402
from rcllm.tropical import TropicalESN  # noqa: E402

RESULTS = os.path.join(HERE, "..", "..", "results", "track_t")
SEGMENTS, WASHOUT, CHUNK = 64, 200, 256


def collect_states(esn, ids: np.ndarray, encode) -> tuple[np.ndarray, np.ndarray]:
    """All post-washout states + next-char targets, in RAM (small N)."""
    seg = as_parallel_segments(ids, SEGMENTS)
    S, T = seg.shape
    rows = S * (T - 1 - WASHOUT)
    X = np.empty((rows, esn.N), dtype=np.float32)
    y = np.empty(rows, dtype=np.uint8)
    ctx = np.empty(rows, dtype=np.int64)          # last-4-char context id
    cur = 0
    Xs = None
    for t0 in range(0, T - 1, CHUNK):
        t1 = min(t0 + CHUNK, T - 1)
        st, Xs = esn.run_batch(encode(seg[:, t0:t1]), washout=0, state=Xs)
        tg = seg[:, t0 + 1 : t1 + 1]
        lo = 0
        if t0 < WASHOUT:
            lo = min(WASHOUT - t0, t1 - t0)
        st, tg = st[:, lo:], tg[:, lo:]
        if st.shape[1] == 0:
            continue
        k = st.shape[0] * st.shape[1]
        X[cur : cur + k] = st.reshape(-1, esn.N)
        y[cur : cur + k] = tg.reshape(-1)
        # context id of the 4 chars ENDING at the input position t (predicting t+1)
        tpos = np.arange(t0 + lo, t1)
        cid = np.zeros((S, len(tpos)), dtype=np.int64)
        for j in range(4):
            cid += seg[:, tpos - 3 + j].astype(np.int64) * 27 ** (3 - j)
        ctx[cur : cur + k] = cid.reshape(-1)
        cur += k
    return X[:cur], y[:cur], ctx[:cur]


def ridge_eval(Xtr, ytr, Xva, yva, Xte, yte, lam=1e-2):
    Y = np.zeros((len(ytr), V), dtype=np.float32)
    Y[np.arange(len(ytr)), ytr] = 1.0
    reg = ChunkedRidge(Xtr.shape[1], V).partial_fit(Xtr, Y)
    reg.solve(lam)
    sv, st_ = reg.predict(Xva), reg.predict(Xte)
    temp = fit_temperature(sv, yva)
    return {"val_bpc": round(scores_bpc(sv, yva, temp), 4),
            "test_bpc": round(scores_bpc(st_, yte, temp), 4),
            "val_correct": (sv.argmax(1) == yva)}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    train, val, test = load_text8()
    train, val, test = train[:1_000_000], val[:200_000], test[:200_000]

    tanh1 = ESN(V, 1000, spectral_radius=0.6, leak_rate=1.0, seed=0)
    trop1 = TropicalESN(V, 1000, cycle_mean=-0.1, input_scale=1.0, seed=0)
    tanh2 = ESN(V, 2000, spectral_radius=0.6, leak_rate=1.0, seed=0)

    data = {}
    for name, esn, enc in [("tanh1k", tanh1, one_hot),
                           ("trop1k", trop1, lambda s: s),
                           ("tanh2k", tanh2, one_hot)]:
        data[name] = tuple(
            collect_states(esn, split, enc) for split in (train, val, test))
        print(f"collected {name}", flush=True)

    res, correct = {}, {}
    for name in ("tanh1k", "trop1k", "tanh2k"):
        (Xtr, ytr, _), (Xva, yva, cva), (Xte, yte, _) = data[name]
        r = ridge_eval(Xtr, ytr, Xva, yva, Xte, yte)
        correct[name] = r.pop("val_correct")
        res[name] = r
        print(f"{name}: {r}", flush=True)

    (Xtr1, ytr, _), (Xva1, yva, cva), (Xte1, yte, _) = data["tanh1k"]
    (Xtr2, _, _), (Xva2, _, _), (Xte2, _, _) = data["trop1k"]
    r = ridge_eval(np.hstack([Xtr1, Xtr2]), ytr, np.hstack([Xva1, Xva2]),
                   yva, np.hstack([Xte1, Xte2]), yte)
    correct["concat"] = r.pop("val_correct")
    res["concat"] = r
    print(f"concat: {r}", flush=True)

    a, b = correct["tanh1k"], correct["trop1k"]
    ea, eb = ~a, ~b
    jacc = float((ea & eb).sum() / (ea | eb).sum())
    xor = a ^ b
    recover = float(correct["concat"][xor].mean())
    res["error_structure"] = {
        "parents_error_jaccard": round(jacc, 4),
        "xor_frac": round(float(xor.mean()), 4),
        "concat_recovers_xor": round(recover, 4),
        "delta_concat_vs_tanh2k": round(res["concat"]["val_bpc"]
                                        - res["tanh2k"]["val_bpc"], 4)}

    # frequency-decile slice
    tr_ctx = np.zeros(27**4, dtype=np.int64)
    tr = train.astype(np.int64)
    keys = np.zeros(len(tr) - 4, dtype=np.int64)
    for j in range(4):
        keys += tr[j : len(tr) - 4 + j] * 27 ** (3 - j)
    np.add.at(tr_ctx, keys, 1)
    freq = tr_ctx[cva]
    dec = np.searchsorted(np.quantile(freq, np.linspace(0.1, 0.9, 9)), freq)
    slice_tab = {}
    for name in ("tanh1k", "trop1k"):
        slice_tab[name] = [round(float(correct[name][dec == d].mean()), 4)
                           for d in range(10)]
    res["freq_decile_accuracy"] = slice_tab
    print("freq-decile acc (rare→common):", json.dumps(slice_tab, indent=1))

    with open(os.path.join(RESULTS, "t8_concat_duel.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {RESULTS}/t8_concat_duel.json")


if __name__ == "__main__":
    main()
