"""T11: the interpolation-signature knife (registered).

On rare-context val rows where exactly one pond errs: does the erring
pond's wrong answer match the SHORTER-context (trigram) modal
prediction — the smoothed answer? Registered ordinal: tanh's
error-matches-trigram-mode rate >= 1.5x tropical's. Confirms
"isolation protects the rare" (tanh over-smooths; the FSM cannot).

    .venv/bin/python experiments/track_t/t11_interp_signature.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
from rcllm import ESN, ChunkedRidge  # noqa: E402
from rcllm.data import V, load_text8, one_hot  # noqa: E402
from rcllm.readouts import fit_temperature  # noqa: E402
from rcllm.tropical import TropicalESN  # noqa: E402
from t8_concat_duel import collect_states  # noqa: E402

RESULTS = os.path.join(HERE, "..", "..", "results", "track_t")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    train, val, _ = load_text8()
    train, val = train[:1_000_000], val[:200_000]

    preds, correct = {}, {}
    yva = cva = None
    for name, esn, enc in [
        ("tanh", ESN(V, 1000, spectral_radius=0.6, leak_rate=1.0, seed=0),
         one_hot),
        ("trop", TropicalESN(V, 1000, cycle_mean=-0.1, input_scale=1.0,
                             seed=0), lambda s: s),
    ]:
        Xtr, ytr, _ = collect_states(esn, train, enc)
        Xva, yva, cva = collect_states(esn, val, enc)
        Y = np.zeros((len(ytr), V), dtype=np.float32)
        Y[np.arange(len(ytr)), ytr] = 1.0
        reg = ChunkedRidge(Xtr.shape[1], V).partial_fit(Xtr, Y)
        reg.solve(1e-2)
        preds[name] = reg.predict(Xva).argmax(1)
        correct[name] = preds[name] == yva
        print(f"{name} val acc {correct[name].mean():.4f}", flush=True)

    # trigram modal answer per val row: 2-char context (c_{t-1}, c_t) -> c_{t+1}
    trn = train.astype(np.int64)
    tri = np.zeros((27**2, V), dtype=np.int64)
    ctx2 = trn[:-2] * 27 + trn[1:-1]
    np.add.at(tri, ctx2, np.eye(V, dtype=np.int64)[trn[2:]])
    tri_mode = tri.argmax(1)
    # val rows: cva is the 4-char context id; last 2 chars = cva % 27**2
    row_mode = tri_mode[cva % (27**2)]

    # rare rows: bottom 4 deciles by 4-char train count
    ctx_count = np.zeros(27**4, dtype=np.int64)
    k4 = np.zeros(len(trn) - 4, dtype=np.int64)
    for j in range(4):
        k4 += trn[j : len(trn) - 4 + j] * 27 ** (3 - j)
    np.add.at(ctx_count, k4, 1)
    raw = ctx_count[cva]
    dec = np.searchsorted(np.quantile(raw, np.linspace(0.1, 0.9, 9)), raw)
    rare = dec <= 3

    a_wrong = rare & ~correct["tanh"] & correct["trop"]
    b_wrong = rare & ~correct["trop"] & correct["tanh"]
    rate_tanh = float((preds["tanh"][a_wrong] == row_mode[a_wrong]).mean())
    rate_trop = float((preds["trop"][b_wrong] == row_mode[b_wrong]).mean())
    ratio = rate_tanh / max(rate_trop, 1e-9)

    # secondary: all wrong rows (not just exclusive)
    aw = rare & ~correct["tanh"]
    bw = rare & ~correct["trop"]
    sec_tanh = float((preds["tanh"][aw] == row_mode[aw]).mean())
    sec_trop = float((preds["trop"][bw] == row_mode[bw]).mean())

    out = {"n_tanh_exclusive_wrong": int(a_wrong.sum()),
           "n_trop_exclusive_wrong": int(b_wrong.sum()),
           "tanh_error_matches_trigram_mode": round(rate_tanh, 4),
           "trop_error_matches_trigram_mode": round(rate_trop, 4),
           "ratio": round(ratio, 3),
           "secondary_all_wrong": {"tanh": round(sec_tanh, 4),
                                   "trop": round(sec_trop, 4)},
           "registered": "ratio >= 1.5 confirms interpolation bias"}
    out["verdict"] = "CONFIRMED" if ratio >= 1.5 else "NOT_CONFIRMED"
    print(json.dumps(out, indent=2))
    with open(os.path.join(RESULTS, "t11_interp_signature.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS}/t11_interp_signature.json")


if __name__ == "__main__":
    main()
