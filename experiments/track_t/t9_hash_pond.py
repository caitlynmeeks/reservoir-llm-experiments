"""T9: the zero-compute pond, performed.

One dynamics pass over train builds state = table[last-7-chars]; the
build audits full-corpus determinism (repeat suffix => bit-identical
state). At val, positions covered by the table get their state by
lookup instead of dynamics; we verify bit-identity and identical
restricted bpc. The reservoir is then, at inference, a hash table.

    .venv/bin/python experiments/track_t/t9_hash_pond.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from rcllm import ChunkedRidge  # noqa: E402
from rcllm.data import V, load_text8, as_parallel_segments  # noqa: E402
from rcllm.readouts import fit_temperature, scores_bpc  # noqa: E402
from rcllm.tropical import TropicalESN  # noqa: E402

RESULTS = os.path.join(HERE, "..", "..", "results", "track_t")
K, SEGMENTS, WASHOUT, CHUNK = 7, 64, 200, 256


def stream(esn, ids, on_state):
    """Drive esn over (S, T) ids; call on_state(seg, t, states_t) per step
    batch post-washout, where states_t is (S, N) at input position t."""
    seg = as_parallel_segments(ids, SEGMENTS)
    S, T = seg.shape
    X = None
    for t0 in range(0, T - 1, CHUNK):
        t1 = min(t0 + CHUNK, T - 1)
        st, X = esn.run_batch(seg[:, t0:t1], washout=0, state=X)
        for k in range(st.shape[1]):
            t = t0 + k
            if t >= WASHOUT and t >= K - 1:
                on_state(seg, t, st[:, k])
    return seg


def main():
    os.makedirs(RESULTS, exist_ok=True)
    train, val, _ = load_text8()
    train, val = train[:1_000_000], val[:200_000]
    esn = TropicalESN(V, 1000, cycle_mean=-0.1, input_scale=1.0, seed=0)

    # ---- build phase: table + determinism audit + ridge readout ----
    table: dict[bytes, np.ndarray] = {}
    counts: dict[bytes, int] = {}
    audit = {"repeats": 0, "mismatches": 0}
    reg = ChunkedRidge(esn.N, V)

    def build(seg, t, states):
        Y = np.zeros((states.shape[0], V), dtype=np.float32)
        Y[np.arange(states.shape[0]), seg[:, t + 1]] = 1.0
        reg.partial_fit(states, Y)
        for s in range(states.shape[0]):
            key = seg[s, t - K + 1 : t + 1].tobytes()
            if key in table:
                audit["repeats"] += 1
                counts[key] += 1
                if not np.array_equal(table[key], states[s]):
                    audit["mismatches"] += 1
            else:
                table[key] = states[s].copy()
                counts[key] = 1

    stream(esn, train, build)
    beta = reg.solve(1e-2)
    mm = audit["mismatches"] / max(1, audit["repeats"])
    print(f"table: {len(table)} suffix states; {audit['repeats']} repeats, "
          f"{audit['mismatches']} mismatches ({100 * mm:.3f}% — register <1%)",
          flush=True)

    # ---- eval phase: dynamics vs lookup ----
    dyn_states, keys, targets = [], [], []

    def grab(seg, t, states):
        dyn_states.append(states.copy())
        targets.append(seg[:, t + 1].copy())
        keys.append([seg[s, t - K + 1 : t + 1].tobytes()
                     for s in range(states.shape[0])])

    stream(esn, val, grab)
    dyn = np.concatenate(dyn_states)
    tg = np.concatenate(targets)
    flat_keys = [k for row in keys for k in row]
    hit = np.array([k in table for k in flat_keys])
    coverage = float(hit.mean())

    look = np.stack([table[k] for k, h in zip(flat_keys, hit) if h])
    dyn_hit = dyn[hit]
    identical = float((look == dyn_hit).all(axis=1).mean())
    max_diff = float(np.abs(look - dyn_hit).max())

    sc_dyn = reg.predict(dyn, beta)
    temp = fit_temperature(sc_dyn, tg)
    bpc_dyn_hits = scores_bpc(sc_dyn[hit], tg[hit], temp)
    sc_look = reg.predict(look, beta)
    bpc_look_hits = scores_bpc(sc_look, tg[hit], temp)

    out = {"table_entries": len(table),
           "build_repeats": audit["repeats"],
           "build_mismatch_frac": round(mm, 6),
           "val_coverage": round(coverage, 4),
           "bit_identical_frac": round(identical, 6),
           "max_abs_state_diff": max_diff,
           "bpc_dynamics_on_hits": round(bpc_dyn_hits, 4),
           "bpc_lookup_on_hits": round(bpc_look_hits, 4)}
    print(json.dumps(out, indent=2))
    with open(os.path.join(RESULTS, "t9_hash_pond.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS}/t9_hash_pond.json")


if __name__ == "__main__":
    main()
