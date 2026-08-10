"""T5 (chat-Fable): is the tropical pond literally a finite-state
automaton over suffix classes — and if so, did the dynamics MERGE
classes (a step toward a minimal automaton, found by physics)?

Method, on 20k sampled (state, 12-char-suffix) pairs per system:
- state ids via exact row equality (float bit-match — collisions are
  exact in max-plus, so this is legitimate);
- for each depth k: group samples by their last-k suffix;
- determinism at k = fraction of suffix classes WITH >= 2 SAMPLES whose
  members share one state (singleton classes are trivially deterministic
  and excluded — the honesty guard);
- merging at k = among determined multi-sample classes, #classes minus
  #distinct states they map to (>0 means different suffixes share one
  state: class merging).
tanh runs as control (expected: zero duplicates, determinism ~0).

    .venv/bin/python experiments/track_t/t5_automaton.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from rcllm.data import V, load_text8  # noqa: E402
from rcllm.dump_states import dump_config  # noqa: E402
from rcllm.esn import ESN  # noqa: E402
from rcllm.tropical import TropicalESN  # noqa: E402

RESULTS = os.path.join(HERE, "..", "..", "results", "track_t")
L = 12


def analyze(states: np.ndarray, suffixes: np.ndarray) -> dict:
    _, state_id = np.unique(states, axis=0, return_inverse=True)
    n_states = int(state_id.max()) + 1
    rows = []
    for k in range(1, L + 1):
        _, cls = np.unique(suffixes[:, -k:], axis=0, return_inverse=True)
        members = defaultdict(set)
        counts = defaultdict(int)
        for c, s in zip(cls, state_id):
            members[c].add(int(s))
            counts[c] += 1
        multi = [c for c in members if counts[c] >= 2]
        det = [c for c in multi if len(members[c]) == 1]
        det_states = {next(iter(members[c])) for c in det}
        rows.append({
            "k": k,
            "suffix_classes": len(members),
            "multi_sample_classes": len(multi),
            "determined_frac": round(len(det) / max(1, len(multi)), 4),
            "merged_classes": len(det) - len(det_states),
        })
    return {"n_samples": len(state_id), "n_distinct_states": n_states,
            "by_depth": rows}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    _, val, _ = load_text8()
    ids = val[:200_000]
    out = {}
    for name, esn, enc in [
        ("tropical", TropicalESN(V, 1000, cycle_mean=-0.1, input_scale=1.0,
                                 seed=0), lambda s: s),
        ("tanh", ESN(V, 1000, spectral_radius=0.6, leak_rate=1.0, seed=0),
         None),
    ]:
        kw = {"esn": esn, "suffix_len": L}
        if enc is not None:
            kw["encode"] = enc
        d = dump_config(ids, **kw)
        out[name] = analyze(d["states"], d["suffixes"])
        r = out[name]
        print(f"{name}: {r['n_distinct_states']} distinct states / "
              f"{r['n_samples']} samples")
        for row in r["by_depth"]:
            print(f"  k={row['k']:2d}: {row['suffix_classes']:6d} classes "
                  f"({row['multi_sample_classes']:6d} multi-sample), "
                  f"determined {row['determined_frac']:.3f}, "
                  f"merged {row['merged_classes']}", flush=True)

    with open(os.path.join(RESULTS, "t5_automaton.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS}/t5_automaton.json")


if __name__ == "__main__":
    main()
