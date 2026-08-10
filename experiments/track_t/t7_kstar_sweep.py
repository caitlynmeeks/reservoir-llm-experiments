"""T7: determinism depth k* as a function of the tropical cycle mean.

Pre-registered form: k*(lambda) ~ 1/|lambda| (k*·|lambda| roughly
constant). k* = smallest k with multi-sample determinism >= 0.99;
None if not reached by k=12.

    .venv/bin/python experiments/track_t/t7_kstar_sweep.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
from rcllm.data import V, load_text8  # noqa: E402
from rcllm.dump_states import dump_config  # noqa: E402
from rcllm.tropical import TropicalESN  # noqa: E402
from t5_automaton import analyze  # noqa: E402

RESULTS = os.path.join(HERE, "..", "..", "results", "track_t")
LAMBDAS = [-0.4, -0.3, -0.2, -0.1, -0.05, -0.02]


def main():
    _, val, _ = load_text8()
    ids = val[:200_000]
    out = {}
    for lam in LAMBDAS:
        d = dump_config(ids, esn=TropicalESN(V, 1000, cycle_mean=lam,
                                             input_scale=1.0, seed=0),
                        encode=lambda s: s, suffix_len=12)
        r = analyze(d["states"], d["suffixes"])
        kstar = next((row["k"] for row in r["by_depth"]
                      if row["determined_frac"] >= 0.99), None)
        out[str(lam)] = {"kstar": kstar,
                         "kstar_times_abs_lambda":
                             None if kstar is None else round(kstar * abs(lam), 3),
                         "n_distinct_states": r["n_distinct_states"]}
        print(f"lambda={lam}: k*={kstar}  k*|lambda|="
              f"{out[str(lam)]['kstar_times_abs_lambda']}  "
              f"states={r['n_distinct_states']}", flush=True)
    with open(os.path.join(RESULTS, "t7_kstar_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS}/t7_kstar_sweep.json")


if __name__ == "__main__":
    main()
