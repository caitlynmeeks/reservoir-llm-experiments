"""U6: the linear-pond null. How much tree-ness comes free with fading
memory alone? Identity-activation pond at the t3 control knobs (N=1k,
rho=0.6, leak=1) through the U1/U2 instruments, alongside the tanh and
tropical numbers for comparison.

    .venv/bin/python experiments/track_u/u6_linear_null.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
from rcllm.data import V, load_text8  # noqa: E402
from rcllm.dump_states import dump_config  # noqa: E402
from rcllm.esn import ESN  # noqa: E402
from u1_faithfulness import faithfulness  # noqa: E402

RESULTS = os.path.join(HERE, "..", "..", "results", "track_u")


def main():
    _, val, _ = load_text8()
    esn = ESN(V, 1000, spectral_radius=0.6, leak_rate=1.0, seed=0,
              activation="linear")
    d = dump_config(val[:200_000], esn=esn)          # alignment gate inside
    rng = np.random.default_rng(42)
    f = faithfulness(d["states"], d["suffixes"], 20_000, rng)
    idx = np.random.default_rng(7).choice(d["states"].shape[0], 2000,
                                          replace=False)
    dist = pdist(d["states"][idx].astype(np.float64))
    coph = float(cophenet(linkage(dist, method="single"), dist)[0])
    out = {"linear": {"faithfulness": round(f, 4), "cophenetic": round(coph, 4)},
           "tanh_reference": {"faithfulness": 0.448, "cophenetic": 0.985},
           "tropical_reference": {"faithfulness": 0.454, "cophenetic": 0.970}}
    print(json.dumps(out, indent=2))
    with open(os.path.join(RESULTS, "u6_linear_null.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {RESULTS}/u6_linear_null.json")


if __name__ == "__main__":
    main()
