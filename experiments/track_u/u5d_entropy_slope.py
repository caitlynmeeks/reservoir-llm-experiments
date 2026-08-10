"""U5d: the h-slope bet. Wide-window D across dump cells vs 1/ln(1/rho),
origin regression; slope = the drive's entropy rate h in nats/char if
the corrected Moran (d = h / ln(1/rho)) is right.

    .venv/bin/python experiments/track_u/u5d_entropy_slope.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.spatial.distance import pdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rcllm.dump_states import DUMP_DIR, dump_name  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results", "track_u")
RHOS = [0.2, 0.3, 0.4, 0.5, 0.6]
CEILING = [0.95]


def wide_slope(states: np.ndarray, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(states.shape[0], 3000, replace=False)
    dist = pdist(states[idx].astype(np.float64))
    dist = dist[dist > 0]
    rs = np.geomspace(dist.min(), dist.max(), 400)
    C = np.searchsorted(np.sort(dist), rs) / len(dist)
    keep = C > 0
    x, y = np.log2(rs[keep]), np.log2(C[keep])
    m = (C[keep] > 1e-4) & (C[keep] < 0.5)
    return float(np.polyfit(x[m], y[m], 1)[0])


def main():
    xs, ds = [], []
    out = {"cells": {}}
    for rho in RHOS + CEILING:
        d = np.load(os.path.join(DUMP_DIR, dump_name(5000, rho, 1.0, 1.0, 0,
                                                     200_000)))
        D = wide_slope(d["states"])
        x = 1.0 / np.log(1.0 / rho)
        out["cells"][str(rho)] = {"D": round(D, 3),
                                  "x_inv_ln": round(float(x), 3),
                                  "implied_h_nats": round(D / x, 3),
                                  "ceiling_row": rho in CEILING}
        print(f"rho={rho}: D {D:.3f}  1/ln(1/rho) {x:.3f}  "
              f"implied h {D / x:.3f} nats"
              f"{'  [ceiling row]' if rho in CEILING else ''}", flush=True)
        if rho in RHOS:
            xs.append(x)
            ds.append(D)
    xs, ds = np.array(xs), np.array(ds)
    h = float((xs @ ds) / (xs @ xs))          # origin regression
    resid = float(np.sqrt(np.mean((ds - h * xs) ** 2)))
    out["origin_slope_h_nats"] = round(h, 3)
    out["fit_rms"] = round(resid, 3)
    verdict = "IN BAND" if 1.0 <= h <= 1.4 else "OUT OF BAND"
    out["registered_band"] = [1.0, 1.4]
    out["verdict"] = verdict
    print(f"\norigin slope h = {h:.3f} nats/char (rms {resid:.3f}) — "
          f"registered band [1.0, 1.4]: {verdict}")
    with open(os.path.join(RESULTS, "u5d_entropy_slope.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS}/u5d_entropy_slope.json")


if __name__ == "__main__":
    main()
