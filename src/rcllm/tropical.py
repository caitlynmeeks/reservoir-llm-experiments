"""T1 (PUNCHLIST Track T): max-plus (tropical) echo state network.

Update rule, all in the (max, +) semiring:

    x'_i = max( max_j (W_ij + x_j),  (Win e_c)_i )   then  x' -= max(x')

i.e. each unit keeps the best-scoring path ending at it — a soft-Viterbi
table over the reservoir graph, refreshed by the current input symbol.
The per-step max-subtraction is projectivization in tropical space
(states live in the tropical projective space; without it they drift
linearly at rate lambda).

Stability knob: lambda = maximum mean cycle weight of W (Karp's
algorithm), the tropical spectral radius. lambda < 0 => every recurrent
loop loses score per traversal and input refreshes eventually dominate:
fading memory. lambda = 0 => critical. Weights are shifted uniformly to
hit a target lambda, which shifts every cycle mean by the same amount.

Pure NumPy; edges stored flat (exactly `fanin` per row, row-sorted) so
one step is a single fancy-index + maximum.reduceat.
"""

from __future__ import annotations

import numpy as np


def max_cycle_mean(rows: np.ndarray, cols: np.ndarray, w: np.ndarray,
                   n: int) -> float:
    """Karp's algorithm. Edge k goes cols[k] -> rows[k] with weight w[k].
    Assumes every node has >= 1 incoming edge (true by construction here),
    so no -inf bookkeeping is needed."""
    order = np.argsort(rows, kind="stable")
    r, c, ww = rows[order], cols[order], w[order]
    starts = np.searchsorted(r, np.arange(n))
    F = np.empty((n + 1, n), dtype=np.float64)
    F[0] = 0.0
    for k in range(1, n + 1):
        F[k] = np.maximum.reduceat(ww + F[k - 1][c], starts)
    ks = np.arange(n)[:, None].astype(np.float64)
    ratios = (F[n][None, :] - F[:n]) / (n - ks)
    return float(ratios.min(axis=0).max())


class TropicalESN:
    def __init__(self, n_inputs: int, n_reservoir: int, cycle_mean: float = -0.2,
                 fanin: int = 10, input_scale: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        N = n_reservoir
        self.N, self.K, self.fanin = N, n_inputs, fanin
        self.rows = np.repeat(np.arange(N), fanin)           # row-sorted
        self.cols = rng.integers(0, N, N * fanin)
        w = rng.uniform(-1.0, 0.0, N * fanin)
        lam0 = max_cycle_mean(self.rows, self.cols, w, N)
        w += cycle_mean - lam0                               # every cycle mean shifts equally
        self.w = w.astype(np.float32)
        self.cycle_mean = cycle_mean
        self.row_starts = np.arange(N) * fanin
        self.Win = (rng.uniform(-1.0, 0.0, (N, n_inputs)) * input_scale
                    ).astype(np.float32)

    def step_batch(self, X: np.ndarray, char_ids: np.ndarray) -> np.ndarray:
        """X: (S, N) normalized states; char_ids: (S,) current symbol."""
        M = self.w[None, :] + X[:, self.cols]                # (S, E)
        rec = np.maximum.reduceat(M, self.row_starts, axis=1)
        Xn = np.maximum(rec, self.Win.T[char_ids])
        Xn -= Xn.max(axis=1, keepdims=True)
        return Xn

    def run_batch(self, ids: np.ndarray, washout: int = 100,
                  collect: bool = True, state: np.ndarray | None = None):
        """ids: (S, T) int symbols (NOT one-hot — tropical input is the
        Win column itself). Returns (states, final) like ESN.run_batch."""
        if ids.ndim == 1:
            ids = ids[None, :]
        S, T = ids.shape
        X = (np.zeros((S, self.N), dtype=np.float32) if state is None else state)
        out = (np.empty((S, T - washout, self.N), dtype=np.float32)
               if collect else None)
        for t in range(T):
            X = self.step_batch(X, ids[:, t])
            if collect and t >= washout:
                out[:, t - washout, :] = X
        return out, X


if __name__ == "__main__":  # smoke: does the claimed lambda control hold?
    for target in (-0.5, -0.1):
        esn = TropicalESN(27, 400, cycle_mean=target, seed=1)
        got = max_cycle_mean(esn.rows, esn.cols, esn.w.astype(np.float64), esn.N)
        print(f"target lambda {target}: achieved {got:.6f}")
