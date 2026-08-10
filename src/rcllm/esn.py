"""Leaky echo state network with sparse reservoir.

Design notes (M3 Ultra target):
- Reservoir matrix W is scipy CSR sparse (~`fanin` nonzeros per row), so
  N up to ~100k is cheap to store and step.
- States are float32; readout training happens in rcllm.ridge with
  float64 normal-equation accumulation.
- `run_batch` steps S independent sequences in parallel — this is the
  fast path for text8-scale runs (split the corpus into S segments).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs


class ESN:
    def __init__(
        self,
        n_inputs: int,
        n_reservoir: int,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.3,
        fanin: int = 10,
        input_scale: float = 1.0,
        bias_scale: float = 0.2,
        seed: int = 0,
        dtype=np.float32,
    ):
        rng = np.random.default_rng(seed)
        self.N = n_reservoir
        self.K = n_inputs
        self.leak = np.asarray(leak_rate, dtype=dtype)
        self.dtype = dtype

        density = min(1.0, fanin / n_reservoir)
        W = sp.random(
            n_reservoir,
            n_reservoir,
            density=density,
            random_state=rng,
            data_rvs=lambda n: rng.uniform(-1.0, 1.0, n),
            format="csr",
            dtype=np.float64,
        )
        rho = self._spectral_radius(W, rng)
        self.W = (W * (spectral_radius / max(rho, 1e-12))).astype(dtype).tocsr()
        self.Win = (rng.uniform(-1.0, 1.0, (n_reservoir, n_inputs)) * input_scale).astype(dtype)
        self.b = (rng.uniform(-1.0, 1.0, n_reservoir) * bias_scale).astype(dtype)

    @staticmethod
    def _spectral_radius(W: sp.spmatrix, rng) -> float:
        """Largest |eigenvalue|; ARPACK with power-iteration fallback."""
        try:
            vals = eigs(W, k=1, which="LM", return_eigenvectors=False, maxiter=1000,
                        v0=rng.standard_normal(W.shape[0]))
            return float(np.abs(vals[0]))
        except Exception:
            v = rng.standard_normal(W.shape[0])
            v /= np.linalg.norm(v)
            lam = 1.0
            for _ in range(200):
                v = W @ v
                lam = np.linalg.norm(v)
                if lam == 0:
                    return 1.0
                v /= lam
            return float(lam)

    # ------------------------------------------------------------------
    def step_batch(self, X: np.ndarray, U_t: np.ndarray) -> np.ndarray:
        """One update for S parallel sequences. X: (S, N), U_t: (S, K)."""
        pre = (self.W @ X.T).T + U_t @ self.Win.T + self.b
        return (1.0 - self.leak) * X + self.leak * np.tanh(pre)

    def run_batch(
        self,
        U: np.ndarray,
        washout: int = 100,
        collect: bool = True,
        state: np.ndarray | None = None,
    ):
        """Drive the reservoir with U of shape (S, T, K).

        Returns (states, final_state):
          states: (S, T - washout, N) float32 if collect else None
          final_state: (S, N)
        """
        U = np.asarray(U, dtype=self.dtype)
        if U.ndim == 2:  # single sequence convenience
            U = U[None, :, :]
        S, T, K = U.shape
        assert K == self.K, f"expected {self.K} input dims, got {K}"
        X = np.zeros((S, self.N), dtype=self.dtype) if state is None else state
        out = np.empty((S, T - washout, self.N), dtype=self.dtype) if collect else None
        for t in range(T):
            X = self.step_batch(X, U[:, t, :])
            if collect and t >= washout:
                out[:, t - washout, :] = X
        return out, X
