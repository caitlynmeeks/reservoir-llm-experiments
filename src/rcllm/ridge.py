"""Chunked ridge regression via normal equations.

This is the only "training" in classic reservoir computing. We stream
state chunks through partial_fit, accumulating XtX and XtY in float64,
then solve (XtX + lam*I) beta = XtY once.

Memory math (why the 512GB machine matters):
  d = n_features (+1 bias). XtX is d*d float64.
    d =  20,000  ->   3.2 GB
    d =  50,000  ->  20.0 GB
    d = 100,000  ->  80.0 GB   (use accum_dtype=float32 -> 40 GB)
  Cholesky solve at d=100k is ~3e14 FLOPs — minutes on the M3 Ultra via
  Accelerate-backed NumPy. Do NOT hold all states in RAM at once for
  long corpora; stream chunks of ~1e5 timesteps.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla


class ChunkedRidge:
    def __init__(self, n_features: int, n_targets: int, add_bias: bool = True,
                 accum_dtype=np.float64):
        self.add_bias = add_bias
        d = n_features + (1 if add_bias else 0)
        self.d = d
        self.XtX = np.zeros((d, d), dtype=accum_dtype)
        self.XtY = np.zeros((d, n_targets), dtype=accum_dtype)
        self.n_seen = 0

    def _with_bias(self, X: np.ndarray) -> np.ndarray:
        if not self.add_bias:
            return X
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([X, ones])

    def partial_fit(self, X: np.ndarray, Y: np.ndarray) -> "ChunkedRidge":
        """X: (T, n_features), Y: (T, n_targets). Call repeatedly over chunks."""
        acc = self.XtX.dtype
        Xb = self._with_bias(np.asarray(X, dtype=acc))  # matmul in accum dtype
        self.XtX += Xb.T @ Xb
        self.XtY += Xb.T @ np.asarray(Y, dtype=acc)
        self.n_seen += Xb.shape[0]
        return self

    def solve(self, lam: float = 1e-3) -> np.ndarray:
        """Returns beta of shape (d, n_targets). lam is absolute (not per-sample)."""
        A = self.XtX.astype(np.float64, copy=True)
        A[np.diag_indices_from(A)] += lam
        try:
            c, low = sla.cho_factor(A, overwrite_a=True, check_finite=False)
            self.beta = sla.cho_solve((c, low), self.XtY.astype(np.float64),
                                      check_finite=False)
        except sla.LinAlgError:  # fall back if not PD at this lam
            self.beta = sla.solve(self.XtX.astype(np.float64) + lam * np.eye(self.d),
                                  self.XtY.astype(np.float64), assume_a="sym")
        return self.beta

    def predict(self, X: np.ndarray, beta: np.ndarray | None = None) -> np.ndarray:
        beta = self.beta if beta is None else beta
        return self._with_bias(np.asarray(X, dtype=np.float64)) @ beta


def r_squared(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Squared Pearson correlation (Jaeger's memory-capacity convention)."""
    yp = y_pred.ravel() - y_pred.mean()
    yt = y_true.ravel() - y_true.mean()
    denom = np.linalg.norm(yp) * np.linalg.norm(yt)
    if denom == 0:
        return 0.0
    return float((yp @ yt) ** 2 / denom**2)
