"""I2 (PUNCHLIST Track 0): readouts that consume ANY frozen feature source.

The contract: features are (T, N) float32 rows (ESN states, tropical
states, signature features...), targets are (T,) class ids. Readouts fit
on frozen features only — still reservoir-computing rules, no gradients
into the dynamics.

- RidgeReadout: closed-form ridge to one-hot + val-fitted temperature
  (wraps the exp02 pipeline; kept for parity of interface).
- LogisticReadout: multinomial logistic regression by minibatch Adam,
  direction-native cross-entropy loss, early stopping on val NLL. The
  fairer bpc (P1); NumPy on Accelerate, no torch needed.
"""

from __future__ import annotations

import numpy as np

from .ridge import ChunkedRidge


class LogisticReadout:
    def __init__(self, n_features: int, n_classes: int, lr: float = 3e-3,
                 batch: int = 8192, max_epochs: int = 12, patience: int = 2,
                 seed: int = 0):
        self.N, self.C = n_features, n_classes
        self.lr, self.batch = lr, batch
        self.max_epochs, self.patience = max_epochs, patience
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((n_features + 1, n_classes), dtype=np.float32)

    def _logits(self, X: np.ndarray, W=None) -> np.ndarray:
        W = self.W if W is None else W
        return X @ W[:-1] + W[-1]

    def nll(self, X: np.ndarray, y: np.ndarray, chunk: int = 65536) -> float:
        """Mean negative log-likelihood in nats, streamed."""
        tot, n = 0.0, 0
        for i in range(0, X.shape[0], chunk):
            xb = np.asarray(X[i : i + chunk], dtype=np.float32)
            z = self._logits(xb).astype(np.float64)
            z -= z.max(axis=1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            tot += -logp[np.arange(len(logp)), y[i : i + chunk]].sum()
            n += len(logp)
        return tot / n

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray, verbose: bool = True):
        m = np.zeros_like(self.W)
        v = np.zeros_like(self.W)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t_adam = 0
        best_nll, best_W, bad = np.inf, self.W.copy(), 0
        n = X.shape[0]
        for epoch in range(self.max_epochs):
            order = self.rng.permutation(n)
            for i in range(0, n - self.batch + 1, self.batch):
                idx = order[i : i + self.batch]
                xb = np.asarray(X[idx], dtype=np.float32)
                yb = y[idx]
                z = self._logits(xb)
                z -= z.max(axis=1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(axis=1, keepdims=True)
                p[np.arange(len(yb)), yb] -= 1.0
                p /= len(yb)
                g = np.empty_like(self.W)
                g[:-1] = xb.T @ p
                g[-1] = p.sum(axis=0)
                t_adam += 1
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g * g
                mh = m / (1 - beta1**t_adam)
                vh = v / (1 - beta2**t_adam)
                self.W -= self.lr * mh / (np.sqrt(vh) + eps)
            val_nll = self.nll(X_val, y_val)
            if verbose:
                print(f"  epoch {epoch + 1}: val nll {val_nll:.4f} nats "
                      f"({val_nll / np.log(2):.4f} bpc)", flush=True)
            if val_nll < best_nll - 1e-5:
                best_nll, best_W, bad = val_nll, self.W.copy(), 0
            else:
                bad += 1
                if bad > self.patience:
                    break
        self.W = best_W
        return self

    def bpc(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.nll(X, y) / np.log(2.0))


def scores_bpc(scores: np.ndarray, targets: np.ndarray, temp: float) -> float:
    z = scores / temp
    z -= z.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return float(-logp[np.arange(len(targets)), targets].mean() / np.log(2.0))


def fit_temperature(scores: np.ndarray, targets: np.ndarray) -> float:
    """Golden-section search on val NLL over log-temperature."""
    phi = (np.sqrt(5) - 1) / 2
    a, b = np.log(0.05), np.log(20.0)
    c, d = b - phi * (b - a), a + phi * (b - a)
    f = lambda lt: scores_bpc(scores, targets, np.exp(lt))
    fc, fd = f(c), f(d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return float(np.exp((a + b) / 2))


class RidgeReadout:
    """Interface twin of LogisticReadout over the closed-form path."""

    def __init__(self, n_features: int, n_classes: int, lam: float = 1e-2):
        self.reg = ChunkedRidge(n_features, n_classes)
        self.C, self.lam = n_classes, lam
        self.temp = 1.0

    def fit(self, X, y, X_val, y_val, verbose: bool = True):
        Y = np.zeros((len(y), self.C), dtype=np.float32)
        Y[np.arange(len(y)), y] = 1.0
        self.reg.partial_fit(X, Y)
        self.reg.solve(self.lam)
        self.temp = fit_temperature(self.reg.predict(X_val), y_val)
        return self

    def bpc(self, X, y) -> float:
        return scores_bpc(self.reg.predict(X), y, self.temp)
