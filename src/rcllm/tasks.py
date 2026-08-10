"""Reservoir diagnostics: memory capacity and k-parity.

These run against ANY state-producing system — the ESN, or a frozen
transformer's per-layer hidden states (experiment 01 treats those states
exactly like reservoir states and reuses these routines).
"""

from __future__ import annotations

import numpy as np

from .ridge import ChunkedRidge, r_squared


def memory_capacity_from_states(
    states: np.ndarray,
    u: np.ndarray,
    k_max: int = 100,
    train_frac: float = 0.7,
    lam: float = 1e-3,
):
    """Jaeger memory capacity given precomputed states.

    states: (T, N) — state at time t, driven by scalar input u[t]
    u:      (T,)   — the i.i.d. input sequence that produced the states
    Trains one joint ridge readout to reconstruct u[t-k] for k=1..k_max.
    Returns (mc_total, r2_per_lag ndarray of shape (k_max,)).
    """
    T, _ = states.shape
    X = states[k_max:]                      # ensure all lags exist
    Y = np.stack([u[k_max - k : T - k] for k in range(1, k_max + 1)], axis=1)
    n_train = int(train_frac * X.shape[0])
    reg = ChunkedRidge(X.shape[1], k_max).partial_fit(X[:n_train], Y[:n_train])
    reg.solve(lam)
    pred = reg.predict(X[n_train:])
    r2 = np.array([r_squared(pred[:, k], Y[n_train:, k]) for k in range(k_max)])
    return float(r2.sum()), r2


def esn_memory_capacity(esn, T: int = 20_000, k_max: int = 100, washout: int = 200,
                        seed: int = 0, lam: float = 1e-3):
    """Convenience wrapper: drive an ESN with U(-1,1) noise, measure MC."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, T).astype(np.float32)
    states, _ = esn.run_batch(u[None, :, None], washout=washout)
    return memory_capacity_from_states(states[0], u[washout:], k_max=k_max, lam=lam)


def parity_targets(bits: np.ndarray, k: int) -> np.ndarray:
    """y[t] = XOR of bits[t-k+1..t]; bits in {0,1}, shape (T,). Returns (T-k+1,) in {0,1}."""
    T = bits.shape[0]
    win = np.lib.stride_tricks.sliding_window_view(bits, k)  # (T-k+1, k)
    return win.sum(axis=1) % 2


def parity_accuracy_from_states(states: np.ndarray, bits: np.ndarray, k: int,
                                train_frac: float = 0.7, lam: float = 1e-3) -> float:
    """Linear-readout accuracy on k-parity — a purely NONLINEAR memory probe.

    states: (T, N) aligned with bits: (T,). Ridge to {-1,+1} target, sign readout.
    """
    y = parity_targets(bits, k) * 2.0 - 1.0
    X = states[k - 1 :]
    n_train = int(train_frac * X.shape[0])
    reg = ChunkedRidge(X.shape[1], 1).partial_fit(X[:n_train], y[:n_train, None])
    reg.solve(lam)
    pred = np.sign(reg.predict(X[n_train:])[:, 0])
    return float((pred == y[n_train:]).mean())
