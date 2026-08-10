"""I1 (PUNCHLIST Track 0): dump sampled (state, suffix) pairs for ESN configs.

Reconstructs the exact reservoir from (N, rho, leak, input_scale, seed) —
identical rng path to the sweep runs — drives it over a text8 val slice,
and saves a uniform post-washout sample of states, each paired with its
trailing suffix characters. Never dumps the full stream. Consumed by
tracks U, C, H.

Dump all sweep cells (writes data/state_dumps/*.npz, skips existing):
    .venv/bin/python -m rcllm.dump_states --all-sweep-cells
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .data import V, load_text8, one_hot, as_parallel_segments
from .esn import ESN

DUMP_DIR = os.path.join("data", "state_dumps")


def dump_name(n_reservoir, rho, leak, input_scale, seed, chars,
              suffix_len: int = 8) -> str:
    tail = "" if suffix_len == 8 else f"_L{suffix_len}"
    return (f"states_N{n_reservoir}_r{rho}_a{leak}_is{input_scale}"
            f"_seed{seed}_T{chars}{tail}.npz")


def dump_config(ids: np.ndarray, n_reservoir: int | None = None,
                rho: float | None = None, leak: float | None = None,
                input_scale: float = 1.0, seed: int = 0,
                n_samples: int = 20_000, suffix_len: int = 8,
                segments: int = 16, washout: int = 200, chunk: int = 2000,
                sample_seed: int = 123, esn=None, encode=one_hot) -> dict:
    """Returns {'states': (n, N) f32, 'suffixes': (n, L) u8} for a val slice.

    Pass a prebuilt `esn` (any object with the run_batch contract, e.g.
    TropicalESN) plus its input `encode` (one_hot for tanh, identity for
    tropical) to dump non-default architectures."""
    if esn is None:
        esn = ESN(V, n_reservoir, spectral_radius=rho, leak_rate=leak,
                  input_scale=input_scale, seed=seed)
    seg = as_parallel_segments(ids, segments)          # (S, T)
    S, T = seg.shape

    rng = np.random.default_rng(sample_seed)
    lo = max(washout, suffix_len - 1)
    flat = rng.choice((T - lo) * S, size=n_samples, replace=False)
    samp_s, samp_t = flat % S, lo + flat // S          # uniform over (s, t)

    states_out = np.empty((n_samples, esn.N), dtype=np.float32)
    X = None
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        st, X = esn.run_batch(encode(seg[:, t0:t1]), washout=0, state=X)
        hit = np.where((samp_t >= t0) & (samp_t < t1))[0]
        states_out[hit] = st[samp_s[hit], samp_t[hit] - t0]

    offs = np.arange(-(suffix_len - 1), 1)             # suffix INCLUDES char at t
    suffixes = seg[samp_s[:, None], samp_t[:, None] + offs].astype(np.uint8)
    out = {"states": states_out, "suffixes": suffixes,
           "positions_s": samp_s, "positions_t": samp_t}
    _alignment_gate(esn, seg, out, suffix_len, encode)
    return out


def _alignment_gate(esn, seg: np.ndarray, dump: dict, suffix_len: int,
                    encode=one_hot) -> None:
    """Refuse-to-write self-test: re-drive the reservoir from cold start to
    the EARLIEST sampled position and require the stored state to match
    bit-for-bit (chunked and unchunked runs execute identical float ops).
    Threshold-free, so it can't false-alarm on sluggish (low-leak) cells
    the way a decodability check would."""
    i0 = int(np.argmin(dump["positions_t"]))
    s, t = int(dump["positions_s"][i0]), int(dump["positions_t"][i0])
    _, X = esn.run_batch(encode(seg[s : s + 1, : t + 1]), washout=0,
                         collect=False)
    if not np.array_equal(X[0], dump["states"][i0]):
        raise RuntimeError(f"ALIGNMENT GATE: recomputed state at (s={s}, t={t}) "
                           "differs from stored state — pairing is shifted")
    want = seg[s, t - suffix_len + 1 : t + 1].astype(np.uint8)
    if not np.array_equal(dump["suffixes"][i0], want):
        raise RuntimeError(f"ALIGNMENT GATE: stored suffix at (s={s}, t={t}) "
                           "does not match the corpus — suffix indexing is off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reservoir", type=int, default=5000)
    ap.add_argument("--rhos", type=float, nargs="*",
                    default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.95, 1.1, 1.25, 1.4])
    ap.add_argument("--leaks", type=float, nargs="*", default=[0.1, 0.3, 0.6, 1.0])
    ap.add_argument("--all-sweep-cells", action="store_true")
    ap.add_argument("--input-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chars", type=int, default=200_000)
    ap.add_argument("--n-samples", type=int, default=20_000)
    ap.add_argument("--suffix-len", type=int, default=8)
    ap.add_argument("--validate", action="store_true",
                    help="run the alignment gate against every existing dump "
                         "written with the default segments/washout")
    a = ap.parse_args()
    os.makedirs(DUMP_DIR, exist_ok=True)

    if a.validate:
        _, val, _ = load_text8()
        seg = as_parallel_segments(val[: a.chars], 16)
        n_bad = 0
        files = sorted(f for f in os.listdir(DUMP_DIR) if f.endswith(".npz"))
        for i, fn in enumerate(files):
            d = np.load(os.path.join(DUMP_DIR, fn))
            cfg = dict(kv.split("=") for kv in str(d["config"][0]).split())
            esn = ESN(V, int(cfg["N"]), spectral_radius=float(cfg["r"]),
                      leak_rate=float(cfg["a"]), input_scale=float(cfg["is"]),
                      seed=int(cfg["seed"]))
            try:
                _alignment_gate(esn, seg, {k: d[k] for k in d.files
                                           if k != "config"},
                                d["suffixes"].shape[1])
                print(f"[{i + 1}/{len(files)}] PASS {fn}", flush=True)
            except RuntimeError as e:
                n_bad += 1
                print(f"[{i + 1}/{len(files)}] FAIL {fn}: {e}", flush=True)
        print(f"validation done: {len(files) - n_bad}/{len(files)} passed")
        return

    _, val, _ = load_text8()
    ids = val[: a.chars]
    cells = ([(r, l) for l in a.leaks for r in a.rhos] if a.all_sweep_cells
             else [(a.rhos[0], a.leaks[0])])
    for i, (rho, leak) in enumerate(cells):
        path = os.path.join(DUMP_DIR, dump_name(a.n_reservoir, rho, leak,
                                                a.input_scale, a.seed, a.chars,
                                                a.suffix_len))
        if os.path.exists(path):
            print(f"[{i + 1}/{len(cells)}] exists: {os.path.basename(path)}",
                  flush=True)
            continue
        d = dump_config(ids, a.n_reservoir, rho, leak, a.input_scale, a.seed,
                        n_samples=a.n_samples, suffix_len=a.suffix_len)
        np.savez_compressed(path, **d,
                            config=np.array([f"N={a.n_reservoir} r={rho} "
                                             f"a={leak} is={a.input_scale} "
                                             f"seed={a.seed} T={a.chars}"]))
        print(f"[{i + 1}/{len(cells)}] wrote {os.path.basename(path)} "
              f"({d['states'].nbytes / 1e6:.0f}MB states)", flush=True)


if __name__ == "__main__":
    main()
