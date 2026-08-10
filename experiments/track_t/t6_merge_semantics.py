"""T6: are the tropical automaton's merged suffix classes predictive
twins (approximate causal states) or arbitrary collisions?

Pre-registered design (journal, written before this ran): MERGED pairs =
determined multi-sample k=7 suffix classes sharing a state. NULL pairs =
determined classes NOT sharing a state, matched on shared trailing-
substring length (the critical confound) and context-frequency bin.
Metric: Jensen–Shannon divergence (bits) between corpus next-char
distributions conditioned on each full 7-char context (5M-char train
slice, add-0.1 smoothing, both contexts >= 20 occurrences).
Thresholds: R = median_null / median_merged; R >= 2 with MWU p < 0.01 =>
predictive twins; R <= 1.25 => arbitrary collisions; between => partial.

    .venv/bin/python experiments/track_t/t6_merge_semantics.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from rcllm.data import V, load_text8  # noqa: E402
from rcllm.dump_states import dump_config  # noqa: E402
from rcllm.tropical import TropicalESN  # noqa: E402

RESULTS = os.path.join(HERE, "..", "..", "results", "track_t")
K = 7
MIN_COUNT = 20
ALPHA = 0.1
VOCAB = " abcdefghijklmnopqrstuvwxyz"
rng = np.random.default_rng(2026)


def suffix_key(sfx: np.ndarray) -> int:
    """Base-27 integer for a (K,) uint8 suffix."""
    return int(sum(int(c) * 27 ** (K - 1 - j) for j, c in enumerate(sfx)))


def tail_len(a: np.ndarray, b: np.ndarray) -> int:
    t = 0
    while t < K and a[K - 1 - t] == b[K - 1 - t]:
        t += 1
    return t


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2
    def kl(x, y):
        nz = x > 0
        return float((x[nz] * np.log2(x[nz] / y[nz])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    # ---- automaton classes from the champion dump ----
    _, val, _ = load_text8()
    d = dump_config(val[:200_000],
                    esn=TropicalESN(V, 1000, cycle_mean=-0.1, input_scale=1.0,
                                    seed=0),
                    encode=lambda s: s, suffix_len=12)
    _, state_id = np.unique(d["states"], axis=0, return_inverse=True)
    _, cls, cls_inv = np.unique(d["suffixes"][:, -K:], axis=0,
                                return_index=True, return_inverse=True)
    members, counts = defaultdict(set), defaultdict(int)
    for c, s in zip(cls_inv, state_id):
        members[c].add(int(s))
        counts[c] += 1
    suffix_of = {c: d["suffixes"][cls[i], -K:]
                 for i, c in enumerate(np.unique(cls_inv))}
    determined = [c for c in members if counts[c] >= 2 and len(members[c]) == 1]
    state_of = {c: next(iter(members[c])) for c in determined}

    # ---- corpus next-char tables for 7-char contexts ----
    train, _, _ = load_text8()
    train = train[:5_000_000].astype(np.int64)
    T = len(train)
    keys = np.zeros(T - K, dtype=np.int64)
    for j in range(K):
        keys += train[j : T - K + j] * 27 ** (K - 1 - j)
    combined = keys * 27 + train[K:]
    uniq, ucnt = np.unique(combined, return_counts=True)

    def dist(key7: int):
        lo = np.searchsorted(uniq, key7 * 27)
        hi = np.searchsorted(uniq, key7 * 27 + 27)
        cvec = np.zeros(27)
        for u, n in zip(uniq[lo:hi], ucnt[lo:hi]):
            cvec[u % 27] = n
        tot = cvec.sum()
        return cvec, tot

    dists, totals = {}, {}
    for c in determined:
        cvec, tot = dist(suffix_key(suffix_of[c]))
        if tot >= MIN_COUNT:
            dists[c] = (cvec + ALPHA) / (tot + ALPHA * 27)
            totals[c] = tot
    usable = [c for c in determined if c in dists]
    print(f"{len(determined)} determined classes; {len(usable)} with corpus "
          f"count >= {MIN_COUNT}", flush=True)

    # ---- merged pairs ----
    groups = defaultdict(list)
    for c in usable:
        groups[state_of[c]].append(c)
    merged_pairs = []
    for g in groups.values():
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                merged_pairs.append((g[i], g[j]))
    rng.shuffle(merged_pairs)
    merged_pairs = merged_pairs[:2000]

    def freq_bin(a, b):
        return int(np.log2(min(totals[a], totals[b])))

    # ---- null pools matched on (tail length, frequency bin) ----
    need = defaultdict(int)
    m_rows = []
    for a, b in merged_pairs:
        t = tail_len(suffix_of[a], suffix_of[b])
        m_rows.append((a, b, t, freq_bin(a, b), jsd(dists[a], dists[b])))
        need[(t, freq_bin(a, b))] += 1

    by_tail = defaultdict(list)
    for c in usable:
        for t in range(1, K):
            by_tail[(t, bytes(suffix_of[c][-t:].tolist()))].append(c)
    null_rows = []
    for (t, fb), n_need in need.items():
        cands = [v for k2, v in by_tail.items() if k2[0] == t and len(v) >= 2]
        pool = []
        for v in cands:
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    a, b = v[i], v[j]
                    if (state_of[a] != state_of[b]
                            and tail_len(suffix_of[a], suffix_of[b]) == t
                            and freq_bin(a, b) == fb):
                        pool.append((a, b))
        rng.shuffle(pool)
        for a, b in pool[: n_need]:
            null_rows.append((a, b, t, fb, jsd(dists[a], dists[b])))

    mj = np.array([r[4] for r in m_rows])
    nj = np.array([r[4] for r in null_rows])
    R = float(np.median(nj) / np.median(mj))
    mwu = mannwhitneyu(nj, mj, alternative="greater")
    verdict = ("predictive_twins" if R >= 2.0 and mwu.pvalue < 0.01 else
               "arbitrary_collisions" if R <= 1.25 else "partial")

    print(f"\nmerged pairs: {len(mj)} (median JSD {np.median(mj):.4f} bits)")
    print(f"null pairs:   {len(nj)} (median JSD {np.median(nj):.4f} bits)")
    print(f"R = {R:.2f}, MWU p = {mwu.pvalue:.2e}  ->  VERDICT: {verdict}")

    sample = sorted(m_rows, key=lambda r: r[4])[:6]
    print("\nlowest-JSD merged pairs:")
    for a, b, t, fb, j in sample:
        sa = "".join(VOCAB[x] for x in suffix_of[a])
        sb = "".join(VOCAB[x] for x in suffix_of[b])
        print(f"  {sa!r} ~ {sb!r}  tail={t} JSD={j:.4f}")

    out = {"n_merged": len(mj), "n_null": len(nj),
           "median_jsd_merged": float(np.median(mj)),
           "median_jsd_null": float(np.median(nj)),
           "R": R, "mwu_p": float(mwu.pvalue), "verdict": verdict,
           "per_tail": {str(t): {
               "merged": float(np.median([r[4] for r in m_rows if r[2] == t]) )
                         if any(r[2] == t for r in m_rows) else None,
               "null": float(np.median([r[4] for r in null_rows if r[2] == t]))
                       if any(r[2] == t for r in null_rows) else None}
               for t in range(1, K)}}
    with open(os.path.join(RESULTS, "t6_merge_semantics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS}/t6_merge_semantics.json")


if __name__ == "__main__":
    main()
