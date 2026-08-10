"""Triptych under a COMMON conditioning set (pre-registered): per layer,
each metric partialed on the other two + dpos. Licenses (or not) the
claim "NOW was never ahead, not even at the embedding."

    .venv/bin/python experiments/01_llm_as_reservoir/run_triptych_common.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
from rcllm.data import load_text8  # noqa: E402
from run_which_tree import (VOCAB, LAYERS, B, T, WASH, N_POS, N_PAIRS,  # noqa: E402
                            POS_WINDOW, SUF_L, fivegram_table, ctx4_index,
                            jsd, trailing_match, partial_spearman)

RESULTS = os.path.join(HERE, "..", "..", "results", "exp01")
MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def main():
    rng = np.random.default_rng(0)
    train, val, _ = load_text8()
    text = "".join(VOCAB[c] for c in val[:400_000])
    P5 = fivegram_table(train[:5_000_000])

    from rcllm.activations import load_model, collect_hidden_states
    model, tok = load_model(MODEL)
    try:
        ids_all = tok.encode(text, add_special_tokens=False)
    except TypeError:
        ids_all = tok.encode(text)
        bos = getattr(tok, "bos_token_id", None)
        if bos is not None and ids_all and ids_all[0] == bos:
            ids_all = ids_all[1:]
    piece, lens = {}, np.empty(len(ids_all), dtype=np.int64)
    for k, i in enumerate(ids_all):
        if i not in piece:
            piece[i] = tok.decode([i])
        lens[k] = len(piece[i])
    cum = np.concatenate([[0], np.cumsum(lens)])
    seqs = np.array(ids_all[: B * T]).reshape(B, T)

    bs = rng.integers(0, B, N_POS)
    ts = rng.integers(WASH, T, N_POS)
    char_end = cum[bs * T + ts + 1]
    ok = char_end >= SUF_L + 4
    bs, ts, char_end = bs[ok], ts[ok], char_end[ok]
    n = len(bs)
    sufs = [text[e - SUF_L : e] for e in char_end]
    pext = P5[[ctx4_index(s) for s in sufs]]
    cur_tok = seqs[bs, ts]

    layer_list = [L for L in LAYERS if L != 15]
    hs = collect_hidden_states(model, seqs, layers=layer_list)
    S = {L: hs.pop(L)[bs, ts] for L in layer_list}

    pi, pj = [], []
    while len(pi) < N_PAIRS:
        a = rng.integers(0, n, N_PAIRS)
        b2 = rng.integers(0, n, N_PAIRS)
        keep = (a != b2) & (np.abs(ts[a] - ts[b2]) <= POS_WINDOW)
        pi.extend(a[keep])
        pj.extend(b2[keep])
    pi, pj = np.array(pi[:N_PAIRS]), np.array(pj[:N_PAIRS])
    d_s = np.array([2.0 ** -trailing_match(sufs[i], sufs[j], SUF_L)
                    for i, j in zip(pi, pj)])
    d_p = jsd(pext[pi], pext[pj])
    d_n = (cur_tok[pi] != cur_tok[pj]).astype(float)
    dq = np.abs(ts[pi] - ts[pj]).astype(float)

    out = {}
    print("common-conditioned triptych (each | other two + dpos):")
    for L in layer_list:
        X = S[L].astype(np.float64)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        dst = 1.0 - np.einsum("ij,ij->i", Xn[pi], Xn[pj])
        ps = partial_spearman(dst, d_s, [d_p, d_n, dq])
        pp = partial_spearman(dst, d_p, [d_s, d_n, dq])
        pn = partial_spearman(dst, d_n, [d_s, d_p, dq])
        out[f"L{L}"] = {"past": round(ps, 4), "future": round(pp, 4),
                        "now": round(pn, 4)}
        print(f"  L{L:2d}: past {ps:+.3f}  now {pn:+.3f}  future {pp:+.3f}",
              flush=True)

    with open(os.path.join(RESULTS, "triptych_common.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS}/triptych_common.json")


if __name__ == "__main__":
    main()
