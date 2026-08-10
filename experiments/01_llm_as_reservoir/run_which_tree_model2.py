"""Second-model replication of Finding 8 + the construction-zone question.

Common-conditioned triptych (past/now/future | other two + dpos) per
layer, plus cophenetic tree gates at the most future-keyed layer.
Layers default to {1, n/4, n/2, 3n/4, n-2} (last layer excluded as the
mechanical ceiling). d_p^self omitted entirely (no lm-head dependence).

    .venv/bin/python experiments/01_llm_as_reservoir/run_which_tree_model2.py \
        --model mlx-community/Qwen2.5-1.5B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
from rcllm.data import load_text8  # noqa: E402
from run_which_tree import (VOCAB, B, T, WASH, N_POS, N_PAIRS,  # noqa: E402
                            POS_WINDOW, SUF_L, fivegram_table, ctx4_index,
                            jsd, trailing_match, partial_spearman)

RESULTS = os.path.join(HERE, "..", "..", "results", "exp01")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    train, val, _ = load_text8()
    text = "".join(VOCAB[c] for c in val[:400_000])
    P5 = fivegram_table(train[:5_000_000])

    from rcllm.activations import load_model, collect_hidden_states
    model, tok = load_model(a.model)
    n_layers = len(model.model.layers)
    layers = a.layers or sorted({1, n_layers // 4, n_layers // 2,
                                 3 * n_layers // 4, n_layers - 2})
    print(f"{a.model}: {n_layers} layers; probing {layers}", flush=True)

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
    assert "".join(piece[i] for i in ids_all[:200]) == text[: cum[200]], \
        "token/char offset reconstruction failed for this tokenizer"
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

    hs = collect_hidden_states(model, seqs, layers=layers)
    S = {L: hs.pop(L)[bs, ts] for L in layers}

    pi, pj = [], []
    while len(pi) < N_PAIRS:
        x = rng.integers(0, n, N_PAIRS)
        y = rng.integers(0, n, N_PAIRS)
        keep = (x != y) & (np.abs(ts[x] - ts[y]) <= POS_WINDOW)
        pi.extend(x[keep])
        pj.extend(y[keep])
    pi, pj = np.array(pi[:N_PAIRS]), np.array(pj[:N_PAIRS])
    d_s = np.array([2.0 ** -trailing_match(sufs[i], sufs[j], SUF_L)
                    for i, j in zip(pi, pj)])
    d_p = jsd(pext[pi], pext[pj])
    d_n = (cur_tok[pi] != cur_tok[pj]).astype(float)
    dq = np.abs(ts[pi] - ts[pj]).astype(float)

    out = {"model": a.model, "n_layers": n_layers, "layers": {}}
    for L in layers:
        X = S[L].astype(np.float64)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        dst = 1.0 - np.einsum("ij,ij->i", Xn[pi], Xn[pj])
        ps = partial_spearman(dst, d_s, [d_p, d_n, dq])
        pp = partial_spearman(dst, d_p, [d_s, d_n, dq])
        pn = partial_spearman(dst, d_n, [d_s, d_p, dq])
        out["layers"][f"L{L}"] = {"past": round(ps, 4), "now": round(pn, 4),
                                  "future": round(pp, 4)}
        print(f"  L{L:2d}: past {ps:+.3f}  now {pn:+.3f}  future {pp:+.3f}",
              flush=True)

    Lbest = max(layers, key=lambda L: out["layers"][f"L{L}"]["future"])
    idx = rng.choice(n, 2000, replace=False)
    X2 = S[Lbest][idx].astype(np.float64)
    X2 /= np.linalg.norm(X2, axis=1, keepdims=True) + 1e-9
    dist2 = pdist(X2, metric="cosine")
    coph = float(cophenet(linkage(dist2, "single"), dist2)[0])
    Xn2 = X2.copy()
    for c in range(Xn2.shape[1]):
        rng.shuffle(Xn2[:, c])
    dn2 = pdist(Xn2, metric="cosine")
    null = float(cophenet(linkage(dn2, "single"), dn2)[0])
    out["cophenetic"] = {"layer": Lbest, "value": round(coph, 4),
                         "null": round(null, 4)}
    print(f"cophenetic L{Lbest}: {coph:.3f} (null {null:.3f})")

    tag = a.model.split("/")[-1]
    with open(os.path.join(RESULTS, f"which_tree_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS}/which_tree_{tag}.json")


if __name__ == "__main__":
    main()
