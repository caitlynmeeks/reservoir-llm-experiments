"""F8 follow-ups, pre-registered in the journal:

F8-a  triptych — partial faithfulness to CURRENT-token identity per
      layer (| d_s, d_pext, dpos). Ordinal: peaks at the L4 dip.
F8-b  earn "manifold" — correlation dimension of L12 states (cosine,
      wide window) + smoothness of log C(r) (stairs vs smooth).
F8-c  quantization exoneration — L12 cophenetic on the SAME positions
      from a bf16 model; PR-C gates unchanged.

    .venv/bin/python experiments/01_llm_as_reservoir/run_which_tree_followups.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from rcllm.data import load_text8  # noqa: E402

sys.path.insert(0, HERE)
from run_which_tree import (VOCAB, LAYERS, B, T, WASH, N_POS, N_PAIRS,  # noqa: E402
                            POS_WINDOW, SUF_L, fivegram_table, ctx4_index,
                            jsd, trailing_match, partial_spearman)

RESULTS = os.path.join(HERE, "..", "..", "results", "exp01")
MODEL_4BIT = "mlx-community/Llama-3.2-1B-Instruct-4bit"
MODEL_BF16 = "mlx-community/Llama-3.2-1B-Instruct-bf16"


def main():
    rng = np.random.default_rng(0)                       # SAME stream as main run
    train, val, _ = load_text8()
    text = "".join(VOCAB[c] for c in val[:400_000])
    P5 = fivegram_table(train[:5_000_000])

    from rcllm.activations import load_model, collect_hidden_states
    model, tok = load_model(MODEL_4BIT)
    try:
        ids_all = tok.encode(text, add_special_tokens=False)
    except TypeError:
        ids_all = tok.encode(text)
        bos = getattr(tok, "bos_token_id", None)
        if bos is not None and ids_all and ids_all[0] == bos:
            ids_all = ids_all[1:]
    piece = {}
    lens = np.empty(len(ids_all), dtype=np.int64)
    for k, i in enumerate(ids_all):
        if i not in piece:
            piece[i] = tok.decode([i])
        lens[k] = len(piece[i])
    cum = np.concatenate([[0], np.cumsum(lens)])
    seqs = np.array(ids_all[: B * T]).reshape(B, T)

    bs = rng.integers(0, B, N_POS)
    ts = rng.integers(WASH, T, N_POS)
    g = bs * T + ts
    char_end = cum[g + 1]
    ok = char_end >= SUF_L + 4
    bs, ts, char_end = bs[ok], ts[ok], char_end[ok]
    n = len(bs)
    sufs = [text[e - SUF_L : e] for e in char_end]
    pext = P5[[ctx4_index(s) for s in sufs]]
    cur_tok = seqs[bs, ts]

    layer_list = [L for L in LAYERS if L != 15]
    hs = collect_hidden_states(model, seqs, layers=layer_list)
    S = {L: hs.pop(L)[bs, ts] for L in layer_list}

    # pairs — same construction/rng cadence as the main run
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
    d_pext = jsd(pext[pi], pext[pj])
    dpos = np.abs(ts[pi] - ts[pj]).astype(float)
    d_now = (cur_tok[pi] != cur_tok[pj]).astype(float)

    out = {"triptych": {}}
    print("F8-a triptych — partial faithfulness to NOW (| d_s, d_pext, dpos):")
    for L in layer_list:
        X = S[L].astype(np.float64)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        dst = 1.0 - np.einsum("ij,ij->i", Xn[pi], Xn[pj])
        p_now = partial_spearman(dst, d_now, [d_s, d_pext, dpos])
        out["triptych"][f"L{L}"] = round(p_now, 4)
        print(f"  L{L:2d}: partial_now {p_now:+.3f}", flush=True)

    # F8-b: correlation dimension of L12, cosine, wide window + curve shape
    idx = rng.choice(n, 2000, replace=False)
    X12 = S[12][idx].astype(np.float64)
    X12 /= np.linalg.norm(X12, axis=1, keepdims=True) + 1e-9
    dist = pdist(X12, metric="cosine")
    dist = dist[dist > 0]
    rs = np.geomspace(np.quantile(dist, 1e-3), dist.max(), 300)
    C = np.searchsorted(np.sort(dist), rs) / len(dist)
    keep = C > 0
    x, y = np.log2(rs[keep]), np.log2(C[keep])
    m = (C[keep] > 1e-3) & (C[keep] < 0.5)
    slope = float(np.polyfit(x[m], y[m], 1)[0])
    resid = float(np.sqrt(np.mean(
        (y[m] - np.polyval(np.polyfit(x[m], y[m], 1), x[m])) ** 2)))
    out["dimension_L12"] = {"wide_window_D2": round(slope, 2),
                            "fit_rms_resid_bits": round(resid, 3)}
    print(f"F8-b: L12 correlation dimension {slope:.2f} "
          f"(fit residual {resid:.3f} bits — small = smooth, no stairs)")

    # F8-c: bf16 spot check, same positions
    del model
    model16, _ = load_model(MODEL_BF16)
    hs16 = collect_hidden_states(model16, seqs, layers=[12])
    X16 = hs16[12][bs, ts][idx].astype(np.float64)
    X16 /= np.linalg.norm(X16, axis=1, keepdims=True) + 1e-9
    d16 = pdist(X16, metric="cosine")
    coph16 = float(cophenet(linkage(d16, "single"), d16)[0])
    Xn16 = X16.copy()
    for c in range(Xn16.shape[1]):
        rng.shuffle(Xn16[:, c])
    dn16 = pdist(Xn16, metric="cosine")
    null16 = float(cophenet(linkage(dn16, "single"), dn16)[0])
    out["bf16_cophenetic_L12"] = {"value": round(coph16, 4),
                                  "null": round(null16, 4)}
    print(f"F8-c: bf16 L12 cophenetic {coph16:.3f} (null {null16:.3f}) — "
          f"gates: >=0.90 & >= null+0.15 indicts quantization; <0.75 exonerates")

    with open(os.path.join(RESULTS, "which_tree_followups.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS}/which_tree_followups.json")


if __name__ == "__main__":
    main()
