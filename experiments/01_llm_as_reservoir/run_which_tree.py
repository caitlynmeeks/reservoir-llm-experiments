"""Which tree does the Llama grow? (final design per notes/brief-which-tree.md)

Past tree vs. future tree: for sampled positions in NATURAL text8 text,
compute per state pair (i, j)
  d_s     = 2^-(shared trailing chars)          (suffix / past metric)
  d_s3    = same, truncated at 3 chars          (working-set null)
  d_pext  = JSD of empirical 5-gram next-char distributions  (future,
            external — the headline metric; breaks lm-head circularity)
  d_pself = JSD of the model's own next-token distributions
            marginalized to next-char           (diagnostic only)
  dpos    = |token-position difference|         (RoPE confound control)
then per layer: cosine state distance (Llama's native ruler),
raw Spearman faithfulness to d_s and d_pext, and rank-partials
  partial_p = partial(state, d_pext | d_s, dpos)
  partial_s = partial(state, d_s | d_pext, dpos)
with a 1000-resample bootstrap CI on (partial_p - partial_s), plus the
deep-suffix share partial(state, d_s | d_s3, dpos)/raw_s, cophenetic
tree gates against a coordinate-shuffled empirical null, and the tanh
pond (Euclidean, its own dump) as the past-corner control. Layer 15 is
reported only as the mechanical ceiling. Verdicts printed against the
pre-registered thresholds (PR-A..E in the brief).

    .venv/bin/python experiments/01_llm_as_reservoir/run_which_tree.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from rcllm.data import V, load_text8  # noqa: E402
from rcllm.dump_states import DUMP_DIR, dump_name  # noqa: E402

RESULTS = os.path.join(HERE, "..", "..", "results", "exp01")
MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
VOCAB = " abcdefghijklmnopqrstuvwxyz"
LAYERS = [1, 4, 8, 12, 14, 15]          # 15 = mechanical ceiling only
B, T, WASH = 32, 512, 64
N_POS, N_PAIRS, POS_WINDOW = 6000, 20000, 64
SUF_L, ALPHA = 12, 0.1
BLUE, ORANGE, MUT = "#2a78d6", "#eb6834", "#898781"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#eceae6"


def jsd(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rowwise JSD in bits; P, Q: (n, k) probability rows."""
    M = (P + Q) / 2
    def kl(A, Bm):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = A * np.log2(A / Bm)
        return np.nansum(np.where(A > 0, t, 0.0), axis=1)
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def trailing_match(a: str, b: str, cap: int) -> int:
    t = 0
    while t < cap and t < len(a) and t < len(b) and a[-1 - t] == b[-1 - t]:
        t += 1
    return t


def partial_spearman(x, y, controls):
    """Partial Spearman of (x, y) given controls: residualize ranks by OLS."""
    rx, ry = rankdata(x), rankdata(y)
    Z = np.column_stack([rankdata(c) for c in controls] + [np.ones(len(x))])
    bx, _, _, _ = np.linalg.lstsq(Z, rx, rcond=None)
    by, _, _, _ = np.linalg.lstsq(Z, ry, rcond=None)
    ex, ey = rx - Z @ bx, ry - Z @ by
    d = np.linalg.norm(ex) * np.linalg.norm(ey)
    return float(ex @ ey / d) if d > 0 else 0.0


def fivegram_table(train: np.ndarray) -> np.ndarray:
    tr = train.astype(np.int64)
    Tn = len(tr)
    keys = np.zeros(Tn - 4, dtype=np.int64)
    for j in range(4):
        keys += tr[j : Tn - 4 + j] * 27 ** (3 - j)
    counts = np.bincount(keys * 27 + tr[4:], minlength=(27**4) * 27)
    counts = counts.reshape(-1, 27).astype(np.float64)
    return (counts + ALPHA) / (counts.sum(1, keepdims=True) + ALPHA * 27)


def ctx4_index(s: str) -> int:
    return sum(VOCAB.index(c) * 27 ** (3 - j) for j, c in enumerate(s[-4:]))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(0)

    # ---- corpus + tables ----
    train, val, _ = load_text8()
    text = "".join(VOCAB[c] for c in val[:400_000])
    P5 = fivegram_table(train[:5_000_000])

    # ---- tokenize with exact char offsets ----
    from rcllm.activations import load_model, collect_hidden_states
    import mlx.core as mx
    model, tok = load_model(MODEL)
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
    assert "".join(piece[i] for i in ids_all[:200]) == text[: cum[200]], \
        "token/char offset reconstruction failed"
    need = B * T
    seqs = np.array(ids_all[:need]).reshape(B, T)

    # ---- sample positions ----
    bs = rng.integers(0, B, N_POS)
    ts = rng.integers(WASH, T, N_POS)
    g = bs * T + ts
    char_end = cum[g + 1]
    ok = char_end >= SUF_L + 4
    bs, ts, char_end = bs[ok], ts[ok], char_end[ok]
    n = len(bs)
    sufs = [text[e - SUF_L : e] for e in char_end]
    pext = P5[[ctx4_index(s) for s in sufs]]                  # (n, 27)

    # ---- forward pass, sampled states per layer ----
    hs = collect_hidden_states(model, seqs, layers=LAYERS)
    S = {L: hs.pop(L)[bs, ts] for L in LAYERS}                # (n, 2048) each

    # ---- d_pself: model next-token dist marginalized to next-char ----
    inner = model.model
    x15 = mx.array(S[15])
    logits = np.array(inner.embed_tokens.as_linear(
        inner.norm(x15)).astype(mx.float32))
    z = logits - logits.max(1, keepdims=True)
    prob = np.exp(z)
    prob /= prob.sum(1, keepdims=True)
    vocab_n = prob.shape[1]
    bucket = np.full(vocab_n, 27, dtype=np.int64)
    for i in range(vocab_n):
        p = tok.decode([i])
        if p and p[0] in VOCAB:
            bucket[i] = VOCAB.index(p[0])
    pself = np.zeros((n, 28))
    np.add.at(pself.T, bucket, prob.T)                        # marginalize

    # ---- pairs, position-matched ----
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
    d_s3 = np.array([2.0 ** -trailing_match(sufs[i], sufs[j], 3)
                     for i, j in zip(pi, pj)])
    d_pext = jsd(pext[pi], pext[pj])
    d_pself = jsd(pself[pi], pself[pj])
    dpos = np.abs(ts[pi] - ts[pj]).astype(float)

    # ---- per-layer analysis ----
    def cosine_pair_dist(X):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        return 1.0 - np.einsum("ij,ij->i", Xn[pi], Xn[pj])

    boot_idx = [rng.integers(0, N_PAIRS, N_PAIRS) for _ in range(1000)]
    rows = {}
    for L in LAYERS:
        dst = cosine_pair_dist(S[L].astype(np.float64))
        raw_s = float(spearmanr(dst, d_s).statistic)
        raw_p = float(spearmanr(dst, d_pext).statistic)
        raw_pself = float(spearmanr(dst, d_pself).statistic)
        p_p = partial_spearman(dst, d_pext, [d_s, dpos])
        p_s = partial_spearman(dst, d_s, [d_pext, dpos])
        deep = partial_spearman(dst, d_s, [d_s3, dpos])
        share = deep / raw_s if abs(raw_s) > 1e-6 else float("nan")
        diffs = []
        for bidx in boot_idx[:1000]:
            dd, ds_, dp_, dq_ = dst[bidx], d_s[bidx], d_pext[bidx], dpos[bidx]
            diffs.append(partial_spearman(dd, dp_, [ds_, dq_])
                         - partial_spearman(dd, ds_, [dp_, dq_]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows[L] = {"raw_s": round(raw_s, 4), "raw_pext": round(raw_p, 4),
                   "raw_pself": round(raw_pself, 4),
                   "partial_p": round(p_p, 4), "partial_s": round(p_s, 4),
                   "diff_ci": [round(float(lo), 4), round(float(hi), 4)],
                   "deep_suffix_share": round(share, 4)}
        print(f"L{L:2d}: raw_s {raw_s:+.3f} raw_pext {raw_p:+.3f} | "
              f"partial_s {p_s:+.3f} partial_p {p_p:+.3f} "
              f"(diff CI [{lo:+.3f},{hi:+.3f}]) share {share:.2f}",
              flush=True)

    # ---- cophenetic gates at best non-ceiling layer ----
    Lbest = max((L for L in LAYERS if L != 15),
                key=lambda L: rows[L]["partial_p"])
    idx2 = rng.choice(n, 2000, replace=False)
    X2 = S[Lbest][idx2].astype(np.float64)
    X2n = X2 / (np.linalg.norm(X2, axis=1, keepdims=True) + 1e-9)
    dist2 = pdist(X2n, metric="cosine")
    coph = float(cophenet(linkage(dist2, "single"), dist2)[0])
    Xnull = X2n.copy()
    for c in range(Xnull.shape[1]):
        rng.shuffle(Xnull[:, c])
    dnull = pdist(Xnull, metric="cosine")
    coph_null = float(cophenet(linkage(dnull, "single"), dnull)[0])
    print(f"cophenetic L{Lbest}: {coph:.3f} (shuffle null {coph_null:.3f})")

    # ---- pond control (Euclidean; its own text8-driven dump) ----
    dp = np.load(os.path.join(DUMP_DIR, dump_name(5000, 0.6, 1.0, 1.0, 0,
                                                  200_000)))
    pn = dp["states"].shape[0]
    psufs = ["".join(VOCAB[c] for c in s) for s in dp["suffixes"]]
    ppext = P5[[ctx4_index(s) for s in psufs]]
    pt = dp["positions_t"].astype(float)
    a = rng.integers(0, pn, N_PAIRS * 3)
    b3 = rng.integers(0, pn, N_PAIRS * 3)
    keep = (a != b3) & (np.abs(pt[a] - pt[b3]) <= POS_WINDOW)
    a, b3 = a[keep][:N_PAIRS], b3[keep][:N_PAIRS]
    pds = np.array([2.0 ** -trailing_match(psufs[i], psufs[j], 8)
                    for i, j in zip(a, b3)])
    pdp = jsd(ppext[a], ppext[b3])
    pdq = np.abs(pt[a] - pt[b3])
    pdst = np.linalg.norm(dp["states"][a].astype(np.float64)
                          - dp["states"][b3].astype(np.float64), axis=1)
    pond = {"partial_p": round(partial_spearman(pdst, pdp, [pds, pdq]), 4),
            "partial_s": round(partial_spearman(pdst, pds, [pdp, pdq]), 4)}
    print(f"pond control: partial_s {pond['partial_s']:+.3f} "
          f"partial_p {pond['partial_p']:+.3f}")

    # ---- figure: the (partial_s, partial_p) plane ----
    fig, ax = plt.subplots(figsize=(7.2, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    for sd in ("left", "bottom"):
        ax.spines[sd].set_color(MUT)
    ax.tick_params(colors=MUT, labelcolor=INK2)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    lim = max(0.5, *(abs(rows[L][k]) for L in LAYERS
                     for k in ("partial_s", "partial_p")),
              abs(pond["partial_s"]), abs(pond["partial_p"])) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color=GRID, lw=1)
    path = [L for L in LAYERS if L != 15]
    xs = [rows[L]["partial_s"] for L in path]
    ys = [rows[L]["partial_p"] for L in path]
    ax.plot(xs, ys, color=BLUE, lw=1.5, alpha=0.6, zorder=2)
    ax.scatter(xs, ys, s=55, color=BLUE, zorder=3)
    for L, x, y in zip(path, xs, ys):
        ax.annotate(f"L{L}", (x, y), textcoords="offset points",
                    xytext=(7, 4), color=BLUE, fontsize=9)
    ax.scatter([rows[15]["partial_s"]], [rows[15]["partial_p"]], s=55,
               facecolors="none", edgecolors=BLUE, zorder=3)
    ax.annotate("L15 (ceiling)", (rows[15]["partial_s"],
                rows[15]["partial_p"]), textcoords="offset points",
                xytext=(7, -10), color=MUT, fontsize=8)
    ax.scatter([pond["partial_s"]], [pond["partial_p"]], s=70, color=ORANGE,
               zorder=3)
    ax.annotate("tanh pond", (pond["partial_s"], pond["partial_p"]),
                textcoords="offset points", xytext=(8, -3), color=ORANGE,
                fontsize=9)
    ax.set_xlabel("partial faithfulness to the PAST  (suffix | future, pos)",
                  color=INK2)
    ax.set_ylabel("partial faithfulness to the FUTURE  (5-gram | suffix, pos)",
                  color=INK2)
    ax.set_title("Which tree? The past–future plane (cosine for Llama, "
                 "Euclidean for pond)", color=INK, fontsize=11, loc="left",
                 pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "which_tree_plane.png"), dpi=150)

    out = {"layers": rows, "pond": pond,
           "cophenetic": {"layer": Lbest, "value": coph, "null": coph_null},
           "config": {"model": MODEL, "B": B, "T": T, "n_pos": int(n),
                      "n_pairs": N_PAIRS, "pos_window": POS_WINDOW,
                      "suffix_len": SUF_L}}
    with open(os.path.join(RESULTS, "which_tree.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS}/which_tree.json / which_tree_plane.png")


if __name__ == "__main__":
    main()
