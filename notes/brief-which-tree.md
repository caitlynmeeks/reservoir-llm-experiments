# Design brief: which tree does the Llama grow?

*For chat-Fable, from the bench. 2026-08-10, midday. Design-before-code;
nothing below has run. Ferry back amendments and threshold numbers.*

## Standing results this must respect

- **F5 (substrate-independence):** tanh, max-plus, and (U6) even linear
  ponds grow the same suffix-keyed geometry; faithfulness to the suffix
  ultrametric predicts bpc across 40 configs (−0.83). Tree-correlation
  is free with fading memory (linear pond faithfulness 0.4484 ≡ tanh's
  0.448); nonlinearity only crispens (cophenetic 0.946 → 0.985).
- **F7 (working set, not tape):** the Llama's per-position state holds
  ~2–3 tokens of lag-indexed history, at every layer; positional trace
  *declines* monotonically with depth. Storage lives in the KV cache;
  attention retrieves.

## The question, sharpened

The ponds' tree is a **past tree**: states cluster by what was just
read, because fading memory forces recency to dominate the state. The
Llama is not forced — next-token training plausibly organizes its
states by the **future**: contexts cluster when they predict similar
continuations. So "which tree" =

> Is Llama state geometry keyed to the suffix metric (past tree,
> universality extends into trained systems) or to a prediction metric
> (future tree — "retrieval without storage" made photographic), and
> does the key change with depth?

Candidate outcomes: (a) suffix tree at pond-like faithfulness →
universality; (b) suffix tree but ONLY to depth ~2–3 → the trivial
working-set null, F7 restated — must be excluded by control, not
discovered; (c) prediction-tree dominance → the moat, named; (d) no
tree (low cophenetic) → geometry genuinely non-hierarchical.

## Proposed design (amend freely)

1. **Drive: natural text8 val text** (semantics intact), with the
   random-token drive from F7 as the contrast condition. 2×2 if budget
   allows: {natural, random} × {suffix metric, prediction metric}.
2. **Two reference metrics per state pair (i,j):**
   - suffix ultrametric d_s = 2^(−shared trailing chars), as U1;
   - prediction metric d_p = JSD between the model's own next-token
     distributions at the two positions (computed from our verified
     manual forward + lm head; cheap at sampled positions).
   Faithfulness_s and Faithfulness_p = Spearman(state distance, d_s or
   d_p). NOTE the confound: d_s and d_p correlate (shared suffix ⇒
   similar next-char). Need partial correlations (faithfulness_p
   controlling d_s, and vice versa) — T6 taught us this lesson; the
   tail-matched-null trick generalizes here.
3. **Layers:** 1, 7, 15 minimum (early/mid/late; F7 says positional
   trace is max at L1).
4. **Controls:** (i) truncated suffix metric at k≤3 — how much
   suffix-faithfulness does the working set alone explain? (ii) tanh
   pond driven by the SAME natural text (dumps exist — text8-driven);
   (iii) shuffled-pair null.
5. **Instruments:** I1 dumper already handles Llama states via the
   esn/encode hooks precedent; U1/U2 code reusable as-is; per-position
   logits need one new collection function (small).

## Pre-registration skeletons (thresholds to be set together)

- **PR-A (the key):** at layers ≥ 7, partial faithfulness to d_p
  (controlling d_s) EXCEEDS partial faithfulness to d_s (controlling
  d_p). At layer 1 the order reverses.
- **PR-B (depth trend):** faithfulness_s declines with depth (F7's
  shedding, restated geometrically); faithfulness_p is flat or rising.
- **PR-C (tree-ness):** single-linkage cophenetic of Llama states at
  the layer of max faithfulness_p: threshold TBD — what counts as
  "tree-like" for a trained system, given the pond gradient
  0.946–0.985 and the U6 lesson that raw tree-ness is cheap?
- **PR-D (working-set null):** the truncated (k≤3) suffix metric must
  NOT explain ≥ (threshold)% of whatever faithfulness_s survives, or
  outcome (b) is declared and the headline claim is withheld.

## Round 2 — chat-Fable's amendments, AGREED (final design)

1. **d_p circularity broken.** The last layer's logits are a linear
   projection of its state, so faithfulness_p rises mechanically with
   depth. Fixes adopted: (a) last layer excluded from all headline
   claims, reported only as the mechanical ceiling; (b) primary
   prediction metric is **d_p^ext** = JSD between empirical 5-gram
   next-char distributions at the two positions (tables recomputed from
   our n-gram machinery); the model's own distribution (marginalized to
   next-char by summing token probs grouped by first character) becomes
   d_p^self, a depth-consistency diagnostic only. Headlines quote
   d_p^ext.
2. **Cosine is the Llama's native ruler** (RMSNorm ⇒ direction carries
   information): cosine primary for Llama faithfulness, Euclidean as
   check. Ponds stay Euclidean-primary (amplitude is meaningful — the
   masonry lesson).
3. **Position confound (RoPE):** pairs sampled position-matched within
   a window, and |i−j| partialed out alongside d_s and d_p. Control
   (iv). Layers: {1, 4, 8, 12, 14} with 15 as the ceiling row.
4. **Budget:** natural text fully; random-token drive only calibrates
   the d_s machinery.

**Pre-registered numbers (agreed):**
- PR-A: mid layers, partial_p − partial_s ≥ 0.10 with 1,000-resample
  bootstrap CI excluding zero; layer 1 reversed with the same margin.
- PR-B: over ≥5 layers, Spearman(layer, faithfulness_s) ≤ −0.8;
  faithfulness_p non-decreasing within −0.02 tolerance.
- PR-C: two gates, both required: cophenetic ≥ 0.90 absolute AND
  ≥ position-shuffled empirical null + 0.15. 0.75–0.90 = "hierarchical
  but loose"; < 0.75 = outcome (d).
- PR-D: deep-suffix share = partial(state, d_s | d_s^k≤3) / raw
  faithfulness_s; past-tree-beyond-working-set claimed only if
  share ≥ 0.30 (floor 0.25, arguable to 0.40).
- PR-E: pond natural-text dumps through identical partials; ordinal:
  pond partial_p ≈ 0 once suffix is controlled. Money figure: the
  (partial_s, partial_p) plane — pond in the past corner, Llama's
  layers tracing a path.

## Honesty flag to carry back (scoreboard discipline)

At N=50k the trainable readout is (50k+1)×27 ≈ **1.35M params — more
than the 4-gram's 531k table.** The N=5k "quarter of the parameters"
framing does NOT transfer to the 50k fence-clearing. The transferable
claims: sample efficiency (2.319 test on 1M chars vs the 4-gram's
2.392 on 5M) and zero feature-training. Part II wording must switch
axes accordingly; the params-vs-bpc frontier plot needs the 50k point
placed honestly.
