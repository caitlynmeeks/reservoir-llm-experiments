# Lab journal — the all-nighter of 2026-08-09/10

*Caity + Claude, one pot of coffee. What we built, measured, and learned.*

## The scoreboard (text8 val bpc, lower = better)

| Model | bpc | Trained params | Training time |
|---|---|---|---|
| uniform | 4.75 | 0 | — |
| 1-gram | 4.13 | 27 | seconds |
| 2-gram | 3.45 | ~7×10² | seconds |
| **tropical ESN N=1k, λ=−0.1** | **3.10** | 27k | 0.003 s solve |
| tanh ESN N=1k (matched control) | 3.10 | 27k | 0.003 s solve |
| 3-gram | 2.91 | ~2×10⁴ | seconds |
| tanh ESN N=5k, untuned defaults | 2.80 | 135k | 0.16 s solve |
| **tanh ESN N=5k, ρ=0.6, leak=1 (champion)** | **2.70** | 135k | 0.16 s solve |
| **tanh ESN N=5k + logistic readout, ρ=0.95 (2M chars)** | **2.45** | 135k | ~2 min SGD |
| **tanh ESN N=5k + logistic readout, ρ=0.95 (5M chars)** | **2.398** | 135k | ~3.5 min SGD |
| 4-gram | 2.392 | ~5×10⁵ | seconds |
| 5-gram | 2.06 | ~1.4×10⁷ | seconds |
| small trained transformer (lit.) | ~1.4–1.6 | 1–10M | hours |

## Finding 1 — Language doesn't want the edge (at this scale)

Classic RC lore (and our lesson demo 3): memory capacity peaks at ρ≈1.
Language voted otherwise. The 10×4 sweep (`results/exp02/
sweep_rho_leak_N5000_T2000000_10x4.png`) is monotone in leak (fast
states win everywhere) with a broad interior optimum at **ρ≈0.5–0.6,
leak=1.0**. Text8 next-char prediction is a short-range game (a 4-char
5-gram hits 2.06), and the reservoir does best as a short sharp pond.

## Finding 2 — The pond grows a suffix tree (Track U)

- **U1**: one number per sweep cell — Spearman between state-space
  distance and suffix ultrametric distance ("faithfulness") — correlates
  −0.83 with bpc across all 40 cells. Acceptance criterion met: geometry
  explains the heatmap. Decomposition caveat: faithfulness explains the
  *leak* axis; within leak=1.0 it's flat while bpc still varies with ρ —
  that residual is Finding 3.
- **U2**: single-linkage dendrogram of champion states, cophenetic
  correlation **0.990** — the state cloud is very nearly an exact tree,
  and its top-level branches ARE the last character, with the previous
  character splitting each branch one level down
  (`results/track_u/u2_dendrogram.png`).
- **U3**: the state as a numeral in base 1/ρ whose digits are recent
  characters. Decode depth: linear in −1/ln ρ at small ρ (numeral story
  verified), saturating at high ρ (digit crosstalk)
  (`results/track_u/u3_digit_decay_L20.png`).
- U2 and U3 are one fact photographed twice: the dendrogram's top-level
  split on the last character IS the numeral's most-significant digit.

## Finding 3 — The interference cost of superposition: memory that helps probes hurts prediction

The night's sharpest result. Decodable memory depth is **monotone
increasing** in ρ (7.5 chars at the champion ρ=0.6 → 12.7 at ρ=1.25),
while language performance **worsens** past ρ≈0.55. The extra past is
demonstrably in the state — a dedicated probe reads it out — but the
*single* next-char readout pays for it as interference: old characters
are stored in superposition, and a shared linear readout cannot avoid
paying crosstalk on features it doesn't need. (This puts the pond in
direct conversation with the interpretability literature on features in
superposition, and reframes experiment 01: attention retrieves the past
without storing it in a crowded state — a mechanism for *dodging* the
superposition tax.)

**Acid test (P1, running now) — PRE-REGISTERED before the curve
completed** (at write time only ρ=0.4, 0.5 had finished; argmin
unknown):

- *Pinned*: logistic argmin ρ ∈ [0.5, 0.6] → the tax is a property of
  the state, deeper than the readout.
- *Migrated*: logistic argmin ρ ≥ 0.8 → the tax was largely ridge's
  poverty; deep memory was usable all along.
- *Partial*: argmin in (0.6, 0.8) → the likeliest single outcome and
  the easiest to story-tell after the fact — which is exactly why it is
  named in advance.

Second statistic, same run: the **penalty slope past the optimum** —
Δbpc at ρ=1.1 relative to each curve's own minimum, ridge vs logistic,
and their ratio (the fraction of the high-ρ tax the better readout
could not remove). Location and cost are separate axes; Finding 3's
final wording quotes both.

*Protocol (honesty check, verified in code before results):* headline
deltas are TEST-split numbers; val is spent only on early stopping
(logistic) and temperature (ridge); train/val/test slices, washout
(200), and segment counts (64/8/8) are identical between the ridge
sweep cells and the logistic runs; ridge λ fixed at 1e-2, not tuned.

**VERDICT (arrived after pre-registration): MIGRATED.** Logistic argmin
ρ=0.95 (ridge: 0.6); the logistic advantage GROWS with ρ (val delta
+0.03 at ρ=0.4 → +0.40 at ρ=1.1). Test-split headline: logistic
ρ=0.95 → **2.448 bpc** vs ridge-champion ρ=0.6 → 2.755, both at 2M
chars. Slope statistic: penalty at ρ=1.1 above own minimum — ridge
0.112, logistic 0.047, **ratio 0.42** (reading scale: ~1 = tax lives in
the state; ~0 = it lived in the readout; between = the pre-named
middle). So: the *location* migrated fully — deep memory was usable all
along, ridge couldn't afford it — while a ~40% residue of the high-ρ
*cost* survives the better readout. Finding 3 as revised: the
superposition tax is real but is paid at the readout interface roughly
as much as in the state; a direction-native readout renegotiates the
memory-sharpness trade substantially, and an attention-style readout
(P2/P3 territory) is the natural next escalation.
(`results/track_p/p1_acid_test.png`)

**Coronation (full budget):** logistic at ρ=0.95, 5M chars → val 2.338,
**test 2.398** — a statistical dead heat with the 4-gram (2.392), which
holds the fence by 0.006 bits on the honest split, against ~4× the
trainable parameters. Ridge at the same knobs and budget: 2.755 test.
The readout upgrade alone was worth 0.36 bits.
(`results/track_p/p1_acid_test_full5M.json`)

## Finding 4 — The semiring doesn't matter (Track T)

A **tropical (max-plus) reservoir** — states are soft-Viterbi path
scores; the only operations are max and + — matches the tanh ESN at
matched size, data, and readout: **3.0999 vs 3.0996 val bpc** (N=1k, 1M
chars). It has its own interior stability optimum (max cycle mean
λ≈−0.1) and its own fading-memory behavior, verified: max-plus
trajectories don't converge, they **collide exactly in finite time**
(`results/track_t/tropical_esp.png`) — textbook max-plus coupling
behavior, i.e., the implementation reproduces known theory where theory
exists. As far as we know nobody has run this animal on text8 before.
**Seed replication (addendum):** 5 seeds/architecture at N=1k, 1M chars:
tanh 3.108 ± 0.010, tropical 3.103 ± 0.007 (means ± sd). The difference
is one standard error — a statistical dead heat, with tropical nominally
ahead. Fairness re-tune: the tanh optimum at N=1k sits at ρ≈0.5
(3.094, seed 0), inside seed noise of everything else. Note the
consistency with Finding 3: the smaller pond prefers even shallower
memory (ρ 0.6→0.5 going 5k→1k), predicting that optimal ρ migrates
RIGHT as N grows — a falsifiable target for the E2-T2 N-sweep.

## Finding 5 — Substrate-independence of the performance-governing quantity (T3/T4)

The tropical champion's states, run through Track U's instruments, show
the SAME representation geometry as the tanh pond's: faithfulness +0.454
vs +0.448, cophenetic tree-likeness 0.970 vs 0.985, same last-char
top-level partition (`results/track_t/t3_geometry.png`). It's the
geometry the task wanted.

**The strong form (referee-proofed).** The deflationary reading — any
fading-memory system over a discrete stream embeds a suffix tree, so
tree-shaped geometry is just what forgetting looks like — is true and is
not the claim. The claim: both algebras' *tuned optima co-locate at
maximal tree-faithfulness*, and faithfulness (not architecture) is what
predicts performance across all 41 measured configurations. The task
pulled two incompatible arithmetics to the same geometric spot.
Convergence isn't the news; substrate-independence of the
performance-governing quantity is.

**Same tree, different masonry (T4).** tanh stores the past in
superposition — graded, everything attenuated but present; max-plus
stores it columnarly — a character survives in a coordinate only while
it is still the max there, then is erased outright. Both predictions
from that mechanism held (`results/track_t/t4_masonry.png`):
(a) *boring*: 293/2000 tropical states are exact duplicates (27% of
pairwise distances tied, tanh: 0.0%) — exact collisions mechanically
depress tropical's cophenetic statistic, so part of the 0.970-vs-0.985
gap is artifact; (b) *fun*: tropical's digit decay is all-or-nothing — a
one-lag cliff (0.94 → 0.39) where tanh fades over three lags. Corollary
of (a): at these knobs the max-plus pond has effectively quantized into
a finite-state machine over recent suffixes, while tanh realizes the
same tree as an infinite-precision fractal (a contracting reservoir is
an iterated function system over symbols — "Markovian architectural
bias" in the tanh literature; the tropical pond is a max-plus IFS,
apparently new). Also note: tropical remembers ~2 fewer characters than
tanh yet reaches the same bpc — Finding 3's tax, seen from the other
side.

## Finding 6 — The pond that became a lookup table (T5)

Finite coupling time + finite alphabet means the tropical state isn't
*approximately* determined by recent suffixes — it's *exactly*
determined. Measured (20k samples, honesty guard: determinism counted
only on suffix classes with ≥2 samples): determinism rises 0.35 (k=4) →
0.84 (k=5) → 0.997 (k=6) → **1.000 at k=7** — the tropical champion is,
within sample, an order-≤7 finite-state machine, `state =
table[suffix]`; the zero-compute pond. The tanh control under the same
analysis: 20,000 distinct states in 20,000 samples, determinism 0.000
at every depth — the smooth pond never merges two histories; same tree,
carved discrete vs. continuous.

And the sharper half: **370 merged classes at k=7** — distinct states
run FEWER than distinct suffixes. The dynamics merged suffix classes on
its own, a step toward a minimal automaton found by physics rather than
by algorithm. The merges are linguistically right: `' s the '` ∪
`'as the '` ∪ `'by the '` (after " the ", the prefix is irrelevant);
`'cation '` ∪ `'dition '` ∪ `'ration '` (derivational endings sharing
their tail). A randomly wired max-plus graph, tuned only for bpc,
rediscovered context-tree pruning. (`results/track_t/t5_automaton.json`;
191 merged groups, largest spans 26 suffixes.)

## Methodological coda

The journal doubled as a **pre-registration instrument**: predictions
were timestamped by position on the page ahead of their results (the
P1 thresholds and slope scale, T4's two masonry predictions, T3b's
design correction, the N-sweep ρ-migration prediction still open).
This practice, more than any single finding, is what makes the writeup
trustworthy without running one extra experiment.

## Infrastructure that made it honest

- `run_ngram_baseline.py` — the n-gram ladder on our exact splits.
- `dump_states.py` (I1) — sampled (state, suffix) pairs with a
  **bit-for-bit alignment gate** (refuses to write on any pairing shift;
  post-hoc validation passed 40/40). Credit: chat-Fable's paranoia,
  upgraded from a 99% threshold to exact equality.
- Config-suffixed result filenames — the "never overwrite" convention is
  now enforced by construction.
- Resumable sweep driver; per-cell JSONs cached and reused.

## Experiment 01 — PRE-REGISTERED predictions (written before any probe ran)

Setup: Llama-3.2-1B-Instruct-4bit (16 layers, D=2048), random sequences
over a 64-word single-token subset, B=32 × T=512, washout 64; ridge
probes of token identity at lag k per layer; ESN controls at N=D and
N=4D fed the model's own (dequantized) embeddings, runner defaults
(ρ=0.95, leak 0.5).

- **E1-PR1 (depth):** token-recall accuracy vs. layer is an inverted U;
  the best layer lies in the middle third of the stack (layers 5–10 of
  16). Early layers under-mix; late layers compress toward next-token
  features.
- **E1-PR2 (memory shape):** at the best transformer layer, recall at
  lag 32 exceeds 80% (attention retrieves; roughly lag-flat), while the
  matched-D ESN control is below 10% at lag 32 (fading memory; chance
  = 1/64 ≈ 1.6%). This is Finding 3's mechanism claim — retrieval
  without storage-crowding — stated as a measurable gap.
- **E1-PR3 (projectivity, P2):** after normalizing states to unit L2
  norm, define δ = mean over lags 1..32 of (acc_raw − acc_normalized)
  at each system's best layer/config. Predict δ_transformer < 0.05
  (transformer memory lives in directions) and δ_ESN > 0.15 (reservoir
  memory partly lives in amplitude).

## Finding 7 — The residual stream is a working set, not a tape (E1)

Verdicts on the pre-registered predictions, from the λ-selected probe
run (three-way split; λ chosen per lag on validation, accuracy reported
on test; a first run with fixed λ=1e-2 was diagnosed as
under-regularized — lag-0 decode 1.000 everywhere ruled out alignment
bugs, and a λ sweep + standardization moved the ESN-4D control from
0.05 to 0.41 at lag 8 while moving the transformer nowhere):

- **E1-PR1 (inverted-U over depth): FALSIFIED.** Lag-2 recall declines
  monotonically with depth — 0.89 (L1) → 0.56 (L15). No mid-stack
  sweet spot; the shallowest layer carries the most positional trace.
- **E1-PR2 (transformer ≥80% at lag 32; ESN <10%): FALSIFIED,
  INVERTED.** The frozen Llama's per-position state holds a ~3-token
  window: lag 1 ≈ 1.00, lag 2 = 0.56–0.89, lag 3 ≈ 0.2–0.3, lag 8 ≈
  0.03, lag 32 = chance (1/64), at every layer. The ESN controls carry
  an order of magnitude more: ESN-4D = 0.94 (lag 3), 0.40 (lag 8),
  0.16 (lag 16). The reservoir out-remembers the transformer state at
  every lag ≥ 2.
- **E1-PR3 (projectivity): HALF-FALSIFIED.** δ_transformer ≈ 0.01 ✓ —
  but δ_ESN ≈ 0.01 as well ✗. Unit-normalization is nearly free for
  BOTH systems: each stores its memory in directions, not amplitude.
  The ablation failed to discriminate; "features are directions" is
  the shared regime.

Interpretation: the transformer does not store lag-indexed history in
its running state at all beyond ~2–3 tokens. Storage lives at the
source positions (the KV cache), retrieved by attention on demand —
the architecture *separates* storage from state, which is precisely
the mechanism that dodges Finding 3's superposition tax. SPEC
hypotheses H1/H2 (residual stream as a long-memory reservoir) are
dead; the honest scope is "as measured by position-independent linear
probes on random token sequences," which is exactly what
reservoir-likeness means. Figure:
`results/exp01/probe_recall_figures.png` — a working set vs. a tape.

## E1-T5 (k-parity) — PRE-REGISTERED prediction (written before the run)

Setup: binary sequences over 2 single-token words, B=32 × T=512,
washout 64; parity of the last k bits (k = 2..8), ridge probe with
per-k λ selection on a validation slice, sign readout, test-split
accuracy; chance = 0.5. Same layers and ESN controls as Finding 7.

- **E1-PR4 (aggressive form of SPEC H3):** the ESN does not merely
  close the parity gap — it wins outright. Transformer layers exceed
  0.55 only for k ≤ 3 (the working-set window); for k ≥ 4 every layer
  sits within noise of chance. ESN N=8192 holds ≥ 0.9 through k = 6.
  Rationale: parity of the last k bits requires simultaneously held,
  nonlinearly mixed lag information — precisely what a fading tape has
  and a retrieval architecture's *state* does not.

**E1-PR4 VERDICT (parity): ordinal claim CONFIRMED, both quantitative
thresholds missed.** ESN-4D beats the best transformer layer at every
k ≥ 4 (1.00/0.97 at k=4, 0.96/0.72 at k=5, 0.78/0.55 at k=6, 0.58/0.51
at k=7) — SPEC H3 exceeded: the reservoir doesn't close the nonlinear
gap, it owns it. But the transformer's window was wider than
pre-registered (above 0.55 through k=5; the binary alphabet is a
different attention regime than 64-token recall — small vocab, heavy
repetition), and ESN-4D fell short of the ≥0.9-through-k=6 claim
(0.78). Depth trend replicates a third time: parity accuracy declines
monotonically with layer (k=4: 0.97 at L3 → 0.74 at L15).
`results/exp01/parity_Llama-3.2-1B-Instruct-4bit_B32_T512_seed0.json`

## T6 (merge semantics) — PRE-REGISTERED design and thresholds
(written before any distribution was computed)

Question (chat-Fable): are the tropical automaton's merged suffix
classes "smart" (predictive twins — approximating the task's causal
states, ε-machine-style) or "dumb" (arbitrary collisions)?

Design. Universe: multi-sample, determined k=7 suffix classes from the
tropical champion dump. MERGED pairs: classes sharing a state. NULL
pairs: classes NOT sharing a state, matched to the merged pairs on
(a) shared trailing-substring length — the critical confound, since
next-char distributions are dominated by recent characters and merged
pairs share long tails — and (b) context frequency bin (log2 of the
rarer context's corpus count). Per pair: Jensen–Shannon divergence
between corpus next-char distributions conditioned on the full 7-char
context (train slice, 5M chars; add-α smoothing α=0.1; both contexts
must have ≥20 occurrences or the pair is dropped).

Thresholds (R = median JSD_null / median JSD_merged, Mann–Whitney
p < 0.01 required for either positive verdict):
- R ≥ 2.0 → merges are predictive twins; "partial ε-machine found by
  physics" is claimable.
- R ≤ 1.25 → merges are arbitrary collisions; quantify their bpc cost
  instead.
- Between → partial; report the number without the lineage claim.

**T6 VERDICT: arbitrary_collisions (R = 1.17; MWU p = 6.5e-05).**
Median JSD 0.028 bits (merged) vs 0.033 (tail-matched null) — both
tiny. The merge criterion is structural, not distributional: sharing a
long tail already makes contexts predictive near-twins, and the
dynamics adds only a marginal (~17%, statistically real) refinement
beyond that. The merges are FREE, not clairvoyant: their bpc cost is
negligible, but "physics approximates causal states" is NOT claimable.
Note the counterfactual: without the pre-registered tail-matched null,
uncontrolled random pairs (JSD ~ tenths of bits) would have produced
R ≈ 10 and the night's biggest overclaim. Limitation: only 249 null
pairs matched (thin tail/frequency cells).
`results/track_t/t6_merge_semantics.json`

**k*(λ) sweep — PRE-REGISTERED form (before running):** determinism
depth k*(λ) (smallest k with determinism ≥ 0.99) scales like 1/|λ|,
i.e. k*·|λ| ≈ constant across λ; measured point so far: k*=7 at
λ=−0.1.

**T7 VERDICT: form FALSIFIED (ordinal survives, again).** k*·|λ|
declines monotonically (1.2 → 0.45 across λ = −0.4 … −0.05); the data
instead fit k* ≈ 2·log₂(1/|λ|) (measured 3/4/4/6/9 vs 2.6/3.5/4.6/
6.6/8.6). Post-hoc rationale (flagged as such): path count grows
exponentially with depth, so extreme-value competition among paths
shaves 1/|λ| to a log law. Bonus observation: distinct states swell
2,684 → 19,206 (of 20k samples) as λ → 0⁻ — the automaton melts
continuously into the fractal at criticality. λ=−0.02 never reaches
determinism by k=12. `results/track_t/t7_kstar_sweep.json`

## U4/U5 (the melting) — PRE-REGISTERED form (before running)

The tanh pond at leak 1 is an iterated function system: each symbol c
applies the contraction f_c(x) = tanh(Wx + Win e_c + b), and the state
after a history is the composition along the suffix. IFS theory (Moran)
predicts the attractor's fractal dimension for V equally-branching maps
of contraction ratio ρ:

    D ≈ ln V / ln(1/ρ)        (V = 27)

- **U5-PR (form + ordinal, exact numbers held loosely per this lab's
  track record):** correlation dimension D₂ of sampled states is
  strictly increasing in ρ, and for ρ ≤ 0.6 tracks ln27/ln(1/ρ)
  (predicting ≈ 2.0, 2.7, 3.6, 4.8, 6.5 at ρ = 0.2…0.6) within
  estimator error; above ρ ≈ 0.6 the estimate saturates at the
  finite-sample ceiling (~6–7 for 2k points), which is a measurement
  limit, not physics.
- **Melting picture:** the tropical pond's distinct-state count and the
  tanh pond's fractal dimension are the same divergence seen from the
  discrete and continuous sides: both blow up approaching criticality
  (λ → 0⁻, ρ → 1). Discreteness is distance from the edge.

**U5-PR VERDICT: FALSIFIED — including the ordinal (first full miss of
the night).** Measured D₂ DECREASES with ρ: 97.5 / 59.1 / 28.2 / 16.0 /
10.9 / 8.0 / 7.9 at ρ = 0.2…0.95. Diagnosis (estimator artifact, not
physics): the attractor is near-ultrametric (U2, cophenetic 0.99) —
its scales are QUANTIZED, so C(r) is a staircase (plateaus between
cluster levels, concentration-sharpened cliffs at them), and a
fixed-quantile slope window lands on a cliff; smaller ρ → wider stairs
→ sharper cliffs → absurd slopes. D₂ = 97 in 2k points is the tell.

**AMENDED PRE-REGISTRATION (before the staircase analysis):** log-log
C(r) shows discrete stairs; horizontal spacing per stair ≈ ln(1/ρ),
vertical drop ≈ ln 27, so stair-geometry slope ≈ Moran's
ln27/ln(1/ρ). A wide window spanning ≥2 levels approaches Moran for
ρ ≤ 0.4; stairs blur progressively as ρ → 1 (the smooth-side
melting: the staircase, not the set, is what melts).

**U5b VERDICT: staircase CONFIRMED structurally, naive Moran numbers
missed low (slopes 0.62 / 0.89 / 1.55 / 3.11 vs 2.05 / 3.60 / 6.45 /
64 at ρ = 0.2/0.4/0.6/0.95).** The stairs exist exactly as predicted
and melt smooth by ρ=0.95 (`results/track_u/u5b_staircase.png` — one
stair = one suffix-tree level). Two identified corrections, both
pushing the slope below naive Moran: (1) the per-level contraction is
the trajectory-Jacobian's effective factor, stronger than ρ itself
(tanh' < 1), widening the treads; (2) the branching measure is English
text, not uniform — the vertical drop per stair measures the corpus's
Rényi-2 collision entropy (~3 bits/char), not log₂27 = 4.75. The true
law to test next session: slope = H₂(corpus) / log(1/c_eff(ρ)), with
both factors measured independently (level-spacing regression).
Caveat: stairs below r/r_max ≈ 2⁻²⁰ are float32 quantization, not
structure. Meta: quantitative miss #6; structural/ordinal claims hold
after amendment.

## OVERNIGHT RUNS (launched while Caity sleeps) — PRE-REGISTERED

**N-sweep under the logistic readout** (ρ ∈ {0.6, 0.8, 0.95, 1.1} at
N ∈ {1k, 2k, 10k, 20k}, 2M chars; N=50k at ρ ∈ {0.8, 0.95, 1.1}, 1M
chars, reduced eval — memory ceiling):
- **PR (ordinal, the standing ρ-drift bet):** the logistic argmin over
  ρ is non-decreasing in N, with argmin(1k) ≤ argmin(5k)=0.95 ≤
  argmin(20k), at least one inequality strict across the sweep.
  Rationale: bigger ponds can afford deeper memory (Finding 3's tax is
  capacity-relative).
- **PR (frontier):** val bpc at fixed ρ=0.95 improves monotonically in
  N with visibly diminishing returns per doubling.

**Uniform-drive staircase control** (champion pond driven by i.i.d.
uniform symbols instead of text8):
- **PR:** the correlation-staircase vertical drops steepen — the
  wide-window slope at matched ρ EXCEEDS the text8-driven slope by a
  factor ≈ log₂27 / H₂(text8) (≈ 1.5–1.9×), because stair heights
  measure the drive's collision entropy, not the alphabet size.
  Unigram Rényi-2 of text8 computed alongside as the reference.

**MORNING VERDICTS (runs completed while Caity slept):**

- **ρ-drift bet: CONFIRMED** (the ordinal streak continues). Logistic
  argmin over ρ by N: 0.8 (1k) → 0.8 (2k) → 0.8 (10k, tied with 0.95
  within ~0.005) → **1.1 (20k)**, with 5k's acid-test 0.95 in between.
  Both registered inequalities hold, both strict. At N=20k the optimum
  is SUPERCRITICAL: with a rich readout and enough neurons, the best
  language reservoir sits past the edge. Finding 1's final form is now
  measured across a factor of 20 in N.
- **Frontier: diminishing returns confirmed, and sharper — data
  saturation.** Best val bpc: 2.642 (1k) → 2.525 (2k) → 2.347 (10k) →
  2.348 (20k) at 2M chars: 10k→20k gains nothing. 2M training chars is
  the binding constraint at N ≥ 10k. Follow-up queued: N=20k at 4M
  chars. PR: val bpc < 2.30 if data was the ceiling; ≈ 2.35 if the
  reservoir itself has saturated.
- **N=50k FAILED — SGD divergence, not physics** (val bpc 5.5–9.7,
  worse than uniform: optimizer blow-up; lr=3e-3 untuned for 50k-dim
  features). Corrected run queued at lr=3e-4. PR: with the lower lr,
  N=50k lands within [2.25, 2.45] on 1M chars.
- **U5c (uniform control): direction SPECTACULARLY confirmed,
  magnitude miss #7.** Uniform-drive stair drops measure 4.7–4.8 bits
  — log₂27 to within reading error — vs text8's ~2–3.5; slope ratios
  2.19/2.28 (ρ=0.2/0.6), ABOVE the registered 1.5–1.9 band because the
  operative text8 quantity is the CONDITIONAL Rényi-2 rate (implied
  ≈ 2.1 bits/char), lower than the unigram 3.741 used for the band.
  The staircase is a ruler: treads measure the dynamics, drops measure
  the drive. `results/track_u/u5c_uniform_control.png`

**MORNING CHAIN 2 VERDICTS:** N=50k at lr=3e-4: divergence cured; val
2.363 / 2.268 / **2.2306** at ρ = 0.8 / 0.95 / 1.1 — new overall
champion, on 1M chars, landing just BELOW the registered [2.25, 2.45]
band (miss #8, favorable). N=20k @ 4M chars: **2.2986** — under the
pre-registered 2.30 threshold: the flattening was data ceiling, not
reservoir saturation. Argmin ρ = 1.1 at both 20k and 50k: firmly
supercritical and still drifting right.

**MORNING CHAIN 3 — PRE-REGISTERED (before launch):**
- N=50k, ρ=1.25, 1M chars: PREDICT interior optimum — 1.25 comes back
  WORSE than 1.1 (the drift has a ceiling; supercritical mixing
  eventually destroys usable structure even for a rich readout).
- N=50k, ρ=1.1, 2M chars: PREDICT val ≤ 2.18 (data scaling continues;
  the 5-gram fence at 2.06 val comes into sight but does not fall
  tonight).

**CHAIN 3 VERDICTS: 2/2.** ρ=1.25 @ 50k: 2.2636 > 2.2306 — interior
optimum CONFIRMED at ρ≈1.1; the drift has a ceiling. 50k @ 2M: val
2.1742 (registered ≤ 2.18 ✓), test 2.2285 — 5-gram fence (2.095 test)
in sight, not fallen.

## U6 (linear-pond null) — PRE-REGISTERED (before running)

Identity-activation pond (activation an option in esn.py), N=1k,
ρ=0.6, leak=1 — same knobs as the t3 tanh control — through the U1/U2
instruments. **PREDICT the deflationary outcome:** the linear pond
grows the same tree (faithfulness within ±0.05 of tanh's +0.448;
cophenetic ≥ 0.95). Recency geometry is linear-algebraic; tanh is not
required for tree-ness. Either way Finding 5's strong form survives —
it rests on co-location of tuned optima across algebras, not on
tree-ness, and this null turns "some of it is generic" into a
measured baseline.

**U6 VERDICT: deflationary outcome confirmed, with a residual.**
Linear-pond faithfulness 0.4484 — identical to tanh's 0.448 to three
decimals (prediction dead-center): the suffix-correlation structure is
fully linear-algebraic. Cophenetic 0.9459, missing the registered
≥0.95 bar by 0.004 (near-miss #9): nonlinearity contributes a real,
small tree-CRISPNESS gradient — 0.946 (linear) → 0.970 (tropical) →
0.985 (tanh) — saturation sharpens soft recency correlation into
cleanly nested clusters. Finding 5's strong form unaffected.
`results/track_u/u6_linear_null.json`; esn.py gained an `activation`
option ("tanh" | "linear").

## Params-matched duel (chat-Fable's weaponized honesty flag) — PRE-REGISTERED

N=20k readout = (20k+1)×27 ≈ 540k trainable params — the 4-gram's own
weight class (531k). Protocol: ρ ∈ {0.95, 1.1} at 5M chars, config
chosen on val, its TEST quoted. Note for the record: at 4M chars the
val-chosen config (ρ=1.1) tests at 2.3907 vs the 4-gram's 2.392 — a
dead heat; ρ=0.95's cleaner 2.359 was not val-chosen and is not
quotable. **PR: at 5M chars the val-chosen config's test bpc < 2.37**
(a ≥0.02 margin — matched parameters, no feature training, same data
budget, better bpc).

*Procedural note:* first attempt VOID — Adam at lr=3e-3 diverged at
epoch 3 in both cells (val 2.42→5.21 bpc in one epoch; early stopping
salvaged epoch-2 weights, yielding meaningless 2.39/2.44). Same
instability class as the N=50k blow-up; stochastic at N=20k (the 4M
run survived on batch-order luck). Rerun at lr=1e-3; the
pre-registration is unchanged — this is optimizer hygiene, diagnosed
from the epoch traces before any verdict was read.

**DUEL VERDICT (stable rerun, lr=1e-3): WON, decisively.** Val-chosen
ρ=0.95 → val 2.1576, **test 2.2253** vs the 4-gram's 2.392 — the
registered 2.37 line cleared by 0.145. The earned sentence: matched
parameters (540k vs 531k), same 5M chars, no feature training, better
bpc by 0.17. This is also the NEW OVERALL CHAMPION (beats 50k@2M's
2.1742 val / 2.2285 test). Honesty riders: (1) the earlier "dead
heats" at 20k were optimizer-limited, and ALL lr=3e-3 large-N results
are lower bounds — the N-sweep's bpc column understates big ponds;
(2) with stable optimization the 20k argmin is 0.95–1.1 (val gap
0.001, a tie): "firmly supercritical at 20k" softens to "at or above
0.95"; the ρ-drift bet's inequalities still hold. The 50k row (lr
3e-4) stands.

## Finding 8 — The Llama grows no tree: a future manifold (which-tree run)

Verdicts against the brief's pre-registered thresholds:

- **PR-A (future beats past at mid layers): CONFIRMED at L12
  (diff +0.317, CI [+0.298, +0.335]), marginal at L8 (+0.095, CI
  [+0.074, +0.114] — excludes zero, straddles the 0.10 line). The
  layer-1 reversal: FALSIFIED — the future wins at L1 TOO (+0.166 vs
  +0.059, CI [+0.086, +0.128]).** The transformer is future-keyed at
  every depth, embedding-adjacent layers included (distributional
  embeddings are already predictive objects).
- **PR-B (monotone trends): FALSIFIED as stated.** raw_s is flat and
  tiny (~0.08) at all layers — there is no past tree to shed. raw_pext
  is non-monotone: 0.17 (L1) → dip 0.09 (L4) → peak 0.37 (L12) → 0.32
  (L14), with the mechanical ceiling at 0.40 (L15).
- **PR-C (tree gates): OUTCOME (d).** Cophenetic at L12 = 0.684 vs
  coordinate-shuffled null 0.687 — the state cloud has NO hierarchical
  structure beyond chance. Not a crisp tree, not a loose tree: no tree.
- **PR-D:** deep-suffix share 0.29–0.38, but raw_s ≈ 0.08 makes the
  question moot — there is no meaningful past tree at any depth.
- **PR-E (pond control): CONFIRMED.** Pond partial_s +0.442,
  partial_p +0.062 — the past corner, as registered.

**The finding (wording per chat-Fable, adopted):** Fading memory files
states by the past — the suffix tree is that filing system, and it is
substrate-independent because forgetting forces it. Prediction
training files states by the future, at every depth we measured,
embedding included, and its filing system is not a tree: **the recent
past stays readable but never organizes the geometry — the working
set is carried, not filed.** (Probes measure what is extractable;
faithfulness measures what organizes the metric.) F5 scopes to
fading-memory systems; the transformer isn't an exception to the law
— it is outside its jurisdiction. "Manifold" is HELD pending a
dimension measurement (registered below); until then: predictive
geometry. The pond–Llama figure:
`results/exp01/which_tree_plane.png`. Caveats: one model (1B, 4-bit —
quantization check registered below), natural-text drive,
|Δt| ≤ 64 position matching, rank-OLS partials, d_p^ext at 5-gram
order (L8's 0.095 stands as "cleared zero, missed meaningful" under
the registered ruler; any higher-order d_p is a new instrument with
its own registration).

**Follow-ups — PRE-REGISTERED (before running):**
- **F8-a (triptych):** add partial faithfulness to CURRENT-token
  identity per layer (controlling d_s, d_pext, dpos). ORDINAL: it
  peaks at the d_p dip (L4) — the arc is past-less → present-busy →
  future-keyed, and the dip is the state being about *now*.
- **F8-b (earn "manifold"):** correlation-dimension estimate of
  Llama-L12 states (cosine, wide window; also inspect log C(r) for
  smoothness vs. stairs). ORDINAL: intrinsic dimension well above the
  pond's Moran-scale values, and the curve is smooth (no quantized
  scales) — or interestingly not.
- **F8-c (quantization exoneration):** recompute L12 cophenetic on the
  SAME positions with a bf16 (unquantized) model. Registered gates:
  apply PR-C unchanged — if bf16 cophenetic ≥ 0.90 and ≥ null + 0.15,
  quantization is indicted and "no tree" is withdrawn pending; if it
  stays < 0.75, exonerated.

**F8 FOLLOW-UP VERDICTS:**
- **F8-a: falsified as stated, better story found.** partial_now does
  NOT peak at L4 — it is flat-highest early (L1 +0.116, L4 +0.115,
  statistically tied) and declines monotonically (L8 +0.108, L12
  +0.062, L14 +0.045). The revised arc: **the state trades NOW for
  NEXT as depth increases** — present-signal decays while
  future-signal grows. The L4 future-dip stays unexplained (no
  present-bump there); open question for a second model.
- **F8-b: ordinal CONFIRMED — "manifold" provisionally earned.** L12
  correlation dimension 7.82 under the same wide-window estimator that
  gave the pond ~1.5, with a SMOOTH log C(r) (fit residual 0.205 bits;
  no staircase). Finite, smooth, un-quantized scales: predictive
  manifold is now a licensed noun, with estimator caveats.
- **F8-c: quantization EXONERATED.** bf16 L12 cophenetic 0.647 vs null
  0.657 — fails the tree gates identically to 4-bit. "No tree" stands
  on full-precision weights.

**CALIBRATION LEDGER (meta-finding, chat-Fable's sentence, logged
verbatim):** the ordinal record was 5-for-5 *inside pond physics* and
is 0-for-2 *across the substrate boundary* — the L1 reversal and the
triptych bump both died from importing fading-memory intuitions into a
prediction-trained system. **Substrate-independence failed for our
priors before it failed for anything else.**

**Corrected Moran (chat-Fable, adopted):** the naive ln27/ln(1/ρ)
assumes all digits equally used and freely combined; the measure's
dimension is entropy-rate-over-forgetting, d ≈ h/ln(1/ρ) with h in
nats — **the pond's fractal dimension is the entropy rate of English
divided by the log of the forgetting rate; the geometry is carved by
the language, not the alphabet.** This retroactively explains the
pond's measured ~1.5 at ρ=0.6 (h ≈ 1.0–1.4 nats ⇒ d ≈ 2.0–2.8,
minus finite-sample low bias).

**Two PRE-REGISTRATIONS (before running, existing data):**
- **Triptych, common conditioning:** recompute all three partials per
  layer with the union control set (each metric | the other two +
  dpos). The "NOW was never ahead, not even at the embedding" claim is
  licensed only if common-conditioned NOW < NEXT at L1.
- **The h-slope bet:** wide-window D across dump cells ρ ∈ {0.2, 0.3,
  0.4, 0.5, 0.6} is linear through the origin in 1/ln(1/ρ). ORDINAL:
  D rises with ρ (we win). QUANTITATIVE: the slope lands in
  h ∈ [1.0, 1.4] nats/char (history says we lose, but this formula
  has the right physics in it — the most informative quantitative bet
  on the books). ρ=0.95 reported as ceiling row only.

**H-SLOPE VERDICT: ordinal WIN, quantitative out-of-band with a
mechanism.** Origin slope h = 0.833 nats (band [1.0, 1.4]); implied h
per cell declines smoothly as D grows (0.998 → 0.767 across ρ =
0.2 → 0.6) — the signature of the estimator's dimension-dependent low
bias — and the least-biased cell (ρ=0.2) reads h = 0.998 nats, on the
band's floor and in Shannon's neighborhood. The corrected-Moran
physics stands; the estimator owes the difference.
`results/track_u/u5d_entropy_slope.json`

**COMMON-CONDITIONED TRIPTYCH VERDICT:** with one conditioning set
(each metric | other two + dpos): PAST collapses to ~0.03 at every
layer (its residual was mostly the current token — the luggage was
lighter than we thought). At L1: FUTURE +0.152 > NOW +0.116 — **the
future leads from the very first layer; licensed.** At L4: NOW +0.115
> FUTURE +0.068 — the ONLY layer where the present outranks the
future, which resolves the L4 dip and resurrects the construction-zone
story at exactly the layer where its unequal-conditioning version
died. The arc, final form: future-leaning at the embedding,
present-centered once (L4), then future-dominant with depth.
`results/exp01/triptych_common.json`

**Part II centerpiece framing (adopted):** a ~1.5-dimensional fractal
tree that files by the past vs. an ~8-dimensional smooth manifold that
files by the future, the present decaying into luggage between them.
**The pond remembers; the Llama anticipates.** Note also: both
cophenetic runs landed marginally BELOW their nulls (0.684/0.687,
0.647/0.657) — not weak hierarchy; zero hierarchy, twice, on solid
weights.

**Optimizer hygiene, codified (chat-Fable):** any run whose val
worsens by > 0.5 bpc in one epoch AUTO-VOIDS, verdicts unread;
gradient clipping (global norm 1.0) added to LogisticReadout alongside
the lr=1e-3 retry policy. The duel line stays at 2.37, test column,
no rounding mercy.

## Afternoon slate — PRE-REGISTERED (before running)

**Second model (Qwen2.5-1.5B-Instruct-4bit — different family, dense):**
- ORDINAL, core replication: future leads past at every probed layer
  under common conditioning; cophenetic tree gates fail (no tree).
- ORDINAL, construction zone: some early-mid layer has NOW > FUTURE
  (an L4-analog exists). If absent, the dip is Llama-idiosyncratic and
  the arc's middle beat is demoted to "sometimes."

**N=100k frontier probe (1M chars, lr=1.5e-4 with clipping):**
- ORDINAL: beats 50k@1M (val < 2.2306). QUANTITATIVE (history says
  no): val ≤ 2.15.

**T3b (concat ponds vs size-matched null; ridge, 1M chars):**
- PR-i (no mean gain): concat(tanh1k ⊕ trop1k) val bpc within ±0.01
  of the lone tanh-2k null.
- PR-ii (complementary failures): the parents' top-1 error sets on val
  have Jaccard < 0.8, and concat recovers > 50% of the positions where
  exactly one parent is correct.
- Slicing: accuracy by 4-char-context corpus-frequency decile, per
  system — T4's cliff-vs-fade masonry predicts tropical relatively
  stronger on frequent contexts, tanh on rare ones.

**SECOND-MODEL VERDICT (Qwen2.5-1.5B, 28 layers):** core replication
CONFIRMED — future leads past at every probed layer (L1 +0.107 vs
+0.064; L26 +0.378 vs +0.010; past declines monotonically), and the
tree gates fail again (cophenetic 0.860 vs null 0.872 — BELOW null, a
third time, second family). Finding 8 is now cross-family. The
construction zone: NOT clearly replicated — Qwen's only NOW ≥ FUTURE
moment is L1 by +0.005 (a tie, not a dip); no interior L4-analog. The
Llama L4 dip is demoted to "family-idiosyncratic pending more models";
the arc's universal spine is: past never organizes, future
strengthens with depth, present decays.
`results/exp01/which_tree_Qwen2.5-1.5B-Instruct-4bit.json`

**T3b VERDICT:** PR-i FALSIFIED narrowly in the interesting direction:
concat (2.9445 val) beats the size-matched tanh-2k null (2.9596) by
0.0151 — outside the ±0.01 "no mean gain" band. Algebraic diversity
IS worth something beyond size. PR-ii mixed: parents' error Jaccard
0.860 (registered < 0.8 — errors more shared than bet), but the 8.7%
disagreement zone exists and concat recovers 58.9% of it (> 50% ✓) —
which is exactly where the mean gain comes from. Slice bet
DIRECTIONALLY WRONG: tropical is relatively stronger on RARE contexts,
not frequent ones (masonry intuition inverted — the FSM's exact
suffix-classes apparently generalize rare contexts better than
superposed fading). `results/track_t/t8_concat_duel.json`

**Hash-table demo — PRE-REGISTERED (before running):** build
state = table[last-7-chars] from one dynamics pass over 1M train
chars; audit full-corpus determinism during the build (same suffix
reappearing must give a bit-identical state; register mismatches < 1%
of repeats). At val: coverage > 50% of positions; on covered
positions, table states bit-identical to dynamics states (≥ 99.9%)
and restricted bpc identical to 4 decimals. The zero-compute pond,
performed.

**T9 VERDICT: CLEAN SWEEP — all four registrations hit, quantitative
included (a first).** 302,597 suffix states from one pass; 684,539
repeats, ZERO mismatches (full-corpus exact determinism); val coverage
71.9%; lookup states bit-identical (frac 1.0, max diff 0.0); bpc on
hits 2.9065 = 2.9065. The zero-compute pond is performed: at
inference, 72% of the recurrent dynamics is a hash table.
`results/track_t/t9_hash_pond.json`

**Rare-context pooling hypothesis — REGISTERED for next (the T3b
surprise):** the tropical FSM's mid-rare-context advantage is
BACKOFF-BY-MERGING — exact shared-suffix classes pool statistics in
the readout like variable-order smoothing, which fuzzy superposed
states cannot. Test: re-slice the tanh-vs-tropical accuracy gap by
CLASS-POOLED frequency (sum of train counts across each FSM state's
merged suffixes) instead of raw context frequency. PREDICT (ordinal):
the tropical advantage collapses under pooled-frequency slicing.
If confirmed: "max-plus reservoirs implement backoff smoothing in
hardware" — connects Track T to classical n-gram smoothing theory.

*Operationalization (added before running):* restrict to val rows
whose FSM class was seen in train (report the excluded fraction).
(1) Reproduce the raw-count gap: mean tropical−tanh accuracy gap over
raw-frequency deciles 1–4. (2) The knife: split those same rare-raw
rows at their median POOLED class count; registered prediction — the
tropical advantage in the pooled-HIGH half exceeds the pooled-LOW
half by at least 2× (backoff mechanism), with the pooled-LOW half's
gap statistically ≈ 0 (|gap| < 2 se). If instead the gap is uniform
across pooled halves, the mechanism is not pooling and the
winner-take-all robustness story takes the floor.

**FLY-POND VERDICT (N=100k, 1M chars, lr=1.5e-4, clipped):** val
2.2025 / test 2.2793, no void. Ordinal CONFIRMED (beats 50k@1M's
2.2306); quantitative MISSED (registered ≤ 2.15). Neuron-scaling is
thinning (0.028 bits for the 50k→100k doubling); the 20k@5M duel
champion (2.1576 val / 2.2253 test) retains the crown. At current
scales, data beats neurons; the 5-gram fence (2.095 test) is a data
problem now. `results/track_p/p1_acid_test_N100000_probe.json`

**T10 VERDICT: NOT_POOLING — backoff hypothesis dead by its own
registered knife, inverted.** Rare-raw gap reproduces (+0.0116 ±
0.0027, 68k rows); the pooled-LOW half carries +0.0154 vs pooled-high
+0.0076 — the tropical advantage is LARGEST where the FSM class has
the LEAST training data. Opposite of strength-sharing. New floor
mechanism: **isolation protects the rare** — ridge on the smooth tanh
manifold interpolates rare contexts toward data-rich neighbors
(over-smoothing); the FSM's isolated, equal-loudness states resist
interpolation bias. Discreteness as a regularizer — the inverse of
the classical smoothing tradition.
`results/track_t/t10_pooling_test.json`

**Next knife — REGISTERED, not yet run (interpolation-signature
test):** on rare rows where exactly one pond errs, tanh's wrong
argmax should match the SHORTER-context (3-gram) modal answer — the
smoothed answer — more often than tropical's wrong argmax does.
ORDINAL: tanh's error-matches-trigram-mode rate exceeds tropical's by
≥ 1.5×. Confirms the interpolation-bias mechanism if it holds.

**T11 VERDICT: NOT_CONFIRMED at the registered bar; directionally
supported at ~9σ.** tanh's exclusive rare-context errors match the
trigram modal (smoothed) answer 43.8% vs tropical's 33.2% — the
predicted asymmetry, clearly real — but ratio 1.32 < the registered
1.5×. On shared errors the rates are identical (0.59 vs 0.58): the
signature lives only in the exclusive-error zone, as a real mechanism
should. Status of "isolation protects the rare": supported, not
confirmed; the effect is real and moderate.
`results/track_t/t11_interp_signature.json`

**5-gram fence assault — PRE-REGISTERED (before running):** N=20k,
ρ ∈ {0.95, 1.1} val-chosen, lr=1e-3, 20M training chars with stride-4
state sampling (4× textual diversity at constant 200GB RAM; decimation
of time-correlated rows should cost little). Anchors: the standing
5-gram@5M (test 2.095) AND a same-data 5-gram@20M (to be computed —
the primary, honest comparison). ORDINAL: beats the 20k@5M champion
(val < 2.1576). QUANTITATIVE (the fence bet, history laughing
already): test < the same-data 5-gram@20M's test bpc.

**FENCE VERDICT: ordinal CONFIRMED (new champion — val 2.0983 / test
2.1841, dethroning 2.1576/2.2253; stride-4 diversity worked),
quantitative LOST (same-data 5-gram@20M fence 2.015 stands by 0.17).**
Data keeps paying (~0.04 test per 4× stride-diversified); the road to
the fence needs another ~2 doublings of text (stride 8–16 at 40–80M
chars, RAM-constant) or a streaming readout. Scoreboard ordering, test
column: 5-gram@20M 2.015 < 5-gram@5M 2.095 < POND 2.184 < 4-gram@20M
2.357 — the pond has now beaten every 4-gram configuration and trails
the weakest 5-gram by 0.09. Next honest step recorded, no run queued.
`results/track_p/p1_acid_test_N20000_T20M_stride4.json`

**FENCE ASSAULT II — PRE-REGISTERED (before launch):** N=20k, 80M
train chars, stride 16 (5M rows, RAM-constant), lr=1e-3. Protocol
change, registered here: ρ fixed at 0.95 — chosen by the two standing
val decisions at this N (5M and 20M both chose 0.95), not by peeking;
single run, ~2.5–3h. Anchors: same-data 5-gram@80M = **test 1.984**
(the fence, taller again); 5-gram@20M (2.015) and @5M (2.095) as
history. ORDINAL: new champion (val < 2.0983). QUANTITATIVE (the
order: "take the fence"): test < 1.984. Extrapolation says ~2.14 and
the fence survives; the lab's quantitative record says trust nothing,
including that.

**T11 SEED REPLICATION — PRE-REGISTERED (before running):** seeds 1–3
added to seed 0 (new reservoir draws, both ponds; data fixed).
ORDINAL: the directional gap (tanh exclusive-error trigram-mode rate >
tropical's) holds in EVERY seed, 4/4. SEVERITY: mechanism upgrades
from "supported" to CONFIRMED only if the pooled ratio (all seeds,
counts summed) ≥ the original 1.5 bar; the pooled ratio and its
per-seed spread are reported either way.

**T11 SEED REPLICATION VERDICT: registered 4/4 FALSIFIED (3/4; seed 3
is a dead null at ratio 1.003), pooled ratio 1.197 — severity bar not
met.** The pooled effect is real (+0.067 gap, 12.9σ over 34k exclusive
errors) but heterogeneous across reservoir draws. Demotion with
precision: if the mechanism were purely algebraic every wiring would
show it — one in four doesn't — so the signature is WIRING-MEDIATED,
prime suspect the seed-dependent merge structure (each random graph
protects different rare classes). Status: weakly supported,
heterogeneous; next instrument must condition on each seed's own merge
map. Per-seed ratios 1.319 / 1.200 / 1.281 / 1.003.
`results/track_t/t11_interp_signature_seed{1,2,3}.json`

**FENCE ASSAULT II VERDICT: ordinal CONFIRMED by a nose (val 2.0938,
new champion by 0.0045; test 2.1734), the fence STANDS by 0.19.**
The real yield is the bent curve: 4× text at constant rows bought
0.041 test last time, 0.011 this time — stride scaling has saturated;
ROWS are the binding constraint now, not text. Ceiling under a
5M-row readout: ≈ 2.17 test. The road to 1.984 is a streaming readout
(re-drive reservoir per epoch, all 80M rows, ~17h) — mapped, not
launched. `results/track_p/p1_acid_test_N20000_T80M_stride16.json`

**FENCE ASSAULT III (the streaming campaign) — PRE-REGISTERED (before
launch):** streaming logistic readout — the reservoir is re-driven
over the full 80M characters EVERY epoch, one Adam step per 8,192-row
chunk as states are produced (S=256 segments × 32-step chunks = one
exact batch; nothing materialized). N=20k, ρ=0.95, leak=1, lr=1e-3,
clip 1.0, max 8 epochs, patience 2, auto-void rule armed. All ~80M
rows per epoch — 16× the rows any readout has seen. Budget: ~60–90
min/epoch, expect 5–10h wall-clock. ORDINAL: new champion (val <
2.0938). THE ORDER (quantitative): test < 1.984 — the fence. Honest
expectation from row-scaling: lands 2.07–2.12 and the fence survives;
the lab's quantitative record is 1-for-11 and that one was the hash
table, so the fence's insurers may price accordingly.

**ASSAULT III FINAL VERDICT: ordinal CONFIRMED (val 2.0374, −0.056
over the stride champion; test 2.1330, −0.040); THE FENCE STANDS by
0.149.** Eight epochs, ~19.5h, no voids; plateau reached at epoch 5
and refined at epoch 7. Honest riders: the val→test gap widened to
+0.096 (best-of-8 selection optimism on val); the expectation band
(2.07–2.12) missed high by 0.013. THE MEASURED RESULT: with neurons
(100k probe), text (80M), and now rows (all of them) each saturated in
turn, the linear/logistic-readout paradigm's ceiling on this corpus at
N=20k is ≈ 2.13 test bpc. The original 5-gram@5M (2.095) sits 0.038
away; the same-data fence (1.984) needs a different *kind* of readout
(attention-style, P3 territory) or feature learning — i.e., leaving
readout-only land, which is the paradigm's honest boundary, now
surveyed from inside. Campaign totals: three assaults, three
champions, fence intact, every constraint named and priced.
`results/track_p/streaming_N20000_T80M.json`

## Open threads (in rough order)

1. **I2 readout harness** (Track 0) — any features × any readout; then
   P1 logistic readout (predicted +0.1–0.25 bpc) closes the honest-bpc
   gap, and P2's cosine ablation asks "is transformer memory projective?"
2. **T2 at N=5k** — does the tropical tie survive scale?
3. **E2-T2 N-sweep** {1k, 20k, 50k} at champion knobs — the frontier
   plot. fp32-on-GPU Gram accumulation (validated against fp64 at N=5k)
   is the on-ramp to 100k neurons.
4. Experiment 01 proper — the frozen-Llama probes, now with sharper
   questions to ask.
