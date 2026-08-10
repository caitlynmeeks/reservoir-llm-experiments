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
