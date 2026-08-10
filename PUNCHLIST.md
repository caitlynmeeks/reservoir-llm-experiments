# PUNCHLIST — strange-math tracks for the reservoir LM

Drop this file at the repo root next to CLAUDE.md, and add one line there:
"Extended program lives in PUNCHLIST.md; work track by track." Each track
teaches one mathematical idea *by building something*, and each builds
tools the next track spends. Order: cheapest and most concrete first,
wildest last. Baseline to beat/explain throughout: 2.73 bpc
(N=5k, leak=1.0, rho=0.6) and the n-gram ladder (3g 2.92 / 4g 2.39 / 5g 2.09).

Conventions as in CLAUDE.md: every run takes --seed, results as JSON+PNG
under results/<track>/, never overwrite. Tracks are independent after
Track 0; within a track, do tasks in order.

---

## Track 0 — shared infrastructure (build once)

- [x] I1 `src/rcllm/dump_states.py` — DONE 2026-08-10. Alignment verified
      (state at t decodes char at t with acc 1.000, prev char at chance).
      40-cell dump to data/state_dumps/. Used by U, C, H.
- [~] I2 Readout harness: `src/rcllm/readouts.py` DONE 2026-08-10 —
      LogisticReadout (minibatch Adam, early stop on val NLL) and
      RidgeReadout (ridge + temperature) behind one fit/bpc interface
      over (T, N) feature rows from any source; state collection lives
      in experiments/track_p/p1_logistic.py:collect. REMAINING: promote
      collect() into the library + unify the results schema across run
      scripts. Subsumes E2-T2b.

## Track U — ultrametric / p-adic (the distance lesson)
*Greek to internalize: an ultrametric satisfies d(x,z) <= max(d(x,y),
d(y,z)) — every triangle is isosceles with a short base, and such spaces
are exactly trees. Suffix distance d = 2^-(shared suffix length) is one;
n-gram models are nearest-neighbor predictors under it.*

- [x] U1 — DONE 2026-08-10 over 40 cells (grid grew): Spearman(
      faithfulness, bpc) = -0.83 (p=3e-11) — ACCEPT met. Decomposition
      caveat: faithfulness explains the LEAK axis (rows cluster by
      faithfulness); within leak=1.0 faithfulness is flat (~0.45) while
      bpc spans 2.73-3.04, so the rho axis is NOT geometric
      unfaithfulness — it's the U3 depth/interference story. Residual
      documented in results/track_u/u1_faithfulness.json.
- [x] U2 — DONE: cophenetic correlation 0.990 at the champion cell; the
      dendrogram's top split IS the last character (contiguous labeled
      blocks), prev char fragments each block one level down.
      results/track_u/u2_dendrogram.png — frame it.
- [x] U3 — DONE 2026-08-10 (two passes; the L=8 suffix window ceilinged
      at rho>=0.6, redumped leak=1.0 row at L=20). Decode depth is
      MONOTONE in rho: 5.7 (rho .4) / 7.5 (champion .6) / 9.3 / 10.5 /
      11.7 / 12.7 (rho 1.25) chars — linear in -1/ln(rho) at low rho
      (numeral picture holds), sublinear cap at high rho (digit
      crosstalk). Punchline vs the heatmap: decodable memory RISES with
      rho while bpc WORSENS past ~0.55 — the past is present in the
      state but is interference for the single next-char readout.
      results/track_u/u3_digit_decay_L20.png

## Track P — projective & hyperbolic readouts (the direction lesson)
*Greek: after temperature calibration only the DIRECTION of the score
vector matters — the task is projective. RMSNorm'd transformers live on
spheres too ("features are directions"). Hyperbolic space = the
Cayley–Klein construction, distance from a log of a cross-ratio; trees
embed in it with low distortion. Reading: "Hyperbolic Neural Networks"
(Ganea et al.).*

- [x] P1 — DONE 2026-08-10, prediction EXCEEDED: logistic beats ridge by
      +0.22 at rho=0.6 and +0.39 at rho=0.95 (val, 2M chars; predicted
      0.1-0.25). ACID TEST VERDICT (pre-registered thresholds):
      MIGRATED — logistic argmin rho=0.95 vs ridge 0.6; slope ratio 0.42
      (40% of the high-rho tax survives the better readout). Test
      headline: 2.448 bpc @2M. results/track_p/p1_acid_test.png
      REMAINING from original item: rerun N-sweep best-N under logistic.
- [ ] P2 Cosine ablation in experiment 01: rerun layer x lag probes on
      length-normalized states, transformer vs. ESN. PREDICT: transformer
      probes survive normalization (its memory is stored in angles), ESN
      probes degrade (tanh amplitude is informative). A yes/no answer to
      "is transformer memory projective?"
- [ ] P3 Hyperbolic multinomial readout on frozen states (Poincare ball,
      Ganea-style MLR; geoopt or hand-rolled Mobius ops, torch-MPS).
      Same harness, matched params vs. P1. If U2 showed tree-like states,
      hyperbolic target space should fit them better than flat.

## Track S — path signatures (the canonical-pond lesson)
*Greek: the signature = iterated integrals of a path; theorem: linear
readouts on signatures approximate any nice sequence functional. It is
the mathematically canonical fixed feature map — the platonic reservoir.
Reading: "A Primer on the Signature Method in Machine Learning"
(Chevyrev & Kormilitzin).*

- [ ] S1 Truncated signatures of the one-hot character path: level 2
      (~756 features) and level 3 (~20.4k features), same ridge harness.
      TEACHABLE PREDICT: for symbol streams, level-k signature terms are
      weighted ordered-cooccurrence counters — n-grams in disguise — so
      level-3 should land near 3-gram bpc (~2.9). Verify or refute; either
      way you now understand both objects.
- [ ] S2 Randomized signatures: random projections standing in for levels
      4–5 at ~20k features, vs. ESN N=20k at ITS tuned knobs. The
      canonical pond vs. the random pond, matched budget, one plot.

## Track T — tropical reservoir (the semiring lesson)
*Greek: tropical algebra replaces (+, x) with (max, +). Viterbi/HMM
decoding is tropical matrix multiplication; shortest paths are tropical
linear algebra. The tropical spectral radius of W is its maximum mean
cycle weight — that's the knob playing rho's role. Reading: "Tropical
Geometry of Deep Neural Networks" (Zhang, Naitzat, Lim); Baccelli et al.
for the spectral theory.*

- [x] T1 `src/rcllm/tropical.py` — DONE 2026-08-10. Karp lambda control
      exact to 6 decimals; fading memory verified and then some: max-plus
      trajectories COLLIDE exactly in finite time (t=1/5/11 at
      lambda=-0.5/-0.1/0), even at lambda=0 under input drive. Figure:
      results/track_t/tropical_esp.png
- [~] T2 pilot DONE at N=1000, 1M chars (grid lambda x input_scale,
      results/track_t/tropical_N1000_T1000000_seed0.json): best
      lambda=-0.1, is=1.0 -> val 3.0999 — a DEAD HEAT with the matched
      tanh ESN control (val 3.0996, same N/data/readout). Beats 2-gram
      (3.45), not just the 1-gram sanity bar. Interior optimum in lambda.
      Tropical state pass ~3x slower (reduceat vs Accelerate spmv).
      Seed replication DONE (5 seeds each): tanh 3.108±0.010 vs tropical
      3.103±0.007 — dead heat confirmed, tropical nominally ahead. Tanh
      re-tune at N=1k: optimum drifts to rho~0.5 (3.094), within noise.
      REMAINING: N=5k grid per original spec; logistic readout via I2.
- [x] T3 (reshaped per chat-Fable): geometry convergence test — DONE
      2026-08-10. Tropical champion states through U1/U2 instruments:
      faithfulness +0.454 (tanh control +0.448), cophenetic 0.970 (tanh
      0.985), same last-char top partition. Two algebras, one tree —
      Findings 2+4 fused. results/track_t/t3_geometry.png
- [ ] T3b Original stretch: concatenate tropical + tanh states, one
      joint readout. Given T3's convergent geometry, PREDICT: near-zero
      gain (the ponds hold overlapping information). DESIGN (chat-Fable):
      the null MUST be a size-matched lone pond — concat 2x N=1k vs a
      single N=2k — else concat wins trivially on feature count. And
      slice errors by context (suffix frequency, required lag): given
      T4's cliff-vs-fade masonry, "no mean gain but complementary
      failures" is the interesting middle outcome.
- [x] T4 (wedge, chat-Fable): masonry test — DONE 2026-08-10. Tropical
      digit decay is a one-lag cliff (all-or-nothing) vs tanh's 3-lag
      fade; 293/2000 exact duplicate tropical states (27% tied distances,
      tanh 0%) explain part of the cophenetic gap mechanically and mean
      the tropical pond is effectively a finite-state machine over recent
      suffixes. results/track_t/t4_masonry.png
- [x] T5 (chat-Fable): automaton analysis — DONE 2026-08-10. Tropical
      champion is EXACTLY suffix-determined at k=7 (determinism 1.000 on
      multi-sample classes; tanh control 0.000 everywhere), with 370
      MERGED classes at k=7 (191 groups, largest 26 suffixes, e.g.
      "* the ", "*tion ") — partial minimal-automaton by physics;
      context-tree pruning rediscovered. results/track_t/t5_automaton.json

## Track C — coded reservoir (the interference lesson)
*Greek: superposed memories interfere like codewords in a noisy channel;
coding theory designs vector families with guaranteed separation
(minimum distance, low coherence). Related lineage: memory traces via
compressed sensing (Ganguli & Sompolinsky) — analysis exists, the
CONSTRUCTIVE version doesn't.*

- [ ] C1 Measure the enemy: at winner knobs, compute cross-correlations
      within the family {rho^j * Win e_c} (all chars c, lags j <= 6) for
      random Win. This coherence is the crosstalk noise floor. Histogram it.
- [ ] C2 Design Win to minimize it: start with Hadamard/Reed-Muller rows
      or a Grassmannian line packing over the 27 columns; same N, same
      knobs. Compare bpc AND the U3 digit-decay curve (coded should decode
      deeper). ACCEPT: coded >= random. A null result still teaches
      coherence and RIP; a positive one is a paper.

## Track H — sheaf on the de Bruijn graph (the gluing lesson)
*Greek: the de Bruijn graph's nodes are k-char contexts, edges are
one-char extensions; n-gram models are walks on it. A cellular sheaf
attaches local data (here: next-char score vectors) to nodes with
consistency maps on edges; the sheaf Laplacian's harmonic structure (H^1)
measures where local predictions cannot glue globally — irreducible
ambiguity, as cohomology. Reading: "Toward a Spectral Theory of Cellular
Sheaves" (Hansen & Ghrist). Most speculative track: the sheaf DESIGN is
itself a research task. Timebox it.*

- [ ] H1 Build the k=2 and k=3 de Bruijn graphs from text8 with empirical
      next-char distributions per node (this is just your n-gram tables,
      reshaped). Propose and implement a sheaf: stalks = score vectors,
      restriction maps = overlap consistency between adjacent contexts.
      Compute the sheaf Laplacian (sparse) and its harmonic sections.
- [ ] H2 Interpret: does gluing-defect energy track the bpc gap between
      k-gram and (k+1)-gram? If yes you have a cohomological reading of
      "how much prediction is k-locally impossible" — compare the
      intrinsic-ambiguity estimate to the 5-gram's 2.09.

---

## Why this order
U is an afternoon against data you already have and teaches the metric
mindset everything else uses. P is already promised, shares I2, and its
normalization idea literally reappears inside T1. S reuses the same
harness and demystifies "canonical features." T is the boldest new build.
C needs U3's decay picture to interpret. H is open-ended research —
last, timeboxed, and allowed to fail interestingly.

Rough effort: U ~1 day | P ~1–2 days | S ~1 day | T ~weekend |
C ~weekend | H ~timebox a week, expect surprises.
