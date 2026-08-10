# Writeup spine — the reservoir all-nighter

*(Authored by chat-Fable, 2026-08-10; transcribed verbatim. One editor's
note from the bench appended to Act V's pending slot — see ⚠ below.)*

Suggested repo home: `writeup/SPINE.md`. This is architecture, not prose:
each act has a job, two or three key sentences written out (keep, cut, or
rewrite them — they're gifts, not orders), its figures, and its claim
discipline. Blog version first; arXiv-note reorganization at the bottom.
One seat is held open in Act V for `p1_acid_test_full5M.json`.

## Title options
1. **Two Algebras, One Tree** — strongest single-finding title; subtitle
   carries the rest: *"a night of strange geometry with the laziest
   learner in machine learning."*
2. **The Pond Reads Wikipedia** — friendliest; matches the metaphor the
   whole piece runs on.
3. **Language and the Edge of Chaos: a reversal in one night** — leads
   with the twist; best if Act V's champion clears the 4-gram.
Recommendation: (1) for the blog, (3)'s content as the subtitle hook.

## The arc in one paragraph
A random, untrained dynamical system with a one-shot linear readout is
made to read Wikipedia. Tuned honestly, it contradicts thirty years of
folklore (no edge of chaos wanted). Probed with deliberately foreign
mathematics — ultrametrics, tropical algebra — it turns out to have
grown a suffix tree; a max-plus twin grows the *same* tree; the twin
then collapses, provably, into a finite automaton. A pre-registered
acid test with a better readout reverses the folklore verdict halfway —
the optimum leaps back toward the edge, and 42% of the memory tax
survives. The reversal is the point: the thresholds were written before
the data, so the night ends with the folklore half-rescued, the method
vindicated, and a pond that beat the trigram in 0.16 seconds of training.

## Act 0 — cold open
**Job:** hook + stakes in four sentences, no jargon.
**Key sentences:**
- "By four in the morning, one of our neural networks had quietly become
  a lookup table — and it read Wikipedia exactly as well as its smooth,
  infinite-state twin."
- "A reservoir computer is the laziest learner in machine learning: a
  big random dynamical system nobody trains, plus a single readout that
  learns in one stroke of algebra. Ours took 0.16 seconds."
**Figures:** none. **Hedge:** none needed — both sentences are literal.

## Act I — the ruler and the sweep (Finding 1)
**Job:** establish text8, bpc, and the n-gram ladder as the honest
ruler; then the 10×4 sweep and the folklore upset. Scoreboard table
lands here, early.
**Key sentences:**
- "Every model in this story is measured against the dumbest possible
  competitors: count-based n-grams, each one the exact cash value of k
  characters of history."
- "Tuned honestly, the pond rejected the folklore: the best language
  reservoir wasn't at the edge of chaos — it was a short, sharp puddle."
**Figures:** `sweep_rho_leak_N5000_T2000000_10x4.png`; scoreboard table.
**Claims:** interior optimum ρ≈0.5–0.6, leak 1.0, under a *ridge*
readout at N=5k — scope it now; Act V pays this sentence off.
**Foreshadow line:** "Hold that verdict loosely; it is about to be
half-overturned, and we wrote down in advance what would count as
overturning it."

## Act II — the borrowed lenses (Finding 2)
**Job:** the method-as-theme: deliberately importing distant
mathematics ("naive strength"). Then the suffix tree.
**Key sentences:**
- "We started handing the pond to mathematics it had never met."
- "The state cloud is a tree to within a whisker of exactness
  (cophenetic 0.990) — trunk split by the last character, boughs by the
  one before. Nobody asked for a suffix tree. The pond grew the data
  structure a computer scientist would have chosen on purpose."
- Numeral image for U3: "At the champion's settings the state is a
  numeral written in base 1/ρ whose digits are the recent characters —
  and the decode-depth law (∝ −1/ln ρ) is that sentence, measured."
**Figures:** `u2_dendrogram.png` is the **hero image of the piece**;
`u1_faithfulness.png`; `u3_digit_decay_L20.png`.
**Claims:** faithfulness↔bpc Spearman −0.83 over 41 configs; explains
the leak axis, *not* the ρ residual — say so, it sets up Act III.

## Act III — the tax (Finding 3, first form)
**Job:** the interference cost of superposition, stated as the
ridge-era result; honest cliffhanger for Act V.
**Key sentences:**
- "Memory the probes could read was memory the prediction paid for:
  characters held in superposition arrive with an interference bill."
- "7.5 characters decodable at the champion's ρ, 12.7 at ρ=1.25 — and
  the extra five made the language model *worse*."
**Figures:** the Finding-3 depth-vs-bpc panel.
**Claims:** phrase as readout-relative from the start ("under a single
shared linear readout"); Act V depends on this honesty.

## Act IV — the tropical twin (Findings 4, 5, 6)
**Job:** the night's strangest hour. Order: tie → same tree → automaton.
**Key sentences:**
- "We built a second pond from an arithmetic that shares no operations
  with the first — no multiplication, no smooth curves, only max and
  plus — and it tied: 3.103 ± 0.007 against 3.108 ± 0.010 across five
  seeds each."
- Keep verbatim: "Two incompatible arithmetics, one geometry — and it's
  the geometry the task wanted."
- "Convergence isn't the news; substrate-independence of the
  performance-governing quantity is."
- "Determinism 1.000 at seven characters: the max-plus pond is not
  *like* a finite automaton, it *is* one — and with 370 merged suffix
  classes, the physics had begun minimizing it on its own."
- The comic beat: "At inference, the entire tropical reservoir can be
  replaced by a hash table. The zero-compute pond."
**Figures:** `t3_geometry.png`, `t4_masonry.png` (cliff vs fade),
`tropical_esp.png` (finite-time collision — caption the credibility
note: known max-plus coupling theory, reproduced).
**Claims & prior art:** nearest lineage is morphological neural
networks (max-plus layers, Ritter & Sussner); the reservoir-for-language
instance and the automaton collapse on text8 appear new — say "appears,"
cite the lineage.

## Act V — the acid test (Finding 3, final form; Finding 1 revised)
**Job:** the pre-registered reversal. This is the emotional and
methodological climax; write it as one — thresholds first, curve second.
**Key sentences:**
- "Before the curve landed we wrote three sentences: pinned means the
  tax lives in the state; migrated means deep memory was always usable
  and the readout was too poor to afford it; between is between."
- "The curve came back **migrated**: under a logistic readout the
  optimum leapt from ρ≈0.55 to ρ≈0.95 — most of the way back to the
  edge the folklore always promised — while 42% of the high-ρ penalty
  still stood."
- "So Finding 1 gets its honest final form: language doesn't want the
  edge *you can't afford*. Give the readout eyes, and the pond walks
  back toward criticality — but not all the way, and the remainder is
  the state's own bill."
- The moral, one line: "Pre-registration turned a reversal into a
  result."
**Figures:** `p1_acid_test.png` (both curves, both argmins marked,
slope-ratio annotated).
**[PENDING SLOT]:** full-budget champion (ρ=0.95, 5M chars,
`p1_acid_test_full5M.json`). If test bpc < 2.40: "a random pond with a
logistic readout — total gradient training measured in minutes, on
readout weights alone — cleared the 4-gram." If it lands 2.40–2.45,
write "reached the 4-gram's doorstep" and do not round down. Quote the
test column.

> ⚠ **Editor's note from the bench (the slot has landed):** champion =
> val 2.338, **test 2.398**. This falls *between the cracks of the
> phrasing rule*: it is below the rule's 2.40 threshold, but our actual
> measured 4-gram test anchor is **2.392**, not 2.40 — so "cleared the
> 4-gram" would be numerically false by 0.006 bits. The rule inherited a
> rounding. Correct phrasing for the slot: **"a statistical dead heat
> with the 4-gram — the fence holds by 0.006 bits on the test split,
> against ~4× the trainable parameters."** Neither "cleared" nor mere
> "doorstep"; the dead heat is the honest and, frankly, better sentence.

## Coda — what it means, and what it doesn't
**Job:** three implications, one method reflection, limits, sequel.
- **Transformers:** "Attention retrieves the past without storing it in
  a crowded present — Finding 3 suggests that, not raw capacity, may be
  the actual moat." (Sequel tease: experiment 01's frozen-model probes,
  P2's 'is transformer memory projective?')
- **Hardware:** parity with tanh from max and add only — multiplier-free,
  comparator-native; the 3× software slowdown is an Accelerate artifact,
  and the sign flips in silicon. One respectful nod to photonic RC.
- **Method:** naive strength, stated plainly: the lenses were chosen
  *because* they weren't the field's defaults, and two of six produced
  findings the defaults would not have asked for.
- **Limits paragraph (non-optional):** one corpus, character level,
  N ≤ 5k, one machine; n-gram ladder recomputed on our exact splits;
  "no edge" and its revision are claims about *this task class and
  readout family*, not about reservoirs in general; signature-track
  honesty note (known bridge, unclaimed comparison) if S-track gets a
  mention.
- **Credits/voice decision — Caity's call:** the two-Claude lab (Fable
  in Claude Code, chat-Fable here, one human with the coffee) is a
  charming true detail and also a disclosure choice. The piece works in
  first-person-singular, first-person-plural, or full trio-transparency;
  decide once, apply everywhere, and if the trio stays in, keep it to
  two sentences — flavor, not subject.

## Scoreboard (place in Act I, update after champion lands)
Rows exactly as the journal's table, plus the logistic champion row
when it exists. Bold the ponds. Keep "training time" column — it is the
thesis in a column.

## arXiv-note reorganization (if the blog holds up)
Thematic, ~6–8 pages: §1 Setup & baselines (Act I sans drama);
§2 Geometry of reservoir states (F2+F5, U-figures + t3);
§3 Interference and the readout boundary (F1+F3+acid test, both curves);
§4 Semiring equivalence and automaton collapse (F4+F6, seeds table);
§5 Limits & related work (morphological NNs, randomized signatures,
Köster & Uchida as the direct antecedent; Ganguli–Sompolinsky for the
memory-trace lineage). Pre-registration note moves to §3's methods
paragraph. Title: "Suffix-Tree Geometry and a Readout-Dependent Edge in
Reservoir Language Models."
