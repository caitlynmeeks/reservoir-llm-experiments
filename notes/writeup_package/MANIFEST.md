# Writeup package — reservoir all-nighter, 2026-08-09/10

For chat-Fable's spine draft. Everything referenced by
`lab-journal-2026-08-10.md` (this folder), organized by finding.
The held seat is filled: `json/p1_acid_test_full5M.json` — logistic
champion at ρ=0.95, 5M chars: val 2.338, TEST 2.398. The 4-gram fence
(2.392) holds by 0.006 bits on the honest split; a statistical dead
heat against ~4x the trainable parameters.

## Finding 1 — Language doesn't want the edge
- figures/sweep_rho_leak_N5000_T2000000_10x4.png, json/ same stem
- json/ngram_baseline_train5000000.json (the ladder)
- json/esn_text8_N5000_r0.6_..._T5000000.json (ridge champion, 2.70)

## Finding 2 — The pond grows a suffix tree
- figures/u1_faithfulness.png / json/u1_faithfulness.json (41-config claim)
- figures/u2_dendrogram.png / json/u2_dendrogram.json (cophenetic 0.990)
- figures/u3_digit_decay_L20.png / json/u3_digit_decay_L20.json

## Finding 3 — Interference cost of superposition + acid test
- figures/p1_acid_test.png / json/p1_acid_test.json
  (pre-registered thresholds in journal; verdict MIGRATED, slope ratio 0.42)

## Finding 4 — The semiring doesn't matter
- figures/tropical_esp.png (finite-time collision / max-plus coupling)
- json/tropical_N1000_T1000000_seed{0..4}.json (5 tropical seeds; seed0's
  champion cell is runs[2]: cycle_mean=-0.1, input_scale=1.0)
- json/esn_text8_N1000_r0.6_..._seed{0..4}_T1000000.json (5 tanh seeds)
  Means ± sd: tanh 3.108 ± 0.010, tropical 3.103 ± 0.007.

## Finding 5 — Substrate-independence of the performance-governing quantity
- figures/t3_geometry.png / json/t3_geometry.json (two algebras, one tree)
- figures/t4_masonry.png / json/t4_masonry.json (cliff vs fade; tie stats)

## Finding 6 — The pond that became a lookup table
- json/t5_automaton.json (determinism 1.000 at k=7; 370 merged classes;
  tanh control 0.000 everywhere; merged-group examples in journal)

## Lesson figures (context/pedagogy, incl. edge-of-chaos MC curve)
- figures/esp_convergence.png, memory_curves.png, edge_of_chaos.png,
  parity.png
