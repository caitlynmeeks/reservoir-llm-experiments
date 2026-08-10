# Experiment 02 — ESN vs. tiny transformer as character-level language models

## Question
Where does a pure reservoir LM sit on the (trainable params, compute,
bits-per-character) frontier relative to a small transformer, in the
spirit of Köster & Uchida (arXiv:2507.15779)? Then: how do spectral
radius and leak rate shape language performance ("edge of chaos" curve)?

## Setup
- Corpus: text8 (27-symbol vocab, 90M/5M/5M split), metric = bpc.
- Reservoir path: one-hot char -> ESN(N) -> readout to next-char scores.
  Batched via `data.as_parallel_segments` (e.g. S=256 parallel segments,
  washout 200 each) so the Python step-loop runs at (T/S) iterations.
- Readout A (primary): `ChunkedRidge` to one-hot targets, closed form.
  Convert scores to probabilities with a val-fitted temperature+bias
  calibration (2 scalars, fit by minimizing val NLL) before reporting bpc.
- Readout B (optional, still "readout-only"): multinomial logistic
  regression on frozen states, mini-batch SGD (NumPy or torch-MPS). This
  is the fairer bpc comparison; report both.
- Transformer baseline: char-level GPT in MLX (~2 configs, e.g. 1M and
  10M params), trained on the same split with a fixed token budget.
  Adapt mlx-examples' transformer LM or write minimal (see E2-T4).

## Comparisons (report ALL axes — this is the honest part)
1. bpc vs. TRAINABLE params: ESN trainable = readout (N+1)×27 (+2 calib)
   vs. transformer = all params.
2. bpc vs. TOTAL params (ESN reservoir counts here).
3. bpc vs. wall-clock to result on the M3 Ultra (reservoir: state pass +
   one solve; transformer: full training run). Log wall-clock per phase.

## Sweeps
- N ∈ {1k, 5k, 20k, 50k} (100k stretch goal — 40GB fp32 Gram; check free
  RAM, LM Studio models likely need unloading).
- Spectral radius ρ ∈ {0.6 … 1.4}, leak a ∈ {0.1, 0.3, 0.6, 1.0} at fixed
  N=5k: heatmap of val bpc. Expect best near ρ≈1 — the edge-of-chaos plot.
- Input scaling matters at one-hot inputs; sweep {0.3, 1.0, 3.0} once.

## Deliverables
- `results/exp02/`: bpc tables (JSON), frontier plot (bpc vs. trainable
  params, both model families), edge-of-chaos heatmap, wall-clock table.
- A short RESULTS.md summarizing where the reservoir frontier sits.

## Reference anchors (sanity)
- Char-level small transformers reach ~1.4–1.6 bpc on text8-like data at
  the ~1–10M scale; classic n-gram ≈ 2.0+; uniform = log2(27) ≈ 4.75.
  Expect the ridge-readout ESN between n-gram and small transformer;
  treat large deviations as bugs first.
