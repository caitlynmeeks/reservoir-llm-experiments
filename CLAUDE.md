# Reservoir computing × LLM experiments

Two experiments probing how reservoir computing ideas interact with
transformers, designed for a Mac Studio M3 Ultra (512GB unified memory).
Read the SPEC.md inside each experiment folder before touching its code —
the specs are the source of truth for method and metrics.

- `experiments/01_llm_as_reservoir/` — treat a frozen mlx-lm model's
  residual stream as a reservoir; run classic RC diagnostics (token-recall
  memory curves, k-parity) per layer vs. a matched echo state network.
- `experiments/02_esn_vs_transformer/` — ESN with ridge readout as a
  char-level LM on text8 vs. a small MLX GPT; params/compute/bpc frontier
  plus spectral-radius × leak-rate ("edge of chaos") sweeps.

## Environment
- Target machine: macOS, Apple silicon (M3 Ultra, 512GB). MLX code paths
  are Mac-only; everything else is pure NumPy/SciPy and runs anywhere.
- LM Studio often has large models resident (~342GB). Before big-N runs
  (N ≥ 50k reservoir, or the 100k stretch goal), check free memory and
  unload LM Studio models if needed. Memory math is in `src/rcllm/ridge.py`.
- Setup: `python3 -m venv .venv && source .venv/bin/activate &&
  pip install -r requirements.txt && pip install mlx mlx-lm` (last two on
  the Mac only; verify current mlx-lm version — its internals move).

## Code map
- `src/rcllm/esn.py` — sparse leaky ESN, batched over parallel sequences.
- `src/rcllm/ridge.py` — chunked normal-equation ridge (the only training).
- `src/rcllm/tasks.py` — memory capacity, k-parity (work on any states).
- `src/rcllm/activations.py` — mlx-lm hidden-state hooks. **UNVERIFIED.**
- `src/rcllm/data.py` — text8 download/encoding, segment batching.

## Status: verified vs. not
Verified on the Mac (2026-08-09, .venv = Python 3.13 + mlx 0.32.0 +
mlx-lm 0.31.3): NumPy path (`run_esn_lm.py --synthetic` end-to-end) and
the mlx path — `activations.py` checked against
mlx-community/Llama-3.2-1B-Instruct-4bit (manual layer loop reproduces
model logits, mask causal, embedding_table dequantizes on quantized
models). Earlier container smoke tests: chunked ridge matches direct
solve; memory-capacity and parity diagnostics.

NOT yet run anywhere: transformer path of `run_probe.py` end-to-end,
text8 download, `run_tiny_gpt.py` (stub).

Extended program lives in PUNCHLIST.md; work track by track.

## Task list
E1 — experiment 01
- [x] E1-T1 Verify `activations.py` against installed mlx-lm + chosen
      model — DONE 2026-08-09: model = mlx-community/Llama-3.2-1B-
      Instruct-4bit (16 layers, D=2048); fixed embedding_table (was
      returning bit-packed quantized weights); added guard against
      mixed-attention-type models.
- [ ] E1-T2 Implement `build_single_token_subset(tokenizer, M)` and
      replace the placeholder id range in `run_probe.py`.
- [ ] E1-T3 Plots: (layer × lag) accuracy heatmap + lag curves vs. both
      ESN controls; save PNG + JSON to `results/exp01/`.
- [ ] E1-T4 Layer sweep on the chosen model (every 2nd layer), B=32,
      T=512; write up which depth is most reservoir-like.
- [ ] E1-T5 k-parity probes (binary token subset) for model + ESNs.

E2 — experiment 02
- [x] E2-T1 First real text8 run — DONE 2026-08-09: N=5k defaults gave
      val 2.80 / test 2.81 bpc (state pass 181s, solve 0.16s). Sanity
      anchors computed on the same slices via run_ngram_baseline.py:
      3-gram 2.91, 4-gram 2.40, 5-gram 2.06. Untuned ESN beats trigram;
      reaching the spec's ~2.0 hope is what the E2-T3 sweep is for.
- [ ] E2-T2 N sweep {1k, 5k, 20k, 50k}; log wall-clock per phase.
- [ ] E2-T2b Optional logistic readout on frozen states (fairer bpc than
      ridge-to-one-hot); NumPy SGD or torch-MPS; still readout-only.
- [x] E2-T3 Sweep driver — DONE 2026-08-10: sweep_rho_leak.py (resumable,
      per-cell JSONs). 10×4 grid at N=5k, 2M chars: monotone in leak
      (1.0 best everywhere); interior optimum at rho≈0.5-0.6, leak=1.0,
      val 2.73. NO edge-of-chaos peak for language — short-range task
      prefers crisp shallow memory. Heatmap:
      results/exp02/sweep_rho_leak_N5000_T2000000_10x4.png
      Champion rerun at 5M chars: val 2.695 / test 2.696.
- [ ] E2-T4 Implement `run_tiny_gpt.py` per the contract in its docstring
      (adapt ml-explore/mlx-examples transformer LM); 1M and 10M configs.
- [ ] E2-T5 Frontier + comparison plots per SPEC; write RESULTS.md.

## Conventions
- Determinism: every run takes `--seed`; record full config in the JSON.
- States float32; normal equations float64 (float32 accum only at N=100k).
- Results as JSON + PNG under `results/expNN/`; never overwrite, suffix
  by config. Keep runs re-creatable from the JSON config alone.
- Style: plain NumPy/SciPy, no framework abstractions; small pure
  functions; keep the streaming discipline (never materialize all states).
