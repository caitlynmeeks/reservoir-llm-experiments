# Experiment 01 — The LLM as a reservoir

## Question
How "reservoir-like" is a frozen transformer's residual stream? Measure it
with the classic RC yardsticks — linear memory capacity and nonlinear
memory (k-parity) — per layer, and compare against a real ESN given the
same inputs and a matched state dimension.

## Hypotheses
- H1: Linear token-recall accuracy vs. lag falls off much more slowly in
  mid layers of the transformer than in a matched-D ESN (attention ≈
  content-addressable memory vs. fading memory).
- H2: There is a depth "sweet spot": early layers under-mix, late layers
  over-compress toward next-token features. Expect an inverted-U over depth.
- H3: On k-parity the ESN closes much of the gap — recurrent nonlinear
  mixing is what reservoirs are good at.

## Method
1. **Inputs.** Random token sequences: sample uniformly from a fixed
   subset of M=64 single-token ids (build the subset by encoding common
   short words; keep only those that tokenize to exactly one id). i.i.d.
   sequences kill linguistic structure — we are probing the *dynamics*,
   not language. B=32 sequences × T=512 tokens, washout 64.
2. **Transformer states.** `rcllm.activations.collect_hidden_states` on a
   small-to-mid DENSE mlx-community model. Chosen + verified (E1-T1):
   `mlx-community/Llama-3.2-1B-Instruct-4bit` — 16 layers, D=2048, all
   full attention, already in the local HF cache. (Weights are 4-bit but
   activations compute in fp16; swap to the bf16 repo if we ever want to
   rule quantization noise out of the dynamics.) Subsample layers, e.g.
   every 2nd.
3. **Probes.** For each layer ℓ and lag k ∈ {1..64}: ridge probe from
   h_t^(ℓ) to the one-hot identity of token t−k (`ChunkedRidge`, M
   targets, argmax accuracy on a held-out 30% of timesteps). Chance = 1/64.
4. **ESN control.** Feed the SAME token sequences to an `ESN` whose input
   is the model's own embedding vectors (`embedding_table`), state dim
   N = D (matched) plus one larger N = 4D variant. Same probes.
5. **Nonlinear probe.** Binary sequences over 2 chosen token ids; k-parity
   accuracy per layer vs. ESN (`tasks.parity_accuracy_from_states`).

## Metrics & deliverables
- Heatmap: probe accuracy over (layer × lag) for the transformer.
- Curves: accuracy vs. lag for best/median transformer layer vs. ESN(D)
  vs. ESN(4D); same for parity vs. k.
- JSON results in `results/exp01/` + PNGs. Every run records model id,
  seed, and config.

## Risks / verify first
- `activations.py` VERIFIED (E1-T1, 2026-08-09): manual pass matches
  model logits; mask causal; embedding_table dequantizes. Re-verify if
  changing model family or mlx-lm version.
- Tokenizer quirks: enforce single-token membership of the M-subset.
- Memory: (B·(T−washout)) × D per layer is small; fine even with LM Studio
  models resident. Free-memory check still worth doing before big sweeps.
