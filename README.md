# reservoir-llm-experiments

Reservoir computing × transformers, sized for a Mac Studio M3 Ultra.

- **Exp 01** — probe a frozen LLM's residual stream with classic reservoir
  diagnostics (memory curves, parity) vs. a matched echo state network.
- **Exp 02** — echo state network as a char-level language model on text8
  vs. a tiny MLX GPT: bpc / params / wall-clock frontier + edge-of-chaos sweeps.

Start with `CLAUDE.md` (project brief + task list), then the `SPEC.md` in
each experiment folder. Quick smoke test on any machine:

    pip install -r requirements.txt
    python experiments/02_esn_vs_transformer/run_esn_lm.py --synthetic \
        --train-chars 200000 --eval-chars 20000 --n-reservoir 500 --segments 16
