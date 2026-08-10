"""Experiment 02 baseline: minimal char-level GPT in MLX (Mac only).

STATUS: STUB. Task E2-T4 in CLAUDE.md — implement by adapting the
transformer LM example from ml-explore/mlx-examples (llms/transformer_lm)
or writing a minimal decoder (~150 lines): token+pos embed, k blocks of
(MHA, MLP, pre-LN), tied output head.

Contract this stub must satisfy (so comparison plots stay uniform):
- CLI: --params-budget {1e6, 1e7}  --train-chars N  --seed S
- Trains on the SAME text8 split/prefix as run_esn_lm.py.
- Writes results/exp02/gpt_text8_P{budget}.json with keys:
    config, val_bpc, test_bpc, trainable_params, wall_clock{train_s}
- Logs tokens/sec so wall-clock comparisons are interpretable.
"""

raise SystemExit("stub — see CLAUDE.md task E2-T4")
