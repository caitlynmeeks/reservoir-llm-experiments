"""Hidden-state extraction from mlx-lm models (Apple silicon only).

STATUS: VERIFIED (E1-T1, 2026-08-09) against mlx-lm 0.31.3 with
mlx-community/Llama-3.2-1B-Instruct-4bit: manual layer loop reproduces
model(ids) logits, mask is causal, embedding_table dequantizes correctly
on quantized models. Written against the llama-family layout
(model.model.embed_tokens / .layers / .norm, layer(x, mask, cache));
other families (Gemma norm placement, MoE, sliding-window attention) need
re-verification — collect_hidden_states raises on mixed attention types.

Everything else in this repo is pure NumPy and runs anywhere.
"""

from __future__ import annotations

# Deliberately soft imports so the rest of the package works on any machine.
try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from mlx_lm.models.base import create_attention_mask

    MLX_AVAILABLE = True
except ImportError:  # pragma: no cover
    MLX_AVAILABLE = False

import numpy as np


def load_model(model_id: str):
    """Load an mlx-community model + tokenizer. Pick a small-to-mid DENSE
    model for probing (dense residual streams are cleaner than MoE for a
    first pass). Verify the exact repo id on huggingface.co/mlx-community."""
    if not MLX_AVAILABLE:
        raise RuntimeError("mlx / mlx-lm not installed (Apple silicon only)")
    return mlx_load(model_id)


def collect_hidden_states(model, token_ids: np.ndarray, layers: list[int] | None = None):
    """Manual forward pass collecting post-block residual-stream states.

    token_ids: (B, T) int array.
    Returns dict {layer_index: np.ndarray (B, T, D)} — float32 on host.
    Layer 0 = after first transformer block; use layer -0.5 = embeddings
    via include_embeddings if needed later.

    Verified (E1-T1) against mlx-lm 0.31.3 + Llama-3.2-1B-Instruct-4bit:
    matches the model's own forward exactly (final norm applied only after
    the last block, so these are true residual-stream states) and the mask
    is causal (perturbing token t leaves states at <t bit-identical).
    """
    inner = model.model                      # llama-style: LlamaModel
    layer_types = set(getattr(inner, "layer_types", None) or ["full_attention"])
    if layer_types != {"full_attention"}:
        raise ValueError(
            f"model mixes attention types {layer_types}; this single-mask "
            "forward is only correct for all-full-attention models"
        )
    x = inner.embed_tokens(mx.array(token_ids))
    mask = create_attention_mask(x, None)
    keep = set(range(len(inner.layers))) if layers is None else set(layers)
    out: dict[int, np.ndarray] = {}
    for i, layer in enumerate(inner.layers):
        x = layer(x, mask, cache=None)
        if i in keep:
            out[i] = np.array(x.astype(mx.float32))
    return out


def embedding_table(model) -> np.ndarray:
    """Token embedding matrix (V, D) as NumPy — used to give the ESN control
    the SAME input representation the transformer sees.

    Looks rows up through the embedding layer itself rather than reading
    `.weight`: on quantized models `.weight` holds bit-packed uint32 data
    (e.g. (V, 256) for a 4-bit D=2048 model), and the lookup dequantizes."""
    emb = model.model.embed_tokens
    vocab = emb.weight.shape[0]
    return np.array(emb(mx.arange(vocab)).astype(mx.float32))


if __name__ == "__main__":  # smoke test — run on the Mac, not in CI
    import sys

    model_id = sys.argv[1] if len(sys.argv) > 1 else "REPLACE/with-mlx-community-model"
    model, tok = load_model(model_id)
    ids = np.array([tok.encode("the quick brown fox jumps over the lazy dog")])
    hs = collect_hidden_states(model, ids, layers=[0])
    (i, h), = hs.items()
    print(f"ok: layer {i} states {h.shape}, dtype {h.dtype}, mean {h.mean():.4f}")
