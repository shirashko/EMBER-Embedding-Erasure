"""HuggingFace (+ TransformerLens, for PISCES) model loading."""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ember.erasure import log


def _resolve_device_map() -> str:
    """device_map from the EMBER_DEVICE env var ("cpu"/"cuda"), else "auto"."""
    device = os.environ.get("EMBER_DEVICE", "").lower()
    return device if device in ("cpu", "cuda") else "auto"


DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def pick_dtype(name: str) -> torch.dtype:
    """Map a config string (``"bf16"``/``"fp16"``/``"fp32"``) to a torch dtype."""
    if name not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype {name!r}; expected one of {list(DTYPE_MAP)}")
    return DTYPE_MAP[name]


def load_hf_model(
        model_name: str,
        cache_dir: str = "",
        dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load an HF causal LM and its tokenizer.

    - ``pad_token := eos_token`` so batched generation works.
    - ``device_map="auto"`` to put the model on the available GPU.
    - For Gemma-2, untie ``lm_head`` from the input embeddings so editing
      the embedding does not corrupt the output projection.
    - ``use_cache=False`` for gradient-checkpointing-compatible training.
    """
    cache_dir_arg = cache_dir if cache_dir else None
    log.info("loading HF model %s (dtype=%s)", model_name, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir_arg)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=_resolve_device_map(),
        torch_dtype=dtype,
        cache_dir=cache_dir_arg,
    )

    if "gemma-2" in model_name.lower():
        with torch.no_grad():
            model.lm_head.weight = torch.nn.Parameter(
                model.lm_head.weight.detach().clone()
            )
        model.config.tie_word_embeddings = False

    model.config.use_cache = False

    n_params = sum(p.numel() for p in model.parameters())
    log.info("loaded %s | %d params | device=%s", model_name, n_params,
             next(model.parameters()).device)
    return model, tokenizer


def load_tl_model(
        model_name: str,
        cache_dir: str = "",
        dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Any, AutoTokenizer]:
    """Load a TransformerLens ``HookedTransformer`` + tokenizer.

    Used only by the PISCES method, which edits MLP weights in TL space and
    then syncs them into an HF model for evaluation. ``fold_ln=False`` keeps
    the block structure PISCES' ``unlearn_concept`` expects.
    """
    from transformer_lens import HookedTransformer  # lazy: TL only needed for PISCES

    cache_dir_arg = cache_dir if cache_dir else None
    device = os.environ.get("EMBER_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("loading TL model %s (dtype=%s)", model_name, dtype)

    tl_model = HookedTransformer.from_pretrained(
        model_name, device=device, cache_dir=cache_dir_arg,
        fold_ln=False, dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir_arg)
    tokenizer.pad_token = tokenizer.eos_token

    log.info("loaded TL %s | %d params", model_name,
             sum(p.numel() for p in tl_model.parameters()))
    return tl_model, tokenizer


__all__ = ["DTYPE_MAP", "pick_dtype", "load_hf_model", "load_tl_model"]
