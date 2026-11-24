"""HF decoder layer access helpers for causal LMs (incl. Qwen3.5 text-only)."""
from __future__ import annotations

from typing import Any, Sequence

def is_qwen(model_name: str) -> bool:
    n = model_name.lower()
    return "qwen3.5" in n or "qwen3_5" in n


def decoder_layers(hf_model: Any) -> Sequence[Any]:
    """Return the transformer block ModuleList for a causal LM."""
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model.layers
    lm = getattr(hf_model, "language_model", None)
    if lm is not None:
        if hasattr(lm, "layers"):
            return lm.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
    raise AttributeError(
        f"Could not find decoder layers on {type(hf_model).__name__}"
    )


def n_decoder_layers(hf_model: Any) -> int:
    return len(decoder_layers(hf_model))


def rmu_module_str(hf_model: Any) -> str:
    """Format string for WMDP RMU ``eval(module_str.format(...))``."""
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return "{model_name}.model.layers[{layer_id}]"
    if hasattr(hf_model, "language_model") and hasattr(hf_model.language_model, "layers"):
        return "{model_name}.language_model.layers[{layer_id}]"
    raise AttributeError(
        f"Could not build RMU module_str for {type(hf_model).__name__}"
    )
