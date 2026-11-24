"""MLP-intermediate activations at the site a frozen SNMF dictionary was trained on."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from ember.frozen_snmf.sites import hook_for_layer


def harvest_activations(
    model,
    sentences: Sequence[str],
    layers: Sequence[int],
    batch_size: int,
    ember_mode: str,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """Activations at the hook for ``ember_mode``.

    ``mlp`` (the image of a dictionary site named ``mlp_intermediate``) is
    ``blocks.{layer}.mlp.hook_post``, the down-projection input. Padding and
    BOS positions are dropped. Each activation tensor has shape
    ``(n_tokens, d_mlp)``. The returned token-id tensor has shape ``(n_tokens,)``
    and lines up with every layer, because the mask does not depend on the layer.
    """
    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    hooks = [hook_for_layer(ember_mode, layer) for layer in layers]
    collected: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    token_batches: List[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(sentences), batch_size):
            batch = list(sentences[start:start + batch_size])
            input_ids = model.to_tokens(batch, padding_side="left")
            input_ids = input_ids.to(next(model.parameters()).device)
            mask = input_ids != pad_id
            if bos_id is not None:
                mask = mask & (input_ids != bos_id)
            token_batches.append(input_ids[mask].reshape(-1).detach().cpu().long())
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=hooks,
                stop_at_layer=max(layers) + 1,
            )
            for layer, hook in zip(layers, hooks):
                acts = cache[hook].detach()
                kept = acts[mask.to(acts.device)].reshape(-1, acts.shape[-1])
                collected[layer].append(kept.to("cpu", dtype=torch.float32))
            del cache
    activations = {layer: torch.cat(parts, dim=0) for layer, parts in collected.items()}
    token_ids = torch.cat(token_batches, dim=0) if token_batches else torch.empty(0, dtype=torch.long)
    for layer, acts in activations.items():
        if acts.shape[0] != token_ids.shape[0]:
            raise RuntimeError(
                f"Layer {layer} has {acts.shape[0]} tokens but the mask kept {token_ids.shape[0]}"
            )
    return activations, token_ids
