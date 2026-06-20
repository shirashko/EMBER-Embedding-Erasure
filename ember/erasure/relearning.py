"""Relearning loop - stress test for concept erasure.

Runs N epochs of full-parameter fine-tuning on concept paragraphs and evals
after every epoch. The final qa_acc measures how easily the concept relearns.
Lower is better.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from torch.optim import AdamW

from ember.erasure import log

DEFAULT_LR = 1e-5
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_LEN = 256
DEFAULT_N_EPOCHS = 2


def load_relearn_paragraphs(json_path: Path) -> Dict[str, List[str]]:
    """Read data/relearn_paragraphs.json -> {concept: [paragraph, ...]}.

    File structure: {concept: {"RelearnParagraphs": [...], ...}}.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"relearn paragraphs file not found: {json_path}")

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{json_path}: expected a top-level dict, got {type(raw).__name__}")

    out: Dict[str, List[str]] = {}
    for concept, value in raw.items():
        if not isinstance(value, dict) or "RelearnParagraphs" not in value:
            raise KeyError(f"{json_path}: concept {concept!r} missing 'RelearnParagraphs'")
        paras = [p for p in value["RelearnParagraphs"] if isinstance(p, str) and p.strip()]
        if not paras:
            raise ValueError(f"{json_path}: concept {concept!r} has no usable paragraphs")
        out[concept] = paras
    return out


def _make_optimizer(hf_model: torch.nn.Module, model_name: str,
                    lr: float = DEFAULT_LR) -> torch.optim.Optimizer:
    """8-bit AdamW for Llama (memory pressure), standard AdamW otherwise."""
    if "llama" in model_name.lower():
        import bitsandbytes as bnb
        log.info("optimizer: bnb.AdamW8bit lr=%.2e (llama)", lr)
        return bnb.optim.AdamW8bit(hf_model.parameters(), lr=lr, weight_decay=0.01)
    log.info("optimizer: AdamW lr=%.2e", lr)
    return AdamW(hf_model.parameters(), lr=lr, weight_decay=0.01)


def _run_one_epoch(
        hf_model: torch.nn.Module,
        tokenizer: Any,
        paragraphs: List[str],
        optimizer: torch.optim.Optimizer,
        device: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LEN,
) -> float:
    """One pass of next-token CE over a shuffled paragraph list. Returns mean loss."""
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    hf_model.train()
    data = list(paragraphs)
    random.shuffle(data)

    losses: List[float] = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding="max_length",
                           max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = hf_model(**inputs).logits

        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[:, 1:].contiguous().view(-1)
        loss = ce(shift_logits, shift_labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return sum(losses) / max(1, len(losses))


def run_relearning(
        hf_model: torch.nn.Module,
        tokenizer: Any,
        *,
        paragraphs: List[str],
        model_name: str,
        concept: str,
        eval_fn: Callable[[torch.nn.Module, Any], Dict[str, Any]],
        relearn_csv_path: Path,
        pre_metrics: Optional[Dict[str, Any]] = None,
        max_paragraphs: int = 100,
        n_epochs: int = DEFAULT_N_EPOCHS,
        lr: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LEN,
) -> Dict[str, Any]:
    """Run an n_epochs relearning loop on hf_model in-place.

    Appends per-epoch metrics to relearn_csv_path and returns the final-epoch
    metrics dict. The model is mutated in place; callers must restore if needed.
    """
    paras = [p for p in paragraphs[:max_paragraphs] if p.strip()]
    if not paras:
        log.warning("relearning skipped: no usable paragraphs for %r", concept)
        return {"qa_acc": float("nan")}

    log.info("starting %d-epoch relearning on %d paragraphs", n_epochs, len(paras))

    hf_model.config.use_cache = False
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for p in hf_model.parameters():
        p.requires_grad = True

    optimizer = _make_optimizer(hf_model, model_name, lr=lr)
    device = str(next(hf_model.parameters()).device)

    relearn_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if pre_metrics is not None:
        rows.append({"epoch": 0, "loss": float("nan"), **pre_metrics})

    last_metrics: Dict[str, Any] = pre_metrics or {}
    try:
        for ep in range(1, n_epochs + 1):
            torch.set_grad_enabled(True)
            loss = _run_one_epoch(hf_model, tokenizer, paras, optimizer,
                                  device=device, batch_size=batch_size,
                                  max_length=max_length)
            torch.set_grad_enabled(False)
            hf_model.eval()

            last_metrics = eval_fn(hf_model, tokenizer)
            rows.append({"epoch": ep, "loss": loss, **last_metrics})
            pd.DataFrame(rows).to_csv(relearn_csv_path, index=False)
            log.info("epoch %d loss=%.4f qa_acc=%.4f", ep, loss,
                     float(last_metrics.get("qa_acc", float("nan"))))
    finally:
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        for p in hf_model.parameters():
            p.grad = None
        torch.cuda.empty_cache()

    return last_metrics


__all__ = [
    "DEFAULT_LR", "DEFAULT_BATCH_SIZE", "DEFAULT_N_EPOCHS",
    "load_relearn_paragraphs", "run_relearning",
]
