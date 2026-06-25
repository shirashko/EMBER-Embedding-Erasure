"""CRISP method plug-in (SAE-based concept identification + LoRA unlearning).

Pipeline per concept:

1. Apply per-concept embed edit (if ``ember_step.enabled``) on the base HF
   model wrapped inside a CRISP object.
2. For each layer range ``(layer_lo, layer_hi, layer_step)``:
    a. Construct a fresh CRISP instance (loads SAEs for those layers).
    b. ``crisp.process_multi_texts_batch(forget, retain)`` -- one-time
       SAE-feature extraction for the concept.
    c. For each ``(k_features, alpha, lr)`` triplet:
        - ``crisp.unload_lora()`` -- drop the previous cell's LoRA.
        - ``unlearn_lora(...)`` -- train a new LoRA on this HP.
        - Eval, write row.

Cells within one layer-range share the SAE-loaded base model; only the
LoRA changes between cells. This is much cheaper than full reloads, so
CRISP sets :attr:`requires_full_reload = False` and instead manages its
own per-range cache on the instance.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from ember.erasure import embed_edit, io, log
from ember.erasure.config import RunConfig
from ember.erasure.methods.base import Method, register
from ember.forget_retain import load_coherency_prompts, load_forget_retain

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "external" / "CRISP"))
sys.path.append(str(ROOT_DIR / "external" / "CRISP" / "crisp"))

GEMMA_LAYER_RANGES: List[Tuple[int, int, int]] = [
    (4, 14, 2),
    (5, 15, 2),
    (4, 20, 2),
    (5, 21, 2),
]
LLAMA_LAYER_RANGES: List[Tuple[int, int, int]] = [
    (5, 19, 2),
    (4, 18, 2),
    (5, 29, 2),
    (4, 28, 2),
]


def _layer_ranges(model_name: str) -> List[Tuple[int, int, int]]:
    return LLAMA_LAYER_RANGES if "llama" in model_name.lower() else GEMMA_LAYER_RANGES


def validate_layer_ranges(
    ranges: List[Tuple[int, int, int]] | List[List[int]],
    model_name: str,
) -> List[Tuple[int, int, int]]:
    """Ensure each ``(layer_lo, layer_hi, layer_step)`` is a valid CRISP grid range."""
    allowed = set(_layer_ranges(model_name))
    normalized = [tuple(int(x) for x in r) for r in ranges]
    invalid = [r for r in normalized if r not in allowed]
    if invalid:
        raise ValueError(
            f"Invalid CRISP layer_ranges {invalid} for {model_name!r}; "
            f"allowed: {sorted(allowed)}"
        )
    return normalized


def resolve_layer_ranges(cfg: Any, model_name: str) -> List[Tuple[int, int, int]]:
    """Return configured layer ranges, or the model defaults when unset."""
    if cfg.layer_ranges is None:
        return _layer_ranges(model_name)
    return validate_layer_ranges(cfg.layer_ranges, model_name)


def _layer_cache_valid(layer_path: Path) -> bool:
    """True when a layer dir already holds downloaded SAE weights."""
    if not layer_path.is_dir():
        return False
    return any(
        f.is_file() and f.suffix in (".pt", ".safetensors", ".npz", ".json")
        for f in layer_path.rglob("*")
    )


def _download_saes_once(model_name: str, layer_ranges: List[Tuple[int, int, int]],
                        sae_cache: str) -> None:
    """Fetch any missing SAE checkpoints to ``external/CRISP/crisp/<sae_cache>/``."""
    from external.CRISP.crisp.sae import JumpReLUSAE, TopkSae  # type: ignore

    all_layers = sorted({L for (lo, hi, st) in layer_ranges
                         for L in range(lo, hi + 1, st)})
    cache_dir = ROOT_DIR / "external" / "CRISP" / "crisp" / sae_cache
    cache_dir.mkdir(parents=True, exist_ok=True)

    # hf_hub_download writes to HF cache; use a writable dir if the shared
    # hub cache (e.g. /home/morg/dataset/models) is read-only on compute nodes.
    hub_cache = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        hub_path = Path(hub_cache)
        if not os.access(hub_path, os.W_OK):
            writable = Path.home() / "hf_cache"
            writable.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(writable)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(writable / "hub")

    is_llama = "llama" in model_name.lower()
    for layer in all_layers:
        layer_path = cache_dir / f"layer_{layer}"
        if _layer_cache_valid(layer_path):
            continue
        log.info("CRISP: downloading missing SAE for layer %d", layer)
        if is_llama:
            TopkSae.download_and_save(layer=layer, save_path=cache_dir)
        else:
            JumpReLUSAE.download_and_save(layer=layer, save_path=cache_dir)


def _resolve_sae_cache(model_name: str, configured: str) -> str:
    if "llama" in model_name.lower() and configured == "gemma_sae_cache":
        return "llama_sae_cache"
    return configured


class CRISPMethod(Method):
    """CRISP: SAE-based concept feature identification + LoRA unlearning."""

    name = "crisp"
    requires_full_reload = False

    def __init__(self) -> None:
        self._crisp: Any = None
        self._concept_for_crisp: Optional[str] = None
        self._range_for_crisp: Optional[Tuple[int, int, int]] = None
        self._embed_snap: Any = None
        self._forget: Optional[List[str]] = None
        self._retain: Optional[List[str]] = None
        self._coher: Optional[List[str]] = None
        self._saes_downloaded: bool = False

    # ------------------------------------------------------------------ #
    def enumerate_hps(self, common: RunConfig) -> Iterable[Dict[str, Any]]:
        cfg = common.crisp
        ranges = resolve_layer_ranges(cfg, common.model_name)
        for (lo, hi, step) in ranges:
            for k in cfg.k_features_grid:
                for alpha in cfg.alpha_grid:
                    for lr in cfg.lr_grid:
                        yield {
                            "k_features": int(k),
                            "alpha": float(alpha),
                            "lr": float(lr),
                            "layer_lo": int(lo),
                            "layer_hi": int(hi),
                            "layer_step": int(step),
                            "num_epochs": int(cfg.num_epochs),
                            "lora_rank": int(cfg.lora_rank),
                        }

    def hp_key_columns(self) -> List[str]:
        return ["delta_embed", "k_features", "alpha", "lr",
                "layer_lo", "layer_hi", "layer_step"]

    def hp_columns(self) -> List[str]:
        return io.EMBED_COLUMNS + io.CRISP_HP_COLUMNS

    # ------------------------------------------------------------------ #
    def on_concept_start(self, hf_model: Any, concept: str,
                         common: RunConfig) -> None:
        if not self._saes_downloaded:
            sae_cache = _resolve_sae_cache(common.model_name, common.crisp.sae_cache)
            layer_ranges = resolve_layer_ranges(common.crisp, common.model_name)
            _download_saes_once(common.model_name, layer_ranges, sae_cache)
            self._saes_downloaded = True

        max_len = common.crisp.max_len
        corpora = load_forget_retain(concept, common, max_len=max_len)
        self._forget, self._retain = corpora["forget"], corpora["retain"]
        self._coher = load_coherency_prompts(concept, common)

    def on_concept_end(self, hf_model: Any, concept: str,
                       common: RunConfig) -> None:
        self._free_crisp()
        self._forget = self._retain = self._coher = None

    # ------------------------------------------------------------------ #
    def snapshot(self, hf_model: Any) -> Any:
        return None

    def restore(self, hf_model: Any, snap: Any) -> None:
        return None

    # ------------------------------------------------------------------ #
    def apply(
            self,
            hf_model: Any,
            tokenizer: Any,
            hp: Dict[str, Any],
            concept: str,
            common: RunConfig,
    ) -> Dict[str, Any]:
        from external.CRISP.crisp.crisp import CRISP, CRISPConfig  # type: ignore
        from external.CRISP.crisp.unlearn import unlearn_lora, UnlearnConfig  # type: ignore

        layer_range = (int(hp["layer_lo"]), int(hp["layer_hi"]), int(hp["layer_step"]))
        layers = list(range(layer_range[0], layer_range[1] + 1, layer_range[2]))
        info: Dict[str, Any] = {}
        delta_embed = float(hp.get("delta_embed", 0.0))

        if (self._crisp is None
                or self._concept_for_crisp != concept
                or self._range_for_crisp != layer_range):
            self._free_crisp()
            log.info("CRISP: building wrapper for concept=%r range=%s", concept, layer_range)
            sae_cache = _resolve_sae_cache(common.model_name, common.crisp.sae_cache)
            os.environ.setdefault("CRISP_SAE_CACHE", sae_cache)
            self._crisp = CRISP(
                CRISPConfig(layers=layers, model_name=common.model_name, bf16=True),
            )
            base_model = self._crisp.model

            if delta_embed != 0.0:
                info.update(embed_edit.apply_concept_embed_edit_factored(
                    hf_model=base_model,
                    model_name=common.model_name,
                    concept_name=concept,
                    delta_embed=delta_embed,
                    rank=common.rank,
                    seed=common.seed,
                    ratio_thresh=common.selection.ratio_thresh,
                ))
            else:
                info.update({"delta_embed": 0.0, "k_features_embed": 0,
                             "n_tokens_edited": 0})
            self._embed_snap = embed_edit.snapshot(base_model)

            # data_config=None disables CRISP's processed-features cache so
            # crisp and crisp+EMBER don't collide on the same filename.
            self._crisp.process_multi_texts_batch(
                text_target=self._forget, text_benign=self._retain,
                data_config=None, batch_size=common.crisp.batch_size,
            )
            self._concept_for_crisp = concept
            self._range_for_crisp = layer_range
        else:
            self._crisp.unload_lora()
            embed_edit.restore(self._crisp.model, self._embed_snap)
            info.update({"delta_embed": delta_embed,
                         "k_features_embed": 0, "n_tokens_edited": 0})

        ucfg = UnlearnConfig(
            data_type="concept",
            learning_rate=float(hp["lr"]),
            num_epochs=int(hp["num_epochs"]),
            batch_size=common.crisp.lora_batch_size,
            k_features=int(hp["k_features"]),
            alpha=float(hp["alpha"]),
            beta=common.crisp.beta,
            gamma=common.crisp.gamma,
            lora_rank=int(hp["lora_rank"]),
            save_model=False,
            verbose=None,
            coherency_prompts=self._coher,
        )
        log.info("CRISP: unlearn_lora (k=%d alpha=%g lr=%g range=%s)",
                 hp["k_features"], hp["alpha"], hp["lr"], layer_range)
        with torch.enable_grad():
            unlearn_lora(crisp=self._crisp,
                         text_target=self._forget,
                         text_benign=self._retain,
                         config=ucfg,
                         data_config=None)
        return info

    # ------------------------------------------------------------------ #
    def get_model_to_eval(self) -> Optional[Any]:
        return self._crisp.model if self._crisp is not None else None

    def _free_crisp(self) -> None:
        if self._crisp is not None:
            try:
                self._crisp.unload_lora()
            except Exception:
                pass
            try:
                self._crisp.model.to("cpu")
            except Exception:
                pass
            del self._crisp
            self._crisp = None
            self._embed_snap = None
            self._concept_for_crisp = None
            self._range_for_crisp = None
            torch.cuda.empty_cache()


register(CRISPMethod())
