"""PISCES erasure method (feature editing via TransformerLens).

PISCES (https://github.com/yoavgur/PISCES) erases a concept by patching MLP
``W_out`` in SAE feature space, exposed as the ``unlearn_concept`` context
manager. To keep the rest of the pipeline HF-only, this method owns a
TransformerLens (TL) model internally (the same self-contained pattern RMU and
CRISP use for their aux state): per HP cell it edits the TL model's MLP, then
**syncs the edited ``W_out`` into the pipeline's HF model's ``down_proj``**.
The pipeline then evaluates the HF model normally via ``WrappedHFModel``
(single-BOS, batched) -- no pipeline/eval changes needed.

The optional embedding pre-step (``ember_step``) is applied directly to the HF
model via :mod:`embed_edit`. Per-concept feature specs live in
``data/pisces_concept_features_{gemma,llama}.json``; the grid sweeps ``k``
(sparsity threshold) x ``value`` (edit magnitude).
"""
from __future__ import annotations

import json
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from ember.erasure import embed_edit, io, log, mlp_edit
from ember.erasure.config import RunConfig
from ember.erasure.methods.base import Method, register

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "external" / "PISCES"))


def _features_file(model_name: str, override: Optional[str]) -> Path:
    """Pick the PISCES feature-defs JSON for ``model_name``."""
    if override:
        return Path(override)
    name = model_name.lower()
    if "gemma" in name:
        return ROOT_DIR / "data" / "pisces_concept_features_gemma.json"
    if "llama" in name:
        return ROOT_DIR / "data" / "pisces_concept_features_llama.json"
    raise ValueError(
        f"PISCES: no default features file for {model_name!r}; "
        f"set pisces.features_json in the config."
    )


def _load_features(features_path: Path,
                   concepts: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Parse the features JSON to ``{concept: feature_specs}``; error on missing."""
    if not features_path.exists():
        raise FileNotFoundError(f"PISCES features file not found: {features_path}")
    raw = json.loads(features_path.read_text(encoding="utf-8"))
    by_name = {str(c["name"]): list(c.get("features", []))
               for c in raw if isinstance(c, dict) and "name" in c}
    missing = [c for c in concepts if c not in by_name]
    if missing:
        raise ValueError(
            f"PISCES: concepts missing from {features_path.name}: {missing}. "
            f"Available: {sorted(by_name)}"
        )
    return {c: by_name[c] for c in concepts}


def _build_pisces_concept(concept_name: str, feature_specs: List[Dict[str, Any]],
                          k: float, value: float) -> Any:
    """Wrap ``feature_specs`` as a PISCES ``Concept`` (Feature 3rd arg = is_positive)."""
    from external.PISCES.editor import Concept, Feature  # type: ignore
    feats = [
        Feature(int(spec["layer"]), int(spec["id"]), not bool(spec.get("neg", False)))
        for spec in feature_specs
    ]
    return Concept(name=concept_name, k=float(k), value=float(value), features=feats)


def copy_tl_to_hf_weights(tl_model: Any, hf_model: Any) -> None:
    """Copy every MLP ``W_out`` from the TL model into the HF model's ``down_proj``.

    TL ``blocks[L].mlp.W_out`` is [d_mlp, d_model]; HF ``down_proj.weight`` is
    [d_model, d_mlp]; the relation is ``down_proj.weight == W_out.T``.
    """
    with torch.no_grad():
        for L in range(tl_model.cfg.n_layers):
            W_out = tl_model.blocks[L].mlp.W_out.data          # [d_mlp, d_model]
            proj = hf_model.model.layers[L].mlp.down_proj      # [d_model, d_mlp]
            proj.weight.data.copy_(
                W_out.T.to(device=proj.weight.device, dtype=proj.weight.dtype)
            )


class PISCESMethod(Method):
    """Feature-based concept editing via PISCES' ``unlearn_concept`` context."""

    name = "pisces"
    requires_full_reload = False

    def __init__(self) -> None:
        self._features: Dict[str, List[Dict[str, Any]]] = {}
        self._ctx_mgr: Optional[AbstractContextManager] = None  # active unlearn_concept
        self._tl_model: Optional[Any] = None
        self._tl_tokenizer: Optional[Any] = None

    # ------------------------------------------------------------------ #
    def enumerate_hps(self, common: RunConfig) -> Iterable[Dict[str, Any]]:
        cfg = common.pisces
        for v in cfg.values:
            for k in cfg.ks:
                yield {
                    "k_pisces": float(k),
                    "value_pisces": float(v),
                    "ratio_thresh": common.selection.ratio_thresh,
                }

    def hp_key_columns(self) -> List[str]:
        return ["delta_embed", "k_pisces", "value_pisces"]

    def hp_columns(self) -> List[str]:
        return io.EMBED_COLUMNS + io.PISCES_HP_COLUMNS

    # ------------------------------------------------------------------ #
    def on_concept_start(self, hf_model: Any, concept: str,
                         common: RunConfig) -> None:
        if not self._features:
            path = _features_file(common.model_name, common.pisces.features_json)
            log.info("PISCES: loading features from %s", path)
            self._features = _load_features(path, common.concepts)
        if self._tl_model is None:
            from ember.erasure.model_loader import load_tl_model, pick_dtype
            log.info("PISCES: loading TransformerLens model %s", common.model_name)
            self._tl_model, self._tl_tokenizer = load_tl_model(
                common.model_name, common.cache_dir, dtype=pick_dtype("bf16"),
            )

    def on_concept_end(self, hf_model: Any, concept: str,
                       common: RunConfig) -> None:
        self._exit_active_context()

    def before_relearning(self, hf_model: Any, concept: str,
                          common: RunConfig) -> None:
        # The TL edit is already synced to the HF model; release the TL model so
        # the relearning optimizer fits. Reloaded on the next concept's grid.
        self._free_tl_model()

    def _free_tl_model(self) -> None:
        self._exit_active_context()
        if self._tl_model is not None:
            del self._tl_model
            self._tl_model = None
            self._tl_tokenizer = None
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    def snapshot(self, hf_model: Any) -> Any:
        # Snapshot the HF model's embeddings + down_proj (what apply mutates).
        return (embed_edit.snapshot(hf_model), mlp_edit.snapshot_down_only(hf_model))

    def restore(self, hf_model: Any, snap: Any) -> None:
        self._exit_active_context()  # revert TL W_out (unlearn_concept teardown)
        if snap is not None:
            embed_snap, mlp_snap = snap
            embed_edit.restore(hf_model, embed_snap)
            mlp_edit.restore_down_only(hf_model, mlp_snap)

    def _exit_active_context(self) -> None:
        if self._ctx_mgr is not None:
            try:
                self._ctx_mgr.__exit__(None, None, None)
            finally:
                self._ctx_mgr = None

    # ------------------------------------------------------------------ #
    def apply(
            self,
            hf_model: Any,
            tokenizer: Any,
            hp: Dict[str, Any],
            concept: str,
            common: RunConfig,
    ) -> Dict[str, Any]:
        from external.PISCES.editor import unlearn_concept  # type: ignore

        self._exit_active_context()
        if self._tl_model is None or not self._features:
            self.on_concept_start(hf_model, concept, common)

        info: Dict[str, Any] = {}

        # Step 1: embedding edit on the HF model (only for the _ef variant).
        delta_embed = float(hp.get("delta_embed", 0.0))
        if delta_embed != 0.0:
            info.update(embed_edit.apply_concept_embed_edit_factored(
                hf_model=hf_model,
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

        # Step 2: PISCES block edit on the TL model, then sync W_out -> HF.
        # PISCES paper: linscale=True for Gemma, False for Llama.
        linscale = "gemma" in common.model_name.lower()
        pisces_concept = _build_pisces_concept(
            concept_name=concept,
            feature_specs=self._features[concept],
            k=float(hp["k_pisces"]),
            value=float(hp["value_pisces"]),
        )
        self._ctx_mgr = unlearn_concept(self._tl_model, pisces_concept, linscale=linscale)
        self._ctx_mgr.__enter__()
        copy_tl_to_hf_weights(self._tl_model, hf_model)

        info["k_pisces"] = float(hp["k_pisces"])
        info["value_pisces"] = float(hp["value_pisces"])
        info["ratio_thresh"] = common.selection.ratio_thresh
        return info


register(PISCESMethod())
