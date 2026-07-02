"""RMU (Representation Misdirection Unlearning) method plug-in.

A fine-tuning baseline from the WMDP paper. For each HP cell:

1. Apply the per-concept best embedding edit (if ``ember_step.enabled``).
2. Load a fresh HF model from disk (RMU mutates weights via gradient
   updates; cannot snapshot/restore cheaply).
3. Run RMU's ``external.wmdp.rmu.unlearn.run_rmu`` with this HP's
   ``lr`` / ``alpha`` / ``steering`` / layer settings on the
   concept's forget/retain split.
4. Hand the trained model to the pipeline for eval.

Sets :attr:`Method.requires_full_reload` so the pipeline reloads from
disk between cells.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from ember.erasure import embed_edit, io, log
from ember.erasure.config import RunConfig
from ember.erasure.methods.base import Method, register
from ember.erasure.model_loader import load_hf_model
from ember.local_datasets import ConceptDataset

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "external" / "wmdp"))
sys.path.append(str(ROOT_DIR / "external" / "wmdp" / "rmu"))

UpdateSetting = Tuple[str, int, List[int]]

GEMMA_UPDATE_SETTINGS: List[UpdateSetting] = [
    ("S1_lid7_L567",  7, [5, 6, 7]),
    ("S2_lid8_L678",  8, [6, 7, 8]),
    ("S3_lid6_L456",  6, [4, 5, 6]),
]
LLAMA_UPDATE_SETTINGS: List[UpdateSetting] = [
    ("S1_lid7_L567",    7,  [5, 6, 7]),
    ("S2_lid9_L789",    9,  [7, 8, 9]),
    ("S3_lid11_L91011", 11, [9, 10, 11]),
]
FIXED_PARAM_IDS: List[int] = [6]


def _allowed_update_settings(model_name: str) -> List[UpdateSetting]:
    return LLAMA_UPDATE_SETTINGS if "llama" in model_name.lower() else GEMMA_UPDATE_SETTINGS


def _setting_key(setting: UpdateSetting) -> Tuple[str, int, Tuple[int, ...]]:
    return (setting[0], int(setting[1]), tuple(int(x) for x in setting[2]))


def _normalize_update_setting(entry: Dict[str, Any] | UpdateSetting) -> UpdateSetting:
    if isinstance(entry, dict):
        return (
            str(entry["setting_name"]),
            int(entry["layer_id"]),
            [int(x) for x in str(entry["layer_ids"]).split(",")],
        )
    name, layer_id, layer_ids = entry
    return (str(name), int(layer_id), [int(x) for x in layer_ids])


def validate_update_settings(
    settings: List[Dict[str, Any]] | List[UpdateSetting],
    model_name: str,
) -> List[UpdateSetting]:
    """Ensure each RMU update setting matches a valid grid preset."""
    allowed = {_setting_key(s) for s in _allowed_update_settings(model_name)}
    normalized = [_normalize_update_setting(s) for s in settings]
    invalid = [s for s in normalized if _setting_key(s) not in allowed]
    if invalid:
        raise ValueError(
            f"Invalid RMU update_settings {invalid} for {model_name!r}; "
            f"allowed: {sorted(allowed)}"
        )
    return normalized


def resolve_update_settings(cfg: Any, model_name: str) -> List[UpdateSetting]:
    """Return configured update settings, or the model defaults when unset."""
    if cfg.update_settings is None:
        return _allowed_update_settings(model_name)
    return validate_update_settings(cfg.update_settings, model_name)


def _grids_and_settings(common: RunConfig
                        ) -> Tuple[List[float], List[float], List[float],
                                   List[UpdateSetting]]:
    cfg = common.rmu
    is_llama = "llama" in common.model_name.lower()
    if is_llama:
        lrs = cfg.lr_grid or [1e-5, 1e-4, 3e-4]
        alphas = cfg.alpha_grid or [30.0, 50.0, 100.0, 300.0]
        steerings = cfg.steering_grid or [30.0, 100.0, 300.0, 1000.0]
    else:
        lrs = cfg.lr_grid or [1e-5, 1e-4, 3e-4]
        alphas = cfg.alpha_grid or [10.0, 30.0, 50.0, 100.0]
        steerings = cfg.steering_grid or [30.0, 100.0, 300.0, 1000.0]
    settings = resolve_update_settings(cfg, common.model_name)
    return lrs, alphas, steerings, settings


def _build_rmu_data(
        concept_name: str,
        min_len: int,
        max_len: int,
        batch_size: int,
        *,
        seed: int,
) -> Tuple[List[List[List[str]]], List[List[List[str]]]]:
    """Return batched forget/retain lists in the format ``run_rmu`` expects."""
    data = ConceptDataset(concept_name).as_forget_retain(seed=seed)
    forget, retain = data["forget"], data["retain"]
    if max_len and max_len > 0:
        forget = [s for s in forget if len(s) <= max_len]
        retain = [s for s in retain if len(s) <= max_len]
    if min_len and min_len > 0:
        forget = [s for s in forget if len(s) >= min_len]
        retain = [s for s in retain if len(s) >= min_len]

    def _batchify(texts: List[str]) -> List[List[str]]:
        return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)
                if texts[i:i + batch_size]]

    return [_batchify(forget)], [_batchify(retain)]


class RMUMethod(Method):
    """Representation Misdirection Unlearning (HF, fine-tuning)."""

    name = "rmu"
    requires_full_reload = True

    def __init__(self) -> None:
        self._frozen_model: Any = None
        self._tokenizer: Any = None
        self._embed_snap: Any = None
        self._working_model: Any = None

    # ------------------------------------------------------------------ #
    def enumerate_hps(self, common: RunConfig) -> Iterable[Dict[str, Any]]:
        lrs, alphas, steerings, settings = _grids_and_settings(common)
        for lr in lrs:
            for alpha in alphas:
                for steering in steerings:
                    for setting_name, layer_id, layer_ids in settings:
                        yield {
                            "lr": float(lr),
                            "alpha": float(alpha),
                            "steering": float(steering),
                            "setting_name": setting_name,
                            "layer_id": int(layer_id),
                            "layer_ids": ",".join(map(str, layer_ids)),
                            "param_ids": ",".join(map(str, FIXED_PARAM_IDS)),
                        }

    def hp_key_columns(self) -> List[str]:
        return ["delta_embed", "lr", "alpha", "steering", "setting_name"]

    def hp_columns(self) -> List[str]:
        return io.EMBED_COLUMNS + io.RMU_HP_COLUMNS

    # ------------------------------------------------------------------ #
    def on_concept_start(self, hf_model: Any, concept: str,
                         common: RunConfig) -> None:
        if self._frozen_model is None:
            self._frozen_model = hf_model
        self._embed_snap = embed_edit.snapshot(self._frozen_model)

    def on_concept_end(self, hf_model: Any, concept: str,
                       common: RunConfig) -> None:
        if self._embed_snap is not None:
            embed_edit.restore(self._frozen_model, self._embed_snap)
        self._embed_snap = None
        self._free_working_model()
        self._frozen_model = None

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
        from external.wmdp.rmu.unlearn import run_rmu  # type: ignore

        cfg = common.rmu
        info: Dict[str, Any] = {}

        delta_embed = float(hp.get("delta_embed", 0.0))
        if delta_embed != 0.0:
            embed_edit.restore(self._frozen_model, self._embed_snap)
            embed_info = embed_edit.apply_concept_embed_edit_factored(
                hf_model=self._frozen_model,
                model_name=common.model_name,
                concept_name=concept,
                delta_embed=delta_embed,
                rank=common.rank,
                seed=common.seed,
                ratio_thresh=common.selection.ratio_thresh,
            )
            info.update(embed_info)
        else:
            info.update({"delta_embed": 0.0, "k_features_embed": 0,
                         "n_tokens_edited": 0})

        self._free_working_model()
        log.info("RMU: loading fresh working model for this cell")
        self._working_model, working_tokenizer = load_hf_model(
            common.model_name, common.cache_dir,
        )
        if delta_embed != 0.0:
            embed_edit.apply_concept_embed_edit_factored(
                hf_model=self._working_model,
                model_name=common.model_name,
                concept_name=concept,
                delta_embed=delta_embed,
                rank=common.rank,
                seed=common.seed,
                ratio_thresh=common.selection.ratio_thresh,
            )

        forget_data, retain_data = _build_rmu_data(
            concept,
            cfg.min_len,
            cfg.max_len,
            cfg.batch_size,
            seed=int(common.seed),
        )

        layer_ids = [int(x) for x in str(hp["layer_ids"]).split(",")]
        run_args = argparse.Namespace(
            layer_id=int(hp["layer_id"]),
            layer_ids=layer_ids,
            param_ids=FIXED_PARAM_IDS,
            alpha=[float(hp["alpha"])],
            steering_coeff_list=[float(hp["steering"])],
            lr=float(hp["lr"]),
            max_num_batches=int(cfg.max_num_batches),
            alpha_str=str(hp["alpha"]),
            steering_coeffs_str=str(hp["steering"]),
            setting_name=str(hp["setting_name"]),
            model_name_or_path=common.model_name,
            module_str="{model_name}.model.layers[{layer_id}]",
            seed=int(common.seed),
            verbose=False,
            output_dir=None,
        )
        log.info("RMU: run_rmu (lr=%g alpha=%g steering=%g setting=%s)",
                 hp["lr"], hp["alpha"], hp["steering"], hp["setting_name"])
        with torch.enable_grad():
            run_rmu(self._working_model, self._frozen_model, working_tokenizer,
                    forget_data, retain_data, run_args)

        return info

    # ------------------------------------------------------------------ #
    def get_model_to_eval(self) -> Optional[Any]:
        return self._working_model

    def _free_working_model(self) -> None:
        if self._working_model is not None:
            try:
                self._working_model.to("cpu")
            except Exception:
                pass
            del self._working_model
            self._working_model = None
            torch.cuda.empty_cache()


register(RMUMethod())
