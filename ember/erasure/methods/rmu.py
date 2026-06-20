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


def _grids_and_settings(common: RunConfig
                        ) -> Tuple[List[float], List[float], List[float],
                                   List[UpdateSetting]]:
    cfg = common.rmu
    is_llama = "llama" in common.model_name.lower()
    if is_llama:
        lrs = cfg.lr_grid or [1e-5, 1e-4, 3e-4]
        alphas = cfg.alpha_grid or [30.0, 50.0, 100.0, 300.0]
        steerings = cfg.steering_grid or [30.0, 100.0, 300.0, 1000.0]
        settings = LLAMA_UPDATE_SETTINGS
    else:
        lrs = cfg.lr_grid or [1e-5, 1e-4, 3e-4]
        alphas = cfg.alpha_grid or [10.0, 30.0, 50.0, 100.0]
        steerings = cfg.steering_grid or [30.0, 100.0, 300.0, 1000.0]
        settings = GEMMA_UPDATE_SETTINGS
    return lrs, alphas, steerings, settings


def _build_rmu_data(concept_name: str, min_len: int, max_len: int, batch_size: int,
                    seed: int):
    """Returns batched forget/retain lists in the format run_rmu expects."""
    data = ConceptDataset(concept_name).as_forget_retain(seed=seed)
    forget, retain = data["forget"], data["retain"]
    if max_len and max_len > 0:
        forget = [s for s in forget if len(s) <= max_len]
        retain = [s for s in retain if len(s) <= max_len]

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
            concept, cfg.min_len, cfg.max_len, cfg.batch_size, int(common.seed),
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
