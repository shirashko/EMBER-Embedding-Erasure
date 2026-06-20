"""EMBER erasure method: factored embedding edit.

Subtracts the factorization's direct concept contribution from each eligible
token embedding::

    c_i = F'[:, k'] @ G'[i, k']   (concept contribution for token i)
    e_i_new = e_i − δ · c_i

where F'[:, k'] are the potential-feature columns of the Sparse MF embedding
factor F (shape d_model × K) and G'[i, k'] are the corresponding rows of
the per-token activation matrix G (shape |V'| × K).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from ember.erasure import embed_edit, features, io, log
from ember.erasure.config import RunConfig
from ember.erasure.methods.base import Method, register


class EMBERMethod(Method):
    """Erase concept by subtracting the factorization's concept contribution."""

    name = "ember"
    requires_full_reload = False

    def __init__(self) -> None:
        self._no_features: bool = False

    # ------------------------------------------------------------------ #
    # Concept lifecycle                                                   #
    # ------------------------------------------------------------------ #

    def on_concept_start(self, hf_model: Any, concept: str,
                         common: RunConfig) -> None:
        self._no_features = False
        try:
            _, pot_csv = features._embedding_paths(
                common.model_name, concept, common.rank, common.seed,
            )
            df = pd.read_csv(pot_csv)
            if df.empty:
                log.info("ember: no potential features for %r; skipping grid", concept)
                self._no_features = True
        except FileNotFoundError:
            log.info("ember: potential_features.csv missing for %r; skipping grid", concept)
            self._no_features = True

    def on_concept_end(self, hf_model: Any, concept: str,
                       common: RunConfig) -> None:
        self._no_features = False

    # ------------------------------------------------------------------ #
    # HP schema                                                           #
    # ------------------------------------------------------------------ #

    def enumerate_hps(self, common: RunConfig) -> Iterable[Dict[str, Any]]:
        if self._no_features:
            return
        for d in common.ember.deltas:
            yield {"delta_embed": float(d)}

    def hp_key_columns(self) -> List[str]:
        return ["delta_embed"]

    def hp_columns(self) -> List[str]:
        return io.EMBED_COLUMNS

    # ------------------------------------------------------------------ #
    # Apply / snapshot                                                    #
    # ------------------------------------------------------------------ #

    def apply(
            self,
            hf_model: Any,
            tokenizer: Any,
            hp: Dict[str, Any],
            concept: str,
            common: RunConfig,
    ) -> Dict[str, Any]:
        return embed_edit.apply_concept_embed_edit_factored(
            hf_model=hf_model,
            model_name=common.model_name,
            concept_name=concept,
            delta_embed=float(hp["delta_embed"]),
            rank=common.rank,
            seed=common.seed,
            ratio_thresh=common.selection.ratio_thresh,
        )

    def snapshot(self, hf_model: Any) -> Any:
        return embed_edit.snapshot(hf_model)

    def restore(self, hf_model: Any, snap: Any) -> None:
        if snap is not None:
            embed_edit.restore(hf_model, snap)

    # ------------------------------------------------------------------ #
    # Grid behavior                                                       #
    # ------------------------------------------------------------------ #

    def writes_topk_csv(self) -> bool:
        return False

    def skip_eval_for_hp(self, hp: Dict[str, Any]) -> bool:
        return float(hp.get("delta_embed", 0.0)) == 0.0

    def grid_eval_kwargs(self, common: RunConfig) -> Dict[str, Any]:
        return {
            "eval_alpaca": True,
            "min_mmlu": 0.90,
            "max_qa_acc": None,
            "min_alpaca": 0.9,
        }

    def pick_best_hp_row(self, hps_df: Any) -> Optional[Any]:
        if hps_df is None or hps_df.empty:
            return None
        return io.pick_best_embed_delta(hps_df)

    def after_concept_grid(self, concept: str, hp_csv_path: Path,
                           grid_out_dir: Path, common: RunConfig) -> None:
        """Append this concept's best-delta row to the shared ``best_embed.csv``."""
        cols = io.full_columns_for(self.name, common.train_eval, is_test=False)
        if self._no_features:
            zero_row = {
                "concept": concept,
                "delta_embed": 0.0,
                "k_features_embed": 0,
                "n_tokens_edited": 0,
                "efficacy": 0.0,
                "specificity": 0.0,
                "coherence": 0.0,
                "harmonic": 0.0,
                "harmonic_alpaca": 0.0,
            }
            io.append_csv_row(hp_csv_path, zero_row, columns=cols)
            io.append_csv_row(grid_out_dir / "best_embed.csv", zero_row, columns=cols)
            log.info("ember: no-features concept %r: wrote delta=0.0", concept)
            return

        df = io.read_csv_safe(hp_csv_path)
        if df.empty:
            return
        best = io.pick_best_embed_delta(df)
        io.append_csv_row(grid_out_dir / "best_embed.csv", best.to_dict(), columns=cols)


register(EMBERMethod())
