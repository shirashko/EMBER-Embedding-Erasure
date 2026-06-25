"""The unified erasure pipeline.

Four stages per run:

    Step 0  Load HF model + compute baselines.
    Step 1  Resolve the EMBER embedding-edit delta per concept (load cache if
            present, run the EMBER grid if not). Skipped for method=ember.
    Step 2  Run the method's HP grid. One row per HP cell, top-K selected.
    Step 3  Validate top-K with strict eval (alpaca + tighter thresholds).
    Step 4  Final test: apply best HP, eval both test_mc AND test_open, write
            final_test_mc.csv and final_test_open.csv. Single relearning run
            (when enabled) populates relearning_qa_mc / relearning_qa_open.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ember.erasure import eval as eval_mod
from ember.erasure import checkpoints
from ember.erasure import io
from ember.erasure import log
from ember.erasure import model_loader
from ember.erasure import relearning as relearn_mod
from ember.erasure.config import RunConfig
from ember.erasure.methods import base as methods_base
from ember.timing import Timer

_ALPACA_BATCH_SIZE_DEFAULT = 50

# Per-concept wall-clock per stage, accumulated during one run() call and
# written to timing.json in the run root. Keyed concept -> stage -> seconds.
_STAGE_TIMES: Dict[str, Dict[str, float]] = {}
_TIMED_STAGES = ("grid", "validate", "final_test", "relearning")


def _t_add(concept: str, stage: str, secs: float) -> None:
    _STAGE_TIMES.setdefault(concept, {})
    _STAGE_TIMES[concept][stage] = _STAGE_TIMES[concept].get(stage, 0.0) + secs


def _write_timing_json(run_root: Path, cfg: RunConfig, method_name: str) -> None:
    """Write/merge per-concept per-stage wall times to <run_root>/timing.json."""
    if not _STAGE_TIMES:
        return
    path = run_root / "timing.json"
    per_concept: Dict[str, Dict[str, float]] = {}
    if path.exists():
        try:
            per_concept = json.loads(path.read_text()).get("per_concept", {})
        except Exception:
            per_concept = {}
    for c, d in _STAGE_TIMES.items():
        per_concept[c] = {k: round(v, 2) for k, v in d.items()}

    stage_totals = {s: round(sum(d.get(s, 0.0) for d in per_concept.values()), 2)
                    for s in _TIMED_STAGES}
    payload = {
        "model": cfg.model_name, "method": method_name,
        "rank": cfg.rank, "seed": cfg.seed,
        "per_concept": per_concept,
        "stage_totals_s": stage_totals,
        "total_s": round(sum(stage_totals.values()), 2),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    log.info("wrote timing.json -> %s (total %.1fs)", path, payload["total_s"])


# ========================================================================== #
# Entry point                                                                 #
# ========================================================================== #

def run(cfg: RunConfig) -> None:
    """Run the full pipeline for cfg."""
    log.set_context(method=cfg.method)
    log.info("starting pipeline: model=%s rank=%d train_eval=%s seed=%d concepts=%s",
             cfg.model_name, cfg.rank, cfg.train_eval, cfg.seed, cfg.concepts)

    _seed_everything(cfg.seed)
    _STAGE_TIMES.clear()
    method = methods_base.get(cfg.method)

    with log.stage("load_model"):
        hf_model, tokenizer = model_loader.load_hf_model(
            cfg.model_name, cfg.cache_dir,
            dtype=model_loader.pick_dtype("bf16"),
        )

    alpaca_bs = cfg.eval.alpaca_batch_size or _ALPACA_BATCH_SIZE_DEFAULT
    log.info("alpaca_batch_size=%d", alpaca_bs)
    if cfg.eval.skip_llm_judge:
        log.info("skip_llm_judge=True: skipping Alpaca, open-QA judging, and validate")

    baselines = _compute_all_baselines(cfg, hf_model, tokenizer, alpaca_bs)
    best_embed_map = _resolve_ember_step(cfg, hf_model, tokenizer, baselines, alpaca_bs)

    with log.stage("grid"):
        _run_method_grid(method, cfg, hf_model, tokenizer,
                         baselines, best_embed_map, alpaca_bs)

    with log.stage("validate"):
        _run_validate_topk(method, cfg, hf_model, tokenizer,
                           baselines, best_embed_map, alpaca_bs)

    if cfg.run_tests_after_train:
        with log.stage("final_test"):
            _run_final_test(method, cfg, hf_model, tokenizer,
                            baselines, best_embed_map, alpaca_bs)

    use_embed_suffix = (method.name != "ember") and bool(cfg.ember_step.enabled)
    _write_timing_json(
        io.run_root(method.name, cfg.model_name, cfg.rank, cfg.seed,
                    embed_step_enabled=use_embed_suffix),
        cfg, method.name)
    log.info("pipeline complete")


# ========================================================================== #
# Helpers                                                                     #
# ========================================================================== #

def _seed_everything(seed: int) -> None:
    import random as _r
    import numpy as _np
    torch.manual_seed(seed)
    _np.random.seed(seed)
    _r.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _identity_cols(cfg: RunConfig, concept: str, method_name: str) -> Dict[str, Any]:
    return {
        "model": cfg.model_name,
        "concept": concept,
        "method": method_name,
        "rank": cfg.rank,
        "seed": cfg.seed,
        "embed_step_enabled": bool(cfg.ember_step.enabled),
    }


def _hp_key(hp: Dict[str, Any], key_cols: List[str]) -> Tuple[Any, ...]:
    out: List[Any] = []
    for c in key_cols:
        v = hp[c]
        if any(c.startswith(p) for p in ("layer_", "k_features", "n_tokens_edited")):
            out.append(int(v))
        elif any(c.endswith(s) for s in ("_lo", "_hi", "_step", "_id")):
            out.append(int(v))
        elif c == "setting_name":
            out.append(str(v))
        else:
            out.append(float(v))
    return tuple(out)


# ========================================================================== #
# Step 0: baselines                                                           #
# ========================================================================== #

def _data_eval_kwargs(cfg: RunConfig) -> Dict[str, str]:
    return {
        "data_source": cfg.data.source,
        "wmdp_data_root": cfg.data.wmdp.data_root,
    }


def _is_wmdp(cfg: RunConfig) -> bool:
    return cfg.data.source.lower() == "wmdp"


def _test_modes(cfg: RunConfig) -> List[str]:
    """Final-test / test-baseline modes. WMDP has no open-QA eval."""
    modes = ["test_mc", "test_open"]
    if _is_wmdp(cfg):
        modes = [m for m in modes if not m.endswith("_open")]
    return modes


def _compute_all_baselines(cfg: RunConfig, hf_model: Any, tokenizer: Any,
                           alpaca_bs: int) -> Dict[str, Dict[str, Any]]:
    train_mode = f"train_{cfg.train_eval}"
    out: Dict[str, Dict[str, Any]] = {}

    with log.stage(f"baseline:{train_mode}"):
        out[train_mode] = eval_mod.get_baselines(
            hf_model, train_mode,
            model_name=cfg.model_name,
            tokenizer=tokenizer,
            alpaca_batch_size=alpaca_bs,
            required_concepts=cfg.concepts,
            skip_llm_judge=cfg.eval.skip_llm_judge,
            **_data_eval_kwargs(cfg),
        )

    if cfg.run_tests_after_train:
        for tm in _test_modes(cfg):
            with log.stage(f"baseline:{tm}"):
                out[tm] = eval_mod.get_baselines(
                    hf_model, tm,
                    model_name=cfg.model_name,
                    tokenizer=tokenizer,
                    alpaca_batch_size=alpaca_bs,
                    required_concepts=cfg.concepts,
                    skip_llm_judge=cfg.eval.skip_llm_judge,
                    **_data_eval_kwargs(cfg),
                )

    torch.cuda.empty_cache()
    return out


# ========================================================================== #
# Step 1: EMBER embedding-edit delta resolution                               #
# ========================================================================== #

def _ember_cache_path(cfg: RunConfig) -> Path:
    """Path to best-embed CSV: results/ember/.../train_<mode>/best_embed.csv."""
    return io.train_dir("ember", cfg.model_name, cfg.rank,
                        cfg.seed, cfg.train_eval) / "best_embed.csv"


def _resolve_ember_step(cfg: RunConfig, hf_model: Any, tokenizer: Any,
                        baselines: Dict[str, Dict[str, Any]],
                        alpaca_bs: int) -> Dict[str, float]:
    """Return {concept: best_delta_embed} for the configured concepts.

    Loading order:
        1. method == "ember" -> this IS the grid; step 1 is a no-op.
        2. ember_step.enabled == False -> {concept: 0.0}.
        3. Cache CSV covers every concept -> load it.
        4. Otherwise run the EMBER grid for missing concepts, then load.
    """
    if cfg.method == "ember":
        return {c: 0.0 for c in cfg.concepts}
    if not cfg.ember_step.enabled:
        log.info("ember_step disabled; skipping step 1")
        return {c: 0.0 for c in cfg.concepts}

    cache_path = _ember_cache_path(cfg)
    cached = io.load_best_embed_map(cache_path)
    missing = [c for c in cfg.concepts if c not in cached]
    if not missing:
        log.info("ember cache hit at %s (%d concepts)", cache_path, len(cached))
        return {c: cached[c] for c in cfg.concepts}

    log.info("ember cache misses for %d concepts: %s", len(missing), missing)
    with log.stage("ember_grid"):
        ember_method = methods_base.get("ember")
        sub_cfg = _override_concepts(cfg, missing)
        _run_method_grid(
            ember_method, sub_cfg, hf_model, tokenizer, baselines,
            best_embed_map={c: 0.0 for c in missing},
            alpaca_bs=alpaca_bs,
        )

    cached = io.load_best_embed_map(cache_path)
    missing_after = [c for c in cfg.concepts if c not in cached]
    if missing_after:
        log.warning("ember grid did not produce a best row for %s; "
                    "defaulting delta_embed to 0.0", missing_after)
    return {c: cached.get(c, 0.0) for c in cfg.concepts}


def _override_concepts(cfg: RunConfig, concepts: List[str]) -> RunConfig:
    out = copy.copy(cfg)
    out.concepts = list(concepts)
    return out


# ========================================================================== #
# Step 2: method grid                                                         #
# ========================================================================== #

def _run_method_grid(method: methods_base.Method, cfg: RunConfig,
                     hf_model: Any, tokenizer: Any,
                     baselines: Dict[str, Dict[str, Any]],
                     best_embed_map: Dict[str, float],
                     alpaca_bs: int) -> None:
    train_mode = f"train_{cfg.train_eval}"
    use_embed_suffix = (method.name != "ember") and bool(cfg.ember_step.enabled)
    out_dir = io.train_dir(method.name, cfg.model_name, cfg.rank,
                           cfg.seed, cfg.train_eval,
                           embed_step_enabled=use_embed_suffix)

    method_is_ember = method.name == "ember"
    snap = None if method.requires_full_reload else method.snapshot(hf_model)

    for concept in cfg.concepts:
        with log.concept(concept):
            cdir = io.concept_dir(out_dir, concept)
            hp_csv = cdir / "hps.csv"
            log.info("method grid -> %s", hp_csv)

            method.on_concept_start(hf_model, concept, cfg)
            done = (io.load_done_set(hp_csv, method.hp_key_columns())
                    if not cfg.overwrite else set())

            for hp in method.enumerate_hps(cfg):
                if not method_is_ember and "delta_embed" not in hp:
                    hp = {**hp, "delta_embed": float(best_embed_map.get(concept, 0.0))}

                key = _hp_key(hp, method.hp_key_columns())
                if key in done:
                    continue

                if not method.requires_full_reload:
                    method.restore(hf_model, snap)

                with Timer() as _t:
                    info = method.apply(hf_model, tokenizer, hp, concept, cfg)
                    eval_model = method.get_model_to_eval() or hf_model

                    if method.skip_eval_for_hp(hp):
                        log.info("skipping eval (no-op HP); writing zero metrics row")
                        metrics = {"efficacy": 0.0, "specificity": 0.0,
                                   "coherence": 0.0, "harmonic": 0.0,
                                   "harmonic_alpaca": 0.0}
                    else:
                        metrics, _ = eval_mod.evaluate_model(
                            eval_model,
                            baselines=baselines[train_mode],
                            concept_name=concept,
                            mode=train_mode,
                            tokenizer=tokenizer,
                            alpaca_batch_size=alpaca_bs,
                            skip_llm_judge=cfg.eval.skip_llm_judge,
                            **_data_eval_kwargs(cfg),
                            **method.grid_eval_kwargs(cfg),
                        )
                _t_add(concept, "grid", _t["elapsed"])

                row = {**_identity_cols(cfg, concept, method.name),
                       **hp, **info, **metrics, "wall_time_s": round(_t["elapsed"], 3)}
                io.append_csv_row(hp_csv, row,
                                  columns=io.full_columns_for(
                                      method.name, cfg.train_eval, is_test=False))
                log.info("wrote row -> %s harmonic=%.4f",
                         hp_csv, metrics.get("harmonic", float("nan")))

            method.restore(hf_model, snap)
            method.on_concept_end(hf_model, concept, cfg)
            if method.writes_topk_csv():
                io.write_topk(hp_csv, cdir / "top_hps.csv", concept, k=cfg.topk)
                log.info("wrote top-%d to %s", cfg.topk, cdir / "top_hps.csv")
            method.after_concept_grid(concept, hp_csv, out_dir, cfg)

    if not method.requires_full_reload:
        method.restore(hf_model, snap)


# ========================================================================== #
# Step 3: validate top-K                                                      #
# ========================================================================== #

def _run_validate_topk(method: methods_base.Method, cfg: RunConfig,
                       hf_model: Any, tokenizer: Any,
                       baselines: Dict[str, Dict[str, Any]],
                       best_embed_map: Dict[str, float],
                       alpaca_bs: int) -> None:
    if not method.writes_topk_csv():
        log.info("method %s does not write top-K; skipping validate", method.name)
        return

    if cfg.eval.skip_llm_judge:
        log.info("skip_llm_judge: skipping validate (Gemini Alpaca re-score)")
        return

    train_mode = f"train_{cfg.train_eval}"
    use_embed_suffix = (method.name != "ember") and bool(cfg.ember_step.enabled)
    out_dir = io.train_dir(method.name, cfg.model_name, cfg.rank,
                           cfg.seed, cfg.train_eval,
                           embed_step_enabled=use_embed_suffix)

    snap = None if method.requires_full_reload else method.snapshot(hf_model)

    for concept in cfg.concepts:
        with log.concept(concept):
            cdir = io.concept_dir(out_dir, concept)
            top_csv = cdir / "top_hps.csv"
            valid_csv = cdir / "top_hps_valid.csv"
            if not top_csv.exists():
                log.warning("no top_hps.csv for concept; skipping validate")
                continue

            top_df = io.read_csv_safe(top_csv)
            if top_df.empty:
                log.info("top_hps.csv empty; skipping validate")
                continue

            done = (io.load_done_set(valid_csv, method.hp_key_columns())
                    if not cfg.overwrite else set())
            log.info("validate -> %s (%d top rows, %d already done)",
                     valid_csv, len(top_df), len(done))

            method.on_concept_start(hf_model, concept, cfg)

            for _, top_row in top_df.iterrows():
                hp: Dict[str, Any] = {c: top_row[c] for c in method.hp_columns()
                                      if c in top_row.index}
                if "delta_embed" not in hp and "delta_embed" in top_row.index:
                    hp["delta_embed"] = float(top_row["delta_embed"])

                key = _hp_key(hp, method.hp_key_columns())
                if key in done:
                    continue

                if not method.requires_full_reload:
                    method.restore(hf_model, snap)

                with Timer() as _t:
                    info = method.apply(hf_model, tokenizer, hp, concept, cfg)
                    eval_model = method.get_model_to_eval() or hf_model

                    if method.skip_eval_for_hp(hp):
                        log.info("skipping validate eval (no-op HP); writing zeros")
                        metrics = {"efficacy": 0.0, "specificity": 0.0,
                                   "coherence": 0.0, "harmonic": 0.0,
                                   "harmonic_alpaca": 0.0}
                    else:
                        precomputed = {k: top_row[k] for k in
                                       ("mmlu_acc", "mmlu_frac", "mmlu_invalid",
                                        "qa_acc", "qa_frac", "qa_invalid",
                                        "simdom_acc", "simdom_frac", "simdom_invalid")
                                       if k in top_row.index}
                        metrics, _ = eval_mod.evaluate_model(
                            eval_model,
                            baselines=baselines[train_mode],
                            concept_name=concept,
                            mode=train_mode,
                            eval_alpaca=True,
                            eval_mmlu=False,
                            eval_qa=False,
                            eval_simdom=False,
                            precomputed_metrics=precomputed,
                            min_mmlu=None,
                            max_qa_acc=None,
                            tokenizer=tokenizer,
                            alpaca_batch_size=alpaca_bs,
                            skip_llm_judge=cfg.eval.skip_llm_judge,
                            **_data_eval_kwargs(cfg),
                        )
                _t_add(concept, "validate", _t["elapsed"])

                row = {**_identity_cols(cfg, concept, method.name),
                       **hp, **info, **metrics, "wall_time_s": round(_t["elapsed"], 3)}
                io.append_csv_row(valid_csv, row,
                                  columns=io.validate_columns_for(
                                      method.name, cfg.train_eval))
                log.info("validate row -> %s harmonic_alpaca=%.4f",
                         valid_csv, metrics.get("harmonic_alpaca", float("nan")))

            method.restore(hf_model, snap)
            method.on_concept_end(hf_model, concept, cfg)

    if not method.requires_full_reload:
        method.restore(hf_model, snap)


# ========================================================================== #
# Step 4: final test                                                          #
# ========================================================================== #

def _pick_best_hp_for_test(
        method: methods_base.Method, hp_csv: Path,
        valid_csv: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    if valid_csv is not None and valid_csv.exists():
        vdf = io.read_csv_safe(valid_csv)
        if not vdf.empty and "harmonic_alpaca" in vdf.columns:
            top = io.topk_per_concept(vdf, k=1, primary="harmonic_alpaca")
            if not top.empty:
                return top.iloc[0].to_dict()
    df = io.read_csv_safe(hp_csv)
    if df.empty:
        return None
    row = method.pick_best_hp_row(df)
    return None if row is None else row.to_dict()



def _final_test_row_complete(row: Any, *, is_mc: bool, skip_llm_judge: bool) -> bool:
    import pandas as pd

    if row is None:
        return False
    cols = ["mmlu_frac"]
    if is_mc:
        cols.append("qa_acc")
    if not skip_llm_judge:
        cols.extend(["harmonic_alpaca", "alp_instr_frac"])
    for col in cols:
        if col not in row or pd.isna(row.get(col, float("nan"))):
            return False
    return True


def _classify_concept_state(
        concept: str,
        df_mc: Any, df_open: Any,
        relearning_enabled: bool,
        *,
        skip_llm_judge: bool = False,
) -> str:
    """Return "done", "partial", or "fresh" for resume logic."""
    import pandas as pd

    def _row(df, c):
        if df is None or df.empty or "concept" not in df.columns:
            return None
        m = df["concept"].astype(str) == str(c)
        if not m.any():
            return None
        return df[m].iloc[-1]

    row_mc = _row(df_mc, concept)
    row_open = _row(df_open, concept)
    if not _final_test_row_complete(row_mc, is_mc=True, skip_llm_judge=skip_llm_judge):
        return "fresh"
    if not _final_test_row_complete(row_open, is_mc=False, skip_llm_judge=skip_llm_judge):
        return "fresh"

    if not relearning_enabled:
        return "done"

    rl_mc = row_mc.get("relearning_qa_mc", float("nan"))
    rl_open = row_open.get("relearning_qa_open", float("nan"))
    if pd.isna(rl_mc) or pd.isna(rl_open):
        return "partial"
    return "done"


def _drop_concept_rows(csv_path: Path, concept: str) -> None:
    if not csv_path.exists():
        return
    df = io.read_csv_safe(csv_path)
    if df.empty or "concept" not in df.columns:
        return
    df[df["concept"].astype(str) != str(concept)].to_csv(csv_path, index=False)


def _run_final_test(method: methods_base.Method, cfg: RunConfig,
                    hf_model: Any, tokenizer: Any,
                    baselines: Dict[str, Dict[str, Any]],
                    best_embed_map: Dict[str, float],
                    alpaca_bs: int) -> None:
    use_embed_suffix = (method.name != "ember") and bool(cfg.ember_step.enabled)
    train_out_dir = io.train_dir(method.name, cfg.model_name, cfg.rank,
                                 cfg.seed, cfg.train_eval,
                                 embed_step_enabled=use_embed_suffix)
    test_out_dir = io.test_dir(method.name, cfg.model_name, cfg.rank,
                               cfg.seed, cfg.train_eval,
                               embed_step_enabled=use_embed_suffix)
    test_out_dir.mkdir(parents=True, exist_ok=True)

    final_mc = test_out_dir / "final_test_mc.csv"
    final_open = test_out_dir / "final_test_open.csv"

    snap = None if method.requires_full_reload else method.snapshot(hf_model)

    for concept in cfg.concepts:
        with log.concept(concept):
            df_mc = io.read_csv_safe(final_mc)
            df_open = io.read_csv_safe(final_open)
            state = (_classify_concept_state(concept, df_mc, df_open,
                                              cfg.relearning.enabled,
                                              skip_llm_judge=cfg.eval.skip_llm_judge)
                     if not cfg.overwrite else "fresh")
            log.info("concept state = %s", state)

            if state == "done":
                log.info("final test already complete; skip")
                continue

            cdir_train = io.concept_dir(train_out_dir, concept)
            best_hp = _pick_best_hp_for_test(
                method, cdir_train / "hps.csv", cdir_train / "top_hps_valid.csv"
            )
            if best_hp is None:
                log.warning("no train rows for concept; cannot run final test")
                continue

            if not method.requires_full_reload:
                method.restore(hf_model, snap)
            method.on_concept_start(hf_model, concept, cfg)

            apply_hp = {c: best_hp[c] for c in method.hp_columns() if c in best_hp}
            if "delta_embed" not in apply_hp and method.name != "ember":
                apply_hp["delta_embed"] = float(best_embed_map.get(concept, 0.0))
            with Timer() as _ta:
                info = method.apply(hf_model, tokenizer, apply_hp, concept, cfg)
            eval_model = method.get_model_to_eval() or hf_model
            skip_eval = method.skip_eval_for_hp(apply_hp)
            ft_secs = _ta["elapsed"]

            checkpoints.save_unlearned_checkpoint(
                eval_model, tokenizer, cfg, concept, hyperparameters=apply_hp,
            )

            if state == "fresh":
                _drop_concept_rows(final_mc, concept)
                _drop_concept_rows(final_open, concept)

                test_csv = {"test_mc": final_mc, "test_open": final_open}
                for mode in _test_modes(cfg):
                    csv_path = test_csv[mode]
                    with Timer() as _te:
                        if skip_eval:
                            log.info("skipping %s eval (no-op HP); writing zeros", mode)
                            metrics = {"efficacy": 0.0, "specificity": 0.0,
                                       "coherence": 0.0, "harmonic": 0.0,
                                       "harmonic_alpaca": 0.0}
                        else:
                            log.info("eval %s", mode)
                            metrics, _ = eval_mod.evaluate_model(
                                eval_model,
                                baselines=baselines[mode],
                                concept_name=concept,
                                mode=mode,
                                eval_alpaca=not cfg.eval.skip_llm_judge,
                                min_mmlu=None,
                                max_qa_acc=None,
                            tokenizer=tokenizer,
                            alpaca_batch_size=alpaca_bs,
                            skip_llm_judge=cfg.eval.skip_llm_judge,
                            **_data_eval_kwargs(cfg),
                        )
                    ft_secs += _te["elapsed"]
                    relearn_col = ("relearning_qa_mc" if mode == "test_mc"
                                   else "relearning_qa_open")
                    test_eval = "mc" if mode == "test_mc" else "open"
                    row = {**_identity_cols(cfg, concept, method.name),
                           **apply_hp, **info, **metrics, relearn_col: float("nan"),
                           "wall_time_s": round(_te["elapsed"], 3)}
                    io.append_csv_row(csv_path, row,
                                      columns=io.full_columns_for(
                                          method.name, test_eval, is_test=True))
                    log.info("wrote final-test row to %s", csv_path)

            _t_add(concept, "final_test", ft_secs)

            if cfg.relearning.enabled and not skip_eval:
                method.before_relearning(hf_model, concept, cfg)
                with Timer() as _tr:
                    _run_relearning_and_update(
                        model_to_relearn=eval_model,
                        method=method, cfg=cfg, concept=concept,
                        hf_model=hf_model, tokenizer=tokenizer,
                        baselines=baselines, alpaca_bs=alpaca_bs,
                        final_mc=final_mc, final_open=final_open,
                        test_out_dir=test_out_dir,
                    )
                _t_add(concept, "relearning", _tr["elapsed"])

            method.on_concept_end(hf_model, concept, cfg)
            method.restore(hf_model, snap)
            torch.cuda.empty_cache()


# ========================================================================== #
# Relearning helper                                                           #
# ========================================================================== #

def _run_relearning_and_update(
        *,
        method: methods_base.Method,
        cfg: RunConfig,
        concept: str,
        hf_model: Any,
        model_to_relearn: Any,
        tokenizer: Any,
        baselines: Dict[str, Dict[str, Any]],
        alpaca_bs: int,
        final_mc: Path,
        final_open: Path,
        test_out_dir: Path,
) -> None:
    paragraphs_map = relearn_mod.load_relearn_paragraphs(
        Path(cfg.relearning.csv_path).expanduser()
    )
    paras = paragraphs_map.get(concept)
    if not paras:
        log.warning("no relearning paragraphs for %r; skipping relearning", concept)
        return

    log.info("relearning: %d paragraphs (cap=%d)", len(paras), cfg.relearning.max_paragraphs)

    def _epoch_eval(model, tok) -> Dict[str, Any]:
        mode = f"test_{cfg.train_eval}"
        metrics, _ = eval_mod.evaluate_model(
            model, baselines=baselines[mode],
            concept_name=concept, mode=mode,
            eval_alpaca=False, eval_mmlu=False,
            min_mmlu=None, max_qa_acc=None,
            tokenizer=tok, alpaca_batch_size=alpaca_bs,
            skip_llm_judge=cfg.eval.skip_llm_judge,
            **_data_eval_kwargs(cfg),
        )
        return metrics

    relearn_csv = io.concept_dir(test_out_dir, concept) / "relearning.csv"
    relearn_mod.run_relearning(
        model_to_relearn, tokenizer,
        paragraphs=paras,
        model_name=cfg.model_name,
        concept=concept,
        eval_fn=_epoch_eval,
        relearn_csv_path=relearn_csv,
        max_paragraphs=cfg.relearning.max_paragraphs,
    )

    relearned_qa: Dict[str, float] = {}
    for mode in _test_modes(cfg):
        log.info("relearning post-eval: %s", mode)
        metrics, _ = eval_mod.evaluate_model(
            model_to_relearn, baselines=baselines[mode],
            concept_name=concept, mode=mode,
            eval_alpaca=False, eval_mmlu=False,
            min_mmlu=None, max_qa_acc=None,
            tokenizer=tokenizer, alpaca_batch_size=alpaca_bs,
            skip_llm_judge=cfg.eval.skip_llm_judge,
            **_data_eval_kwargs(cfg),
        )
        relearned_qa[mode] = float(metrics.get("qa_acc", float("nan")))

    relearn_cols = {
        "test_mc": (final_mc, "relearning_qa_mc"),
        "test_open": (final_open, "relearning_qa_open"),
    }
    for mode_key in _test_modes(cfg):
        csv_path, col = relearn_cols[mode_key]
        df = io.read_csv_safe(csv_path)
        if df.empty or "concept" not in df.columns:
            continue
        mask = df["concept"].astype(str) == str(concept)
        if not mask.any():
            continue
        df.loc[mask, col] = relearned_qa[mode_key]
        df.to_csv(csv_path, index=False)
        log.info("patched %s with %s=%.4f", csv_path.name, col, relearned_qa[mode_key])


__all__ = ["run"]
