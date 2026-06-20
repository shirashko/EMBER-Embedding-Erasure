"""Output paths, CSV helpers, and the unified column schema.

The directory layout for every method is::

    results/<method[_ef]>/<model_safe>/rank<R>/seed<S>/
        train_<mode>/
            best_embed.csv              # best EMBER delta per concept
            <concept_safe>/
                hps.csv                 # one row per HP cell visited
                top_hps.csv             # top-K rows by harmonic
                top_hps_valid.csv       # top-K re-evaluated with full alpaca
        test_<mode>/
            final_test_mc.csv           # rolled-up across concepts
            final_test_open.csv         # rolled-up across concepts
            <concept_safe>/
                relearning.csv          # per-epoch relearning trace
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ember.utils import _safe_concept, _safe_model_name

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = ROOT_DIR / "data"


# ========================================================================== #
# Path helpers                                                                #
# ========================================================================== #

def method_dir_name(method: str, embed_step_enabled: bool) -> str:
    """Return the method subdir name.

    When ``embed_step_enabled`` is True and method is not "ember", appends
    ``_ef`` to signal EMBER was used as a pre-step.
    """
    name = method.lower()
    if name == "ember":
        return name
    if not embed_step_enabled:
        return name
    return f"{name}_ef"


def run_root(method: str, model_name: str, rank: int, seed: int,
             embed_step_enabled: bool = False) -> Path:
    """``results/<method[_ef]>/<model_safe>/rank<R>/seed<S>``."""
    return (RESULTS_DIR / method_dir_name(method, embed_step_enabled)
            / _safe_model_name(model_name)
            / f"rank{rank}"
            / f"seed{seed}")


def train_dir(method: str, model_name: str, rank: int, seed: int,
              train_eval: str, embed_step_enabled: bool = False) -> Path:
    """Directory holding train-time grid/validate outputs."""
    return run_root(method, model_name, rank, seed, embed_step_enabled) / f"train_{train_eval}"


def test_dir(method: str, model_name: str, rank: int, seed: int,
             train_eval: str, embed_step_enabled: bool = False) -> Path:
    """Directory holding final-test outputs."""
    return run_root(method, model_name, rank, seed, embed_step_enabled) / f"test_{train_eval}"


def concept_dir(parent: Path, concept: str) -> Path:
    return parent / _safe_concept(concept)


def baseline_dir(model_name: str) -> Path:
    return DATA_DIR / "baselines" / _safe_model_name(model_name)


# ========================================================================== #
# CSV I/O                                                                     #
# ========================================================================== #

def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read CSV or return an empty DataFrame. Never raises on missing/corrupt."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[io] WARN: failed to read {path}: {type(e).__name__}: {e}")
        return pd.DataFrame()


def append_csv_row(path: Path, row: Dict[str, Any],
                   columns: Optional[Sequence[str]] = None) -> None:
    """Append one row to ``path``, creating parent dirs + header on first write.

    If ``columns`` is given, the row is re-indexed to that column order; cells
    missing from ``row`` become NaN.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if columns is not None:
        df = df.reindex(columns=list(columns))
    df.to_csv(path,
              mode="a" if path.exists() else "w",
              header=not path.exists(),
              index=False)


def save_json_atomic(path: Path, obj: Any) -> None:
    """Atomic JSON write via tmp-file + rename. Survives mid-write crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ========================================================================== #
# Done-set tracking (resume support)                                          #
# ========================================================================== #

_INT_COL_PREFIXES = ("layer_", "k_features", "n_tokens_edited")
_INT_COL_SUFFIXES = ("_lo", "_hi", "_step", "_id")


def _cast_key_cell(col: str, value: Any) -> Any:
    """Cast a key cell to the right type so set lookups work after CSV round-trip."""
    if any(col.startswith(p) for p in _INT_COL_PREFIXES):
        return int(value)
    if any(col.endswith(s) for s in _INT_COL_SUFFIXES):
        return int(value)
    if col == "setting_name":
        return str(value)
    return float(value)


def load_done_set(hp_csv: Path, key_cols: Sequence[str]) -> set:
    """Return the set of HP-tuples already present in ``hp_csv``.

    Used to skip already-evaluated grid cells when resuming a partial run.
    """
    if not hp_csv.exists():
        return set()
    df = read_csv_safe(hp_csv)
    if df.empty:
        return set()
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{hp_csv} missing key columns {missing}; has {list(df.columns)}")

    done: set = set()
    for _, r in df.iterrows():
        tpl: List[Any] = []
        valid = True
        for c in key_cols:
            v = r[c]
            if pd.isna(v):
                valid = False
                break
            tpl.append(_cast_key_cell(c, v))
        if valid:
            done.add(tuple(tpl))
    return done


# ========================================================================== #
# Selection helpers                                                           #
# ========================================================================== #

_TIEBREAK_COLS = [
    "delta_embed",
    "delta_in", "layer_lo_in", "layer_hi_in",
    "delta_out", "layer_lo_out", "layer_hi_out",
    "k_features", "alpha", "lr",
    "layer_lo", "layer_hi", "layer_step",
]


def topk_per_concept(df: pd.DataFrame, k: int,
                     primary: str = "harmonic") -> pd.DataFrame:
    """Return top-K rows sorted by ``primary`` descending, with stable tie-breakers."""
    if df.empty:
        return df
    if primary not in df.columns:
        raise ValueError(f"DataFrame missing primary column {primary!r}")
    tie_cols = [c for c in _TIEBREAK_COLS if c in df.columns]
    sort_cols = [primary] + tie_cols
    ascending = [False] + [True] * len(tie_cols)
    return df.sort_values(by=sort_cols, ascending=ascending, kind="mergesort").head(k)


def write_topk(hp_csv: Path, out_path: Path, concept: str, k: int = 20) -> None:
    """Write the top-K rows for ``concept`` from ``hp_csv`` to ``out_path``."""
    df = read_csv_safe(hp_csv)
    if df.empty:
        return
    if "concept" in df.columns:
        df = df[df["concept"].astype(str) == str(concept)].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    topk_per_concept(df, k=k).to_csv(out_path, index=False)


def pick_best_embed_delta(df: pd.DataFrame) -> pd.Series:
    """Return the single best EMBER-grid row for one concept.

    Primary key: ``harmonic_alpaca`` (descending). Ties broken by smaller
    ``delta_embed``. If all rows have zero harmonic_alpaca, returns the row
    with the smallest delta_embed and forces delta_embed to 0.0.
    """
    work = df.copy()
    work["delta_embed"] = pd.to_numeric(work["delta_embed"], errors="coerce")
    work = work.dropna(subset=["delta_embed"])
    if work.empty:
        raise ValueError("pick_best_embed_delta: no rows with a valid delta_embed")

    if (work["harmonic_alpaca"].fillna(0.0) == 0.0).all():
        row = work.sort_values(by="delta_embed", kind="mergesort").iloc[0].copy()
        row["delta_embed"] = 0.0
        return row

    return work.sort_values(by=["harmonic_alpaca", "delta_embed"],
                            ascending=[False, True],
                            kind="mergesort").iloc[0]


def load_best_embed_map(best_embed_csv: Path) -> Dict[str, float]:
    """Return ``{concept: best_delta_embed}`` from a saved best-embed CSV."""
    df = read_csv_safe(best_embed_csv)
    if df.empty or "concept" not in df.columns or "delta_embed" not in df.columns:
        return {}
    return {str(r["concept"]): float(r["delta_embed"]) for _, r in df.iterrows()}


# ========================================================================== #
# Unified column schema                                                       #
# ========================================================================== #

ID_COLUMNS: List[str] = [
    "model", "concept", "method", "rank", "seed", "embed_step_enabled",
]

EMBED_COLUMNS: List[str] = [
    "delta_embed", "k_features_embed", "n_tokens_edited",
]

SNMF_HP_COLUMNS: List[str] = [
    "w_mode", "feature_source",
    "ratio_thresh", "coverage_thresh", "neurons_thresh",
    "delta_in", "layer_lo_in", "layer_hi_in", "k_features_mlp_in",
    "delta_out", "layer_lo_out", "layer_hi_out", "k_features_mlp_out",
]

RMU_HP_COLUMNS: List[str] = [
    "lr", "alpha", "steering",
    "setting_name", "layer_id", "layer_ids", "param_ids",
]

CRISP_HP_COLUMNS: List[str] = [
    "k_features", "alpha", "lr",
    "layer_lo", "layer_hi", "layer_step",
    "num_epochs", "lora_rank",
]

PISCES_HP_COLUMNS: List[str] = [
    "k_pisces", "value_pisces", "ratio_thresh",
]

TRAIN_OPEN_METRICS: List[str] = [
    "mmlu_acc", "mmlu_frac", "mmlu_invalid",
    "qa_acc", "qa_frac",
    "simdom_acc", "simdom_frac",
    "efficacy", "specificity", "harmonic",
]

TRAIN_MC_METRICS: List[str] = [
    "mmlu_acc", "mmlu_frac", "mmlu_invalid",
    "qa_acc", "qa_frac", "qa_invalid",
    "simdom_acc", "simdom_frac", "simdom_invalid",
    "efficacy", "specificity", "harmonic",
]

ALPACA_METRICS: List[str] = [
    "alpaca_instr", "alp_instr_frac",
    "alpaca_flu", "alp_flu_frac",
    "coherence", "harmonic_alpaca",
]

TEST_OPEN_METRICS: List[str] = TRAIN_OPEN_METRICS + ALPACA_METRICS + ["relearning_qa_open"]
TEST_MC_METRICS:   List[str] = TRAIN_MC_METRICS   + ALPACA_METRICS + ["relearning_qa_mc"]


def metric_keys_for_mode(mode: str) -> List[str]:
    return {
        "train_open": TRAIN_OPEN_METRICS,
        "train_mc":   TRAIN_MC_METRICS,
        "test_open":  TEST_OPEN_METRICS,
        "test_mc":    TEST_MC_METRICS,
    }[mode]


def normalize_metrics_for_mode(mode: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Reindex ``metrics`` to the canonical key list for ``mode``."""
    keys = metric_keys_for_mode(mode)
    out: Dict[str, Any] = {k: metrics.get(k, float("nan")) for k in keys}
    for k in ALPACA_METRICS:
        if k in metrics:
            out[k] = metrics[k]
    return out


def hp_columns_for(method: str) -> List[str]:
    return {
        "snmf":  SNMF_HP_COLUMNS,
        "rmu":   RMU_HP_COLUMNS,
        "crisp": CRISP_HP_COLUMNS,
        "pisces": PISCES_HP_COLUMNS,
        "ember": [],
    }[method.lower()]


_ALPACA_IN_GRID_METHODS = {"ember"}


def full_columns_for(method: str, train_eval_mode: str,
                     is_test: bool = False) -> List[str]:
    """Concatenate ID + embed + method-HP + metric columns in canonical order."""
    mode_prefix = "test" if is_test else "train"
    mode = f"{mode_prefix}_{train_eval_mode}"
    cols = (ID_COLUMNS
            + EMBED_COLUMNS
            + hp_columns_for(method)
            + metric_keys_for_mode(mode))
    if method.lower() in _ALPACA_IN_GRID_METHODS and not is_test:
        cols = cols + ALPACA_METRICS
    return cols + ["wall_time_s"]


def validate_columns_for(method: str, train_eval_mode: str) -> List[str]:
    """Columns for the validate-top-K stage (always includes alpaca metrics)."""
    train_mode = f"train_{train_eval_mode}"
    return (ID_COLUMNS
            + EMBED_COLUMNS
            + hp_columns_for(method)
            + metric_keys_for_mode(train_mode)
            + ALPACA_METRICS
            + ["wall_time_s"])


__all__ = [
    "ROOT_DIR", "RESULTS_DIR", "DATA_DIR",
    "method_dir_name", "run_root", "train_dir", "test_dir",
    "concept_dir", "baseline_dir",
    "read_csv_safe", "append_csv_row", "save_json_atomic",
    "load_done_set",
    "topk_per_concept", "write_topk", "pick_best_embed_delta", "load_best_embed_map",
    "ID_COLUMNS", "EMBED_COLUMNS",
    "SNMF_HP_COLUMNS", "RMU_HP_COLUMNS", "CRISP_HP_COLUMNS", "PISCES_HP_COLUMNS",
    "TRAIN_OPEN_METRICS", "TRAIN_MC_METRICS", "TEST_OPEN_METRICS", "TEST_MC_METRICS",
    "ALPACA_METRICS",
    "metric_keys_for_mode", "normalize_metrics_for_mode",
    "hp_columns_for", "full_columns_for", "validate_columns_for",
]
