"""Run configuration: YAML schema + dataclasses + CLI.

A run is specified by:

    1. A YAML file at ``configs/<method>_<model>.yaml`` holding everything
       invariant for a (method, model) combination: grid values, feature-
       selection thresholds, eval settings.

    2. A tiny CLI for the per-invocation choices that vary across runs:
       ``--concepts``, ``--train-eval``, ``--seed``, ``--overwrite``.

Example invocation::

    python -m ember.run_erasure \\
        --config configs/snmf_ember_gemma.yaml \\
        --train-eval mc \\
        --concepts "Harry Potter"

The single CLI entry-point (:func:`parse_args`) returns a fully-resolved
:class:`RunConfig`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

import yaml

from ember.erasure.io import ROOT_DIR

METHODS = ("snmf", "rmu", "crisp", "ember", "pisces")
TRAIN_EVAL_MODES = ("mc", "open")

_EMBER_DELTAS = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]


# ========================================================================== #
# Sub-configs                                                                 #
# ========================================================================== #

@dataclass
class EMBERStepConfig:
    """Pre-step EMBER grid (run before the main method grid).

    Skipped when ``method == "ember"`` (EMBER is the method, not a pre-step).
    For other methods, when ``enabled=True`` a cached best-embed CSV from a
    prior ember-only run is reused if present; otherwise the grid is run here.
    """
    enabled: bool = False
    deltas: List[float] = field(default_factory=lambda: list(_EMBER_DELTAS))


@dataclass
class SelectionConfig:
    """Feature-selection thresholds used by SNMF."""
    ratio_thresh: Optional[float] = None
    coverage_thresh: Optional[float] = None
    neurons_thresh: Optional[int] = None


@dataclass
class SNMFGridConfig:
    w_mode: str = "out"
    feature_source: str = "all"
    in_deltas: List[float] = field(default_factory=lambda: [1.0, 4.0, 7.0, 10.0])
    out_deltas: List[float] = field(default_factory=lambda: [1.0, 4.0, 7.0, 10.0])
    layer_ranges_in: Optional[List[List[int]]] = None
    layer_ranges_out: Optional[List[List[int]]] = None
    dtype: str = "bf16"


@dataclass
class RMUGridConfig:
    lr_grid: Optional[List[float]] = None
    alpha_grid: Optional[List[float]] = None
    steering_grid: Optional[List[float]] = None
    update_settings: Optional[List[Dict[str, Any]]] = None
    batch_size: int = 4
    max_num_batches: int = 150
    min_len: int = 50
    max_len: int = 2000


@dataclass
class WMDPDataConfig:
    """Settings used when ``data.source`` is ``wmdp``.

    Training reads prepared ``*_dataset_cleaned.jsonl`` files only. Run
    ``scripts/prepare_wmdp_corpora.py`` once to build them (CRISP paper
    preprocessing: clean, right-truncate to 1000 chars, bio sample 5000).
    """

    retain_type: Optional[str] = None
    data_root: str = "data/wmdp"
    n_examples: Optional[int] = None
    wiki_retain_max_len: int = 1000

    def resolved_retain_type(self, domain: str) -> str:
        """Return the retain corpus id; default matches the hazard domain."""
        if self.retain_type is not None:
            return self.retain_type.lower()
        return domain.lower()


@dataclass
class DataConfig:
    """Forget/retain corpus backend for unlearning methods."""

    source: str = "ember"
    wmdp: WMDPDataConfig = field(default_factory=WMDPDataConfig)


@dataclass
class CRISPGridConfig:
    k_features_grid: List[int] = field(default_factory=lambda: [5, 10, 20])
    alpha_grid: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0, 50.0])
    lr_grid: List[float] = field(default_factory=lambda: [5e-5, 1e-4, 5e-4])
    layer_ranges: Optional[List[List[int]]] = None
    batch_size: int = 16
    num_epochs: int = 2
    lora_batch_size: int = 1
    beta: float = 0.99
    gamma: float = 0.01
    lora_rank: int = 4
    sae_cache: str = "gemma_sae_cache"
    max_len: int = 2000


@dataclass
class EMBERConfig:
    """Grid config for the EMBER factored embedding erasure method."""
    deltas: List[float] = field(default_factory=lambda: list(_EMBER_DELTAS))


@dataclass
class PISCESGridConfig:
    """Grid config for the PISCES feature-editing method.

    ``features_json`` defaults (None) to data/pisces_concept_features_<model>.json.
    The grid sweeps ``k`` (sparsity threshold) x ``value`` (edit magnitude).
    """
    features_json: Optional[str] = None
    ks: List[float] = field(default_factory=lambda: [
        0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    values: List[float] = field(default_factory=lambda: [
        4.0, 7.0, 10.0, 13.0, 18.0, 21.0, 24.0, 30.0, 36.0, 42.0, 50.0, 60.0])
    dtype: str = "bf16"


@dataclass
class EvalConfig:
    alpaca: bool = True
    min_mmlu: float = 0.7
    max_qa_acc: float = 0.6
    alpaca_batch_size: Optional[int] = None
    skip_llm_judge: bool = False


@dataclass
class RelearningConfig:
    enabled: bool = True
    max_paragraphs: int = 100
    csv_path: str = "data/relearn_paragraphs.json"


@dataclass
class CheckpointConfig:
    """Optional export of the post-unlearning model weights."""
    enabled: bool = False
    root: str = "unlearned_checkpoints"


# ========================================================================== #
# Top-level RunConfig                                                         #
# ========================================================================== #

@dataclass
class RunConfig:
    """Top-level run configuration.

    Fields with defaults can be omitted from YAML. ``concepts`` and
    ``train_eval`` are always set from the CLI, not YAML.
    """
    method: str = "snmf"
    model_name: str = "google/gemma-2-2b-it"
    cache_dir: str = ""
    rank: int = 100
    seed: int = 42
    topk: int = 20
    overwrite: bool = False
    run_tests_after_train: bool = True
    features_source: str = "hf"  # "hf": download features from the HF dataset; "local": read mf_outputs/ as-is

    data: DataConfig = field(default_factory=DataConfig)
    ember_step: EMBERStepConfig = field(default_factory=EMBERStepConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    relearning: RelearningConfig = field(default_factory=RelearningConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    snmf: SNMFGridConfig = field(default_factory=SNMFGridConfig)
    rmu: RMUGridConfig = field(default_factory=RMUGridConfig)
    crisp: CRISPGridConfig = field(default_factory=CRISPGridConfig)
    ember: EMBERConfig = field(default_factory=EMBERConfig)
    pisces: PISCESGridConfig = field(default_factory=PISCESGridConfig)

    # CLI-supplied (not from YAML):
    concepts: List[str] = field(default_factory=list)
    train_eval: str = "mc"

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unknown method {self.method!r}; expected one of {METHODS}")
        if self.train_eval not in TRAIN_EVAL_MODES:
            raise ValueError(f"Unknown train_eval {self.train_eval!r}; "
                             f"expected one of {TRAIN_EVAL_MODES}")
        if not self.concepts:
            raise ValueError("RunConfig.concepts is empty (pass --concepts on CLI)")
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.features_source not in ("hf", "local"):
            raise ValueError(f"features_source must be 'hf' or 'local', got "
                             f"{self.features_source!r}")
        if self.data.source not in ("ember", "wmdp"):
            raise ValueError(f"data.source must be 'ember' or 'wmdp', got "
                             f"{self.data.source!r}")
        if self.data.source == "wmdp":
            from ember.wmdp.corpora import WMDP_DOMAINS, WMDP_RETAIN_TYPES
            bad = [c for c in self.concepts if c.lower() not in WMDP_DOMAINS]
            if bad:
                raise ValueError(
                    f"data.source='wmdp' requires concepts in {sorted(WMDP_DOMAINS)}, "
                    f"got {bad}"
                )
            if self.data.wmdp.retain_type is not None:
                rt = self.data.wmdp.retain_type.lower()
                if rt not in WMDP_RETAIN_TYPES:
                    raise ValueError(
                        f"data.wmdp.retain_type must be one of "
                        f"{sorted(WMDP_RETAIN_TYPES)}, got {rt!r}"
                    )
            if self.train_eval != "mc":
                raise ValueError(
                    "data.source='wmdp' requires --train-eval mc "
                    "(WMDP evaluation uses logits MCQ, not open QA)"
                )
        if (self.selection.neurons_thresh is not None
                and self.selection.coverage_thresh is not None):
            raise ValueError("Use at most one of selection.neurons_thresh / coverage_thresh")
        if self.method == "ember" and self.ember_step.enabled:
            self.ember_step.enabled = False
        if self.method == "crisp" and self.crisp.layer_ranges is not None:
            from ember.erasure.methods.crisp import validate_layer_ranges
            validate_layer_ranges(self.crisp.layer_ranges, self.model_name)
        if self.method == "rmu" and self.rmu.update_settings is not None:
            from ember.erasure.methods.rmu import validate_update_settings
            validate_update_settings(self.rmu.update_settings, self.model_name)
        if self.method == "snmf":
            from ember.erasure.methods.snmf import validate_snmf_layer_ranges
            validate_snmf_layer_ranges(
                self.snmf.layer_ranges_in,
                self.snmf.layer_ranges_out,
                self.model_name,
            )


# ========================================================================== #
# YAML loader                                                                 #
# ========================================================================== #

def _unwrap_optional(tp: Any) -> Any:
    """Resolve ``Optional[T]`` / ``T | None`` to ``T`` for nested dataclass detection."""
    if get_origin(tp) is Union:
        non_none = [a for a in get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return tp


def _instantiate(cls, raw: Any) -> Any:
    """Recursively build a dataclass tree from a dict."""
    if not is_dataclass(cls):
        return raw
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(raw).__name__}")

    hints = get_type_hints(cls)
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys in {cls.__name__}: {sorted(unknown)}")

    kwargs: Dict[str, Any] = {}
    for name, f in known.items():
        if name not in raw:
            continue
        field_type = _unwrap_optional(hints.get(name, f.type))
        if is_dataclass(field_type):
            kwargs[name] = _instantiate(field_type, raw[name])
        else:
            kwargs[name] = raw[name]
    return cls(**kwargs)


def load_yaml(path: Path) -> RunConfig:
    """Parse a YAML file into a :class:`RunConfig`."""
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        raw = {}

    sub_classes = {
        "data": DataConfig,
        "ember_step": EMBERStepConfig,
        "selection": SelectionConfig,
        "eval": EvalConfig,
        "relearning": RelearningConfig,
        "checkpoint": CheckpointConfig,
        "snmf": SNMFGridConfig,
        "rmu": RMUGridConfig,
        "crisp": CRISPGridConfig,
        "ember": EMBERConfig,
        "pisces": PISCESGridConfig,
    }

    known = {f.name for f in fields(RunConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys in RunConfig: {sorted(unknown)}")

    kwargs: Dict[str, Any] = {}
    for k, v in raw.items():
        if k in sub_classes:
            kwargs[k] = _instantiate(sub_classes[k], v)
        else:
            kwargs[k] = v
    return RunConfig(**kwargs)


# ========================================================================== #
# CLI                                                                         #
# ========================================================================== #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ember",
        description="Run the EMBER erasure pipeline from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path,
                   help="Path to a YAML config (see configs/).")
    p.add_argument("--concepts", nargs="+", required=True,
                   help="One or more concept names.")
    p.add_argument("--train-eval", choices=TRAIN_EVAL_MODES, default="mc",
                   help="Which question format drives the grid search.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override the config's seed.")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Re-run cells already present in the HP CSV.")
    p.add_argument("--rank", type=int, default=None,
                   help="Override the config's rank.")
    p.add_argument("--features-source", choices=["hf", "local"], default=None,
                   help="Where to get features: 'hf' downloads them from the "
                        "ClSu/ember-features dataset, 'local' reads mf_outputs/ as-is.")
    return p


def parse_args(argv: Optional[List[str]] = None) -> RunConfig:
    """Parse CLI args, merge with YAML config, and return a validated RunConfig."""
    args = build_parser().parse_args(argv)
    cfg = load_yaml(args.config)
    cfg.concepts = list(args.concepts)
    cfg.train_eval = args.train_eval
    if args.seed is not None:
        cfg.seed = args.seed
    if args.overwrite:
        cfg.overwrite = True
    if args.rank is not None:
        cfg.rank = args.rank
    if args.features_source is not None:
        cfg.features_source = args.features_source
    cfg.validate()
    return cfg


__all__ = [
    "METHODS", "TRAIN_EVAL_MODES",
    "EMBERStepConfig", "SelectionConfig", "DataConfig", "WMDPDataConfig",
    "SNMFGridConfig", "RMUGridConfig", "CRISPGridConfig", "EMBERConfig",
    "PISCESGridConfig",
    "EvalConfig", "RelearningConfig", "CheckpointConfig", "RunConfig",
    "load_yaml", "build_parser", "parse_args",
]
