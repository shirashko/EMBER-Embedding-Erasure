#!/usr/bin/env python3
"""Generate per-run erasure configs from optimal_unlearning_hyperparams.yaml.

Each output YAML pins the grid to the previously selected optimal
hyperparameters so ``ember.run_erasure`` runs a single HP cell and can save a
reproducible unlearned checkpoint.

Example:
    python scripts/generate_reproduce_configs.py

    python -m ember.run_erasure \\
        --config configs/reproduce_optimal_configs/snmf/gemma/golf.yaml \\
        --concepts "Golf" --train-eval mc --features-source local
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT / "configs" / "reproduce_optimal_configs" / "optimal_unlearning_hyperparams.yaml"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "configs" / "reproduce_optimal_configs"

MODEL_SPECS: Dict[str, Tuple[str, int]] = {
    "Gemma": ("google/gemma-2-2b-it", 100),
    "Llama": ("meta-llama/Llama-3.1-8B-Instruct", 200),
}

METHOD_SLUGS = {
    "PISCES": "pisces",
    "RMU": "rmu",
    "CRISP": "crisp",
    "SNMF": "snmf",
}

TRAIN_EVAL_BY_METHOD = {
    "PISCES": "open",
    "RMU": "mc",
    "CRISP": "mc",
    "SNMF": "mc",
}

# Keep in sync with ember.erasure.methods.crisp.{GEMMA,LLAMA}_LAYER_RANGES
_CRISP_LAYER_RANGES: Dict[str, List[Tuple[int, int, int]]] = {
    "gemma": [(4, 14, 2), (5, 15, 2), (4, 20, 2), (5, 21, 2)],
    "llama": [(5, 19, 2), (4, 18, 2), (5, 29, 2), (4, 28, 2)],
}

# Keep in sync with ember.erasure.methods.rmu.{GEMMA,LLAMA}_UPDATE_SETTINGS
_RMU_UPDATE_SETTINGS: Dict[str, List[Tuple[str, int, List[int]]]] = {
    "gemma": [
        ("S1_lid7_L567", 7, [5, 6, 7]),
        ("S2_lid8_L678", 8, [6, 7, 8]),
        ("S3_lid6_L456", 6, [4, 5, 6]),
    ],
    "llama": [
        ("S1_lid7_L567", 7, [5, 6, 7]),
        ("S2_lid9_L789", 9, [7, 8, 9]),
        ("S3_lid11_L91011", 11, [9, 10, 11]),
    ],
}


# Keep in sync with ember.erasure.methods.snmf.{GEMMA,LLAMA}_LAYER_RANGES_{IN,OUT}
_SNMF_LAYER_RANGES: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
    "gemma": {
        "in": [(0, 25), (0, 8), (0, 12)],
        "out": [(0, 8), (9, 17), (13, 25)],
    },
    "llama": {
        "in": [(0, 31), (0, 10), (0, 16)],
        "out": [(0, 10), (11, 21), (16, 31)],
    },
}


def _validate_crisp_layer_range(model_key: str, lo: int, hi: int, step: int) -> None:
    allowed = set(_CRISP_LAYER_RANGES[model_key.lower()])
    layer_range = (lo, hi, step)
    if layer_range not in allowed:
        raise ValueError(
            f"Invalid CRISP layer range {layer_range} for {model_key}; "
            f"allowed: {sorted(allowed)}"
        )


def _validate_rmu_update_setting(
    model_key: str, setting_name: str, layer_id: int, layer_ids: str,
) -> None:
    allowed = {
        (name, lid, tuple(ids))
        for name, lid, ids in _RMU_UPDATE_SETTINGS[model_key.lower()]
    }
    key = (setting_name, int(layer_id), tuple(int(x) for x in layer_ids.split(",")))
    if key not in allowed:
        raise ValueError(
            f"Invalid RMU update setting {key} for {model_key}; "
            f"allowed: {sorted(allowed)}"
        )


def _validate_snmf_layer_range(model_key: str, side: str, lo: int, hi: int) -> None:
    allowed = set(_SNMF_LAYER_RANGES[model_key.lower()][side])
    layer_range = (lo, hi)
    if layer_range not in allowed:
        raise ValueError(
            f"Invalid SNMF layer_ranges_{side} {layer_range} for {model_key}; "
            f"allowed: {sorted(allowed)}"
        )


_BASE_CONFIG_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def load_base_config(method: str, model_key: str) -> Dict[str, Any]:
    """Load configs/<method>_<model>.yaml (e.g. crisp_llama.yaml)."""
    method_slug = METHOD_SLUGS[method]
    model_slug = model_key.lower()
    cache_key = (method_slug, model_slug)
    if cache_key not in _BASE_CONFIG_CACHE:
        path = REPO_ROOT / "configs" / f"{method_slug}_{model_slug}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Missing base config: {path}")
        _BASE_CONFIG_CACHE[cache_key] = (
            yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        )
    return copy.deepcopy(_BASE_CONFIG_CACHE[cache_key])


def concept_slug(concept: str) -> str:
    return concept.strip().lower().replace(" ", "_").replace("-", "_")


def _selection_block(hp: Dict[str, Any]) -> Dict[str, Any]:
    block: Dict[str, Any] = {}
    if hp.get("ratio_thresh") is not None:
        block["ratio_thresh"] = float(hp["ratio_thresh"])
    if hp.get("coverage_thresh") is not None:
        block["coverage_thresh"] = float(hp["coverage_thresh"])
    if hp.get("neurons_thresh") is not None:
        block["neurons_thresh"] = int(hp["neurons_thresh"])
    return block


def _ember_step_block(hp: Dict[str, Any]) -> Dict[str, Any]:
    enabled = bool(hp.get("embed_step_enabled", False))
    block: Dict[str, Any] = {"enabled": enabled}
    delta = float(hp.get("delta_embed", 0.0) or 0.0)
    if enabled and delta > 0.0:
        block["deltas"] = [delta]
    return block


def _pin_method_hps(
    method: str, model_key: str, hp: Dict[str, Any], block: Dict[str, Any],
) -> Dict[str, Any]:
    """Pin optimal hyperparameters onto the base method block."""
    if method == "PISCES":
        block["ks"] = [float(hp["k_pisces"])]
        block["values"] = [float(hp["value_pisces"])]
        return block

    if method == "RMU":
        setting_name = str(hp["setting_name"])
        layer_id = int(float(hp["layer_id"]))
        layer_ids = str(hp["layer_ids"])
        _validate_rmu_update_setting(model_key, setting_name, layer_id, layer_ids)
        block["lr_grid"] = [float(hp["lr"])]
        block["alpha_grid"] = [float(hp["alpha"])]
        block["steering_grid"] = [float(hp["steering"])]
        block["update_settings"] = [{
            "setting_name": setting_name,
            "layer_id": layer_id,
            "layer_ids": layer_ids,
        }]
        return block

    if method == "CRISP":
        layer_range = [
            int(hp["layer_lo"]),
            int(hp["layer_hi"]),
            int(hp["layer_step"]),
        ]
        _validate_crisp_layer_range(model_key, *layer_range)
        block["k_features_grid"] = [int(hp["k_features"])]
        block["alpha_grid"] = [float(hp["alpha"])]
        block["lr_grid"] = [float(hp["lr"])]
        block["layer_ranges"] = [layer_range]
        if "num_epochs" in hp:
            block["num_epochs"] = int(hp["num_epochs"])
        if "lora_rank" in hp:
            block["lora_rank"] = int(hp["lora_rank"])
        return block

    if method == "SNMF":
        layer_lo_in = int(hp["layer_lo_in"])
        layer_hi_in = int(hp["layer_hi_in"])
        layer_lo_out = int(hp["layer_lo_out"])
        layer_hi_out = int(hp["layer_hi_out"])
        _validate_snmf_layer_range(model_key, "in", layer_lo_in, layer_hi_in)
        _validate_snmf_layer_range(model_key, "out", layer_lo_out, layer_hi_out)
        block["w_mode"] = str(hp.get("w_mode", block.get("w_mode", "both")))
        block["feature_source"] = str(
            hp.get("feature_source", block.get("feature_source", "all"))
        )
        block["in_deltas"] = [float(hp["delta_in"])]
        block["out_deltas"] = [float(hp["delta_out"])]
        block["layer_ranges_in"] = [[layer_lo_in, layer_hi_in]]
        block["layer_ranges_out"] = [[layer_lo_out, layer_hi_out]]
        return block

    raise ValueError(f"Unsupported method: {method}")


def build_run_config(
    model_key: str,
    method: str,
    concept: str,
    hp: Dict[str, Any],
) -> Dict[str, Any]:
    base = load_base_config(method, model_key)
    model_name, default_rank = MODEL_SPECS[model_key]
    method_slug = METHOD_SLUGS[method]
    train_eval = TRAIN_EVAL_BY_METHOD[method]

    selection = dict(base.get("selection") or {})
    selection.update(_selection_block(hp))

    cfg: Dict[str, Any] = {
        "method": method_slug,
        "model_name": model_name,
        "rank": int(hp.get("rank", default_rank)),
        "seed": int(hp.get("seed", base.get("seed", 42))),
        "topk": 1,
        "run_tests_after_train": base.get("run_tests_after_train", True),
        "features_source": "local",
        "selection": selection,
        "ember_step": _ember_step_block(hp),
        method_slug: _pin_method_hps(
            method, model_key, hp, dict(base.get(method_slug) or {}),
        ),
        "eval": dict(base.get("eval") or {}),
        "relearning": dict(base.get("relearning") or {}),
        "checkpoint": {
            "enabled": True,
            "root": "unlearned_checkpoints",
        },
        "_reproduce_meta": {
            "source": "optimal_unlearning_hyperparams.yaml",
            "base_config": f"configs/{method_slug}_{model_key.lower()}.yaml",
            "model": model_key,
            "method": method,
            "concept": concept,
            "train_eval": train_eval,
        },
    }
    return cfg


def config_relpath(method: str, model_key: str, concept: str) -> str:
    """Relative path under configs/reproduce_optimal_configs/."""
    return (
        f"{METHOD_SLUGS[method]}/{model_key.lower()}/{concept_slug(concept)}.yaml"
    )


def config_path(output_dir: Path, method: str, model_key: str, concept: str) -> Path:
    return output_dir / config_relpath(method, model_key, concept)


def _yaml_header(model_key: str, method: str, concept: str, train_eval: str) -> str:
    rel = f"configs/reproduce_optimal_configs/{config_relpath(method, model_key, concept)}"
    return (
        f"# Reproduce optimal unlearning: {model_key} / {method} / {concept}\n"
        f"#\n"
        f"# Usage:\n"
        f"#   python -m ember.run_erasure --config {rel} \\\n"
        f"#       --concepts \"{concept}\" --train-eval {train_eval} \\\n"
        f"#       --features-source local\n"
    )


def write_config(path: Path, model_key: str, method: str, concept: str,
                 cfg: Dict[str, Any]) -> None:
    meta = cfg.pop("_reproduce_meta")
    header = _yaml_header(model_key, method, concept, meta["train_eval"])
    body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False).rstrip()
    path.write_text(f"{header}\n{body}\n", encoding="utf-8")


def generate_configs(
    input_path: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> List[Path]:
    tree = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    if not isinstance(tree, dict):
        raise ValueError(f"Expected mapping in {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for model_key, methods in tree.items():
        if model_key not in MODEL_SPECS:
            continue
        if not isinstance(methods, dict):
            continue
        for method, concepts in methods.items():
            if method not in METHOD_SLUGS:
                continue
            if not isinstance(concepts, dict):
                continue
            for concept, payload in concepts.items():
                hp = payload.get("hyperparameters", {})
                if not hp:
                    continue
                cfg = build_run_config(model_key, method, concept, hp)
                out_path = config_path(output_dir, method, model_key, concept)
                if dry_run:
                    written.append(out_path)
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                write_config(out_path, model_key, method, concept, cfg)
                written.append(out_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproduce-optimal erasure configs from the metadata YAML.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to optimal_unlearning_hyperparams.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated run configs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths that would be written without creating files",
    )
    args = parser.parse_args()

    paths = generate_configs(args.input, args.output_dir, dry_run=args.dry_run)
    action = "Would write" if args.dry_run else "Wrote"
    print(f"{action} {len(paths)} config(s) to {args.output_dir}")
    for path in sorted(paths):
        print(f"  {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
