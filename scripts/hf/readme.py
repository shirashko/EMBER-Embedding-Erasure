"""Generate Hugging Face model card READMEs for unlearned checkpoints."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


PRIMARY_METRICS = (
    ("efficacy", "Efficacy"),
    ("specificity", "Specificity"),
    ("harmonic", "Harmonic mean"),
    ("relearning_qa_mc", "Relearning QA (MC)"),
)

FULL_EVAL_METRICS = (
    ("qa_acc", "QA accuracy"),
    ("qa_frac", "QA fraction"),
    ("simdom_acc", "SimDom accuracy"),
    ("simdom_frac", "SimDom fraction"),
    ("mmlu_acc", "MMLU accuracy"),
    ("mmlu_frac", "MMLU fraction"),
)


def is_adapter_checkpoint(concept_dir: Path) -> bool:
    return (concept_dir / "adapter_config.json").exists() or (
        concept_dir / "adapter_model.safetensors"
    ).exists()


def _load_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").replace("NaN", "null")
    return json.loads(text)


def _parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    try:
        number = float(stripped)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def _format_number(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _format_hyperparameter_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    if isinstance(value, bool):
        return str(value)
    return str(value)


def _load_score_comparison(concept_dir: Path) -> dict[str, dict[str, float | None]]:
    csv_path = concept_dir / "evaluation" / "score_comparison.csv"
    if not csv_path.exists():
        return {}

    rows: dict[str, dict[str, float | None]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric = row.get("metric", "").strip()
            if not metric:
                continue
            rows[metric] = {
                "baseline_train": _parse_number(row.get("baseline_train")),
                "train_after_unlearning": _parse_number(row.get("train_after_unlearning")),
                "baseline_test": _parse_number(row.get("baseline_test")),
                "test_after_unlearning": _parse_number(row.get("test_after_unlearning")),
            }
    return rows


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _hyperparameter_rows(metadata: dict[str, Any]) -> list[list[str]]:
    hyperparameters = metadata.get("hyperparameters") or metadata.get("selected_hyperparameters") or {}
    rows: list[list[str]] = []
    for key in sorted(hyperparameters):
        value = hyperparameters[key]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        rows.append([f"`{key}`", _format_hyperparameter_value(value)])
    return rows


def _resolve_base_model(metadata: dict[str, Any], base_model_raw: str) -> str:
    model_name = metadata.get("model_name")
    if model_name:
        return str(model_name)
    if "/" in base_model_raw:
        return base_model_raw
    if base_model_raw.startswith("google_"):
        return base_model_raw.replace("google_", "google/", 1).replace("_", "-", 1)
    if base_model_raw.startswith("meta-llama_"):
        return base_model_raw.replace("meta-llama_", "meta-llama/", 1).replace("_", "-")
    return base_model_raw.replace("_", "/")


def generate_readme(
    concept_dir: Path,
    *,
    method: str,
    base_model_raw: str,
    base_model_clean: str,
    concept: str | None = None,
) -> str:
    metadata_path = concept_dir / "unlearned_checkpoints.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = _load_json(metadata_path)

    resolved_concept = metadata.get("concept") or concept or concept_dir.name
    resolved_method = metadata.get("method") or method
    base_model = _resolve_base_model(metadata, base_model_raw)
    is_adapter = is_adapter_checkpoint(concept_dir)
    library_tag = "peft" if is_adapter else "transformers"
    checkpoint_type = "LoRA Adapter" if is_adapter else "Full Model Weights"

    rank = metadata.get("rank")
    seed = metadata.get("seed")
    train_eval = metadata.get("train_eval")

    score_rows = _load_score_comparison(concept_dir)

    summary_rows = [
        ["**Unlearning method**", resolved_method.upper()],
        ["**Base model**", f"`{base_model}`"],
        ["**Target concept**", str(resolved_concept)],
        ["**Checkpoint type**", checkpoint_type],
    ]
    if rank is not None:
        summary_rows.append(["**Rank / seed**", f"{rank} / {seed}"])
    elif seed is not None:
        summary_rows.append(["**Seed**", str(seed)])
    if train_eval:
        summary_rows.append(["**Train eval protocol**", str(train_eval)])

    sections = [
        "---",
        "tags:",
        "- unlearning",
        f"- {resolved_method}",
        f"- {base_model_clean}",
        f"library_name: {library_tag}",
        "pipeline_tag: text-generation",
        f"base_model: {base_model}",
        "language:",
        "- en",
        "metrics:",
        "- efficacy",
        "- specificity",
        "- harmonic",
        "---",
        "",
        "# Unlearned Checkpoint",
        "",
        _markdown_table(["Field", "Value"], summary_rows),
        "",
    ]

    hyperparameter_rows = _hyperparameter_rows(metadata)
    if hyperparameter_rows:
        sections.extend(
            [
                "---",
                "",
                "## Unlearning Configuration",
                "",
                "Selected hyperparameters (from `unlearned_checkpoints.json`):",
                "",
                _markdown_table(["Parameter", "Value"], hyperparameter_rows),
                "",
            ]
        )

    primary_table_rows: list[list[str]] = []
    for metric_key, metric_label in PRIMARY_METRICS:
        row = score_rows.get(metric_key)
        if not row:
            continue
        if all(row[column] is None for column in row):
            continue
        primary_table_rows.append(
            [
                f"**{metric_label}**",
                _format_number(row["train_after_unlearning"]),
                f"**{_format_number(row['test_after_unlearning'])}**",
            ]
        )

    if primary_table_rows:
        sections.extend(
            [
                "---",
                "",
                "## Primary Unlearning Metrics (held-out test, MC protocol)",
                "",
                "Headline scores used for checkpoint selection:",
                "",
                _markdown_table(
                    ["Metric", "Train (after unlearning)", "**Test (after unlearning)**"],
                    primary_table_rows,
                ),
                "",
            ]
        )

    full_eval_rows: list[list[str]] = []
    for metric_key, metric_label in FULL_EVAL_METRICS:
        row = score_rows.get(metric_key)
        if not row:
            continue
        if all(row[column] is None for column in row):
            continue
        full_eval_rows.append(
            [
                metric_label,
                _format_number(row["baseline_train"]),
                _format_number(row["train_after_unlearning"]),
                _format_number(row["baseline_test"]),
                f"**{_format_number(row['test_after_unlearning'])}**",
            ]
        )

    if full_eval_rows:
        sections.extend(
            [
                "---",
                "",
                "## Full Evaluation (baseline → unlearned)",
                "",
                "From `evaluation/score_comparison.csv`:",
                "",
                _markdown_table(
                    [
                        "Metric",
                        "Baseline (train)",
                        "After unlearn (train)",
                        "Baseline (test)",
                        "**After unlearn (test)**",
                    ],
                    full_eval_rows,
                ),
                "",
            ]
        )

    eval_dir = concept_dir / "evaluation"
    file_rows = [["`unlearned_checkpoints.json`", "Checkpoint metadata & hyperparameters"]]
    if eval_dir.exists():
        file_rows.extend(
            [
                ["`evaluation/evaluation_summary.json`", "Full evaluation payload (train/test/relearning)"],
                ["`evaluation/score_comparison.csv`", "Baseline vs. unlearned comparison table"],
            ]
        )

    sections.extend(
        [
            "---",
            "",
            "## Files in This Repository",
            "",
            _markdown_table(["File", "Description"], file_rows),
            "",
        ]
    )

    return "\n".join(sections)


def write_readme(
    concept_dir: Path,
    *,
    method: str,
    base_model_raw: str,
    base_model_clean: str,
    concept: str | None = None,
) -> Path:
    readme_path = concept_dir / "README.md"
    readme_path.write_text(
        generate_readme(
            concept_dir,
            method=method,
            base_model_raw=base_model_raw,
            base_model_clean=base_model_clean,
            concept=concept,
        ),
        encoding="utf-8",
    )
    return readme_path
