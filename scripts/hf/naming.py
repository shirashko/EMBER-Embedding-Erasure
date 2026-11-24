"""Hugging Face repo naming helpers for unlearned checkpoints."""

from __future__ import annotations

BASE_MODEL_SLUGS: dict[str, str] = {
    "google_gemma-2-2b-it": "gemma-2-2b-it",
    "meta-llama_Llama-3.1-8B-Instruct": "llama-3.1-8b-instruct",
    "Qwen_Qwen3.5-2B": "qwen3.5-2b",
    "Qwen_Qwen2.5-3B-Instruct": "qwen2.5-3b-instruct",
    "Qwen_Qwen3-1.7B": "qwen3-1.7b",
}


def short_model_slug(base_model_raw: str) -> str:
    if base_model_raw in BASE_MODEL_SLUGS:
        return BASE_MODEL_SLUGS[base_model_raw]
    slug = base_model_raw.replace("/", "_").replace(".", "-")
    if slug.startswith("google_"):
        slug = slug.removeprefix("google_")
    if slug.startswith("meta-llama_"):
        slug = slug.removeprefix("meta-llama_").replace("_", "-").lower()
        if not slug.endswith("-instruct"):
            slug = f"{slug}-instruct"
    return slug.lower()


def concept_slug(concept: str) -> str:
    return concept.replace("_", "-").lower()


def repo_name(method: str, base_model_raw: str, concept: str) -> str:
    return f"{short_model_slug(base_model_raw)}-{method.lower()}-{concept_slug(concept)}"


def repo_id_for(username: str, repo_name_value: str) -> str:
    return f"{username}/{repo_name_value}"
