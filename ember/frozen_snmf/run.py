"""Drive concept-detection and promotion scoring for a frozen SNMF dictionary."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
_SNMF_ROOT = ROOT_DIR / "external" / "snmf"
if str(_SNMF_ROOT) not in sys.path:
    sys.path.insert(0, str(_SNMF_ROOT))

from ember.frozen_snmf.activations import harvest_activations  # noqa: E402
from ember.frozen_snmf.concepts import concept_sentences, neutral_sentences, select_concepts  # noqa: E402
from ember.frozen_snmf.factors import (  # noqa: E402
    dictionary_site,
    discover_layers,
    infer_model_name,
    load_dictionary,
    parse_layer_spec,
)
from ember.frozen_snmf.sites import EMBER_MODE_HOOK, ember_mode_for_site  # noqa: E402
from ember.frozen_snmf.interpret import judge_shortlist  # noqa: E402
from ember.frozen_snmf.output import layer_csv_path, merge_concept_tables, write_layer_scores  # noqa: E402
from ember.frozen_snmf.scoring import exclusive_vocab, score_layer  # noqa: E402
from ember.local_datasets import resolve_neutral_path  # noqa: E402
from ember.utils import _safe_concept, resolve_device  # noqa: E402
from llm_utils.activation_generator import ActivationGenerator  # noqa: E402


def _layers_still_needed(pending, chunk_layers):
    return [
        layer for layer in chunk_layers
        if any(concept_layer == layer for _name, concept_layer in pending)
    ]


def _concept_layers_needed(concept_name, needed, outdir, overwrite):
    concept_dir = outdir / _safe_concept(concept_name)
    return [
        layer for layer in needed
        if overwrite or not layer_csv_path(concept_dir, layer).is_file()
    ]


def _write_run_config(outdir, factors_dir, model_name, concepts, layer_files, args, neutral_path, concept_path):
    payload = {
        "factors_dir": str(factors_dir),
        "model_name": model_name,
        "concepts": concepts,
        "layers": [layer for layer, _path in layer_files],
        "eps": args.eps,
        "g_iters": args.g_iters,
        "reg": args.reg,
        "top_k": args.top_k,
        "k_proj": args.k_proj,
        "skip_interpret": bool(args.skip_interpret),
        "confidence_thresh": args.confidence_thresh,
        "judge_top_tokens": args.judge_top_tokens,
        "neutral_path": str(neutral_path),
        "concept_path": str(concept_path),
    }
    (outdir / "run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_model(model_name: str, device: str, ember_mode: str):
    print(f"Loading {model_name} on {device} (EMBER mode {ember_mode})")
    generator = ActivationGenerator(model_name, model_device=device, data_device="cpu", mode=ember_mode)
    model = generator.model
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.eval()
    return model


def _score_concept_layers(
    *,
    model,
    concept_name,
    concept_acts,
    neutral_acts,
    concept_token_ids,
    neutral_token_ids,
    concept_ids,
    neutral_ids,
    layers,
    by_layer,
    outdir,
    model_name,
    args,
    device,
):
    token_ids = torch.cat([concept_token_ids, neutral_token_ids], dim=0)
    concept_dir = outdir / _safe_concept(concept_name)
    for layer in layers:
        print(f"Scoring {concept_name!r} layer {layer} ({by_layer[layer].name})")
        factors, site = load_dictionary(by_layer[layer])
        if site != args.dictionary_site:
            raise SystemExit(
                f"{by_layer[layer]} records site {site!r}, but {args.factors_dir} is {args.dictionary_site!r}"
            )
        rows = score_layer(
            model=model,
            factors=factors,
            concept_acts=concept_acts[layer],
            neutral_acts=neutral_acts[layer],
            token_ids=token_ids,
            concept_ids=concept_ids,
            neutral_ids=neutral_ids,
            layer=layer,
            concept_name=concept_name,
            model_name=model_name,
            eps=args.eps,
            g_iters=args.g_iters,
            reg=args.reg,
            k_proj=args.k_proj,
            fit_device=device,
        )
        write_layer_scores(layer_csv_path(concept_dir, layer), rows)
        merge_concept_tables(concept_dir, args.top_k)
        best = max(rows, key=lambda row: row["s_cd"])
        promoted = max(rows, key=lambda row: row["promotion_score"])
        print(
            f"  best S_CD={best['s_cd']:.3f} at feature {best['feature']}; "
            f"best promotion={promoted['promotion_score']:.3f} at feature {promoted['feature']} "
            f"({promoted['best_concept_token']})"
        )


def score_dictionary(args) -> None:
    """Score ``args.concepts`` against the dictionary in ``args.factors_dir``."""
    factors_dir = args.factors_dir.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    concept_path = args.concept_path.expanduser().resolve()
    neutral_path = resolve_neutral_path(args.neutral_path)
    if not factors_dir.is_dir():
        raise SystemExit(f"--factors-dir is not a directory: {factors_dir}")

    concepts = select_concepts(args.concepts, concept_path)
    args.dictionary_site = dictionary_site(factors_dir)
    args.ember_mode = ember_mode_for_site(args.dictionary_site)
    print(
        f"Dictionary site {args.dictionary_site!r} -> EMBER mode {args.ember_mode!r} "
        f"({EMBER_MODE_HOOK[args.ember_mode]})"
    )
    layer_files = discover_layers(factors_dir, args.factors_filename, parse_layer_spec(args.layers))
    model_name = infer_model_name(factors_dir, args.model_name)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_run_config(outdir, factors_dir, model_name, concepts, layer_files, args, neutral_path, concept_path)

    pending = [
        (concept_name, layer)
        for concept_name in concepts
        for layer, _path in layer_files
        if args.overwrite or not layer_csv_path(outdir / _safe_concept(concept_name), layer).is_file()
    ]
    if not pending:
        print("Every requested layer already has a CSV. Rebuilding the summary tables.")
        for concept_name in concepts:
            merge_concept_tables(outdir / _safe_concept(concept_name), args.top_k)
    else:
        device = resolve_device(args.device)
        model = _load_model(model_name, device, args.ember_mode)
        shared_neutral = neutral_sentences(neutral_path)
        sentences = {name: concept_sentences(concept_path, name) for name in concepts}
        vocabs = {
            name: exclusive_vocab(model.tokenizer, sentences[name], shared_neutral)
            for name in concepts
        }
        for name, (concept_ids, neutral_ids) in vocabs.items():
            print(
                f"{name}: {len(sentences[name])} concept sentences, "
                f"{len(shared_neutral)} neutral sentences, "
                f"{len(concept_ids)} concept-only tokens, {len(neutral_ids)} neutral-only tokens"
            )

        layers = [layer for layer, _path in layer_files]
        by_layer = {layer: path for layer, path in layer_files}
        chunk = max(1, args.layer_chunk)
        for start in range(0, len(layers), chunk):
            needed = _layers_still_needed(pending, layers[start:start + chunk])
            if not needed:
                continue
            print(f"Harvesting neutral activations for layers {needed}")
            neutral_acts, neutral_token_ids = harvest_activations(
                model, shared_neutral, needed, args.batch_size, args.ember_mode,
            )
            for concept_name in concepts:
                concept_needed = _concept_layers_needed(concept_name, needed, outdir, args.overwrite)
                if not concept_needed:
                    continue
                print(f"Harvesting {concept_name!r} activations for layers {concept_needed}")
                concept_acts, concept_token_ids = harvest_activations(
                    model, sentences[concept_name], concept_needed, args.batch_size, args.ember_mode,
                )
                concept_ids, neutral_ids = vocabs[concept_name]
                _score_concept_layers(
                    model=model,
                    concept_name=concept_name,
                    concept_acts=concept_acts,
                    neutral_acts=neutral_acts,
                    concept_token_ids=concept_token_ids,
                    neutral_token_ids=neutral_token_ids,
                    concept_ids=concept_ids,
                    neutral_ids=neutral_ids,
                    layers=concept_needed,
                    by_layer=by_layer,
                    outdir=outdir,
                    model_name=model_name,
                    args=args,
                    device=device,
                )
                del concept_acts, concept_token_ids
            del neutral_acts, neutral_token_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.skip_interpret:
        print(f"Wrote concept scores under {outdir}")
        return

    for concept_name in concepts:
        judge_shortlist(
            outdir / _safe_concept(concept_name),
            gemini_model=args.gemini_model,
            confidence_thresh=args.confidence_thresh,
            judge_top_tokens=args.judge_top_tokens,
            max_workers=args.max_workers,
            resume=not args.overwrite,
        )
    print(f"Wrote concept scores under {outdir}")
