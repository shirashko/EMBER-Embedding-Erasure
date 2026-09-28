#!/usr/bin/env python3
"""Score a frozen SNMF dictionary against this repo's concept and neutral sentences.

The dictionary is the shared factorization already trained on generic text
(``layer_*/snmf_factors.pt``), not a per-concept fitted trained SNMF features.
Each name must be one of the concepts in ``data/concept_sentences.json``
(the concepts this repo has concept and neutral sentences for). For each
requested concept the script:

1. Reads MLP-intermediate activations of the concept sentences and of
   ``data/neutral_sentences.json`` at ``blocks.{layer}.mlp.hook_post``
   (the down-projection input the dictionary was trained on).
2. Holds ``F`` fixed and solves for the nonnegative coefficients ``y`` with the
   same Semi-NMF ``G`` update used in training, so ``A ≈ F y^T``.
3. Scores every feature with the input concept-detection score

       S_CD(k) = log( max_{t in concept} y_{k,t} / (max_{t in neutral} y_{k,t} + eps) )

4. Decodes every column through the unembedding,

       logits_k = W_U( LN(W_down F_{:,k}) )

   and ranks features by how strongly they promote concept-specific vocabulary
   over neutral-only vocabulary:

       promotion(k) = max_{t in V_concept} logits_k[t] - max_{t in V_neutral} logits_k[t]

5. Keeps the top ``--top-k`` features per layer on each ranking. Gemini, using
   the prompts in ``ember.interpret_features``, describes ``activating_tokens``
   for the concept-detection shortlist and ``projection_top_tokens`` for the
   promotion shortlist. ``potential_features.csv`` keeps rows with
   ``is_member`` true and confidence at least ``--confidence-thresh``.
   ``metric_score`` is the linear activation ratio for input rows and
   ``promotion_score`` for output rows.

``V_concept`` is the set of token ids that occur in the concept sentences and
not in the neutral sentences. ``V_neutral`` is the reverse. The natural log is
used; the ranking by ``S_CD`` does not depend on the base.

Example::

    python -m ember.score_frozen_snmf \\
        --config configs/frozen_snmf/llama3_1_8b_it_rank200_init_random_mlp_intermediate.yaml

``--config`` must list every hyperparameter. CLI flags remain available when
``--config`` is omitted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ember.frozen_snmf.config import load_score_config
from ember.frozen_snmf.run import ROOT_DIR, score_dictionary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank a frozen SNMF dictionary by concept detection and logit-lens promotion."
    )
    parser.add_argument("--factors-dir", type=Path, required=True,
                        help="Directory of layer_*/snmf_factors.pt from the shared SNMF training run.")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Dedicated directory for per-concept scores and the selected feature tables.")
    parser.add_argument(
        "--concepts",
        nargs="+",
        required=True,
        help="Concepts to score. Each must be a concept in data/concept_sentences.json.",
    )
    parser.add_argument("--concept-path", type=Path, default=ROOT_DIR / "data" / "concept_sentences.json")
    parser.add_argument("--neutral-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None,
                        help="Base model. Default: model_path in factors-dir/config.json, else Llama-3.1-8B-Instruct.")
    parser.add_argument("--factors-filename", type=str, default="snmf_factors.pt")
    parser.add_argument("--layers", type=str, default=None, help="Subset such as 0-5,10. Default: every layer_* directory.")
    parser.add_argument("--layer-chunk", type=int, default=1, help="How many layers to harvest in one forward pass.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--g-iters", type=int, default=40, help="Frozen-F coefficient updates per layer.")
    parser.add_argument("--reg", type=float, default=1e-4, help="Ridge term for the least-squares coefficient initialization.")
    parser.add_argument("--top-k", type=int, default=10, help="Features kept per layer for both shortlists and the Gemini judge.")
    parser.add_argument("--k-proj", type=int, default=30, help="Activating and logit-lens tokens stored on each feature.")
    parser.add_argument("--judge-top-tokens", type=int, default=20, help="Tokens from each list sent to Gemini.")
    parser.add_argument("--confidence-thresh", type=float, default=0.85, help="Minimum membership confidence kept in potential_features.csv.")
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--skip-interpret", action="store_true", help="Write score tables only. Skip the Gemini membership judge.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--overwrite", action="store_true", help="Recompute layers and Gemini rows that already exist.")
    return parser.parse_args()


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--config":
        if len(argv) != 2:
            raise SystemExit("Usage: python -m ember.score_frozen_snmf --config <file.yaml>")
        score_dictionary(load_score_config(Path(argv[1])))
        return
    score_dictionary(parse_args())


if __name__ == "__main__":
    main()
