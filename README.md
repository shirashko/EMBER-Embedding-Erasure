# EMBER

**EMBedding ERasure (EMBER)** is a plug-and-play module for erasing concepts from
language models. It uses Sparse Matrix Factorization to remove concept-related
features directly from the token embeddings, and can be combined with other
unlearning methods (e.g., PISCES, CRISP, RMU, SNMF). Augmenting those methods with
EMBER improves erasure efficacy and specificity and substantially increases
robustness to relearning, with minimal coherence loss.

Paper: [Don't Forget Your Embeddings: Robust Knowledge Erasure via Precise Editing of Embeddings](https://arxiv.org/abs/2606.03695)

This repository covers the full pipeline: extracting and interpreting concept
features, running erasure with any method (with or without EMBER), and evaluating
efficacy, specificity, coherency and robustness to relearning. Two models are
supported: `google/gemma-2-2b-it` and `meta-llama/Llama-3.1-8B-Instruct`.

[`demo.ipynb`](demo.ipynb) walks through erasing *Harry Potter* from Gemma with
EMBER + SNMF.

## Setup

```bash
git clone --recurse-submodules https://github.com/ClarSu/EMBER-Embedding-Erasure.git
cd EMBER-Embedding-Erasure

conda create -n ember python=3.10 -y
conda activate ember
pip install -r requirements.txt
pip install -e .

cp .env.example .env   # then add your HF_TOKEN and GEMINI_API_KEY
```

The vendored methods live in `external/` as submodules (snmf, CRISP, wmdp, PISCES).
`HF_TOKEN` is needed to download gated models (e.g. Llama-3.1-8B-Instruct);
`GEMINI_API_KEY` is needed for feature interpretation and the Alpaca coherence judge
(the erasure grids run without it). See `.env.example` for details.

## Concept features

Erasure reads precomputed concept features from `mf_outputs/`. Each concept has two
factorizations:

- **Embedding features**: a sparse factorization of the token-embedding matrix. This
  is all EMBER needs.
- **MLP features**: Semi-NMF over MLP activations, used by the SNMF erasure method.

Each is paired with an LLM-written interpretation that selects the concept-related
features (`potential_features.csv`).

### Provided features (recommended)

We publish all features for both models on the Hugging Face Hub:
[**ClSu/ember-features**](https://huggingface.co/datasets/ClSu/ember-features).
Erasure runs download them automatically (scoped to the run's model, rank, seed and
concepts; files already present are skipped), so no manual step is needed. To
pre-fetch:

```python
from huggingface_hub import snapshot_download
snapshot_download("ClSu/ember-features", repo_type="dataset", local_dir="mf_outputs",
                  allow_patterns=["google_gemma-2-2b-it/**/Harry_Potter/**"])
```

### Re-training the features

You can also regenerate the features. The factorization step is seeded and
reproducible; the interpretation step calls an LLM and may select a slightly
different feature set between runs.

The training has two tracks. EMBER only needs the **embedding** track; add the
**MLP** track as well if you also want to run SNMF erasure.

```bash
# Factorize. Drop --skip-mlp to also build the MLP track (for SNMF).
python -m ember.train_mf_features --concepts "Harry Potter" --ranks 100 \
    --model-name google/gemma-2-2b-it --seed 42 --skip-mlp

# Interpret and select concept features (needs GEMINI_API_KEY).
python -m ember.interpret_features --concepts "Harry Potter" --rank 100 \
    --model-name google/gemma-2-2b-it --seed 42 --tracks embedding
```

For Llama, use rank 200 and its model name:

```bash
python -m ember.train_mf_features --concepts "Harry Potter" --ranks 200 \
    --model-name meta-llama/Llama-3.1-8B-Instruct --seed 42 --skip-mlp
python -m ember.interpret_features --concepts "Harry Potter" --rank 200 \
    --model-name meta-llama/Llama-3.1-8B-Instruct --seed 42 --tracks embedding
```

To use whatever is already in `mf_outputs/` and never contact the Hub, pass
`--features-source local` to the erasure command (below) or set
`features_source: local` in the config.

## Running erasure

Each run is driven by a YAML config in `configs/` plus CLI overrides:

```bash
python -m ember.run_erasure --config configs/snmf_ember_gemma.yaml \
    --concepts "Harry Potter" --train-eval mc
```

**Methods** (the `method` field / config prefix): `snmf`, `rmu`, `crisp`, `pisces`,
`ember`. Config names follow `<method>_<model>.yaml` for the method alone and
`<method>_ember_<model>.yaml` to augment it with the EMBER embedding edit, where
`<model>` is `gemma` or `llama`. For example, `configs/rmu_ember_llama.yaml` runs RMU
with EMBER on Llama.

**Modes**: `--train-eval {mc,open}` selects the question format that drives the grid
search. The pipeline runs a grid search, validates the top configurations, evaluates
the best one on the held-out test set, and (if enabled) measures relearning.

**Results** are written under:

```
results/<method>[_ef]/<model>/rank<R>/seed<S>/
    train_<mode>/   # grid search + validation
    test_<mode>/    # final-test metrics and relearning
```

The `_ef` suffix marks runs that used EMBER as a pre-step (method + EMBER). Metrics
include efficacy, specificity, MMLU retention, Alpaca instruction-following and
fluency, the harmonic aggregate, and post-relearning accuracy.

## Citation

```bibtex
@misc{suslik2026dontforgetembeddingsrobust,
  title         = {Don't Forget Your Embeddings: Robust Knowledge Erasure via Precise Editing of Embeddings},
  author        = {Clara Haya Suslik and Or Shafran and Mor Geva},
  year          = {2026},
  eprint        = {2606.03695},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2606.03695},
}
```
