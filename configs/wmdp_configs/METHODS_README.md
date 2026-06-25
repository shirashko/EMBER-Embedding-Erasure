# CRISP on WMDP

CRISP uses SAEs to identify features that are highly active on the forget data but not on the retain data. These salient features represent the specific knowledge we want to unlearn.

Optimal hyperparameters for WMDP-Bio and WMDP-Cyber from the CRISP paper (*Persistent Concept Unlearning via Sparse Autoencoders*):

| Model | Domain | SAE layers | k | λ (`alpha`) | LoRA rank | lr |
|-------|--------|------------|---|-------------|-----------|-----|
| Gemma-2-2B | Cyber | [4, 6, 8, 10, 12, 14] | 50 | 20 | 4 | 4×10⁻⁵ |
| Gemma-2-2B | Bio | [4, 6, 8, 10, 12, 14] | 30 | 30 | 8 | 4×10⁻⁵ |
| Llama-3.1-8B | Cyber | [4, 6, 8, …, 18] | 50 | 30 | 4 | 4×10⁻⁵ |
| Llama-3.1-8B | Bio | [4, 6, 8, …, 28] | 10 | 40 | 8 | 4×10⁻⁵ |

Layer ranges map to `crisp.layer_ranges: [[lo, hi, step]]` (e.g. `[4, 14, 2]` → layers 4, 6, 8, 10, 12, 14).

## Config files

Pinned single-cell configs (`topk: 1`):

| Config | Path |
|--------|------|
| Gemma / Bio | `configs/wmdp_configs/crisp/gemma/bio.yaml` |
| Gemma / Cyber | `configs/wmdp_configs/crisp/gemma/cyber.yaml` |
| Llama / Bio | `configs/wmdp_configs/crisp/llama/bio.yaml` |
| Llama / Cyber | `configs/wmdp_configs/crisp/llama/cyber.yaml` |

## Data workflow

WMDP runs set `data.source: wmdp`. Corpora are **prepared offline** and reused across runs:

```bash
# Once per machine / corpus version
export HF_TOKEN=...
python scripts/prepare_wmdp_corpora.py --domain all --seed 42
python scripts/prepare_wmdp_mcq.py --domain all

# Every training run
python -m ember.run_erasure \
  --config configs/wmdp_configs/crisp/gemma/cyber.yaml \
  --train-eval mc --concepts cyber
```

## Evaluation

When `data.source: wmdp`, the pipeline uses **WMDP MCQ logits eval** (matching CRISP `eval.py`), not `data/mc_questions.json`:

- **Efficacy:** `bio_mcq.json` / `cyber_mcq.json` from `cais/wmdp`
- **Specificity (SimDom slot):** mean accuracy on high-school + college MMLU subjects for that domain
- **MMLU:** standard EMBER MMLU subset (unchanged)

Prepare eval files with `scripts/prepare_wmdp_mcq.py`. Use `--train-eval mc` only.

Set `eval.skip_llm_judge: true` in the YAML to skip Gemini Alpaca/open-QA judging (WMDP runs use MCQ + MMLU only). No `GEMINI_API_KEY` required.

| `data.source` | Forget | Retain | Coherency | Eval QA |
|---------------|--------|--------|-----------|
| `ember` (default) | `data/concept_sentences.json` | `data/neutral_sentences.json` | `data/coherency_prompts.json` |
| `wmdp` | prepared forget JSONL | prepared JSONL or wiki HF | `data/wmdp/coherency_prompts.json` | WMDP MCQ + MMLU aux |

WMDP-specific options (`data.wmdp`):

| Field | Default | Description |
|-------|---------|-------------|
| `features_source` | `local` in WMDP YAMLs | Skip EMBER `mf_outputs` HF download (`ember_step` is off) |
| `retain_type` | domain name (`bio` / `cyber`) | Benign corpus: `wiki`, `wiki-bio`, `bio`, or `cyber` |
| `data_root` | `data/wmdp` | Root for prepared JSONL (see `data/wmdp/README.md`) |
| `wiki_retain_max_len` | `1000` | Paragraph chunk size for `wiki` / `wiki-bio` retain only |
| `n_examples` | `null` | Optional cap at train time |

Preprocessing knobs (`scripts/prepare_wmdp_corpora.py`): `--max-len 1000`, `--bio-n-examples 5000`, `--seed`.

Checkpoints are written after final-test unlearning when enabled in the YAML:

```yaml
checkpoint:
  enabled: true
  root: unlearned_checkpoints
```

Output: `unlearned_checkpoints/crisp/<model>/<concept>/` (LoRA adapter + tokenizer + `unlearned_checkpoints.json`). Re-run with `--overwrite` to replace an existing checkpoint.

## Code layout

```
scripts/prepare_wmdp_corpora.py   # offline corpus build (HF → cleaned JSONL)
scripts/prepare_wmdp_mcq.py       # WMDP + MMLU MCQ JSON for eval
ember/forget_retain.py            # load_forget_retain(), load_coherency_prompts()
ember/wmdp/mcq_eval.py            # WMDP logits MCQ evaluation
ember/evals/wmdp_baselines.py     # cached WMDP eval baselines
ember/wmdp/prepare_corpora.py     # download + CRISP preprocessing
ember/wmdp/preprocess.py          # clean, right-truncate, sample
ember/wmdp/corpora.py             # load prepared JSONL for training
ember/wmdp/coherency.py           # WMDP coherency prompts
```

CRISP (`ember/erasure/methods/crisp.py`) calls the unified loader in `on_concept_start`, no WMDP-specific branches in the training loop.

## RMU on WMDP

The original WMDP paper evaluates RMU on **Zephyr-7B**, **Mixtral-8×7B**, and **Yi-34B** — not on the Gemma / Llama models used in this repo. The search space and chosen settings they report are nevertheless useful as a reference if you add RMU WMDP configs here.

**Search space**

- **Layer** ℓ for the unlearning loss: grid from layer 3 through the last layer.
- **Training batches** (gradient steps): `{150, 300, 500}`.
- **Target modules:** MLP blocks only (paper argues knowledge is concentrated there).

**Selected hyperparameters (paper)**

| Model | Layer ℓ | Retain weight α | Unlearning coeff. *c* | Max batches |
|-------|---------|-----------------|----------------------|-------------|
| Zephyr-7B | 7 | 1200 | 6.5 | (from grid above) |
| Mixtral-8×7B | 7 | 1600 | 300 | (from grid above) |
| Yi-34B | 15 | 350 | 300 | (from grid above) |

Map these to EMBER’s `rmu` YAML fields (`update_settings`, `alpha_grid`, `steering_grid`, `max_num_batches`, etc.) when pinning configs for Gemma or Llama — expect to re-tune ℓ, α, and *c* for smaller models and different layer counts.