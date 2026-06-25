# WMDP corpora (CRISP unlearning)

Prepared JSONL files live here for WMDP bio/cyber runs (`data.source: wmdp`).

## Two-step workflow

**1. Prepare training corpora** (downloads from Hugging Face or reads local raw JSONL):

```bash
export HF_TOKEN=...   # required for gated forget corpora
python scripts/prepare_wmdp_corpora.py --domain all --seed 42
```

**1b. Prepare evaluation MCQs** (WMDP benchmark + MMLU auxiliary sets):

```bash
python scripts/prepare_wmdp_mcq.py --domain all
```

This writes CRISP-format JSON used by the pipeline when `data.source: wmdp`:

```
data/wmdp/bio/bio_mcq.json
data/wmdp/bio/high_school_bio_mcq.json
data/wmdp/bio/college_bio_mcq.json
data/wmdp/cyber/cyber_mcq.json
...
```

**2. Train** (reuses the same files on every run):

Training corpora layout:

```
data/wmdp/
  coherency_prompts.json          # shipped with the repo
  bio/
    bio_forget_dataset_cleaned.jsonl
    bio_retain_dataset_cleaned.jsonl
    prepared_manifest.json
  cyber/
    cyber_forget_dataset_cleaned.jsonl
    cyber_retain_dataset_cleaned.jsonl
    prepared_manifest.json
```

Bio forget/retain are sampled to 5000 examples; cyber uses the full corpus.
Re-run corpora with `--force` after changing `--max-len` or `--bio-n-examples`.

```bash
python -m ember.run_erasure \
  --config configs/wmdp_configs/crisp/gemma/bio.yaml \
  --train-eval mc --concepts bio
```

## Layout details

Each JSONL row is `{"text": "..."}`.

| Path | Description |
|------|-------------|
| `raw/{domain}/{domain}_{split}_dataset.jsonl` | Optional local raw input (skip HF) |
| `{domain}/{domain}_{split}_dataset_cleaned.jsonl` | Prepared training corpora |
| `{domain}/prepared_manifest.json` | Provenance (sources, counts, seeds) |
| `{domain}/*_mcq.json` | WMDP + MMLU MCQ eval files (`prepare_wmdp_mcq.py`) |

## Evaluation (when `data.source: wmdp`)

The erasure pipeline maps EMBER metrics to WMDP as follows:

| EMBER metric | WMDP source |
|--------------|-------------|
| `qa_acc` / efficacy | Primary WMDP MCQ (`bio_mcq.json` / `cyber_mcq.json`), logits A/B/C/D (CRISP-style) |
| `simdom_acc` / specificity (partial) | Mean of high-school + college MMLU subject MCQs for the domain |
| `mmlu_frac` | Standard EMBER MMLU subset (unchanged) |

Requires `--train-eval mc`. Baselines are cached under `data/baselines/<model>/baseline_wmdp_train_mc.json`.

## Hugging Face sources

The prepare script tries, in order:

1. Local raw JSONL under `data/wmdp/raw/`
2. `cais/wmdp-corpora` subsets (`bio-retain-corpus`, etc.)
3. Gated forget: `cais/wmdp-bio-forget-corpus`, `cais/wmdp-cyber-forget-corpus`

Request access on Hugging Face for the gated forget datasets before running.

## Wiki retain

Configs with `retain_type: wiki` or `wiki-bio` load public HF datasets at **training**
time (not via the prepare script). Domain-matched retain (`retain_type: bio` / `cyber`)
uses the prepared JSONL files above.
