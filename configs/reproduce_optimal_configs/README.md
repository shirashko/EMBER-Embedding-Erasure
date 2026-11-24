# Reproduce optimal unlearning configs

This directory holds **pinned erasure configs** for rerunning the best hyperparameters found during grid search — one per `(method, model, concept)` triplet.

The source of truth is [`optimal_unlearning_hyperparams.yaml`](optimal_unlearning_hyperparams.yaml). Generated run configs live under:

```
<method>/<model>/<concept>.yaml
```

For example: `snmf/gemma/ancient_rome.yaml`.

## Goal

These configs are meant to **reproduce a single optimal unlearning run** and save the resulting checkpoint, not to repeat a full grid search. Each file therefore sets `topk: 1` and collapses method grids to one hyperparameter cell.

Non-grid settings (batch sizes, eval thresholds, etc.) are inherited from the corresponding base config in `configs/` (e.g. `crisp_llama.yaml`).

## Generate configs

```bash
python scripts/generate_reproduce_configs.py
```

Use `--input` to point at another metadata file (e.g. `optimal_unlearning_hyperparams_more_concepts.yaml`).

## Run a config

```bash
python -m ember.run_erasure \
    --config configs/reproduce_optimal_configs/snmf/gemma/ancient_rome.yaml \
    --concepts "Ancient Rome" --train-eval mc \
    --features-source local
```

Use `--train-eval open` for PISCES configs.

## Run all configs on SLURM

Generate configs and the job manifest (if not already present):

```bash
python scripts/generate_reproduce_configs.py
```

Jobs are submitted with the generic array template
`scripts/slurm/ember_array.slurm`, which runs one manifest row per array task.
The manifest at `configs/reproduce_optimal_configs/jobs.manifest.tsv` lists
`config<TAB>concept<TAB>train_eval` per line; task `i` runs line `i + 1`.

Environment used by the template (export in your shell; sbatch forwards it):

| Variable | Required | Meaning |
|---|---|---|
| `MANIFEST` | yes | Manifest TSV, absolute or relative to the repo root |
| `RUNNER` | yes | Runner for that manifest; here `scripts/reproduce_unlearning.sh` |
| `EMBER_ROOT` | no | Repo checkout; defaults to the directory you run `sbatch` from |
| `CONDA_ENV` | no | Conda env name or prefix to activate |
| `CONDA_HOME` | no | Conda install prefix |
| `SKIP_LLM_JUDGE` | no | `1` skips Gemini (MC-only configs; refused for `open` rows) |
| `REPRODUCE_OVERWRITE` | no | `1` re-runs cells that already have results |

Secrets and machine paths (`HF_TOKEN`, `GOOGLE_API_KEY`, `HUGGINGFACE_HUB_CACHE`, ...)
go in `.env` (see `.env.example`). Partition, account and mail address cannot be
set in `#SBATCH` lines, so pass them as sbatch flags, e.g. from your own
`SLURM_PARTITION`, `SLURM_ACCOUNT` and `SLURM_MAIL_USER` shell variables.

Submit one GPU array task per manifest row:

```bash
mkdir -p slurm_outputs slurm_errors
export RUNNER=scripts/reproduce_unlearning.sh
export MANIFEST=configs/reproduce_optimal_configs/jobs.manifest.tsv
N=$(wc -l < "$MANIFEST")
sbatch --array=0-$((N - 1)) \
    --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" \
    --mail-user="$SLURM_MAIL_USER" --mail-type=END,FAIL \
    scripts/slurm/ember_array.slurm
```

Run a single config (example: first manifest entry) with `--array=0`.

Another reproduce manifest (e.g. `jobs.manifest.rmu.gemma.tsv`) works the same way:
set `MANIFEST` to it; the template passes it to the runner as `REPRODUCE_MANIFEST`.

Logs: `slurm_outputs/ember_array_<job>_<task>.out` and `slurm_errors/`.
Checkpoints: `unlearned_checkpoints/<method>/<model>/<concept>/`.

## Method-specific pinning

The standard erasure pipelines sweep some choices via hard-coded grids. For reproduction we added config fields to pin the winning combination directly.

### CRISP — `layer_ranges`

The default CRISP grid searches over a fixed set of LoRA layer spans. Reproduce configs set `layer_ranges` to the single winning triple `(layer_lo, layer_hi, layer_step)`, e.g. `[5, 19, 2]` on Llama.

### RMU — `update_settings`

The default RMU grid searches over preset layer targets. Each entry specifies:

- `setting_name` — label for logging/results (e.g. `S2_lid8_L678`)
- `layer_id` — layer whose activations drive the RMU loss
- `layer_ids` — layers whose weights receive gradient updates (typically MLP `down_proj`)

Example:

```yaml
update_settings:
  - setting_name: S2_lid8_L678
    layer_id: 8
    layer_ids: 6,7,8
```

### SNMF — `layer_ranges_in` / `layer_ranges_out`

The default SNMF grid searches over hard-coded MLP layer spans on both sides of the intervention:

- `layer_ranges_in` — layers where **up-proj** (`W_in`) is edited
- `layer_ranges_out` — layers where **down-proj** (`W_out`) is edited

Reproduce configs pin one span per side, together with single `in_deltas` / `out_deltas` values. Example on Gemma (Ancient Rome):

```yaml
in_deltas: [7.0]
out_deltas: [1.0]
layer_ranges_in: [[0, 8]]
layer_ranges_out: [[13, 25]]
```

Pinned layer ranges and RMU update settings are validated against the allowed grid presets for each model at config load time.

## Checkpoints

Reproduce configs enable checkpoint export:

```yaml
checkpoint:
  enabled: true
  root: unlearned_checkpoints
```

After the final-test unlearning step (and **before** relearning), the pipeline saves the unlearned model to:

```
unlearned_checkpoints/<method>/<model>/<concept>/
```

Each directory contains Hugging Face `save_pretrained` weights, the tokenizer, and `unlearned_checkpoints.json` (run metadata + hyperparameters). Re-run with `--overwrite` to replace an existing checkpoint.

### What gets saved

- **CRISP** trains a small LoRA adapter on top of the base model (via PEFT). The checkpoint contains only that adapter (`adapter_model.safetensors`, etc.), not the full base weights. Load with the original base model:

  ```python
  from peft import PeftModel
  from transformers import AutoModelForCausalLM

  base = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
  model = PeftModel.from_pretrained(base, "unlearned_checkpoints/crisp/.../ancient_rome")
  ```

- **RMU, SNMF, PISCES** edit weights directly in the model. The checkpoint is a standalone full model — load from the directory alone:

  ```python
  from transformers import AutoModelForCausalLM

  model = AutoModelForCausalLM.from_pretrained("unlearned_checkpoints/snmf/.../ancient_rome")
  ```

CRISP checkpoints are much smaller on disk, the others are roughly the size of the full model.

## Gemini (LLM judge) eval

Reproduce configs use the default `eval.skip_llm_judge: false`, so the pipeline
calls Gemini for:

- **Alpaca** coherence scoring (grid, validate, and final test when `eval.alpaca: true`)
- **Open-ended QA** judging (PISCES jobs with `train_eval: open`)

`scripts/reproduce_unlearning.sh` sources `scripts/ember_runner_env.sh`, which loads
`${REPO_ROOT}/.env` and exports `GEMINI_API_KEY` (or falls back to `GOOGLE_API_KEY`).

Set in `.env`:

```bash
GEMINI_API_KEY=...
```

MC-only methods (CRISP, RMU, SNMF) still run MMLU + MCQ without Gemini; Alpaca runs
when the judge is enabled. Cached LLM-judge baselines under `data/baselines/` are
reused when present to avoid recomputing pristine-model Alpaca/open-QA scores.

To skip Gemini (e.g. cluster runs without API access):

```yaml
eval:
  skip_llm_judge: true
```
