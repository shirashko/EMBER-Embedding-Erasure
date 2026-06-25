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

Submit one GPU array task per config (24 jobs by default):

```bash
sbatch slurm/reproduce_unlearning.slurm
```

Run a single config (example: first manifest entry):

```bash
sbatch --array=0 slurm/reproduce_unlearning.slurm
```

The manifest at `configs/reproduce_optimal_configs/jobs.manifest.tsv` lists
`config<TAB>concept<TAB>train_eval` per line. Adjust `#SBATCH --array=0-23` in
`slurm/reproduce_unlearning.slurm` if the manifest length changes.

Logs: `slurm_outputs/reproduce_unlearning/` and `slurm_errors/reproduce_unlearning/`.
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

## Skipping Gemini (LLM judge) eval

Reproduce configs set `eval.skip_llm_judge: true` so the run does not call Gemini. When enabled:

- **Skipped:** Alpaca scoring, open-ended QA judging, and the validate stage (Alpaca re-score).
- **Still runs:** MMLU and multiple-choice QA (for `train_eval: mc` and `test_mc`).
- **Baselines:** Cached LLM-judge baselines are reused if present, otherwise those sets are omitted (no API key, no dummy values).
- **Checkpoints:** Still saved in final test after unlearning, before any remaining eval.

To enable manually:

```yaml
eval:
  skip_llm_judge: true
```
