# Hyperparameter Grids (No EMBER Step)

Grid-search spaces for the non-EMBER configs in `configs/`:

- `rmu_{gemma,llama}.yaml`
- `crisp_{gemma,llama}.yaml`
- `snmf_{gemma,llama}.yaml`
- `pisces_{gemma,llama}.yaml`

All set `ember_step.enabled: false`, so only the method's own HPs are swept.

**Pipeline:** grid (all cells) → **top-K filter** → validate (top K + Alpaca) → final test (best HP).

**Reading the tables below**

- **Swept** = every combination of these values is one grid cell.
- **Source:** `YAML` = listed in the config file; `code default` = used when the YAML key is missing or `null`.
- **Fixed** = same value for every cell (not part of the search).

---

## Shared config keys

| Key | Meaning |
|-----|---------|
| `method` | Algorithm: `rmu`, `crisp`, `snmf`, or `pisces`. |
| `model_name` | HF model id. |
| `rank` | SNMF/embedding factorization rank in `mf_outputs/` (must match trained features; not swept here). |
| `seed` | Reproducibility seed. |
| **`topk`** | After the grid, keep the **best K rows** by train `harmonic` → `top_hps.csv` → validate. Not a grid dimension (base configs: **15**). |
| `run_tests_after_train` | Run held-out `test_mc` / `test_open` after picking best HP. |
| `selection.ratio_thresh` | SNMF/embed: min concept/neutral feature score ratio (higher = fewer features). |
| `eval.min_mmlu` | Grid early-stop if general knowledge drops too much (vs baseline). |
| `eval.max_qa_acc` | Grid early-stop if concept QA stays too high (not enough forgetting). |
| `eval.alpaca` | Alpaca scoring in validate/final test (Gemini; skipped in grid for these methods). |
| `eval.skip_llm_judge` | Skip Gemini (Alpaca + open-QA). |
| `relearning.enabled` | Post-test fine-tune on concept text to measure knowledge return. |

**Defaults:** Gemma `rank: 100`; Llama `rank: 200`. Llama configs also use `eval.alpaca_batch_size: 75`.

**CLI (not YAML):** `--concepts`, `--train-eval mc|open`, `--skip-llm-judge`, `--overwrite`.

---

## RMU — 144 cells

Configs: `configs/rmu_{gemma,llama}.yaml`

### Grid search (swept)

| Parameter | Values | Source |
|-----------|--------|--------|
| `lr_grid` | `1e-5`, `1e-4`, `3e-4` | code default (`lr_grid: null` in YAML) |
| `alpha_grid` | Gemma: `10`, `30`, `50`, `100` · Llama: `30`, `50`, `100`, `300` | code default |
| `steering_grid` | `30`, `100`, `300`, `1000` | code default |
| `update_settings` | 3 presets per model (see below) | code default |

`update_settings` fields: `layer_id` = activation layer for RMU loss; `layer_ids` = layers whose weights are updated (`param_ids: [6]`, WMDP MLP default).

**Gemma-2** (`google/gemma-2-2b-it`):

| Preset | `layer_id` | `layer_ids` |
|--------|------------|-------------|
| `S1_lid7_L567` | 7 | 5, 6, 7 |
| `S2_lid8_L678` | 8 | 6, 7, 8 |
| `S3_lid6_L456` | 6 | 4, 5, 6 |

**Llama-3.1** (`meta-llama/Llama-3.1-8B-Instruct`):

| Preset | `layer_id` | `layer_ids` |
|--------|------------|-------------|
| `S1_lid7_L567` | 7 | 5, 6, 7 |
| `S2_lid9_L789` | 9 | 7, 8, 9 |
| `S3_lid11_L91011` | 11 | 9, 10, 11 |

Presets defined in `ember/erasure/methods/rmu.py`.

**Total:** `3 × 4 × 4 × 3 = 144`

### Fixed (not swept)

| Parameter | Value | Source |
|-----------|-------|--------|
| `batch_size` | `16` | YAML |
| `max_num_batches` | `150` | YAML |
| `min_len` | `50` | YAML |
| `max_len` | `2000` | YAML |

---

## CRISP — 144 cells

Configs: `configs/crisp_{gemma,llama}.yaml`

### Grid search (swept)

| Parameter | Values | Source |
|-----------|--------|--------|
| `k_features_grid` | `5`, `10`, `20` | YAML |
| `alpha_grid` | `5.0`, `10.0`, `20.0`, `50.0` | YAML |
| `lr_grid` | `5e-5`, `1e-4`, `5e-4` | YAML |
| `layer_ranges` | Gemma: `(4,14,2)`, `(5,15,2)`, `(4,20,2)`, `(5,21,2)` · Llama: `(5,19,2)`, `(4,18,2)`, `(5,29,2)`, `(4,28,2)` | code default (key omitted in YAML) |

Each `layer_ranges` entry is `(layer_lo, layer_hi, layer_step)`: attach SAEs and LoRA on layers `lo, lo+step, …, hi`.

**Total:** `3 × 4 × 3 × 4 = 144`

### Fixed (not swept)

| Parameter | Gemma | Llama | Source |
|-----------|-------|-------|--------|
| `num_epochs` | `2` | `2` | YAML |
| `lora_rank` | `4` | `4` | YAML |
| `beta` | `0.99` | `0.99` | YAML |
| `gamma` | `0.01` | `0.01` | YAML |
| `batch_size` | `16` | `16` | YAML |
| `lora_batch_size` | `16` | `8` | YAML |
| `sae_cache` | `gemma_sae_cache` | `llama_sae_cache` | YAML |
| `max_len` | `2000` | `2000` | code default |

---

## SNMF — 144 cells

Configs: `configs/snmf_{gemma,llama}.yaml`

### Grid search (swept)

With `w_mode: both`, the grid is the Cartesian product of in-side and out-side sweeps.

| Parameter | Values | Source |
|-----------|--------|--------|
| `in_deltas` | `1.0`, `4.0`, `7.0`, `10.0` | YAML |
| `out_deltas` | `1.0`, `4.0`, `7.0`, `10.0` | YAML |
| `layer_ranges_in` | Gemma: `(0,25)`, `(0,8)`, `(0,12)` · Llama: `(0,31)`, `(0,10)`, `(0,16)` | code default |
| `layer_ranges_out` | Gemma: `(0,8)`, `(9,17)`, `(13,25)` · Llama: `(0,10)`, `(11,21)`, `(16,31)` | code default |

Each layer range is `(layer_lo, layer_hi)` inclusive.

| Parameter | Value | Source |
|-----------|-------|--------|
| `w_mode` | `both` | YAML (fixed; enables in×out product) |
| `feature_source` | `all` | YAML (fixed) |

**Total:** `(4 × 3) × (4 × 3) = 144`

### Fixed (not swept)

| Parameter | Value | Source |
|-----------|-------|--------|
| `dtype` | `bf16` | YAML |

---

## PISCES — 144 cells

Configs: `configs/pisces_{gemma,llama}.yaml`

### Grid search (swept)

| Parameter | Values | Source |
|-----------|--------|--------|
| `ks` | `0.95`, `0.9`, `0.85`, `0.8`, `0.75`, `0.7`, `0.6`, `0.5`, `0.4`, `0.3`, `0.2`, `0.1` | YAML |
| `values` | `4`, `7`, `10`, `13`, `18`, `21`, `24`, `30`, `36`, `42`, `50`, `60` | YAML |

Feature lists: `data/pisces_concept_features_{gemma,llama}.json` (same for both model YAMLs).

**Total:** `12 × 12 = 144`

### Fixed (not swept)

| Parameter | Value | Source |
|-----------|-------|--------|
| `dtype` | `bf16` | YAML |

---

## Reproduce configs

`configs/reproduce_optimal_configs/**` pin winning HPs (usually **1 cell** per concept), not full 144-cell sweeps.
