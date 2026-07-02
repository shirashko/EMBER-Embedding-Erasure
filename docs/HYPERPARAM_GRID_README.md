# Hyperparameter Grids (No EMBER Step)

Grid-search spaces for the non-EMBER configs in `configs/`:

- `rmu_{gemma,llama}.yaml`
- `crisp_{gemma,llama}.yaml`
- `snmf_{gemma,llama}.yaml`
- `pisces_{gemma,llama}.yaml`

All set `ember_step.enabled: false`, so only the method's own HPs are swept.

**Pipeline:** grid (all cells) → **top-K filter** → validate (top K + Alpaca) → final test (best HP).

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

YAML sets `lr_grid`, `alpha_grid`, `steering_grid` to `null` → code defaults below.

| Swept param | Meaning |
|-------------|---------|
| `lr_grid` | Fine-tuning learning rate. |
| `alpha_grid` | Retain-loss weight (preserve neutral behavior). |
| `steering_grid` | Forget steering-vector magnitude. |
| `update_settings` | Layer preset: activation layer (`layer_id`) + weight-update layers (`layer_ids`). See table below. |

Each preset is `(setting_name, layer_id, layer_ids)`. Weights updated at `layer_ids` use `param_ids: [6]` (WMDP default MLP matrix). Activations for the RMU loss are read from the full block at `layer_id`.

**Gemma-2** (`google/gemma-2-2b-it`, 26 layers):

| Preset | `layer_id` (activations) | `layer_ids` (weight updates) |
|--------|--------------------------|------------------------------|
| `S1_lid7_L567` | 7 | 5, 6, 7 |
| `S2_lid8_L678` | 8 | 6, 7, 8 |
| `S3_lid6_L456` | 6 | 4, 5, 6 |

**Llama-3.1** (`meta-llama/Llama-3.1-8B-Instruct`, 32 layers):

| Preset | `layer_id` (activations) | `layer_ids` (weight updates) |
|--------|--------------------------|------------------------------|
| `S1_lid7_L567` | 7 | 5, 6, 7 |
| `S2_lid9_L789` | 9 | 7, 8, 9 |
| `S3_lid11_L91011` | 11 | 9, 10, 11 |

Defined in `ember/erasure/methods/rmu.py` (`GEMMA_UPDATE_SETTINGS`, `LLAMA_UPDATE_SETTINGS`).

**Gemma defaults:** `lr` ×3, `alpha` ×4 `[10,30,50,100]`, `steering` ×4, `update_settings` ×3 → **144**.

**Llama defaults:** same `lr`/`steering`; `alpha` ×4 `[30,50,100,300]`.

**Fixed (not swept):** `batch_size`, `max_num_batches`, `min_len`, `max_len`.

---

## CRISP — 144 cells

Configs: `configs/crisp_{gemma,llama}.yaml`

| Swept param | Meaning |
|-------------|---------|
| `k_features_grid` | Top SAE features to target per layer. |
| `alpha_grid` | Unlearning loss strength. |
| `lr_grid` | LoRA learning rate. |
| `layer_ranges` | Layer span `(lo, hi, step)` for SAE+LoRA (4 defaults if unset). |

**From YAML:** `k_features` ×3 `[5,10,20]`, `alpha` ×4, `lr` ×3, `layer_ranges` ×4 → **144**.

**Fixed:** `num_epochs`, `lora_rank`, `beta`, `gamma`, `sae_cache`, batch sizes, `max_len`.

---

## SNMF — 144 cells

Configs: `configs/snmf_{gemma,llama}.yaml`

| Swept param | Meaning |
|-------------|---------|
| `in_deltas` / `out_deltas` | Edit strength on MLP up_proj / down_proj. |
| `layer_ranges_in` / `layer_ranges_out` | Layer spans for in/out edits (3 defaults each if unset). |
| `w_mode` | `both` = sweep in and out together. |
| `feature_source` | Which SNMF features to use (`all`, `activation`, etc.). |

**From YAML:** `w_mode: both`, deltas ×4 each side, ranges ×3 each → `(4×3) × (4×3)` = **144**.

---

## PISCES — 144 cells

Configs: `configs/pisces_{gemma,llama}.yaml`

| Swept param | Meaning |
|-------------|---------|
| `ks` | Feature sparsity threshold (how many activations to keep). |
| `values` | Edit magnitude on selected MLP features. |

**From YAML:** 12 × 12 → **144**. Features from `data/pisces_concept_features_{gemma,llama}.json`.

---

## Reproduce configs

`configs/reproduce_optimal_configs/**` pin winning HPs (usually **1 cell** per concept), not full 144-cell sweeps.
