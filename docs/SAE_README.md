## Sparse Autoencoder (SAE) Architectures for Unlearning

This repository leverages Sparse Autoencoders (SAEs) to identify and intervene on specific concepts within Large Language Models (LLMs). We employ two distinct mechanistic unlearning frameworks: **CRISP** and **PISCES**.

Because these methods target different components of the transformer architecture, they rely on entirely different families of SAEs.

### Methodological Overview

The core difference between the two methods lies in **where** they intervene and **how** they update the model's behavior:

| Method | SAE Source Family | Intervention Stream | Edit Mechanism |
| --- | --- | --- | --- |
| **CRISP** | Gemma/Llama Scope (**Residual**) | Residual Stream | Block output hooks + LoRA unlearning |
| **PISCES** | Gemma/Llama Scope (**MLP**) | MLP Output (`W_out`) | Direct weight edits on targeted feature IDs |

---

### 1. CRISP: Residual Stream Interventions

CRISP operates on the **residual stream**—the central "highway" of information passing through the transformer. By hooking into the block outputs, it identifies when a target concept is active in the main stream and uses a Low-Rank Adaptation (LoRA) mechanism to dynamically suppress it.

*Note: Instruct models reuse the base model's SAE weights.*

#### Gemma (`google/gemma-2-2b-it`)

CRISP downloads these SAEs at runtime into local cache directories.

* **SAE Class:** `JumpReLUSAE`
* **Hugging Face Repo:** `google/gemma-scope-2b-pt-res`
* **Target Stream:** Residual (`pt-res`)
* **Dictionary Size:** `width_16k` (16,384 latents)
* **Checkpoint Path:** `layer_{L}/width_16k/{average_l0_*}/params.npz`

#### Llama (`meta-llama/Llama-3.1-8B-Instruct`)

* **SAE Class:** `TopkSae`
* **Hugging Face Repo:** `fnlp/Llama3_1-8B-Base-LXR-8x`
* **Target Stream:** Residual (`LXR`)
* **Expansion Factor:** 8x
* **Checkpoint Path:** `Llama3_1-8B-Base-L{layer}R-8x/`

---

### 2. PISCES: MLP Weight Editing

Unlike CRISP, PISCES targets the **MLP (Feed-Forward) layers**, which function as the model's factual memory banks. It performs surgical, static edits directly on the network's weights (`W_out`) to erase precomputed concept features, requiring no forward-pass hooks during inference. PISCES loads these SAEs via the `sae_lens` library.

#### Gemma (`google/gemma-2-2b-it`)

* **Library:** `sae_lens`
* **Release:** `gemma-scope-2b-pt-mlp`
* **Hook/Type:** MLP Output (`type="mlp"`)
* **Dictionary Size:** `16k` *(Configured as `large=false` in feature JSONs)*
* **SAE ID:** `layer_{L}/width_16k/{average_l0_*}`

#### Llama (`meta-llama/Llama-3.1-8B-Instruct`)

* **Release:** `llama_scope_lxm_8x`
* **Hook/Type:** MLP Output (`type="mlp"`)
* **Dictionary Size:** `32k` *(Maps to 8x expansion in code)*
* **SAE ID:** `l{layer}m_8x`

---

### Understanding the Terminology

To understand the configuration paths above, it is helpful to define a few key terms specific to Mechanistic Interpretability and the Gemma Scope project:

* **Residual vs. MLP Streams:** * *Residual Stream:* The cumulative state of the token at a given layer. Editing here (CRISP) affects the holistic representation of the token before it moves to the next layer.
* *MLP Stream:* The localized factual processing unit of a specific layer. Editing here (PISCES) targets localized factual associations before they are added to the residual stream.


* **Dictionary Size / Expansion (e.g., `16k`, `8x`):** The number of interpretable features (latents) the SAE maps the model's dense activations into. A larger dictionary allows for more granular, monosemantic (single-meaning) concepts.
* **`{average_l0_*}`:** This variable represents the $L_0$ norm (the average number of features that activate per token). Because there is an inherent trade-off between *reconstruction fidelity* (requiring high $L_0$) and *feature interpretability/sparsity* (requiring low $L_0$), models like Gemma-2-2b offer a Pareto frontier of options. For PISCES, the pipeline uses a static dictionary mapping to load the "Canonical SAEs" (e.g., `average_l0_82` for Layer 12), ensuring an optimal balance for identifying and editing target features.

### Summary of SAE Configurations

The following table outlines the exact SAE specifications used across the two models and unlearning frameworks, highlighting the targeted streams and mathematical architectures.

| Framework | Target Model | Target Stream | Mathematical SAE Class | Source Repository / Release |
| :--- | :--- | :--- | :--- | :--- |
| **CRISP** | Gemma-2-2B-IT | Residual | `JumpReLUSAE` | `google/gemma-scope-2b-pt-res` |
| **CRISP** | Llama-3.1-8B-IT | Residual | `TopkSae` | `fnlp/Llama3_1-8B-Base-LXR-8x` |
| **PISCES** | Gemma-2-2B-IT | MLP | `JumpReLUSAE` | `gemma-scope-2b-pt-mlp` |
| **PISCES** | Llama-3.1-8B-IT | MLP | `TopkSae` | `llama_scope_lxm_8x` |