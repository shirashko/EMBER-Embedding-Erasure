#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from llm_utils.activation_generator import ActivationGenerator, extract_token_ids_sample_ids_and_labels
from factorization.seminmf import NMFSemiNMF
from huggingface_hub import login
from transformers import AutoTokenizer

from ember.local_datasets import ConceptDataset
from ember.timing import Timer
from ember.utils import (
    set_seed, resolve_device, _safe_model_name, _safe_concept, _safe_tokens,
    get_pipeline_path, save_df_to_csv, update_timing,
    get_embedding_matrix, get_special_token_ids, vector_to_logits,
    generate_token_contexts, collect_feature_rows_for_layer, fit_with_ridge,
    build_token_label_codes, compute_embedding_stats, collect_feature_rows_for_embeddings,
    compute_mlp_layer_stats,
    SparseMatrixFactorization,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Train MF features: MLP (SNMF) and embedding (Sparse MF) tracks.")

    ap.add_argument("--concepts", nargs="+", default=["Harry Potter"])
    ap.add_argument("--ranks", type=int, nargs="+", default=[100])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--model-name", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--model-device", type=str, default="auto")
    ap.add_argument("--data-device", type=str, default="cpu")
    ap.add_argument("--fitting-device", type=str, default="auto")
    ap.add_argument("--cache-dir", type=str, default=None,
                    help="HuggingFace model cache directory. Defaults to HF_HOME if unset.")
    ap.add_argument("--hf-token", type=str, default=None)

    ap.add_argument("--max-iterations", type=int, default=20_000)
    ap.add_argument("--sparsity", type=float, default=0.01,
                    help="WTA sparsity on F_ for MLP track (fraction of MLP dims kept per feature).")
    ap.add_argument("--g-sparsity", type=float, default=0.01,
                    help="WTA sparsity on G_ for embedding track (fraction of tokens kept per feature).")
    ap.add_argument("--k-proj", type=int, default=30)
    ap.add_argument("--skip-mlp", action="store_true", help="Skip the MLP track.")
    ap.add_argument("--skip-embedding", action="store_true", help="Skip the embedding track.")

    ap.add_argument("--outdir", type=str, default="mf_outputs", help="Root directory for all outputs.")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 1.25, 1.5, 1.75, 2.0])

    return ap.parse_args()


def main():
    from dotenv import load_dotenv
    load_dotenv()
    args = parse_args()
    set_seed(args.seed)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
        except (ValueError, RuntimeError) as e:
            print(f"[hf_token] login() raced and lost ({e}); falling back to HF_TOKEN env var")
            os.environ.setdefault("HF_TOKEN", hf_token)

    model_device = resolve_device(args.model_device)
    data_device = args.data_device
    fit_device = resolve_device(args.fitting_device)
    safe_model = _safe_model_name(args.model_name)

    act_generator = ActivationGenerator(
        args.model_name, model_device=model_device, data_device=data_device,
        mode="mlp",
    )
    model = act_generator.model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, use_fast=True)

    num_layers = (
        int(model.cfg.n_layers) if hasattr(model, "cfg")
        else int(model.n_layers) if hasattr(model, "n_layers")
        else len(model.blocks)
    )
    layers = list(range(num_layers))

    timing_path = Path(args.outdir) / safe_model / "timing.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)

    for concept_name in args.concepts:
        safe_concept = _safe_concept(concept_name)
        print(f"\n{'=' * 80}\n[CONCEPT] {concept_name}\n{'=' * 80}")

        # Fixed-seed neutral ordering for reproducible feature extraction.
        dataset = ConceptDataset(concept_name=concept_name, neutral_sample_seed=42)

        token_ids_all, sample_ids, labels_all = extract_token_ids_sample_ids_and_labels(dataset, act_generator)
        labels_arr = np.array(labels_all, dtype=object)
        is_concept_mask = (labels_arr == concept_name)
        is_neutral_mask = (labels_arr == "Neutral")
        token_ds = generate_token_contexts(token_ids_all, sample_ids, act_generator)

        for rank in args.ranks:
            print(f"\n[RUN] concept='{concept_name}' | rank={rank}")

            # =====================================================================
            # TRACK A: EMBEDDING (Sparse MF)
            # =====================================================================
            if args.skip_embedding:
                print("--> Skipping embedding track (--skip-embedding)")
            else:
                print("--> Starting embedding track")

                embed_pkl_path = get_pipeline_path(
                    args.outdir, safe_model, "pickles", rank,
                    safe_concept, "embedding", "embedding.pkl", seed=args.seed)
                embed_csv_path = get_pipeline_path(
                    args.outdir, safe_model, "csvs", rank,
                    safe_concept, "embedding", "token_features.csv", seed=args.seed)
                embed_stats_path = get_pipeline_path(
                    args.outdir, safe_model, "csvs", rank,
                    safe_concept, "embedding", "stats_embed.csv", seed=args.seed)

                if os.path.exists(embed_pkl_path) and os.path.exists(embed_stats_path):
                    print(f"[SKIP] Embedding track already done for '{concept_name}'.")
                else:
                    with Timer() as t_embed:
                        set_seed(args.seed)
                        token_to_code = build_token_label_codes(concept_name, dataset, tokenizer)
                        special = get_special_token_ids(model)
                        unique_tids = sorted(
                            [int(tid) for tid in set(token_ids_all.tolist()) if int(tid) not in special])

                        E = get_embedding_matrix(model)
                        # E_vprime : (|V'|, d_model)
                        E_vprime = E.index_select(
                            0, torch.tensor(unique_tids, dtype=torch.long, device=E.device)
                        ).detach()
                        # A_embed : (d_model, |V'|) - transposed so F_=(d_model,K) and G_=(|V'|,K)
                        A_embed = E_vprime.T.contiguous()

                        nmf_embed = SparseMatrixFactorization(
                            rank, fitting_device=fit_device, g_sparsity=args.g_sparsity)
                        fit_with_ridge(nmf_embed, A_embed.to(nmf_embed.fitting_device).float(), args.max_iterations)

                        with open(embed_pkl_path, "wb") as f:
                            pickle.dump({"nmf": nmf_embed, "vprime_token_ids": unique_tids}, f)

                        # F_emb : (d_model, K) - dense erasure directions
                        # G_tok : (|V'|, K)   - WTA-sparse signed per-token activation scores
                        F_emb = torch.as_tensor(nmf_embed.F_).detach().cpu()
                        G_tok = torch.as_tensor(nmf_embed.G_).detach().cpu()

                        token_label_map = {
                            tid: "both" if c == 2 else concept_name if c == 1 else "Neutral"
                            for tid, c in token_to_code.items()
                        }
                        embed_rows = collect_feature_rows_for_embeddings(
                            G_tok, unique_tids, token_label_map, model, safe_model, concept_name, rank)

                        # Project F_ columns through W_U: F_ IS the d_model direction
                        V_embed = F_emb.T.to(device=E_vprime.device, dtype=E_vprime.dtype)  # (K, d_model)
                        logits_embed = vector_to_logits(model, V_embed)

                        topk_embed = torch.topk(logits_embed, args.k_proj, dim=-1)
                        bottomk_embed = torch.topk(logits_embed, args.k_proj, dim=-1, largest=False)
                        abs_topk_embed = torch.topk(logits_embed.abs(), args.k_proj * 2, dim=-1)

                        for k, row in enumerate(embed_rows):
                            row["projection_top_tokens"] = _safe_tokens(
                                model.to_str_tokens(torch.tensor(topk_embed.indices[k].tolist(), dtype=torch.long)))
                            row["projection_bottom_tokens"] = _safe_tokens(
                                model.to_str_tokens(torch.tensor(bottomk_embed.indices[k].tolist(), dtype=torch.long)))
                            row["projection_abs_top_tokens"] = _safe_tokens(
                                model.to_str_tokens(torch.tensor(abs_topk_embed.indices[k].tolist(), dtype=torch.long)))

                        save_df_to_csv(pd.DataFrame(embed_rows), embed_csv_path, dedupe_cols=["feature"])

                        df_embed_stats = compute_embedding_stats(G_tok, unique_tids, token_to_code, concept_name)
                        save_df_to_csv(df_embed_stats, embed_stats_path)

                    elapsed = t_embed["elapsed"]
                    print(f"[TIMING] Embedding track: {elapsed:.1f}s")
                    update_timing(timing_path, f"{safe_concept}_rank{rank}_embedding_s", elapsed)

            # =====================================================================
            # TRACK B: MLP (SNMF)
            # =====================================================================
            if args.skip_mlp:
                print("--> Skipping MLP track (--skip-mlp)")
            else:
                print("--> Starting MLP track")

                mlp_csv_path = get_pipeline_path(
                    args.outdir, safe_model, "csvs", rank,
                    safe_concept, "mlp", "token_features.csv", seed=args.seed)
                mlp_stats_path = get_pipeline_path(
                    args.outdir, safe_model, "csvs", rank,
                    safe_concept, "mlp", "stats_concept_vs_neutral.csv", seed=args.seed)

                mlp_acts = None
                all_mlp_stats_dfs = []

                with Timer() as t_mlp:
                    for idx, layer in enumerate(tqdm(layers, desc="Fitting Layers")):
                        layer_pkl_path = get_pipeline_path(
                            args.outdir, safe_model, "pickles", rank,
                            safe_concept, "mlp", f"layer{layer}.pkl", seed=args.seed)

                        skip_layer = False
                        if os.path.exists(layer_pkl_path) and os.path.exists(mlp_csv_path):
                            try:
                                df_check = pd.read_csv(mlp_csv_path, usecols=["layer"])
                                if (df_check["layer"] == layer).any():
                                    skip_layer = True
                            except Exception:
                                pass

                        if skip_layer:
                            continue

                        if mlp_acts is None:
                            set_seed(args.seed)
                            mlp_acts, _ = act_generator.generate_multiple_layer_activations_and_freq(dataset, layers)

                        A_mlp = mlp_acts[idx].T
                        nmf_mlp = NMFSemiNMF(rank, fitting_device=fit_device, sparsity=args.sparsity)
                        fit_with_ridge(nmf_mlp, A_mlp.to(nmf_mlp.fitting_device).float(), args.max_iterations)

                        with open(layer_pkl_path, "wb") as f:
                            pickle.dump(nmf_mlp, f)

                        G_mlp = torch.as_tensor(nmf_mlp.G_).detach().cpu()
                        mlp_rows = collect_feature_rows_for_layer(
                            G_mlp, rank, token_ds, labels_all, layer, safe_model, concept_name)

                        W_out = model.blocks[layer].mlp.W_out
                        F_mlp = torch.as_tensor(nmf_mlp.F_).to(device=W_out.device, dtype=W_out.dtype)
                        logits_mlp = vector_to_logits(model, (F_mlp.T @ W_out))

                        topk_mlp = torch.topk(logits_mlp, args.k_proj, dim=-1)
                        bottomk_mlp = torch.topk(logits_mlp, args.k_proj, dim=-1, largest=False)
                        abs_topk_mlp = torch.topk(logits_mlp.abs(), args.k_proj * 2, dim=-1)

                        for k, row in enumerate(mlp_rows):
                            row["projection_top_tokens"] = _safe_tokens(
                                model.to_str_tokens(torch.tensor(topk_mlp.indices[k].tolist(), dtype=torch.long)))
                            row["projection_bottom_tokens"] = _safe_tokens(
                                model.to_str_tokens(torch.tensor(bottomk_mlp.indices[k].tolist(), dtype=torch.long)))
                            row["projection_abs_top_tokens"] = _safe_tokens(
                                model.to_str_tokens(torch.tensor(abs_topk_mlp.indices[k].tolist(), dtype=torch.long)))

                        save_df_to_csv(pd.DataFrame(mlp_rows), mlp_csv_path, dedupe_cols=["layer", "feature"])

                        df_layer_stats = compute_mlp_layer_stats(
                            G_mlp, is_concept_mask, is_neutral_mask, layer, rank,
                            safe_model, concept_name)
                        all_mlp_stats_dfs.append(df_layer_stats)

                elapsed = t_mlp["elapsed"]
                print(f"[TIMING] MLP track: {elapsed:.1f}s")
                update_timing(timing_path, f"{safe_concept}_rank{rank}_mlp_s", elapsed)

                if all_mlp_stats_dfs:
                    df_mlp_all_stats = pd.concat(all_mlp_stats_dfs, ignore_index=True)
                    save_df_to_csv(df_mlp_all_stats, mlp_stats_path)


if __name__ == "__main__":
    main()
