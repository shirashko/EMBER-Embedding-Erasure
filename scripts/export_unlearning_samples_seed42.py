#!/usr/bin/env python3
"""Export 15 forget + 15 retain sentences actually used at seed 42.

Reconstructs the seed-42 RMU/CRISP corpora from ``ConceptDataset.as_forget_retain``
and writes a JSON file of samples that were used across methods and models.

See the ``notes`` field in the output for caveats (PISCES, llama-CRISP WMDP).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from ember.local_datasets import DATA_DIR, ConceptDataset, resolve_neutral_path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONCEPTS = [
    "Ancient Rome",
    "Golf",
    "Uranium",
    "Culture of Greece",
    "Cannabis",
    "Baseball",
    "Gambling",
    "Republic of Ireland",
    "wmdp-bio",
    "wmdp-cyber",
]

RMU_BATCH_SIZES = {
    "google/gemma-2-2b-it": 16,
    "Qwen/Qwen3.5-2B": 16,
    "meta-llama/Llama-3.1-8B-Instruct": 8,
}
RMU_MIN_LEN = 50
RMU_MAX_LEN = 2000
RMU_MAX_NUM_BATCHES = 150
CRISP_MAX_LEN = 2000
SEED = 42
N_SAMPLES = 15


def _filter_len(texts: Sequence[str], *, min_len: Optional[int], max_len: Optional[int]) -> List[str]:
    out: List[str] = []
    for s in texts:
        if min_len is not None and len(s) < min_len:
            continue
        if max_len is not None and max_len > 0 and len(s) > max_len:
            continue
        out.append(s)
    return out


def _batchify(texts: Sequence[str], batch_size: int) -> List[List[str]]:
    return [list(texts[i:i + batch_size])
            for i in range(0, len(texts), batch_size)
            if texts[i:i + batch_size]]


def rmu_used(forget: Sequence[str], retain: Sequence[str], batch_size: int,
             max_num_batches: int = RMU_MAX_NUM_BATCHES) -> Tuple[List[str], List[str], int]:
    """Prefix of forget/retain batches actually consumed by ``run_rmu``."""
    fb = _batchify(forget, batch_size)
    rb = _batchify(retain, batch_size)
    n_batches = min(max_num_batches, len(fb), len(rb))
    used_f = [s for b in fb[:n_batches] for s in b]
    used_r = [s for b in rb[:n_batches] for s in b]
    return used_f, used_r, n_batches


def unique_in_order(texts: Iterable[str], k: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for s in texts:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= k:
            break
    if len(out) < k:
        raise ValueError(f"Needed {k} unique sentences, found {len(out)}")
    return out


def _index_map(texts: Sequence[str]) -> dict:
    idx = {}
    for i, s in enumerate(texts):
        idx.setdefault(s, i)
    return idx


def records_for(texts: Sequence[str], source_index: dict) -> List[dict]:
    return [{"text": s, "char_len": len(s), "source_index": source_index[s]} for s in texts]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concepts", nargs="+", default=list(DEFAULT_CONCEPTS))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n", type=int, default=N_SAMPLES)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "unlearning_samples_seed42.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wiki_path = resolve_neutral_path("data/neutral_sentences.json")
    wiki_raw = [o["sentence"] for o in json.loads(wiki_path.read_text(encoding="utf-8"))
                if o.get("sentence")]
    wiki_index = _index_map(wiki_raw)

    concept_raw = {e["concept"]: [s for s in e.get("sentences", []) if s]
                   for e in json.loads((DATA_DIR / "concept_sentences.json").read_text(encoding="utf-8"))}

    # Shared retain: RMU min_len filter after the seed-42 shuffle. Forget length is
    # 300 for every concept, so shuffle(forget) consumes the same RNG stream and
    # the shuffled retain order is identical across concepts.
    probe = ConceptDataset(args.concepts[0], neutral_path=wiki_path).as_forget_retain(seed=args.seed)
    shared_retain_rmu = _filter_len(probe["retain"], min_len=RMU_MIN_LEN, max_len=RMU_MAX_LEN)
    for concept in args.concepts:
        data = ConceptDataset(concept, neutral_path=wiki_path).as_forget_retain(seed=args.seed)
        retain_rmu = _filter_len(data["retain"], min_len=RMU_MIN_LEN, max_len=RMU_MAX_LEN)
        if retain_rmu != shared_retain_rmu:
            raise RuntimeError(f"Retain shuffle/filter diverged for {concept!r}")

    retain_15 = unique_in_order(shared_retain_rmu, args.n)

    concepts_out = []
    for concept in args.concepts:
        data = ConceptDataset(concept, neutral_path=wiki_path).as_forget_retain(seed=args.seed)
        forget_crisp = _filter_len(data["forget"], min_len=None, max_len=CRISP_MAX_LEN)
        forget_rmu = _filter_len(data["forget"], min_len=RMU_MIN_LEN, max_len=RMU_MAX_LEN)
        src_index = _index_map(concept_raw[concept])

        used_by_model = {}
        for model, bs in RMU_BATCH_SIZES.items():
            uf, ur, n_batches = rmu_used(forget_rmu, shared_retain_rmu, bs)
            used_by_model[model] = {
                "batch_size": bs,
                "n_batches": n_batches,
                "n_forget_used": len(uf),
                "n_forget_used_unique": len(set(uf)),
                "n_retain_used": len(ur),
                "n_retain_used_unique": len(set(ur)),
            }

        # Llama RMU is the strictest prefix (batch_size=8 -> 264 forget texts).
        llama_forget, _, _ = rmu_used(
            forget_rmu, shared_retain_rmu,
            RMU_BATCH_SIZES["meta-llama/Llama-3.1-8B-Instruct"],
        )
        forget_15 = unique_in_order(llama_forget, args.n)
        missing_crisp = [s for s in forget_15 if s not in set(forget_crisp)]
        if missing_crisp:
            raise RuntimeError(f"{concept}: selected forget not in CRISP corpus")

        concepts_out.append({
            "concept": concept,
            "n_forget_raw": len(concept_raw[concept]),
            "n_forget_raw_unique": len(set(concept_raw[concept])),
            "n_forget_rmu_filtered": len(forget_rmu),
            "n_forget_crisp": len(forget_crisp),
            "rmu_usage": used_by_model,
            "forget": records_for(forget_15, src_index),
        })

    payload = {
        "seed": args.seed,
        "n_per_split": args.n,
        "selection": {
            "forget": (
                "First 15 unique texts in the Llama RMU forget prefix "
                "(shuffled with seed 42, then min_len=50 / max_len=2000, "
                "then the first min(forget_batches, retain_batches, 150) batches "
                "at batch_size=8). That prefix is a subset of Gemma/Qwen RMU "
                "(batch_size=16), CRISP (all 300), and SNMF feature extraction "
                "(all 300)."
            ),
            "retain": (
                "First 15 unique texts in the shared Wikipedia retain list after "
                "the same seed-42 shuffle and RMU length filter. This ordered "
                "list is identical across concepts because every concept has "
                "300 forget sentences, so shuffle(forget) leaves the RNG in "
                "the same state before shuffle(retain)."
            ),
        },
        "notes": [
            "Checkpoints do not store sample lists; this file reconstructs them from code + seed + data.",
            "PISCES does not consume forget/retain sentences (SAE feature edits only), so it does not constrain the set.",
            "SNMF/EMBER unlearning applies precomputed features. Feature extraction used all 300 concept + all 300 Wikipedia neutrals (train_mf_features hardcodes data/neutral_sentences.json).",
            "RMU is the bottleneck: it drops texts shorter than 50 chars and stops when either forget or retain batches run out (retain is shorter, so some forget texts are unused).",
            "Llama CRISP for wmdp-bio and wmdp-cyber used data/cyber_neutral_sentences.json (0 overlap with Wikipedia neutrals). The shared retain list below was NOT used in those two runs.",
            "Gemma/Qwen CRISP and all RMU WMDP runs used Wikipedia neutrals, including wmdp-cyber.",
        ],
        "retain_corpus": str(wiki_path.relative_to(ROOT)),
        "n_retain_raw": len(wiki_raw),
        "n_retain_rmu_filtered": len(shared_retain_rmu),
        "n_retain_rmu_filtered_unique": len(set(shared_retain_rmu)),
        "retain": records_for(retain_15, wiki_index),
        "concepts": concepts_out,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"Wrote {args.out} ({len(args.concepts)} concepts, {args.n} forget + {args.n} retain)")


if __name__ == "__main__":
    main()
