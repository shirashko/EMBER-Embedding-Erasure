#!/usr/bin/env python3
"""Upsert a concept entry into PISCES feature JSON.

Usage examples:
  # From feature_finder output (.concept file with name/k/value/features)
  python scripts/upsert_pisces_features.py \
    --features-json data/pisces_concept_features_gemma.json \
    --concept-file /path/to/gemma_wmdp-bio_coef_signed_linscale_test.csv.concept

  # Optional rename + backup
  python scripts/upsert_pisces_features.py \
    --features-json data/pisces_concept_features_gemma.json \
    --concept-file /path/to/out.concept \
    --concept-name wmdp-bio \
    --backup
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _backup(path: Path) -> None:
    b = path.with_suffix(path.suffix + ".bak")
    b.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[backup] {path} -> {b}")


def _normalize_features(raw_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for f in raw_features:
        if not isinstance(f, dict):
            continue
        if "layer" not in f or "id" not in f:
            continue
        row: Dict[str, Any] = {
            "layer": int(f["layer"]),
            "id": int(f["id"]),
            "neg": bool(f.get("neg", False)),
        }
        if "large" in f:
            row["large"] = bool(f["large"])
        out.append(row)
    if not out:
        raise ValueError("No valid features found (need objects with layer/id).")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-json", required=True, help="Target pisces_concept_features_*.json")
    p.add_argument("--concept-file", required=True, help="feature_finder .concept JSON output")
    p.add_argument("--concept-name", default=None,
                   help="Override concept name written to features JSON.")
    p.add_argument("--backup", action="store_true", help="Write .bak before update.")
    args = p.parse_args()

    features_path = Path(args.features_json).resolve()
    concept_path = Path(args.concept_file).resolve()

    if not features_path.exists():
        raise SystemExit(f"Missing features JSON: {features_path}")
    if not concept_path.exists():
        raise SystemExit(f"Missing concept file: {concept_path}")

    base = _read_json(features_path)
    if not isinstance(base, list):
        raise SystemExit(f"{features_path} must contain a JSON list.")

    concept_obj = _read_json(concept_path)
    if not isinstance(concept_obj, dict):
        raise SystemExit(f"{concept_path} must contain a JSON object.")

    src_name = str(concept_obj.get("name", "")).strip()
    concept_name = (args.concept_name or src_name).strip()
    if not concept_name:
        raise SystemExit("Concept name is empty. Use --concept-name to set it explicitly.")

    k = float(concept_obj.get("k", -1))
    value = float(concept_obj.get("value", -1))
    features = _normalize_features(list(concept_obj.get("features", [])))

    entry = {
        "name": concept_name,
        "k": k,
        "value": value,
        "features": features,
    }

    replaced = False
    out_list: List[Dict[str, Any]] = []
    for item in base:
        if isinstance(item, dict) and str(item.get("name", "")).strip() == concept_name:
            out_list.append(entry)
            replaced = True
        else:
            out_list.append(item)
    if not replaced:
        out_list.append(entry)

    if args.backup:
        _backup(features_path)
    _write_json(features_path, out_list)

    print("[done]")
    print(f"  concept: {concept_name}")
    print(f"  features: {len(features)}")
    print(f"  action: {'replaced' if replaced else 'appended'}")
    print(f"  target: {features_path}")


if __name__ == "__main__":
    main()
