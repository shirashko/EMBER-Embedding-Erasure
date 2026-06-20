#!/usr/bin/env python3
import os
import json
import argparse
import ast
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures

import google.generativeai as gai
import pandas as pd
from tqdm import tqdm

from ember.timing import Timer
from ember.utils import _safe_concept, _safe_model_name, update_timing


# -------------------- Prompts --------------------

TOKENS_DESC_SYS = """
You get tokens that represent a single feature vector (with some noise).
Infer the single most specific, cohesive concept shared by the relevant tokens.
Return only one concise sentence. No preface, no list, no caveats.

Example:
Tokens: ['▁tomorrow','▁tonight','▁yesterday','▁today','▁demain']
Explanation of feature behavior: this vector is related to specific dates and times (e.g., today/tomorrow/yesterday).
"""

TOKENS_DESC_USER = "Tokens: {tokens}\nExplanation of feature behavior:"

MEMBERSHIP_PROMPT = """
You get a concept name and a feature's description.
Decide if this feature describes the given concept. Consider if the feature is DISTINCTIVE for the concept (not broad like "sport" for "basketball") and not too noisy.
A feature should be marked true only if the concept is central and dominant in the description, not just one of several themes.
Return ONLY JSON: {{"is_member": true|false, "confidence": 0..1}}

Example 1:
Concept: Charity
Description: This vector represents elements related to giving and donations to charitable causes.
Answer: {{"is_member": true, "confidence": 0.98}}

Example 2:
Concept: Charity
Description: This vector represents elements related to money, such as gambling, investing and charity.
Answer: {{"is_member": false, "confidence": 0.80}}

Concept: {concept}
Description: {desc}
Answer:
"""


# -------------------- Helpers --------------------

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _parse_list_cell(cell) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)): return []
    if isinstance(cell, list): return [str(x) for x in cell]
    s = str(cell).strip()
    if not s: return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)): return [str(x) for x in v]
        except Exception:
            pass
    sep = "|" if "|" in s else ("," if "," in s else None)
    if sep is None: return [s]
    return [t.strip().strip("'").strip('"') for t in s.split(sep) if t.strip()]


def _load_existing_interpretations(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()


def _existing_keys(df: pd.DataFrame) -> set:
    keys = set()
    if df is None or df.empty: return keys
    for _, r in df.iterrows():
        try:
            keys.add((int(r.get("layer", 0)), int(r["feature"])))
        except Exception:
            continue
    return keys


# -------------------- Gemini Client --------------------

class SimpleGeminiClient:
    def __init__(self, model_name: str = "models/gemini-2.5-flash-lite", max_retries: int = 3, sleep_seconds: float = 5.0):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_TOKEN") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("No Gemini API key found. Set GOOGLE_API_KEY, GEMINI_API_TOKEN, or GEMINI_API_KEY.")
        gai.configure(api_key=api_key)
        self.model = gai.GenerativeModel(model_name)
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds

    def generate(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.model.generate_content(prompt)
                if not resp.candidates: raise RuntimeError("No candidates returned from Gemini.")
                text = getattr(resp, "text", None)
                if not isinstance(text, str) or not text.strip():
                    parts = getattr(getattr(resp.candidates[0], "content", None), "parts", None)
                    if parts:
                        text = "\n".join([getattr(p, "text", "") for p in parts if getattr(p, "text", "").strip()])
                if not isinstance(text, str) or not text.strip(): raise RuntimeError("Gemini returned empty text.")
                return text.strip()
            except Exception as e:
                last_err = e
                print(f"[Gemini] error {e}, attempt {attempt}/{self.max_retries}")
                if attempt < self.max_retries: time.sleep(self.sleep_seconds)
        raise RuntimeError(f"Gemini failed after {self.max_retries} attempts: {last_err}")


# -------------------- Interpretation Logic --------------------

def get_description(client: SimpleGeminiClient, tokens: List[str]) -> str:
    user = TOKENS_DESC_USER.format(tokens=tokens)
    prompt = f"{TOKENS_DESC_SYS.strip()}\n\n{user.strip()}"
    txt = client.generate(prompt).strip()
    s = txt.strip().strip("`").strip()
    try:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                for key in ["Explanation", "explanation", "description"]:
                    if key in obj: return str(obj[key]).strip()
    except Exception:
        pass
    return s


def classify_membership_with_desc(client: SimpleGeminiClient, concept_name: str, description: str) -> Tuple[
        Optional[bool], Optional[float]]:
    prompt = MEMBERSHIP_PROMPT.format(concept=concept_name, desc=description)
    raw = client.generate(prompt).strip()
    try:
        s = raw.strip("`").strip()
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            return bool(obj.get("is_member")), max(0.0, min(1.0, float(obj.get("confidence", 0))))
    except Exception:
        pass
    return None, None


def _process_single_feature(client: SimpleGeminiClient, concept_name: str, row: pd.Series, source_col: str, top_k: int,
                             model_name: str, rank: int, ratio_col: str) -> Optional[Dict[str, Any]]:
    try:
        layer, feature = int(row.get("layer", 0)), int(row["feature"])
        if source_col not in row or pd.isna(row[source_col]): return None

        tokens = _parse_list_cell(row[source_col])[:top_k]
        if not tokens: return None

        description = get_description(client, tokens).strip()
        is_member, conf = classify_membership_with_desc(client, concept_name, description)

        ratio_val = row.get(ratio_col, "")
        try:
            ratio_val = float(ratio_val) if ratio_val != "" and not pd.isna(ratio_val) else ""
        except Exception:
            ratio_val = ""

        return {
            "model": model_name, "concept": concept_name, "rank": int(rank),
            "layer": layer, "feature": feature, "metric_score": ratio_val,
            "source": source_col, "n_tokens_used": len(tokens),
            "tokens_provided": json.dumps(tokens, ensure_ascii=False),
            "description": description,
            "is_member": is_member if is_member is not None else "",
            "confidence": conf if conf is not None else ""
        }
    except Exception as e:
        print(f"[warn] Error interpreting feature (layer={row.get('layer', 0)}, feature={row.get('feature', 'N/A')}): {e}")
        return None


def interpret_source(client: SimpleGeminiClient, df_tokens: pd.DataFrame, concept_name: str, model_name: str, rank: int,
                     source_col: str, out_path: Path, ratio_col: str, top_k: int, save_every: int, resume: bool,
                     max_workers: int) -> pd.DataFrame:
    existing_df = _load_existing_interpretations(out_path) if resume else pd.DataFrame()
    existing = _existing_keys(existing_df)

    rows_to_process = [row for _, row in df_tokens.iterrows() if
                       (int(row.get("layer", 0)), int(row["feature"])) not in existing]
    if not rows_to_process:
        print(f"[{source_col}] All features already interpreted. Skipping.")
        return existing_df

    print(f"[{source_col}] Concurrent interpretation of {len(rows_to_process)} features with {max_workers} workers.")
    new_rows, processed_count = [], 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_process_single_feature, client, concept_name, row, source_col, top_k, model_name, rank,
                            ratio_col): idx for idx, row in enumerate(rows_to_process)}

        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(rows_to_process),
                           desc=f"Interpreting {source_col}"):
            res = future.result()
            processed_count += 1
            if res: new_rows.append(res)

            if save_every and processed_count % save_every == 0:
                merged = pd.concat([existing_df, pd.DataFrame(new_rows)],
                                   ignore_index=True) if not existing_df.empty or new_rows else existing_df
                if not merged.empty:
                    merged = merged.sort_values(["layer", "feature"]).drop_duplicates(
                        subset=["layer", "feature", "source"], keep="first")
                _atomic_write_csv(merged, out_path)
                existing_df, new_rows = merged, []

    merged_final = pd.concat([existing_df, pd.DataFrame(new_rows)],
                             ignore_index=True) if not existing_df.empty or new_rows else existing_df
    if not merged_final.empty:
        merged_final = merged_final.sort_values(["layer", "feature"]).drop_duplicates(
            subset=["layer", "feature", "source"], keep="first")
    _atomic_write_csv(merged_final, out_path)
    return merged_final


# -------------------- Main Driver --------------------

def run_for_concept_and_track(concept_name: str, track: str, client: SimpleGeminiClient, args: argparse.Namespace):
    concept_safe = _safe_concept(concept_name)
    model_safe = _safe_model_name(args.model_name)
    base_dir = Path(args.outdir) / model_safe

    seed_dir = f"seed{args.seed}"
    csv_dir = base_dir / "csvs" / f"rank{args.rank}" / seed_dir / concept_safe / track
    interp_dir = base_dir / "interpretations" / f"rank{args.rank}" / seed_dir / concept_safe / track
    interp_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = csv_dir / "token_features.csv"
    stats_path = csv_dir / ("stats_concept_vs_neutral.csv" if track == "mlp" else "stats_embed.csv")

    if not tokens_path.exists() or not stats_path.exists():
        print(f"[skip] Missing required CSVs for {concept_name} / {track}. Run train_mf_features first.")
        return

    df_tokens = pd.read_csv(tokens_path)
    df_stats = pd.read_csv(stats_path)

    if "layer" not in df_tokens.columns: df_tokens["layer"] = 0
    if "layer" not in df_stats.columns: df_stats["layer"] = 0

    ratio_col = "ratio_abs"
    merge_cols = ["layer", "feature"]
    df_tokens = df_tokens.merge(df_stats[merge_cols + [ratio_col]], on=merge_cols, how="inner")

    df_tokens[ratio_col] = pd.to_numeric(df_tokens[ratio_col], errors="coerce")
    df_filtered = df_tokens[df_tokens[ratio_col] >= args.ratio_thresh].copy()

    if df_filtered.empty:
        print(f"[skip] No {track} features pass threshold {args.ratio_thresh} for {concept_name}.")
        return

    print(f"\n[concept] {concept_name} | [track] {track.upper()} -> {len(df_filtered)} features to process.")

    act_out = interp_dir / "from_activation.csv"
    proj_out = interp_dir / "from_projection.csv"
    potential_out = interp_dir / "potential_features.csv"

    interpret_source(client, df_filtered, concept_name, args.model_name, args.rank, args.activation_col, act_out,
                     ratio_col, args.top_k, args.save_every, args.resume, args.max_workers)

    # Projection tokens only available for MLP track
    if track == "mlp":
        interpret_source(client, df_filtered, concept_name, args.model_name, args.rank, args.projection_col, proj_out,
                         ratio_col, args.top_k, args.save_every, args.resume, args.max_workers)

    def get_potentials(path):
        df = _load_existing_interpretations(path)
        if df.empty: return pd.DataFrame()
        is_mem = (df["is_member"] == True) | (df["is_member"].astype(str).str.strip().str.lower() == "true")
        conf = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0) >= args.confidence_thresh
        return df[is_mem & conf].copy()

    df_pot = pd.concat(
        [get_potentials(act_out), get_potentials(proj_out) if track == "mlp" else pd.DataFrame()],
        ignore_index=True)

    if not df_pot.empty:
        df_pot = df_pot.drop_duplicates(subset=["layer", "feature", "source"], keep="first").reset_index(drop=True)

    df_pot.to_csv(potential_out, index=False)
    print(f"[done] Saved interpretations to {interp_dir}")


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Interpret MF features via Gemini (MLP and embedding tracks).")
    parser.add_argument("--concepts", nargs="+", default=[
        "Ancient Rome", "Artificial intelligence", "Baseball", "Cannabis",
        "COVID-19 pandemic", "Culture of Greece", "Gambling", "Golf", "Gun",
        "Halloween", "Harry Potter", "Heroin", "Nazism", "Pornography",
        "Republic of Ireland", "Uranium", "Valentine's Day", "World War II",
    ])
    parser.add_argument("--tracks", nargs="+", default=["embedding", "mlp"], choices=["mlp", "embedding"])
    parser.add_argument("--rank", type=int, default=100,
                        help="Feature rank. Default 100 for Gemma; use 200 for Llama.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="google/gemma-2-2b-it")

    parser.add_argument("--outdir", type=str, default="mf_outputs")
    parser.add_argument("--ratio-thresh", type=float, default=2.0)
    parser.add_argument("--confidence-thresh", type=float, default=0.85)

    parser.add_argument("--activation-col", type=str, default="activating_tokens")
    parser.add_argument("--projection-col", type=str, default="projection_top_tokens")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--gemini-model", type=str, default="models/gemini-2.5-flash-lite")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Resume from existing interpretation CSVs (use --no-resume to recompute).")

    args = parser.parse_args()
    client = SimpleGeminiClient(model_name=args.gemini_model)

    model_safe = _safe_model_name(args.model_name)
    timing_path = Path(args.outdir) / model_safe / "timing.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)

    for concept in args.concepts:
        print(f"\n{'=' * 80}\n[RUN] Concept: {concept}\n{'=' * 80}")
        with Timer() as t:
            for track in args.tracks:
                try:
                    run_for_concept_and_track(concept, track, client, args)
                except Exception as e:
                    print(f"[error] Failed processing {track} for {concept}: {e}")
        elapsed = t["elapsed"]
        print(f"[TIMING] {concept}: {elapsed:.1f}s")
        safe_concept = _safe_concept(concept)
        tracks_key = "_".join(args.tracks)
        update_timing(timing_path, f"{safe_concept}_rank{args.rank}_interp_{tracks_key}_s", elapsed)


if __name__ == "__main__":
    main()
