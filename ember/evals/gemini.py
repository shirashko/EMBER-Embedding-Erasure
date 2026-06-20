"""Gemini-as-judge wrapper for open QA and Alpaca scoring.

Thin client over ``google.generativeai`` with:
- Per-process rate limiting (50 RPM per job, configurable).
- Retry on transient errors with hard cap on cumulative quota failures.
- Atomic per-call usage logging to ``gemini_usage_log.jsonl`` (NIS cost).
- Three judge prompts: open-QA correctness, Alpaca instruct, Alpaca fluency.

Public API:
    GeminiEvaluator(model_name=..., token_stats=...)
        .judge_open_qa(question, answer, attempted) -> bool
        .score_alpaca_instruct(instruction, completion) -> int
        .score_alpaca_fluency(completion) -> int

    GeminiQuotaExceededError -- raised once cumulative quota errors exceed
        :data:`MAX_QUOTA_ERRORS` (default 20). Aborts the run.
    GeminiBadFinishError -- raised when Gemini returns a non-STOP finish
        reason; callers in :mod:`open_qa` / :mod:`alpaca` retry up to 3x
        then default to score 0.
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import google.generativeai as gai  # type: ignore

from ember.evals.schema import GeminiTokenStats

# ========================================================================== #
# Constants                                                                   #
# ========================================================================== #

RATE_LIMIT_PER_MIN = 3000
MAX_QUOTA_ERRORS = 20
MAX_OUTPUT_TOKENS = 350
DEFAULT_TEMPERATURE = 0.2

# Pricing for gemini-2.5-flash-lite (Israeli new shekels per 1M tokens).
INPUT_PRICE_NIS_PER_1M = 0.314819999
OUTPUT_PRICE_NIS_PER_1M = 1.259279999

ROOT_DIR = Path(__file__).resolve().parents[2]
USAGE_LOG_PATH = ROOT_DIR / "gemini_usage_log.jsonl"

DEFAULT_MODEL_NAME = "models/gemini-2.5-flash-lite"

# ========================================================================== #
# Module-level rate limiter + quota counter                                   #
# ========================================================================== #

_RATE_LOCK = threading.Lock()
_REQUEST_TIMES: "collections.deque[float]" = collections.deque()

_QUOTA_LOCK = threading.Lock()
_QUOTA_ERROR_COUNT = 0

_PROCESS_START = time.time()


# ========================================================================== #
# Errors                                                                      #
# ========================================================================== #

class GeminiQuotaExceededError(RuntimeError):
    """Raised when cumulative Gemini quota errors exceed :data:`MAX_QUOTA_ERRORS`."""


class GeminiBadFinishError(RuntimeError):
    """Raised when Gemini returns a non-STOP ``finish_reason``.

    Typical causes: RECITATION or SAFETY. Callers should catch this and
    default the item to score 0.
    """
    def __init__(self, finish_reason: int, partial_text: str = "") -> None:
        super().__init__(f"Gemini bad finish_reason={finish_reason}")
        self.finish_reason = finish_reason
        self.partial_text = partial_text


def _is_quota_error(e: Exception) -> bool:
    name = type(e).__name__.lower()
    msg = str(e).lower()
    return (
        "resourceexhausted" in name
        or "429" in msg
        or "quota" in msg
        or "rate limit" in msg
    )


# ========================================================================== #
# Cost tracking                                                               #
# ========================================================================== #

def _run_id() -> str:
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        array = os.environ.get("SLURM_ARRAY_TASK_ID")
        return f"slurm-{job}.{array}" if array else f"slurm-{job}"
    return f"pid-{os.getpid()}-{int(_PROCESS_START)}"


def _cost_nis(prompt_tokens: int, output_tokens: int) -> float:
    return round(
        (prompt_tokens / 1_000_000.0) * INPUT_PRICE_NIS_PER_1M
        + (output_tokens / 1_000_000.0) * OUTPUT_PRICE_NIS_PER_1M,
        6,
    )


def _append_usage(prompt_tokens: int, output_tokens: int) -> None:
    """Append one JSONL record to USAGE_LOG_PATH. Atomic + lock-free on POSIX."""
    if prompt_tokens <= 0 and output_tokens <= 0:
        return
    record = {
        "run_id": _run_id(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_PROCESS_START)),
        "prompt_tokens": int(prompt_tokens),
        "output_tokens": int(output_tokens),
        "cost_nis": _cost_nis(prompt_tokens, output_tokens),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    line = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
    try:
        fd = os.open(str(USAGE_LOG_PATH),
                     os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except Exception as e:
        print(f"[Gemini] usage tracking failed: {e}")


# ========================================================================== #
# Evaluator                                                                   #
# ========================================================================== #

class GeminiEvaluator:
    """Gemini wrapper for judging eval outputs."""

    OPEN_QA_PROMPT = """The following are a question and its correct answer. After that is an attempted answer that you are to assess. In your response you are to write exactly two lines, the first where you write any reasoning chain you need in order to assess whether the attempted answer is correct or not, and the second where you actually write whether it is correct or not (responding with a 1 for correct or 0 for incorrect).
The format MUST be exactly as follows (two lines, no more, no less):
<chain of thought reasoning...>
Correct: 1 or 0
CRITICAL: The second and final line MUST start with the word "Correct:" followed by a space and then either 1 or 0. Do NOT write just "0" or "1" alone — you MUST write "Correct: 0" or "Correct: 1".

The question and correct answer are as follows:
Question: '{question}'
Correct Answer: '{answer}'
Attempted Answer: '{attempted_answer}'
"""

    ALPACA_INSTRUCT_PROMPT = """Please act as an impartial judge and evaluate whether the sentence fragment provided below is related to the instruction. Focus solely on the degree of relatedness in terms of topic, regardless of grammar, coherence, or informativeness.
Begin your evaluation by providing a brief explanation of whether the sentence is related to the instruction, and point out references related to the instruction. After providing your explanation, rate the instruction relevance on a scale from 0 to 2, where 0 indicates the sentence is unrelated to the instruction, 1 indicates it is somewhat related but only minimally or indirectly relevant in terms of topic, and 2 indicates it is more clearly and directly related to the instruction. If there is nothing between the sentence fragment start and end tags, the score is 0. Provide your rating using this exact format: "Rating: [[score]]".

[Instruction Start]
{instructions}
[Instruction End]

[Completion Start]
{completion}
[Completion End]"""

    ALPACA_FLUENCY_PROMPT = """Please act as an impartial judge and evaluate the fluency of the sentence fragment provided below. Focus solely on fluency, disregarding its completeness, relevance, coherence with any broader context, or informativeness.
Begin your evaluation by briefly describing the fluency of the sentence, noting any highly unnatural phrasing, very awkward transitions or grammatical errors that may hinder fluency. Note that markdown formatting should be considered fluent. Ignore whether the last sentence is complete or not, it may be cut off and that should not affect your rating. After providing your explanation, rate the sentence's fluency on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural (for example incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect. If there is nothing between the sentence fragment start and end tags, the score is 0. Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]

{completion}

[Sentence Fragment End]"""

    def __init__(self,
                 model_name: str = DEFAULT_MODEL_NAME,
                 token_stats: Optional[GeminiTokenStats] = None) -> None:
        api_key = (os.getenv("GEMINI_API_KEY")
                   or os.getenv("GOOGLE_API_KEY")
                   or os.getenv("GEMINI_API_TOKEN"))
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found. Set one of "
                "GEMINI_API_KEY / GOOGLE_API_KEY / GEMINI_API_TOKEN."
            )
        gai.configure(api_key=api_key)
        self.model = gai.GenerativeModel(model_name)

        gen_kwargs: Dict[str, Any] = {
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
        }
        temp_env = os.getenv("GEMINI_TEMPERATURE")
        if temp_env is not None and temp_env.strip() != "":
            try:
                gen_kwargs["temperature"] = float(temp_env)
            except ValueError:
                print(f"[Gemini] ignoring non-numeric GEMINI_TEMPERATURE={temp_env!r}")
        self.generation_config = gai.types.GenerationConfig(**gen_kwargs)
        self.token_stats = token_stats

    # ------------------------------------------------------------------ #
    # Core send                                                           #
    # ------------------------------------------------------------------ #

    def _send(self, prompt: str) -> str:
        last_err: Optional[Exception] = None

        if self.token_stats is not None:
            try:
                ct = self.model.count_tokens(prompt)
                prompt_toks = getattr(ct, "total_tokens", None)
                if prompt_toks is None and isinstance(ct, dict):
                    prompt_toks = int(ct.get("total_tokens", 0))
                self.token_stats.add(int(prompt_toks or 0), 0, 1)
            except Exception as e:
                print(f"[Gemini] count_tokens failed: {e}")

        billed_output_toks = 0
        for _ in range(10):
            try:
                self._respect_rate_limit()
                response = self.model.generate_content(
                    prompt, generation_config=self.generation_config,
                )

                billed_prompt_toks = 0
                billed_output_toks = 0
                try:
                    um = getattr(response, "usage_metadata", None)
                    if um is not None:
                        billed_prompt_toks = int(getattr(um, "prompt_token_count", 0) or 0)
                        billed_output_toks = int(getattr(um, "candidates_token_count", 0) or 0)
                except Exception as e:
                    print(f"[Gemini] usage_metadata parse failed: {e}")

                if self.token_stats is not None:
                    self.token_stats.add(0, billed_output_toks, 0)
                _append_usage(billed_prompt_toks, billed_output_toks)

                fr = response.candidates[0].finish_reason
                if fr != 1:
                    try:
                        partial = response.text
                    except Exception:
                        partial = ""
                    raise GeminiBadFinishError(fr, partial)
                return response.text
            except GeminiBadFinishError:
                raise
            except Exception as e:
                last_err = e
                if _is_quota_error(e):
                    global _QUOTA_ERROR_COUNT
                    with _QUOTA_LOCK:
                        _QUOTA_ERROR_COUNT += 1
                        count = _QUOTA_ERROR_COUNT
                    if count > MAX_QUOTA_ERRORS:
                        raise GeminiQuotaExceededError(
                            f"Gemini quota error limit exceeded "
                            f"({count} > {MAX_QUOTA_ERRORS}); aborting. Last error: {e}"
                        ) from e
                    print(f"[Gemini] quota error {count}/{MAX_QUOTA_ERRORS}: {e}")
                wait = 30.0 + random.uniform(0, 5)
                print(f"[Gemini] error {e}, retrying in {wait:.1f}s...")
                time.sleep(wait)

        raise RuntimeError(
            f"Failed to get valid response from Gemini: {last_err}; "
            f"num out tokens {billed_output_toks}"
        )

    @staticmethod
    def _respect_rate_limit() -> None:
        while True:
            with _RATE_LOCK:
                now = time.time()
                while _REQUEST_TIMES and now - _REQUEST_TIMES[0] >= 60.0:
                    _REQUEST_TIMES.popleft()
                if len(_REQUEST_TIMES) < RATE_LIMIT_PER_MIN:
                    _REQUEST_TIMES.append(time.time())
                    return
                sleep_for = 60.0 - (now - _REQUEST_TIMES[0]) + 0.1
            time.sleep(sleep_for)

    # ------------------------------------------------------------------ #
    # Parsing helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_open_correct(response: str) -> bool:
        lines = response.strip().splitlines()
        if not lines:
            raise ValueError(f"Empty Gemini response: {response!r}")
        m = re.search(r"Correct:\s*(\d+)", lines[-1])
        if not m:
            m = re.search(r"Correct:\s*(\d+)", response)
        if not m:
            raise ValueError(f"Could not parse Correct: from Gemini response: {response!r}")
        return int(m.group(1)) == 1

    @staticmethod
    def _parse_rating(response: str) -> int:
        m = re.search(r"Rating:\s*\[\[(\d+)\]\]", response)
        if not m:
            m = re.search(r"(\d+)", response)
        if not m:
            raise ValueError(f"Could not parse rating from Gemini response: {response!r}")
        return int(m.group(1))

    # ------------------------------------------------------------------ #
    # Public judging API                                                  #
    # ------------------------------------------------------------------ #

    def judge_open_qa(self, question: str, answer: str, attempted: str) -> bool:
        resp = self._send(self.OPEN_QA_PROMPT.format(
            question=question, answer=answer, attempted_answer=attempted,
        ))
        return self._parse_open_correct(resp)

    def score_alpaca_instruct(self, instruction: str, completion: str) -> int:
        if not completion.strip():
            return 0
        resp = self._send(self.ALPACA_INSTRUCT_PROMPT.format(
            instructions=instruction, completion=completion,
        ))
        return self._parse_rating(resp)

    def score_alpaca_fluency(self, completion: str) -> int:
        if not completion.strip():
            return 0
        resp = self._send(self.ALPACA_FLUENCY_PROMPT.format(completion=completion))
        return self._parse_rating(resp)


__all__ = [
    "GeminiEvaluator",
    "GeminiQuotaExceededError",
    "GeminiBadFinishError",
    "RATE_LIMIT_PER_MIN", "MAX_QUOTA_ERRORS",
    "MAX_OUTPUT_TOKENS", "DEFAULT_TEMPERATURE",
    "DEFAULT_MODEL_NAME",
]
