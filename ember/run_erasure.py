#!/usr/bin/env python3
"""EMBER pipeline entry point.

Tiny wrapper: parse a YAML config + CLI overrides into a :class:`RunConfig`,
configure logging, dispatch to :func:`pipeline.run`.

Usage::

    python -m ember.run_erasure \\
        --config configs/snmf_ember_gemma.yaml \\
        --train-eval mc \\
        --concepts "Harry Potter"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from ember.erasure import log
from ember.erasure.config import parse_args
from ember.erasure import pipeline


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    cfg = parse_args()
    log.configure()
    if cfg.features_source == "hf":
        from ember.data import ensure_features
        ensure_features(cfg.model_name, cfg.concepts, cfg.rank, cfg.seed)
    pipeline.run(cfg)


if __name__ == "__main__":
    main()
