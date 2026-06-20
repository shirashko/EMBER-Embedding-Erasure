"""Structured logging for the erasure pipeline.

A thin wrapper over :mod:`logging` that gives every line a consistent
``[method|concept|stage]`` prefix so it is easy to grep slurm output.

Typical use::

    from ember.erasure import log

    log.configure()
    log.set_context(method="snmf")
    with log.stage("baselines"):
        log.info("loading model")    # -> [snmf|-|baselines] loading model
"""
from __future__ import annotations

import contextlib
import logging
import sys
from typing import Iterator, Optional

_LOGGER_NAME = "ember"
_CONTEXT: dict[str, str] = {"method": "-", "concept": "-", "stage": "-"}


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.method = _CONTEXT.get("method", "-")
        record.concept = _CONTEXT.get("concept", "-")
        record.stage = _CONTEXT.get("stage", "-")
        return True


def configure(level: int = logging.INFO) -> None:
    """Install handler + formatter on the module logger. Idempotent."""
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        logger.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s] [%(method)s|%(concept)s|%(stage)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    handler.addFilter(_ContextFilter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def set_context(*, method: Optional[str] = None,
                concept: Optional[str] = None,
                stage: Optional[str] = None) -> None:
    """Update persistent context fields. ``None`` leaves a field unchanged."""
    if method is not None:
        _CONTEXT["method"] = method
    if concept is not None:
        _CONTEXT["concept"] = concept
    if stage is not None:
        _CONTEXT["stage"] = stage


@contextlib.contextmanager
def stage(name: str) -> Iterator[None]:
    """Context manager that sets and restores the ``stage`` field."""
    prev = _CONTEXT.get("stage", "-")
    _CONTEXT["stage"] = name
    try:
        yield
    finally:
        _CONTEXT["stage"] = prev


@contextlib.contextmanager
def concept(name: str) -> Iterator[None]:
    """Context manager that sets and restores the ``concept`` field."""
    prev = _CONTEXT.get("concept", "-")
    _CONTEXT["concept"] = name
    try:
        yield
    finally:
        _CONTEXT["concept"] = prev


def _logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def info(msg: str, *args) -> None:
    _logger().info(msg, *args)


def warning(msg: str, *args) -> None:
    _logger().warning(msg, *args)


def error(msg: str, *args) -> None:
    _logger().error(msg, *args)


def debug(msg: str, *args) -> None:
    _logger().debug(msg, *args)
