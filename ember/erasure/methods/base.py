"""Method abstract base class.

A "method" is one of: SNMF, RMU, CRISP, EMBER. Each implements the same
small interface so the pipeline driver in :mod:`pipeline` can run them
without method-specific branching.

The interface (see :class:`Method`):
    - :meth:`enumerate_hps` -- yield method-specific HP dicts (one per CSV row).
    - :meth:`hp_key_columns` -- subset of HP keys that uniquely identify a cell.
    - :meth:`hp_columns` -- full list of HP+info columns written to the CSV.
    - :meth:`apply` -- mutate the model for one HP. Returns info columns
      (e.g. ``k_features_mlp_in``) merged into the row.
    - :meth:`snapshot` / :meth:`restore` -- undo :meth:`apply` between HP cells.
      Default no-op for stateless methods.
    - :attr:`requires_full_reload` -- True for fine-tuning methods (RMU/CRISP):
      apply() modifies parameters in ways snapshot/restore can't cheaply undo,
      so the pipeline reloads the model from disk between cells.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

from ember.erasure.config import RunConfig


class Method(ABC):
    """Per-method plug-in.

    Subclasses live in ``ember/erasure/methods/{snmf,rmu,crisp,ember}.py``
    and are looked up by :data:`REGISTRY` keyed on :attr:`name`.
    """

    name: str  # set by subclass; matches RunConfig.method
    requires_full_reload: bool = False  # True for RMU / CRISP (FT methods)

    # ------------------------------------------------------------------ #
    # HP enumeration                                                     #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def enumerate_hps(self, common: RunConfig) -> Iterable[Dict[str, Any]]:
        """Yield method-specific HP dicts.

        Each dict's keys must be a subset of :meth:`hp_columns` and a superset
        of :meth:`hp_key_columns`. Values that get filled in later (e.g.
        ``k_features_mlp_in`` returned by :meth:`apply`) should be omitted.
        """

    @abstractmethod
    def hp_key_columns(self) -> List[str]:
        """Subset of HP columns used as the resume key (must be unique per cell)."""

    @abstractmethod
    def hp_columns(self) -> List[str]:
        """All HP/info columns this method writes to the per-concept HP CSV.

        Does *not* include identity columns (``model``, ``concept``, ...) or
        eval-metric columns -- the pipeline owns those.
        """

    # ------------------------------------------------------------------ #
    # Apply / snapshot                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def apply(
            self,
            hf_model: Any,
            tokenizer: Any,
            hp: Dict[str, Any],
            concept: str,
            common: RunConfig,
    ) -> Dict[str, Any]:
        """Mutate ``hf_model`` according to ``hp`` for ``concept``.

        Returns a dict of *info columns* to merge into the row (e.g.
        ``{"k_features_mlp_in": 240, "k_features_mlp_out": 180}``). Keys here
        must be a subset of :meth:`hp_columns`.

        Methods that need cached per-concept state (e.g. SNMF's
        :class:`features.ConceptContext`) should keep it on ``self`` and
        invalidate it when :attr:`current_concept` changes -- see the SNMF
        implementation.
        """

    def snapshot(self, hf_model: Any) -> Any:
        """Take a snapshot the pipeline can hand back to :meth:`restore`.

        Default no-op (returns ``None``). Stateful methods like SNMF override
        this to checkpoint embeddings + MLP weights.

        For ``requires_full_reload=True`` methods, :meth:`snapshot` is never
        called -- the pipeline reloads the model from disk instead.
        """
        return None

    def restore(self, hf_model: Any, snap: Any) -> None:
        """Reverse :meth:`apply` using the snapshot from :meth:`snapshot`."""

    # ------------------------------------------------------------------ #
    # Optional lifecycle hooks                                           #
    # ------------------------------------------------------------------ #

    def on_concept_start(self, hf_model: Any, concept: str,
                         common: RunConfig) -> None:
        """Called once per concept before any HP cells run."""

    def on_concept_end(self, hf_model: Any, concept: str,
                       common: RunConfig) -> None:
        """Called once per concept after all HP cells for that concept finish."""

    def before_relearning(self, hf_model: Any, concept: str,
                          common: RunConfig) -> None:
        """Called per concept after final-test eval, before relearning.

        Lets a method release heavy auxiliary state (e.g. a second resident
        model) so the relearning optimizer fits. Default: no-op.
        """

    def after_concept_grid(self, concept: str, hp_csv_path: Any,
                           grid_out_dir: Any, common: RunConfig) -> None:
        """Called after the per-concept grid finishes (after ``top_hps.csv`` is written)."""

    # ------------------------------------------------------------------ #
    # Grid-time eval behavior                                            #
    # ------------------------------------------------------------------ #

    def grid_eval_kwargs(self, common: RunConfig) -> Dict[str, Any]:
        """Eval kwargs for the *method grid* stage (not validate / final test).

        Default: skip Alpaca, use the configured min_mmlu / max_qa_acc to
        early-discard bad cells. Methods that need Alpaca during their grid
        (notably EMBER) override this with tighter thresholds.
        """
        return {
            "eval_alpaca": False,
            "min_mmlu": common.eval.min_mmlu,
            "max_qa_acc": common.eval.max_qa_acc,
            "min_alpaca": None,
        }

    def pick_best_hp_row(self, hps_df: Any) -> Optional[Any]:
        """Select the single best row from this method's per-concept hps.csv.

        Default: top row by ``harmonic``. EMBER overrides to use
        ``harmonic_alpaca`` since it always evaluates Alpaca.
        """
        from ember.erasure import io
        if hps_df is None or hps_df.empty:
            return None
        topk = io.topk_per_concept(hps_df, k=1)
        return topk.iloc[0] if not topk.empty else None

    def writes_topk_csv(self) -> bool:
        """Whether to write ``top_hps.csv`` after the grid finishes.

        Default True. EMBER overrides to False -- its grid is just a few
        delta values, and there's no top-K validation stage downstream.
        """
        return True

    def skip_eval_for_hp(self, hp: Dict[str, Any]) -> bool:
        """Return True to skip eval for this HP cell and emit a zero-metric row.

        Default: ``False``. EMBER overrides this to skip delta=0.0 cells.
        """
        return False

    def get_model_to_eval(self) -> Optional[Any]:
        """Return a freshly-mutated model to eval on, if :meth:`apply` produced one.

        Default: ``None`` -- the pipeline evaluates the model it passed to
        :meth:`apply`. RMU and CRISP override this to return the trained
        working copy.
        """
        return None


# Method registry filled in by methods/__init__.py at import time.
# Keys are RunConfig.method strings ("snmf", "rmu", ...).
REGISTRY: Dict[str, "Method"] = {}


def register(method: Method) -> Method:
    """Add ``method`` to the registry under ``method.name``."""
    if not method.name:
        raise ValueError(f"Method subclass {type(method).__name__} has empty .name")
    if method.name in REGISTRY:
        raise ValueError(f"Method {method.name!r} already registered")
    REGISTRY[method.name] = method
    return method


def get(name: str) -> Method:
    """Look up a registered method by name. Imports the methods package first."""
    import ember.erasure.methods  # triggers registration
    if name not in REGISTRY:
        raise KeyError(f"Unknown method {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


__all__ = ["Method", "REGISTRY", "register", "get"]
