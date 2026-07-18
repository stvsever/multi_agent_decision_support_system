"""
Reusable, dataset-agnostic ingestion helpers for COMPASS validation.

Nothing in this package is specific to a single dataset. Each dataset under
``validation/<DATASET>/`` supplies its own ``config.py`` and pipeline scripts
and imports these helpers to:

1. explore raw feature structure (``manifest``),
2. build an LLM-driven, non-redundant subclass ontology (``ontology``),
3. standardise features into deviation scores (``deviation``),
4. emit the four COMPASS participant files (``compass_writer``).

The full-stack engine under ``src/full_stack`` stays completely free of any
dataset-specific logic; all of that lives here in ``validation/``.
"""

from . import manifest, ontology, deviation, compass_writer, llm  # noqa: F401

__all__ = ["manifest", "ontology", "deviation", "compass_writer", "llm"]
