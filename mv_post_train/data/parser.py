"""Unified eval-parser contract: ParseResult + EvalParser abstract base.

The pipeline ingests evals from two upstream sources with very different
shapes — a directory tree of lm-eval-harness `results_*.json` files
(`logs.zip`) and a flat HuggingFace parquet (the Phase 2 dump). Both
get adapted into the same downstream schema (`ItemRow` / `AggregateRow`)
via concrete `EvalParser` subclasses.

Every subclass implements one method (`parse() -> ParseResult`); the base
class handles persistence in a single shared `write_to_data_dir`. New eval
sources only have to fill in the abstract method + declare three class
attributes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from .. import paths
from .io_logs import AggregateRow, ItemRow


class ParseResult(BaseModel):
    """Unified parser output. Both `Phase1Parser.parse()` and
    `Phase2Parser.parse()` return this shape; downstream code
    (analyze) doesn't have to know which parser produced
    the rows.

    `item_rows` and `aggregate_rows` are typed against the canonical
    schemas in `data/io_logs.py`. `audit_rows` is a freeform list of dicts
    used for parser-specific diagnostics (file_selection_log,
    skipped-row counts, etc.) — pydantic doesn't validate its shape, only
    its presence.
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    item_rows: tuple[ItemRow, ...] = ()
    aggregate_rows: tuple[AggregateRow, ...] = ()
    audit_rows: tuple[dict[str, Any], ...] = ()


class EvalParser(ABC):
    """Abstract base for upstream eval-source adapters.

    Each subclass:
      1. takes its upstream inputs in `__init__`,
      2. implements `parse()` to produce a `ParseResult`,
      3. declares three class attributes describing where its output goes:
           items_filename     — basename under `data_dir/` for item rows
           emits_aggregates   — True if parse() returns aggregate_rows
           audit_filename     — basename under `art_dir/` for the audit
                                 CSV, or None to skip

    The concrete `write_to_data_dir` persists ParseResult to disk via
    these attributes; subclasses don't need to repeat I/O boilerplate.
    """

    # Class attributes — subclasses override.
    parser_name: str                       # display name for logging
    items_filename: str | None             # paths.ITEMS_LONG / paths.PHASE2_ITEMS; None to skip
    emits_aggregates: bool                 # True if parse() returns aggregate_rows
    aggregates_filename: str = paths.AGGREGATES  # overridable per subclass
    audit_filename: str | None = None      # CSV name under art_dir, None = skip

    @abstractmethod
    def parse(self) -> ParseResult:
        """Read upstream inputs; return a `ParseResult`.

        Implementations should:
          - Yield `ItemRow` / `AggregateRow` model instances (already
            validated by pydantic at construction).
          - Surface data-quality issues via `_warn` from `io_logs.py`
            (corrupted files, malformed payloads) rather than silently
            dropping records.
          - Use `audit_rows` for any diagnostic state that should ride
            alongside the rows (file selection log, skipped counts, etc.).
        """
        ...

    def write_to_data_dir(self, data_dir: Path,
                           art_dir: Path | None = None) -> int:
        """Persist a fresh `parse()` to disk under `data_dir` / `art_dir`.

        Returns 0 on success (used as the CLI subcommand return code).
        """
        data_dir.mkdir(exist_ok=True, parents=True)
        if art_dir is not None:
            art_dir.mkdir(exist_ok=True, parents=True)

        result = self.parse()

        # Items (skipped for parsers whose source emits no per-item rows)
        if self.items_filename is not None:
            df_items = pd.DataFrame([r.model_dump() for r in result.item_rows])
            items_path = data_dir / self.items_filename
            df_items.to_parquet(items_path, index=False)
            print(f"{self.parser_name}: items rows: {len(df_items)} → {items_path.name}")

        # Aggregates (only the parsers that produce them)
        if self.emits_aggregates:
            df_agg = pd.DataFrame([r.model_dump() for r in result.aggregate_rows])
            agg_path = data_dir / self.aggregates_filename
            df_agg.to_parquet(agg_path, index=False)
            print(f"{self.parser_name}: aggregate rows: {len(df_agg)} → {agg_path.name}")

        # Audit log (audit_filename embeds an `audit/` subdir per paths.py)
        if self.audit_filename and art_dir is not None and result.audit_rows:
            audit_path = art_dir / self.audit_filename
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(list(result.audit_rows)).to_csv(audit_path, index=False)
            print(f"{self.parser_name}: audit rows: {len(result.audit_rows)} → "
                  f"{audit_path.relative_to(art_dir)}")

        return 0
