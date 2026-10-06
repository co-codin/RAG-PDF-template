#!/usr/bin/env python
"""Bulk-ingest a directory (or files) synchronously, bypassing the HTTP API.

    uv run python scripts/ingest_dir.py data/
    uv run python scripts/ingest_dir.py docs/ --recursive --ext .md .pdf

Uses the same Settings/.env and Container as the API. With embedded Chroma, stop the
API server first (embedded Chroma is single-process); with CHROMA_HOST set it is safe
to run while the API is serving - workers pick up the changes via the store revision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_app.container import build_container
from rag_app.core.config import get_settings
from rag_app.core.errors import IngestionError
from rag_app.core.logging import setup_logging


def iter_files(paths: list[Path], recursive: bool, exts: set[str]):
    for p in paths:
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*") if recursive else p.glob("*")):
                if f.is_file() and f.suffix.lower() in exts:
                    yield f


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("-r", "--recursive", action="store_true")
    ap.add_argument("--ext", nargs="*", default=settings.allowed_extensions, help="extensions to include")
    args = ap.parse_args(argv)

    setup_logging("WARNING", json_logs=False)
    container = build_container(settings)
    failures = 0
    try:
        for f in iter_files(args.paths, args.recursive, {e.lower() for e in args.ext}):
            try:
                print(json.dumps(container.ingestion.ingest_path(f).model_dump(exclude_none=True)))
            except IngestionError as exc:
                failures += 1
                print(json.dumps({"source": f.name, "status": "error", "error": str(exc)}))
    finally:
        container.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
