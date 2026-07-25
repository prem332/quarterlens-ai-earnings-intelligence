"""
Hierarchical + semantic section-aware chunking (Deviation #32).

Pipeline:
    python -m data_pipeline.chunking     # chunk filings + transcripts
    python -m data_pipeline.embedding    # embed all child chunks
    python -m data_pipeline.indexer      # recreate index + upload all

Public API preserved from the former single-module chunking.py.
"""
from __future__ import annotations

from .config import get_encoder
from .hierarchy import chunk_filing, chunk_transcript, run

__all__ = ["run", "chunk_filing", "chunk_transcript", "get_encoder"]
