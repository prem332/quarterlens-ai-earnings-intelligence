"""CLI: `python -m data_pipeline.chunking`."""
from __future__ import annotations

import argparse
import logging

from .hierarchy import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical + semantic chunking of parsed filings + transcripts.")
    parser.add_argument("--manifest", default="data/parsed/parsed_manifest.json")
    parser.add_argument("--out", default="data/chunks")
    parser.add_argument("--transcripts-manifest", default=None)
    args = parser.parse_args()
    run(args.manifest, args.out, args.transcripts_manifest)


if __name__ == "__main__":
    main()
