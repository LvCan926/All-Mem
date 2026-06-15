#!/usr/bin/env python3
"""Download the cleaned LongMemEval dataset into the refactored workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download cleaned LongMemEval.")
    parser.add_argument("--repo-id", default="xiaowu0162/longmemeval-cleaned")
    parser.add_argument("--local-dir", default=str(ROOT / "data/longmemeval"))
    args = parser.parse_args()

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        repo_type="dataset",
    )
    print(f"Downloaded {args.repo_id} to {args.local_dir}")


if __name__ == "__main__":
    main()
