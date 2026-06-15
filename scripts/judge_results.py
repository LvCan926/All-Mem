#!/usr/bin/env python3
"""Run LLM judge on All-Mem JSONL results."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from all_mem.eval.judge import DEFAULT_BASE_URL, judge_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge All-Mem JSONL results.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--concurrency", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = asyncio.run(
        judge_jsonl(
            input_file=args.input,
            output_file=args.output,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            concurrency=args.concurrency,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
