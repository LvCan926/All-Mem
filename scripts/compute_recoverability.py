#!/usr/bin/env python3
"""Compute All-Mem recoverability statistics from a cached graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from all_mem.analysis.recoverability import (
    compute_recoverability_report,
    write_per_node_csv,
    write_report_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze All-Mem cached graph recoverability.")
    parser.add_argument("--cache", required=True, help="Path to a pickled All-Mem graph/system cache.")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--max-hops", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = compute_recoverability_report(args.cache, max_hops=args.max_hops)
    print(json.dumps(payload["report"], indent=2, ensure_ascii=False))
    if args.out_json:
        write_report_json(payload, args.out_json)
    if args.out_csv:
        write_per_node_csv(payload, args.out_csv)


if __name__ == "__main__":
    main()
