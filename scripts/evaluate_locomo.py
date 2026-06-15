#!/usr/bin/env python3
"""Run All-Mem on LoCoMo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from all_mem.eval.locomo import evaluate_locomo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate All-Mem on LoCoMo.")
    parser.add_argument("--dataset", default=str(WORKSPACE_ROOT / "data/locomo/locomo10.json"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--backend", default="openai", choices=["openai", "sglang", "ollama"])
    parser.add_argument("--output", default=str(ROOT / "results/all_mem/locomo/results.json"))
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--temperature-c5", type=float, default=0.0)
    parser.add_argument("--sleep-interval-sessions", type=int, default=3)
    parser.add_argument("--cache-dir", default=str(ROOT / "cache"))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--sglang-host", default="http://localhost")
    parser.add_argument("--sglang-port", type=int, default=30000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_locomo(
        dataset_path=args.dataset,
        model=args.model,
        output_path=args.output,
        ratio=args.ratio,
        backend=args.backend,
        retrieve_k=args.retrieve_k,
        temperature_c5=args.temperature_c5,
        sleep_interval_sessions=args.sleep_interval_sessions,
        cache_dir=args.cache_dir,
        api_key=args.api_key,
        api_base=args.api_base,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
    )
    print(json.dumps(result["aggregate_metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
