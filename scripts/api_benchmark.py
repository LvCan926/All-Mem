#!/usr/bin/env python3
"""Simple async API benchmark for OpenAI-compatible endpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import Counter
from typing import Dict, List, Optional

from openai import AsyncOpenAI


def resolve_api_key(explicit_api_key: Optional[str] = None) -> str:
    key = explicit_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")
    return key


async def call_model(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    system_message: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    request_timeout: float,
    request_id: int,
) -> Dict[str, object]:
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                ),
                timeout=request_timeout,
            )
            return {
                "id": request_id,
                "status": "success",
                "latency": time.perf_counter() - started,
                "content": response.choices[0].message.content or "",
            }
        except Exception as exc:
            return {
                "id": request_id,
                "status": "error",
                "latency": time.perf_counter() - started,
                "error": str(exc),
            }


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    rank = (pct / 100.0) * (len(values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def summarize(results: List[Dict[str, object]], concurrency: int) -> Dict[str, object]:
    latencies = [float(r["latency"]) for r in results if r["status"] == "success"]
    errors = Counter(str(r.get("error", "unknown_error")) for r in results if r["status"] == "error")
    return {
        "concurrency": concurrency,
        "total": len(results),
        "success": len(latencies),
        "errors": len(results) - len(latencies),
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency": percentile(latencies, 95),
        "max_latency": max(latencies) if latencies else 0.0,
        "top_errors": errors.most_common(3),
    }


async def run(args: argparse.Namespace) -> List[Dict[str, object]]:
    client = AsyncOpenAI(api_key=resolve_api_key(args.api_key), base_url=args.base_url)
    summaries = []
    try:
        for concurrency in args.concurrency:
            semaphore = asyncio.Semaphore(concurrency)
            tasks = [
                call_model(
                    client,
                    model=args.model,
                    prompt=args.prompt,
                    system_message=args.system_message,
                    temperature=args.temperature,
                    semaphore=semaphore,
                    request_timeout=args.request_timeout,
                    request_id=i,
                )
                for i in range(args.requests)
            ]
            results = await asyncio.gather(*tasks)
            summaries.append(summarize(results, concurrency))
    finally:
        await client.close()
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark OpenAI-compatible API latency.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt", default="Introduce yourself in one sentence.")
    parser.add_argument("--system-message", default="You are a helpful assistant.")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[20])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summaries = asyncio.run(run(args))
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
