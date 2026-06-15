"""LongMemEval evaluation pipeline for All-Mem."""

from __future__ import annotations

import json
import logging
import pickle
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from all_mem import AllMemSystem, LLMController
from all_mem.data.longmemeval import LongMemEvalSample, load_longmemeval_dataset
from all_mem.eval.metrics import (
    aggregate_metrics,
    calculate_id_based_metrics,
    calculate_metrics,
)
from all_mem.eval.locomo import count_tokens, sanitize_name

logger = logging.getLogger("all_mem.longmemeval")
_WRITE_LOCK = threading.Lock()


SYSTEM_PROMPT = """You are an All-Mem assistant with precise memory recall.
Answer using only the retrieved memory logs. If the answer is not present, say the user did not mention it."""


class AllMemLongMemEvalAgent:
    def __init__(
        self,
        model: str,
        backend: str,
        retrieve_k: int,
        temperature: float,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
    ):
        self.llm_controller = LLMController(
            backend=backend,
            model=model,
            api_key=api_key,
            api_base=api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.memory_system = AllMemSystem(self.llm_controller)
        self.retrieve_k = retrieve_k
        self.temperature = temperature

    def add_memory(self, content: str, timestamp: str, source_id: str) -> None:
        self.memory_system.wake_process(content, timestamp=timestamp, source_id=source_id)

    def sleep(self) -> None:
        self.memory_system.sleep_process()

    def answer_question(self, question: str, current_date: str) -> Dict[str, object]:
        context, ranked_source_ids, latency_ms = self.memory_system.get_context_for_query(
            question,
            final_k=self.retrieve_k,
        )
        prompt = f"""
{SYSTEM_PROMPT}

Current reference time:
{current_date}

All-Mem context:
{context}

Question:
{question}

Return JSON with:
{{
  "reasoning": "brief reasoning, including date calculation if needed",
  "final_answer": "concise answer"
}}
"""
        response = self.llm_controller.complete_json(
            prompt,
            response_format={"type": "json_object"},
            temperature=self.temperature,
            max_retries=3,
        )
        total_nodes = self.memory_system.graph.graph.number_of_nodes()
        active_nodes = len(self.memory_system.graph.node_cache)
        return {
            "answer": str(response.get("final_answer", "")).strip(),
            "reasoning": response.get("reasoning", ""),
            "prompt": prompt,
            "context": context,
            "ranked_source_ids": ranked_source_ids,
            "latency_ms": latency_ms,
            "graph_stats": {
                "total_nodes": total_nodes,
                "active_nodes": active_nodes,
                "active_ratio": active_nodes / total_nodes if total_nodes else 0.0,
            },
        }


def build_longmemeval_memory(
    agent: AllMemLongMemEvalAgent,
    sample: LongMemEvalSample,
    sleep_interval_sessions: int = 3,
) -> List[Dict[str, object]]:
    trace: List[Dict[str, object]] = []
    session_counter = 0
    turn_count = 0
    for session_id, session in sample.conversation.sessions.items():
        for turn in session.turns:
            turn_count += 1
            content = f"{turn.speaker}: {turn.text}"
            agent.add_memory(content, timestamp=session.date_time, source_id=str(session_id))
        session_counter += 1
        did_sleep = False
        if sleep_interval_sessions > 0 and session_counter % sleep_interval_sessions == 0:
            agent.sleep()
            did_sleep = True
        trace.append(
            {
                "turn": turn_count,
                "session_idx": session_counter,
                "active_nodes": len(agent.memory_system.graph.node_cache),
                "total_nodes": agent.memory_system.graph.graph.number_of_nodes(),
                "did_sleep": did_sleep,
            }
        )

    if sleep_interval_sessions > 0 and session_counter % sleep_interval_sessions != 0:
        agent.sleep()
        trace.append(
            {
                "turn": turn_count,
                "session_idx": session_counter,
                "active_nodes": len(agent.memory_system.graph.node_cache),
                "total_nodes": agent.memory_system.graph.graph.number_of_nodes(),
                "did_sleep": True,
            }
        )
    return trace


def process_longmemeval_sample(
    sample: LongMemEvalSample,
    model: str,
    backend: str,
    retrieve_k: int,
    temperature: float,
    graph_cache_dir: Path,
    trace_dir: Path,
    jsonl_output_path: Optional[Path],
    sleep_interval_sessions: int,
    api_key: Optional[str],
    api_base: Optional[str],
    sglang_host: str,
    sglang_port: int,
) -> Tuple[Optional[Dict[str, object]], Optional[str], Optional[Dict[str, float]]]:
    qa = sample.qa[0]
    agent = AllMemLongMemEvalAgent(
        model=model,
        backend=backend,
        retrieve_k=retrieve_k,
        temperature=temperature,
        api_key=api_key,
        api_base=api_base,
        sglang_host=sglang_host,
        sglang_port=sglang_port,
    )

    cache_file = graph_cache_dir / f"all_mem_graph_{sample.question_id}.pkl"
    if cache_file.exists():
        with cache_file.open("rb") as f:
            agent.memory_system.graph = pickle.load(f)
        agent.memory_system.graph.rebuild_index()
    else:
        trace = build_longmemeval_memory(agent, sample, sleep_interval_sessions)
        with (trace_dir / f"trace_{sample.question_id}.json").open("w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
        with cache_file.open("wb") as f:
            pickle.dump(agent.memory_system.graph, f)

    started = time.perf_counter()
    answer_payload = agent.answer_question(qa.question, qa.question_date or "Unknown")
    inference_latency = time.perf_counter() - started
    prediction = str(answer_payload["answer"])

    metrics = {
        **calculate_metrics(prediction, qa.final_answer or ""),
        **calculate_id_based_metrics(answer_payload["ranked_source_ids"], qa.evidence),
    }
    metrics["inference_latency"] = inference_latency
    metrics["inference_tokens"] = count_tokens(str(answer_payload["prompt"]))
    metrics["graph_active_nodes"] = answer_payload["graph_stats"]["active_nodes"]
    metrics["graph_total_nodes"] = answer_payload["graph_stats"]["total_nodes"]
    for name, value in answer_payload["latency_ms"].items():
        metrics[f"memlat_{name}"] = float(value)

    result = {
        "question_id": sample.question_id,
        "question_type": sample.question_type,
        "question": qa.question,
        "answer": qa.final_answer,
        "response": prediction,
        "reasoning": answer_payload["reasoning"],
        "metrics": metrics,
        "graph_stats": answer_payload["graph_stats"],
        "source_ids": answer_payload["ranked_source_ids"],
    }

    if jsonl_output_path:
        with _WRITE_LOCK:
            with jsonl_output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result, sample.question_type, metrics


def evaluate_longmemeval(
    dataset_path: str,
    model: str = "gpt-4o-mini",
    output_path: Optional[str] = None,
    ratio: float = 1.0,
    backend: str = "openai",
    workers: int = 4,
    retrieve_k: int = 10,
    temperature: float = 0.0,
    sleep_interval_sessions: int = 3,
    cache_dir: str = "cache",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    sglang_host: str = "http://localhost",
    sglang_port: int = 30000,
) -> Dict[str, object]:
    samples = load_longmemeval_dataset(dataset_path)
    if ratio < 1.0:
        samples = samples[: max(1, int(len(samples) * ratio))]

    run_name = f"{backend}_{sanitize_name(model)}"
    graph_cache_dir = Path(cache_dir) / f"all_mem_longmemeval_{run_name}"
    graph_cache_dir.mkdir(parents=True, exist_ok=True)

    output = Path(output_path) if output_path else None
    result_dir = output.parent if output else Path("results/all_mem/longmemeval")
    result_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = result_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    jsonl_output_path = output.with_suffix(".jsonl") if output else result_dir / "results.jsonl"
    if jsonl_output_path.exists():
        jsonl_output_path.unlink()

    results = []
    all_metrics = []
    all_categories = []
    category_counts = defaultdict(int)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_longmemeval_sample,
                sample,
                model,
                backend,
                retrieve_k,
                temperature,
                graph_cache_dir,
                trace_dir,
                jsonl_output_path,
                sleep_interval_sessions,
                api_key,
                api_base,
                sglang_host,
                sglang_port,
            )
            for sample in samples
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="All-Mem LongMemEval"):
            result, category, metrics = future.result()
            if result and category and metrics:
                results.append(result)
                all_metrics.append(metrics)
                all_categories.append(category)
                category_counts[category] += 1

    final_results = {
        "framework": "All-Mem",
        "model": model,
        "backend": backend,
        "dataset": dataset_path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_questions": len(results),
        "category_distribution": dict(category_counts),
        "aggregate_metrics": aggregate_metrics(all_metrics, all_categories),
        "results": results,
    }

    if output:
        with output.open("w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    return final_results
