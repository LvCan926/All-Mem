"""LoCoMo evaluation pipeline for All-Mem."""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from all_mem import AllMemSystem, LLMController
from all_mem.data.locomo import QA, LoCoMoSample, load_locomo_dataset
from all_mem.eval.metrics import (
    aggregate_metrics,
    calculate_id_based_metrics,
    calculate_metrics,
)

logger = logging.getLogger("all_mem.locomo")


def sanitize_name(value: str) -> str:
    return str(value).replace("/", "_").replace(":", "_")


def count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(str(text).split())


class AllMemLocomoAgent:
    def __init__(
        self,
        model: str,
        backend: str,
        retrieve_k: int,
        temperature_c5: float,
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
        self.temperature_c5 = temperature_c5

    def add_memory(self, content: str, timestamp: str, source_id: str) -> None:
        self.memory_system.wake_process(content, timestamp=timestamp, source_id=source_id)

    def sleep(self) -> None:
        self.memory_system.sleep_process()

    def answer_question(self, qa: QA) -> Dict[str, object]:
        context, ranked_source_ids, latency_ms = self.memory_system.get_context_for_query(
            qa.question,
            final_k=self.retrieve_k,
        )
        active_count = len(self.memory_system.graph.node_cache)
        total_count = self.memory_system.graph.graph.number_of_nodes()
        graph_stats = {
            "total_nodes": total_count,
            "active_nodes": active_count,
            "active_ratio": active_count / total_count if total_count else 0.0,
        }

        if int(qa.category or 0) == 5:
            choices = ["Not mentioned", qa.final_answer or ""]
            random.shuffle(choices)
            prompt = f"""
Use the context to answer the question.

Context:
{context}

Question: {qa.question}

Select the correct answer: {choices[0]} or {choices[1]}.
Return JSON: {{"answer": "..."}}
"""
            temperature = self.temperature_c5
        else:
            prompt = f"""
Task: You are a precise data extraction engine. Answer using only the retrieved All-Mem context.

Rules:
1. If the question asks for a date or time, resolve relative expressions using the date of the log entry that mentions the event.
2. Scan all context lines before answering.
3. Output only the concise final answer in JSON.

Question: {qa.question}

All-Mem context:
{context}

Return JSON: {{"answer": "..."}}
"""
            temperature = 0.0

        response = self.llm_controller.complete_json(
            prompt,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_retries=3,
        )
        return {
            "answer": str(response.get("answer", "")).strip(),
            "prompt": prompt,
            "context": context,
            "ranked_source_ids": ranked_source_ids,
            "graph_stats": graph_stats,
            "latency_ms": latency_ms,
        }


def build_locomo_memory(
    agent: AllMemLocomoAgent,
    sample: LoCoMoSample,
    sleep_interval_sessions: int = 3,
) -> List[Dict[str, int]]:
    evolution_trace: List[Dict[str, int]] = []
    session_counter = 0
    turn_count = 0

    for _, session in sample.conversation.sessions.items():
        for turn in session.turns:
            content = f"{turn.speaker}: {turn.text}"
            agent.add_memory(content, timestamp=str(session.date_time), source_id=turn.dia_id)
            turn_count += 1

        session_counter += 1
        did_sleep = False
        if sleep_interval_sessions > 0 and session_counter % sleep_interval_sessions == 0:
            agent.sleep()
            did_sleep = True

        evolution_trace.append(
            {
                "turn": turn_count,
                "active_nodes": len(agent.memory_system.graph.node_cache),
                "total_nodes": agent.memory_system.graph.graph.number_of_nodes(),
                "session_idx": session_counter,
                "did_sleep": did_sleep,
            }
        )

    if sleep_interval_sessions > 0 and session_counter % sleep_interval_sessions != 0:
        agent.sleep()
        evolution_trace.append(
            {
                "turn": turn_count,
                "active_nodes": len(agent.memory_system.graph.node_cache),
                "total_nodes": agent.memory_system.graph.graph.number_of_nodes(),
                "session_idx": session_counter,
                "did_sleep": True,
            }
        )

    return evolution_trace


def evaluate_locomo(
    dataset_path: str,
    model: str = "gpt-4o-mini",
    output_path: Optional[str] = None,
    ratio: float = 1.0,
    backend: str = "openai",
    retrieve_k: int = 10,
    temperature_c5: float = 0.0,
    sleep_interval_sessions: int = 3,
    cache_dir: str = "cache",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    sglang_host: str = "http://localhost",
    sglang_port: int = 30000,
) -> Dict[str, object]:
    samples = load_locomo_dataset(dataset_path)
    if ratio < 1.0:
        samples = samples[: max(1, int(len(samples) * ratio))]

    run_name = f"{backend}_{sanitize_name(model)}"
    graph_cache_dir = Path(cache_dir) / f"all_mem_locomo_{run_name}"
    graph_cache_dir.mkdir(parents=True, exist_ok=True)

    output = Path(output_path) if output_path else None
    trace_dir = (output.parent if output else Path("results/all_mem/locomo")) / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_metrics = []
    all_categories = []
    category_counts = defaultdict(int)
    total_questions = 0

    for sample_idx, sample in enumerate(samples):
        agent = AllMemLocomoAgent(
            model=model,
            backend=backend,
            retrieve_k=retrieve_k,
            temperature_c5=temperature_c5,
            api_key=api_key,
            api_base=api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        cache_file = graph_cache_dir / f"all_mem_graph_sample_{sample_idx}.pkl"
        if cache_file.exists():
            with cache_file.open("rb") as f:
                agent.memory_system.graph = pickle.load(f)
            agent.memory_system.graph.rebuild_index()
        else:
            trace = build_locomo_memory(agent, sample, sleep_interval_sessions)
            with (trace_dir / f"sample_{sample_idx}.json").open("w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2, ensure_ascii=False)
            with cache_file.open("wb") as f:
                pickle.dump(agent.memory_system.graph, f)

        for qa in sample.qa:
            category = int(qa.category or 0)
            if category not in {1, 2, 3, 4, 5}:
                continue
            total_questions += 1
            category_counts[category] += 1

            started = time.perf_counter()
            answer_payload = agent.answer_question(qa)
            inference_latency = time.perf_counter() - started
            prediction = str(answer_payload["answer"])

            gen_metrics = calculate_metrics(prediction, qa.final_answer or "")
            retrieval_metrics = calculate_id_based_metrics(
                answer_payload["ranked_source_ids"],
                qa.evidence,
            )
            metrics = {**gen_metrics, **retrieval_metrics}
            metrics["inference_latency"] = inference_latency
            metrics["inference_tokens"] = count_tokens(str(answer_payload["prompt"]))
            metrics["graph_active_nodes"] = answer_payload["graph_stats"]["active_nodes"]
            metrics["graph_total_nodes"] = answer_payload["graph_stats"]["total_nodes"]
            for name, value in answer_payload["latency_ms"].items():
                metrics[f"memlat_{name}"] = float(value)

            all_metrics.append(metrics)
            all_categories.append(category)
            results.append(
                {
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": qa.final_answer,
                    "category": category,
                    "metrics": metrics,
                    "ranked_source_ids": answer_payload["ranked_source_ids"],
                }
            )

    final_results = {
        "framework": "All-Mem",
        "model": model,
        "backend": backend,
        "dataset": dataset_path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_questions": total_questions,
        "category_distribution": {str(k): v for k, v in sorted(category_counts.items())},
        "aggregate_metrics": aggregate_metrics(all_metrics, all_categories),
        "individual_results": results,
    }

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    return final_results
