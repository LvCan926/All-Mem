"""LLM-as-judge utilities for All-Mem result JSONL files."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from openai import AsyncOpenAI
from tqdm import tqdm


DEFAULT_BASE_URL = os.getenv("OPENAI_API_BASE") or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"


def resolve_api_key(explicit_api_key: Optional[str] = None) -> str:
    key = explicit_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")
    return key


def infer_abstention(item: Dict[str, object]) -> bool:
    question_id = str(item.get("question_id", "")).lower()
    return "abs" in question_id


def get_answer_check_prompt(
    task: object,
    question: str,
    answer: str,
    response: str,
    abstention: bool = False,
) -> str:
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a model response. "
            "Answer [[yes]] if the model correctly identifies the question as unanswerable; otherwise answer [[no]].\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer [[yes]] or [[no]] only."
        )

    if task in {"temporal-reasoning", 2}:
        extra = (
            "Do not penalize off-by-one errors for counts of days/weeks/months when the response otherwise "
            "contains the correct answer."
        )
    elif task == "knowledge-update":
        extra = (
            "If the response includes older information but clearly gives the updated answer, count it as correct."
        )
    elif task == "single-session-preference":
        return (
            "I will give you a question, a rubric for a desired personalized response, and a model response. "
            "Answer [[yes]] if the response satisfies the rubric; otherwise answer [[no]].\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer [[yes]] or [[no]] only."
        )
    else:
        extra = ""

    return (
        "I will give you a question, a correct answer, and a model response. "
        "Answer [[yes]] if the response contains the correct answer; otherwise answer [[no]]. "
        "If the response is equivalent to the correct answer, answer [[yes]]. "
        f"{extra}\n\n"
        f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
        "Is the model response correct? Answer [[yes]] or [[no]] only."
    )


async def judge_one(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> str:
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""


async def judge_jsonl(
    input_file: str,
    output_file: str,
    model: str = "gpt-4o-mini",
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
    concurrency: int = 10,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    with Path(input_file).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    client = AsyncOpenAI(api_key=resolve_api_key(api_key), base_url=base_url)
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    progress = tqdm(total=len(rows), desc="All-Mem judge")

    async def wrapped(prompt: str) -> str:
        try:
            return await judge_one(client, prompt, model, semaphore)
        finally:
            progress.update(1)

    for row in rows:
        if "llm_judge_single" in row:
            async def existing(value=row["llm_judge_single"]):
                return str(value)
            tasks.append(existing())
            continue
        prompt = get_answer_check_prompt(
            task=row.get("question_type") or row.get("category"),
            question=str(row.get("question", "")),
            answer=str(row.get("answer", "")),
            response=str(row.get("response", "")),
            abstention=infer_abstention(row),
        )
        tasks.append(wrapped(prompt))

    try:
        judge_responses = await asyncio.gather(*tasks)
    finally:
        progress.close()
        await client.close()

    yes_count = 0
    valid_count = 0
    processed = []
    for row, judge_response in zip(rows, judge_responses):
        row["llm_judge_single"] = judge_response
        low = judge_response.lower()
        if "[[yes]]" in low:
            yes_count += 1
            valid_count += 1
        elif "[[no]]" in low:
            valid_count += 1
        processed.append(row)

    processed.sort(key=lambda x: 0 if "[[no]]" in str(x.get("llm_judge_single", "")).lower() else 1)
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in processed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "total": len(rows),
        "valid": valid_count,
        "yes": yes_count,
        "accuracy": yes_count / valid_count if valid_count else 0.0,
        "output_file": str(output),
    }
