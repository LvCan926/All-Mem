"""Evaluation metrics shared by All-Mem experiments."""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    SmoothingFunction = None
    sentence_bleu = None
    meteor_score = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

_SENTENCE_MODEL = None


def simple_tokenize(text: object) -> List[str]:
    return (
        str(text)
        .lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace("!", " ")
        .replace("?", " ")
        .split()
    )


def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    if rouge_scorer is None:
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    if sentence_bleu is None or SmoothingFunction is None:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    pred_tokens = simple_tokenize(prediction)
    ref_tokens = [simple_tokenize(reference)]
    smooth = SmoothingFunction().method1
    weights = [
        (1, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (1 / 3, 1 / 3, 1 / 3, 0),
        (0.25, 0.25, 0.25, 0.25),
    ]
    out = {}
    for idx, weight in enumerate(weights, start=1):
        try:
            out[f"bleu{idx}"] = sentence_bleu(
                ref_tokens,
                pred_tokens,
                weights=weight,
                smoothing_function=smooth,
            )
        except Exception:
            out[f"bleu{idx}"] = 0.0
    return out


def calculate_bert_scores(prediction: str, reference: str) -> Dict[str, float]:
    try:
        from bert_score import score as bert_score

        precision, recall, f1 = bert_score([prediction], [reference], lang="en", verbose=False)
        return {
            "bert_precision": float(precision.item()),
            "bert_recall": float(recall.item()),
            "bert_f1": float(f1.item()),
        }
    except Exception:
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}


def calculate_meteor(prediction: str, reference: str) -> float:
    if meteor_score is None:
        return 0.0
    try:
        return float(meteor_score([reference.split()], prediction.split()))
    except Exception:
        return 0.0


def _get_sentence_model():
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SENTENCE_MODEL


def calculate_sentence_similarity(prediction: str, reference: str) -> float:
    try:
        from sentence_transformers.util import pytorch_cos_sim

        model = _get_sentence_model()
        pred_emb = model.encode([prediction], convert_to_tensor=True)
        ref_emb = model.encode([reference], convert_to_tensor=True)
        return float(pytorch_cos_sim(pred_emb, ref_emb).item())
    except Exception:
        return 0.0


def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    if not prediction or not reference:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "bert_precision": 0.0,
            "bert_recall": 0.0,
            "bert_f1": 0.0,
            "meteor": 0.0,
            "sbert_similarity": 0.0,
        }

    prediction = str(prediction).strip()
    reference = str(reference).strip()
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common = pred_tokens & ref_tokens
    if pred_tokens and ref_tokens and common:
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "exact_match": int(prediction.lower() == reference.lower()),
        "f1": f1,
        **calculate_rouge_scores(prediction, reference),
        **calculate_bleu_scores(prediction, reference),
        **calculate_bert_scores(prediction, reference),
        "meteor": calculate_meteor(prediction, reference),
        "sbert_similarity": calculate_sentence_similarity(prediction, reference),
    }


def calculate_id_based_metrics(
    ranked_retrieved_ids: List[str],
    golden_evidence: List[str],
    k_list: List[int] = None,
) -> Dict[str, float]:
    if k_list is None:
        k_list = [3, 5, 10]
    golden_set = set(str(x) for x in golden_evidence)
    metrics: Dict[str, float] = {}
    if not golden_set:
        for k in k_list:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
        return metrics

    relevance = [1 if str(rid) in golden_set else 0 for rid in ranked_retrieved_ids]
    for k in k_list:
        current = relevance[:k]
        metrics[f"recall@{k}"] = sum(current) / len(golden_set)
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(current) if rel)
        ideal = [1] * min(len(golden_set), k)
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal))
        metrics[f"ndcg@{k}"] = float(dcg / idcg) if idcg else 0.0
    return metrics


def aggregate_metrics(
    all_metrics: List[Dict[str, float]],
    all_categories: List[Union[int, str]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not all_metrics:
        return {}

    overall = defaultdict(list)
    by_category = defaultdict(lambda: defaultdict(list))
    for metrics, category in zip(all_metrics, all_categories):
        for name, value in metrics.items():
            if isinstance(value, (int, float, np.floating)):
                overall[name].append(float(value))
                by_category[category][name].append(float(value))

    result: Dict[str, Dict[str, Dict[str, float]]] = {"overall": {}}
    for name, values in overall.items():
        result["overall"][name] = summarize_values(values)

    for category in sorted(by_category.keys(), key=lambda item: str(item)):
        bucket = f"category_{category}"
        result[bucket] = {
            name: summarize_values(values)
            for name, values in by_category[category].items()
        }
    return result


def summarize_values(values: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def clean_and_parse_json(response_text: str) -> Dict[str, object]:
    text = (response_text or "").replace("```json", "").replace("```", "").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group())
    return {}
