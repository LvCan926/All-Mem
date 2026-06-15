"""Recoverability analysis for All-Mem graph caches."""

from __future__ import annotations

import csv
import json
import pickle
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np


def canonical_edge_type(edge_type: str) -> str:
    if edge_type in {"sibling_split", "sibling"}:
        return "sibling"
    if edge_type in {"revision", "version"}:
        return "version"
    return edge_type


def extract_nx_digraph(obj: Any) -> nx.DiGraph:
    if isinstance(obj, nx.DiGraph):
        return obj
    if hasattr(obj, "graph") and isinstance(getattr(obj, "graph"), nx.DiGraph):
        return getattr(obj, "graph")
    if hasattr(obj, "graph"):
        inner = getattr(obj, "graph")
        if hasattr(inner, "graph") and isinstance(getattr(inner, "graph"), nx.DiGraph):
            return getattr(inner, "graph")
    raise TypeError("Cache must contain a NetworkX DiGraph or an All-Mem graph/system object.")


def get_node_status(graph: nx.DiGraph, node_id: str) -> Optional[str]:
    data = graph.nodes[node_id].get("data")
    return getattr(data, "status", None) if data is not None else None


def build_recovery_adjacency(graph: nx.DiGraph) -> Dict[str, List[str]]:
    adjacency = defaultdict(list)
    for source, target, edge_data in graph.edges(data=True):
        edge_type = canonical_edge_type((edge_data or {}).get("type", ""))
        if edge_type in {"sibling", "version"}:
            adjacency[source].append(target)
            adjacency[target].append(source)
    return adjacency


def archived_node_hop(
    node_id: str,
    adjacency: Dict[str, List[str]],
    is_active: Dict[str, bool],
    max_hops: Optional[int] = None,
) -> Optional[int]:
    queue = deque([(node_id, 0)])
    visited = {node_id}
    while queue:
        current, distance = queue.popleft()
        if distance > 0 and is_active.get(current, False):
            return distance
        if max_hops is not None and distance >= max_hops:
            continue
        for neighbor in adjacency.get(current, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


def summarize_hops(hops: List[Optional[int]], cov_marks: List[int]) -> Dict[str, Any]:
    finite = [hop for hop in hops if hop is not None]
    total = len(hops)
    coverage = {
        int(mark): sum(hop is not None and hop <= mark for hop in hops) / float(total or 1)
        for mark in cov_marks
    }
    if not finite:
        return {
            "archived_count": total,
            "reachable_count": 0,
            "unreachable_count": total,
            "unreachable_ratio": 1.0 if total else 0.0,
            "hv_max": None,
            "hv_mean": None,
            "hv_median": None,
            "hv_p95": None,
            "cov": coverage,
        }
    arr = np.array(finite, dtype=np.float32)
    unreachable = total - len(finite)
    return {
        "archived_count": total,
        "reachable_count": len(finite),
        "unreachable_count": unreachable,
        "unreachable_ratio": unreachable / float(total or 1),
        "hv_max": int(np.max(arr)),
        "hv_mean": float(np.mean(arr)),
        "hv_median": float(np.median(arr)),
        "hv_p95": float(np.percentile(arr, 95)),
        "cov": coverage,
    }


def compute_recoverability_report(
    cache_path: str,
    max_hops: int = 20,
    cov_marks: Optional[List[int]] = None,
) -> Dict[str, Any]:
    if cov_marks is None:
        cov_marks = [0, 1, 2, 3, 4, 5, 10, 20]

    with Path(cache_path).open("rb") as f:
        obj = pickle.load(f)
    graph = extract_nx_digraph(obj)

    is_active = {}
    archived_nodes = []
    for node_id in graph.nodes():
        status = get_node_status(graph, node_id)
        is_active[node_id] = status == "Active"
        if status == "Archived":
            archived_nodes.append(node_id)

    adjacency = build_recovery_adjacency(graph)
    per_node = []
    hops = []
    for node_id in archived_nodes:
        hop = archived_node_hop(node_id, adjacency, is_active, max_hops=max_hops)
        hops.append(hop)
        per_node.append({"node_id": node_id, "h": hop})

    report = {
        "framework": "All-Mem",
        "cache": cache_path,
        "graph_num_nodes": int(graph.number_of_nodes()),
        "graph_num_edges": int(graph.number_of_edges()),
        "active_count": int(sum(is_active.values())),
        "archived_count": int(len(archived_nodes)),
        "max_hops_cutoff": max_hops,
        "definition": "nearest Active node from Archived node over sibling/version edges",
        "summary": summarize_hops(hops, cov_marks),
    }
    return {"report": report, "per_node": per_node}


def write_report_json(payload: Dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_per_node_csv(payload: Dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "h"])
        writer.writeheader()
        for row in payload["per_node"]:
            writer.writerow(row)
