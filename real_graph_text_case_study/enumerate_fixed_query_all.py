#!/usr/bin/env python3
"""
Enumerate ALL exact semantic subgraph matches for a fixed query on a real
text-attributed graph.

This script is intended for the second pass after citation_text_case_study.py has
found and saved a good query, e.g. output_arxiv/fixed_query.json. It does not
apply disjointness constraints and, by default, does not cap the number of
matches. It streams every match to JSONL/CSV and writes a compact summary.

Example:
  python enumerate_fixed_query_all.py \
    --dataset ogbn-arxiv \
    --ogb-root dataset \
    --titleabs dataset/ogbn_arxiv_text/titleabs.tsv.gz \
    --max-nodes 50000 \
    --encoder tfidf-svd \
    --dim 128 \
    --load-query output_arxiv/fixed_query.json \
    --tau 0.70 \
    --out output_arxiv_all
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

# Reuse the data loading, encoding, and query utilities from the case-study script.
import citation_text_case_study as base

Mapping = Dict[str, int]


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def mapping_scores(m: Mapping, score: Dict[Tuple[str, int], float]) -> Tuple[float, float]:
    vals = [score[(qid, v)] for qid, v in m.items()]
    return float(min(vals)), float(sum(vals))


def is_ground_truth(q: base.QueryInstance, m: Mapping) -> bool:
    return all(m.get(qid) == q.source_vertices[i] for i, qid in enumerate(q.qids))


def query_neighbors_and_order(q: base.QueryInstance, domains: Dict[str, List[int]]):
    qedges = base.query_edge_set(q)
    q_neighbors: Dict[str, Set[str]] = {qid: set() for qid in q.qids}
    for a, b in qedges:
        q_neighbors[a].add(b)
        q_neighbors[b].add(a)

    # Same deterministic idea as the case-study matcher: smallest semantic domain first,
    # then higher query degree, then query id. This changes only enumeration order, not results.
    order = sorted(q.qids, key=lambda x: (len(domains[x]), -len(q_neighbors[x]), x))
    return qedges, q_neighbors, order


def row_for_match(
    graph: base.TextGraph,
    q: base.QueryInstance,
    m: Mapping,
    score: Dict[Tuple[str, int], float],
    match_id: int,
) -> Dict[str, object]:
    mn, sm = mapping_scores(m, score)
    row: Dict[str, object] = {
        "match_id": match_id,
        "min_similarity": mn,
        "sum_similarity": sm,
        "is_ground_truth": is_ground_truth(q, m),
    }
    for qid in q.qids:
        v = m[qid]
        row[f"{qid}_node_id"] = graph.node_ids[v]
        row[f"{qid}_similarity"] = score[(qid, v)]
        if graph.labels is not None:
            row[f"{qid}_label"] = graph.labels[v]
        if graph.years is not None:
            row[f"{qid}_year"] = graph.years[v]
        row[f"{qid}_title"] = graph.titles[v]
    return row


def enumerate_all_matches_streaming(
    graph: base.TextGraph,
    q: base.QueryInstance,
    domains: Dict[str, List[int]],
    score: Dict[Tuple[str, int], float],
    outdir: Path,
    top_k: int,
    progress_every: int,
    optional_limit: int,
):
    """Enumerate all matches and stream them to disk.

    optional_limit=0 means no cap. If a nonzero limit is supplied, enumeration stops
    after that many complete matches; this is a manual safety valve and is not used by
    default.
    """
    ensure_dir(outdir)
    jsonl_path = outdir / "all_matches.jsonl"
    csv_path = outdir / "all_matches.csv"

    qedges, q_neighbors, order = query_neighbors_and_order(q, domains)

    # Headers for CSV. JSONL contains the same information without a rigid schema issue.
    fieldnames = ["match_id", "min_similarity", "sum_similarity", "is_ground_truth"]
    for qid in q.qids:
        fieldnames.extend([f"{qid}_node_id", f"{qid}_similarity"])
        if graph.labels is not None:
            fieldnames.append(f"{qid}_label")
        if graph.years is not None:
            fieldnames.append(f"{qid}_year")
        fieldnames.append(f"{qid}_title")

    count = 0
    nodes_visited = 0
    gt_count = 0
    gt_first_match_id = None
    start = time.time()
    top_heap: List[Tuple[Tuple[float, float, int], Mapping]] = []

    def feasible(qid: str, v: int, cur: Mapping, used: Set[int]) -> bool:
        if v in used:
            return False
        for q2, v2 in cur.items():
            if tuple(sorted((qid, q2))) in qedges and v2 not in graph.adj[v]:
                return False
        return True

    def push_top(m: Mapping, match_id: int) -> None:
        if top_k <= 0:
            return
        mn, sm = mapping_scores(m, score)
        # Keep top by min similarity first, then sum similarity, then earlier match id.
        key = (mn, sm, -match_id)
        item = (key, dict(m))
        if len(top_heap) < top_k:
            heapq.heappush(top_heap, item)
        elif key > top_heap[0][0]:
            heapq.heapreplace(top_heap, item)

    with jsonl_path.open("w", encoding="utf-8") as jf, csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()

        def dfs(pos: int, cur: Mapping, used: Set[int]) -> bool:
            # Return True to continue, False to stop due to optional limit.
            nonlocal count, nodes_visited, gt_count, gt_first_match_id
            nodes_visited += 1
            if pos == len(order):
                count += 1
                m = dict(cur)
                row = row_for_match(graph, q, m, score, count)
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                writer.writerow(row)
                push_top(m, count)
                if row["is_ground_truth"]:
                    gt_count += 1
                    if gt_first_match_id is None:
                        gt_first_match_id = count
                if progress_every > 0 and count % progress_every == 0:
                    elapsed = time.time() - start
                    print(f"enumerated {count} matches; visited {nodes_visited} search states; elapsed {elapsed:.1f}s", flush=True)
                if optional_limit and count >= optional_limit:
                    return False
                return True

            qid = order[pos]
            for v in domains[qid]:
                if feasible(qid, v, cur, used):
                    cur[qid] = v
                    used.add(v)
                    ok = dfs(pos + 1, cur, used)
                    used.remove(v)
                    del cur[qid]
                    if not ok:
                        return False
            return True

        completed = dfs(0, {}, set())

    elapsed = time.time() - start
    top_matches = [m for _, m in sorted(top_heap, key=lambda x: x[0], reverse=True)]
    return {
        "count": count,
        "nodes_visited": nodes_visited,
        "ground_truth_count": gt_count,
        "ground_truth_first_match_id": gt_first_match_id,
        "elapsed_seconds": elapsed,
        "completed": completed,
        "enumeration_order": order,
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
        "top_matches": top_matches,
    }


def write_summary(
    outdir: Path,
    graph: base.TextGraph,
    X: np.ndarray,
    q: base.QueryInstance,
    domains: Dict[str, List[int]],
    score: Dict[Tuple[str, int], float],
    enum_info: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    ensure_dir(outdir)
    gt = {qid: q.source_vertices[i] for i, qid in enumerate(q.qids)}
    total_edges = sum(len(a) for a in graph.adj) // 2 if args.match_undirected else sum(len(a) for a in graph.adj)

    def node_meta(v: int) -> str:
        parts = []
        if graph.labels is not None:
            parts.append(f"label={graph.labels[v]}")
        if graph.years is not None:
            parts.append(f"year={graph.years[v]}")
        return "; ".join(parts)

    def mapping_table(m: Mapping) -> str:
        lines = ["| Query vertex | Data vertex | Metadata | Similarity | Title |", "|---|---|---|---:|---|"]
        for qid in q.qids:
            v = m[qid]
            lines.append(
                f"| `{qid}` | `{graph.node_ids[v]}` | {node_meta(v)} | {score[(qid, v)]:.4f} | {base.truncate(graph.titles[v], 160)} |"
            )
        return "\n".join(lines)

    summary_json = {
        "dataset": graph.source_name,
        "num_vertices": len(graph.node_ids),
        "num_edges_used_for_matching": total_edges,
        "encoder": args.encoder,
        "dim": int(X.shape[1]),
        "tau": args.tau,
        "query_file": args.load_query,
        "query_edges": q.query_edges,
        "candidate_counts": {qid: len(domains[qid]) for qid in q.qids},
        "enumeration_order": enum_info["enumeration_order"],
        "total_matches": enum_info["count"],
        "completed_without_limit": enum_info["completed"],
        "optional_limit": args.limit,
        "ground_truth_count": enum_info["ground_truth_count"],
        "ground_truth_first_match_id": enum_info["ground_truth_first_match_id"],
        "elapsed_seconds": enum_info["elapsed_seconds"],
        "all_matches_jsonl": enum_info["jsonl_path"],
        "all_matches_csv": enum_info["csv_path"],
    }
    (outdir / "all_matches_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    md = []
    md.append("# Exhaustive fixed-query semantic subgraph matching\n")
    md.append("## Configuration\n")
    md.append(f"Dataset: `{graph.source_name}`")
    md.append(f"Data vertices: `{len(graph.node_ids)}`")
    md.append(f"Data edges used for matching: `{total_edges}`")
    md.append("Graph structure: real input edges, not text-kNN edges.")
    md.append(f"Text encoder: `{args.encoder}`; dimension: `{X.shape[1]}`")
    md.append(f"Semantic threshold: `{args.tau:.4f}`")
    md.append(f"Loaded fixed query: `{args.load_query}`")
    md.append(f"Optional match limit: `{'none' if args.limit == 0 else args.limit}`")

    md.append("\n## Query\n")
    md.append(f"Query vertices: `{', '.join(q.qids)}`")
    md.append(f"Query edge count: `{len(q.query_edges)}`")
    md.append("Query edges:")
    for a, b in q.query_edges:
        md.append(f"- `({a},{b})`")
    md.append("\nCandidate counts:")
    for qid in q.qids:
        md.append(f"- `{qid}`: `{len(domains[qid])}`")

    md.append("\n## Exhaustive enumeration result\n")
    md.append(f"Total matches enumerated: `{enum_info['count']}`")
    md.append(f"Completed without optional limit: `{enum_info['completed']}`")
    md.append(f"Search states visited: `{enum_info['nodes_visited']}`")
    md.append(f"Elapsed time: `{float(enum_info['elapsed_seconds']):.2f}` seconds")
    md.append(f"Ground-truth match count: `{enum_info['ground_truth_count']}`")
    md.append(f"First ground-truth match id: `{enum_info['ground_truth_first_match_id']}`")
    md.append(f"All matches JSONL: `{Path(enum_info['jsonl_path']).name}`")
    md.append(f"All matches CSV: `{Path(enum_info['csv_path']).name}`")

    md.append("\n## Ground-truth mapping\n")
    gt_mn, gt_sum = mapping_scores(gt, score)
    md.append(f"Minimum similarity: `{gt_mn:.4f}`; sum similarity: `{gt_sum:.4f}`\n")
    md.append(mapping_table(gt))

    top_matches: List[Mapping] = enum_info["top_matches"]  # type: ignore[assignment]
    if top_matches:
        md.append(f"\n## Top {len(top_matches)} matches by minimum similarity\n")
        for i, m in enumerate(top_matches, start=1):
            mn, sm = mapping_scores(m, score)
            md.append(f"### Top match {i}: min similarity `{mn:.4f}`, sum similarity `{sm:.4f}`\n")
            md.append(mapping_table(m))

    (outdir / "all_matches_summary.md").write_text("\n".join(md), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enumerate all matches for a fixed text-attributed graph query")
    p.add_argument("--dataset", default="csv", choices=["csv", "ogbn-arxiv"])
    p.add_argument("--vertices-csv")
    p.add_argument("--edges-csv")
    p.add_argument("--id-col", default="id")
    p.add_argument("--src-col", default="src")
    p.add_argument("--dst-col", default="dst")
    p.add_argument("--text-cols", default=None)
    p.add_argument("--label-col", default="label")
    p.add_argument("--year-col", default="year")

    p.add_argument("--ogb-root", default="dataset")
    p.add_argument("--titleabs", default=None)
    p.add_argument("--nodeidx2paperid", default=None)
    p.add_argument("--label-filter", default=None)
    p.add_argument("--max-nodes", type=int, default=50000)
    p.add_argument("--min-text-words", type=int, default=20)

    p.add_argument("--match-undirected", action="store_true", default=True)
    p.add_argument("--directed-match", dest="match_undirected", action="store_false")

    p.add_argument("--encoder", choices=["tfidf-svd", "sentence-transformers"], default="tfidf-svd")
    p.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--tfidf-features", type=int, default=80000)
    p.add_argument("--tfidf-min-df", type=int, default=2)
    p.add_argument("--tfidf-max-df", type=float, default=0.9)
    p.add_argument("--dim", type=int, default=128)

    p.add_argument("--load-query", required=True, help="fixed_query.json generated by citation_text_case_study.py")
    p.add_argument("--tau", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=20, help="number of best matches to show in the markdown summary")
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--limit", type=int, default=0, help="manual safety cap; 0 means enumerate all matches")
    p.add_argument("--out", default="output_all_matches")
    args = p.parse_args()
    if args.max_nodes == 0:
        args.max_nodes = None
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.out)
    ensure_dir(outdir)

    if args.dataset == "ogbn-arxiv":
        graph = base.load_ogbn_arxiv(args)
    else:
        graph = base.load_csv_graph(args)
    print(f"Loaded {graph.source_name}: {len(graph.node_ids)} vertices")

    print("Encoding text attributes into vectors...")
    X = base.encode_texts(graph.texts, args)
    print(f"Vector matrix: {X.shape}")

    q = base.load_query(args.load_query, graph)
    print(f"Loaded fixed query with {len(q.qids)} vertices and {len(q.query_edges)} edges")

    domains, score = base.build_candidate_domains(q, X, graph, args.tau, source_mode=True)
    print("Candidate counts:", {qid: len(domains[qid]) for qid in q.qids})

    enum_info = enumerate_all_matches_streaming(
        graph=graph,
        q=q,
        domains=domains,
        score=score,
        outdir=outdir,
        top_k=args.top_k,
        progress_every=args.progress_every,
        optional_limit=args.limit,
    )
    write_summary(outdir, graph, X, q, domains, score, enum_info, args)

    print(f"Total matches enumerated: {enum_info['count']}")
    print(f"Ground-truth match count: {enum_info['ground_truth_count']}")
    print(f"Completed without optional limit: {enum_info['completed']}")
    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
