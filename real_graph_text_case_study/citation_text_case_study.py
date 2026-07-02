#!/usr/bin/env python3
"""
Case study pipeline for exact semantic subgraph matching on a real text-attributed graph.

Supported data sources:
  1) OGBN-Arxiv: real citation edges from OGB + raw title/abstract text from OGB's titleabs.tsv.gz.
  2) Generic CSV: vertices.csv with text/title/abstract columns + edges.csv with src,dst.

The script constructs a vector-attributed graph from text, samples a query from the data graph
so that the source subgraph is the ground-truth match, and then runs an exact semantic subgraph
matcher that enforces injectivity, query-edge preservation, and per-vertex cosine threshold.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import re
import sys
import textwrap
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


Edge = Tuple[int, int]
Mapping = Dict[str, int]


@dataclass
class TextGraph:
    node_ids: List[str]
    id_to_idx: Dict[str, int]
    texts: List[str]
    titles: List[str]
    labels: Optional[List[str]]
    years: Optional[List[str]]
    adj: List[Set[int]]
    directed_edges: List[Edge]
    source_name: str


@dataclass
class QueryInstance:
    qids: List[str]
    source_vertices: List[int]
    query_edges: List[Tuple[str, str]]
    query_texts: Dict[str, str]
    source_pair_avg_sim: float
    source_centroid_min_sim: float


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_text(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def truncate(s: str, n: int = 260) -> str:
    s = clean_text(s)
    if len(s) <= n:
        return s
    return s[: max(0, n - 3)].rstrip() + "..."


def read_csv_maybe_gz(path: str | Path, **kwargs) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".gz"):
        return pd.read_csv(path, compression="gzip", **kwargs)
    return pd.read_csv(path, **kwargs)


def find_file(root: str | Path, filename: str) -> Optional[Path]:
    root = Path(root)
    if not root.exists():
        return None
    for p in root.rglob(filename):
        return p
    return None


def load_titleabs(titleabs_path: str | Path) -> pd.DataFrame:
    """Read OGB titleabs.tsv(.gz). Expected columns: paper_id, title, abstract."""
    path = str(titleabs_path)
    compression = "gzip" if path.endswith(".gz") else None
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["paper_id", "title", "abstract"],
        compression=compression,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )
    df["paper_id"] = df["paper_id"].astype(str)
    df["title"] = df["title"].fillna("").map(clean_text)
    df["abstract"] = df["abstract"].fillna("").map(clean_text)
    df["text"] = (df["title"] + ". " + df["abstract"]).map(clean_text)
    return df


def load_ogbn_arxiv(args: argparse.Namespace) -> TextGraph:
    try:
        # OGB versions commonly used for ogbn-arxiv call torch.load() internally.
        # PyTorch >= 2.6 changed the default torch.load(weights_only=True), which
        # breaks OGB's cached preprocessed files. The OGB dataset files are public
        # data downloaded from Stanford SNAP/OGB; for this trusted source we
        # temporarily restore the old behavior only while OGB initializes.
        import torch

        _orig_torch_load = torch.load

        def _torch_load_compat(*load_args, **load_kwargs):
            load_kwargs.setdefault("weights_only", False)
            return _orig_torch_load(*load_args, **load_kwargs)

        torch.load = _torch_load_compat
        from ogb.nodeproppred import NodePropPredDataset
    except Exception as e:
        raise RuntimeError("ogb is required for --dataset ogbn-arxiv. Install with `pip install ogb`.") from e

    try:
        dataset = NodePropPredDataset(name="ogbn-arxiv", root=args.ogb_root)
    finally:
        try:
            torch.load = _orig_torch_load
        except Exception:
            pass
    graph, labels = dataset[0]
    edge_index = graph["edge_index"]
    num_nodes = int(graph["num_nodes"])
    label_arr = np.asarray(labels).reshape(-1)

    # OGB stores nodeidx2paperid.csv.gz under the raw mapping directory.
    mapping_path = args.nodeidx2paperid
    if mapping_path is None:
        mapping_path = find_file(args.ogb_root, "nodeidx2paperid.csv.gz")
    if mapping_path is None or not Path(mapping_path).exists():
        raise FileNotFoundError(
            "Cannot find nodeidx2paperid.csv.gz. Pass --nodeidx2paperid explicitly, "
            "or keep the default OGB directory structure under --ogb-root."
        )
    map_df = read_csv_maybe_gz(mapping_path)
    # Robust column handling.
    if len(map_df.columns) == 1:
        map_df = map_df.reset_index()
        map_df.columns = ["node_idx", "paper_id"]
    else:
        cols = list(map_df.columns)
        if "node idx" in cols:
            map_df = map_df.rename(columns={"node idx": "node_idx"})
        if "node_idx" not in map_df.columns:
            map_df = map_df.rename(columns={cols[0]: "node_idx"})
        if "paper id" in cols:
            map_df = map_df.rename(columns={"paper id": "paper_id"})
        if "paper_id" not in map_df.columns:
            map_df = map_df.rename(columns={list(map_df.columns)[1]: "paper_id"})
    map_df["node_idx"] = map_df["node_idx"].astype(int)
    map_df["paper_id"] = map_df["paper_id"].astype(str)

    titleabs_path = args.titleabs
    if titleabs_path is None:
        raise FileNotFoundError(
            "For ogbn-arxiv text attributes, pass --titleabs path/to/titleabs.tsv.gz. "
            "This is the raw title/abstract mapping linked by the OGB ogbn-arxiv page."
        )
    title_df = load_titleabs(titleabs_path)
    title_by_paper = dict(zip(title_df["paper_id"], title_df["title"]))
    text_by_paper = dict(zip(title_df["paper_id"], title_df["text"]))

    if args.max_nodes and args.max_nodes < num_nodes:
        rng = np.random.default_rng(args.seed)
        if args.label_filter:
            keep_labels = {int(x) for x in args.label_filter.split(",") if x.strip() != ""}
            pool = np.array([i for i, y in enumerate(label_arr) if int(y) in keep_labels], dtype=np.int64)
        else:
            pool = np.arange(num_nodes, dtype=np.int64)
        if len(pool) > args.max_nodes:
            chosen = np.sort(rng.choice(pool, size=args.max_nodes, replace=False))
        else:
            chosen = np.sort(pool)
        old_to_new = {int(v): i for i, v in enumerate(chosen)}
    else:
        chosen = np.arange(num_nodes, dtype=np.int64)
        old_to_new = {i: i for i in range(num_nodes)}

    # Build metadata only for chosen nodes.
    paper_ids_by_node = dict(zip(map_df["node_idx"], map_df["paper_id"]))
    node_ids: List[str] = []
    titles: List[str] = []
    texts: List[str] = []
    labels_out: List[str] = []
    selected_old: List[int] = []
    for old in chosen:
        old = int(old)
        pid = paper_ids_by_node.get(old)
        if pid is None:
            continue
        text = text_by_paper.get(str(pid), "")
        if len(text.split()) < args.min_text_words:
            continue
        selected_old.append(old)
        node_ids.append(str(pid))
        titles.append(title_by_paper.get(str(pid), ""))
        texts.append(text)
        labels_out.append(str(int(label_arr[old])))

    # Remap again after dropping missing-text nodes.
    old_to_new = {old: i for i, old in enumerate(selected_old)}
    adj = [set() for _ in selected_old]
    directed_edges: List[Edge] = []
    srcs = edge_index[0]
    dsts = edge_index[1]
    for s, t in zip(srcs, dsts):
        s = int(s); t = int(t)
        if s in old_to_new and t in old_to_new and s != t:
            u = old_to_new[s]; v = old_to_new[t]
            directed_edges.append((u, v))
            if args.match_undirected:
                adj[u].add(v); adj[v].add(u)
            else:
                adj[u].add(v)

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    return TextGraph(
        node_ids=node_ids,
        id_to_idx=id_to_idx,
        texts=texts,
        titles=titles,
        labels=labels_out,
        years=None,
        adj=adj,
        directed_edges=directed_edges,
        source_name=f"ogbn-arxiv ({len(node_ids)} text-bearing nodes)",
    )


def load_csv_graph(args: argparse.Namespace) -> TextGraph:
    if not args.vertices_csv or not args.edges_csv:
        raise ValueError("CSV mode requires --vertices-csv and --edges-csv.")
    vdf = pd.read_csv(args.vertices_csv)
    edf = pd.read_csv(args.edges_csv)
    id_col = args.id_col
    if id_col not in vdf.columns:
        raise ValueError(f"id column {id_col!r} not found in vertices CSV")
    node_ids = vdf[id_col].astype(str).tolist()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    if args.text_cols:
        text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    else:
        if "text" in vdf.columns:
            text_cols = ["text"]
        else:
            text_cols = [c for c in ["title", "abstract", "keywords", "description"] if c in vdf.columns]
    if not text_cols:
        raise ValueError("No text columns found. Pass --text-cols title,abstract,...")

    texts = vdf[text_cols].fillna("").astype(str).agg(". ".join, axis=1).map(clean_text).tolist()
    if "title" in vdf.columns:
        titles = vdf["title"].fillna("").astype(str).map(clean_text).tolist()
    else:
        titles = [truncate(t, 90) for t in texts]
    labels = vdf[args.label_col].astype(str).tolist() if args.label_col and args.label_col in vdf.columns else None
    years = vdf[args.year_col].astype(str).tolist() if args.year_col and args.year_col in vdf.columns else None

    src_col = args.src_col
    dst_col = args.dst_col
    if src_col not in edf.columns or dst_col not in edf.columns:
        raise ValueError(f"edge columns {src_col!r}, {dst_col!r} not found in edges CSV")
    adj = [set() for _ in node_ids]
    directed_edges: List[Edge] = []
    for _, row in edf.iterrows():
        s = str(row[src_col]); t = str(row[dst_col])
        if s not in id_to_idx or t not in id_to_idx or s == t:
            continue
        u = id_to_idx[s]; v = id_to_idx[t]
        directed_edges.append((u, v))
        if args.match_undirected:
            adj[u].add(v); adj[v].add(u)
        else:
            adj[u].add(v)
    return TextGraph(node_ids, id_to_idx, texts, titles, labels, years, adj, directed_edges, "custom CSV")


def encode_texts(texts: List[str], args: argparse.Namespace) -> np.ndarray:
    if args.encoder == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("Install sentence-transformers or use --encoder tfidf-svd.") from e
        model = SentenceTransformer(args.st_model)
        X = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
        return np.asarray(X, dtype=np.float32)

    # TF-IDF + SVD, deterministic and lightweight.
    vectorizer = TfidfVectorizer(
        max_features=args.tfidf_features,
        min_df=args.tfidf_min_df,
        max_df=args.tfidf_max_df,
        stop_words="english",
        ngram_range=(1, 2),
    )
    A = vectorizer.fit_transform(texts)
    max_dim = min(args.dim, max(1, min(A.shape) - 1))
    if max_dim < 2:
        X = A.toarray().astype(np.float32)
    else:
        svd = TruncatedSVD(n_components=max_dim, random_state=args.seed)
        X = svd.fit_transform(A).astype(np.float32)
    X = normalize(X, norm="l2", axis=1)
    return np.asarray(X, dtype=np.float32)


def extract_keywords(text: str, topk: int = 10) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "have", "has", "had",
        "into", "their", "there", "which", "using", "used", "paper", "study", "results", "show", "based",
        "method", "methods", "model", "models", "approach", "problem", "data", "new", "present", "propose",
    }
    counts: Dict[str, int] = defaultdict(int)
    for w in words:
        if w not in stop and len(w) >= 4:
            counts[w] += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:topk]
    return ", ".join(w for w, _ in ranked)


def make_query_text(text: str, mode: str, keyword_topk: int) -> str:
    text = clean_text(text)
    if mode == "original":
        return text
    if mode == "sentence":
        parts = re.split(r"(?<=[.!?])\s+", text)
        return clean_text(parts[0] if parts else text)[:400]
    kws = extract_keywords(text, topk=keyword_topk)
    return f"paper about {kws}" if kws else truncate(text, 200)


def connected(nodes: Sequence[int], adj: List[Set[int]]) -> bool:
    if not nodes:
        return False
    S = set(nodes)
    seen = {nodes[0]}
    dq = deque([nodes[0]])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v in S and v not in seen:
                seen.add(v); dq.append(v)
    return len(seen) == len(S)


def induced_edges(nodes: Sequence[int], adj: List[Set[int]]) -> List[Tuple[int, int]]:
    S = set(nodes)
    out = []
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            if v in adj[u]:
                out.append((u, v))
    return out


def sample_connected_subgraph(adj: List[Set[int]], size: int, rng: random.Random, tries: int) -> Optional[List[int]]:
    n = len(adj)
    valid_starts = [i for i, a in enumerate(adj) if len(a) > 0]
    if not valid_starts:
        return None
    for _ in range(tries):
        start = rng.choice(valid_starts)
        S = [start]
        frontier = set(adj[start])
        while len(S) < size and frontier:
            v = rng.choice(tuple(frontier))
            frontier.discard(v)
            if v in S:
                continue
            S.append(v)
            frontier.update(adj[v] - set(S))
        if len(S) == size and connected(S, adj):
            return S
    return None


def semantic_cohesion(nodes: Sequence[int], X: np.ndarray) -> Tuple[float, float]:
    Y = X[list(nodes)]
    C = Y @ Y.T
    vals = [float(C[i, j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]
    pair_avg = float(np.mean(vals)) if vals else 1.0
    centroid = Y.mean(axis=0, keepdims=True)
    centroid = normalize(centroid, norm="l2", axis=1)
    coss = (Y @ centroid.T).reshape(-1)
    return pair_avg, float(np.min(coss))


def build_query_from_nodes(nodes: List[int], graph: TextGraph, X: np.ndarray, args: argparse.Namespace) -> QueryInstance:
    qids = [f"q{i+1}" for i in range(len(nodes))]
    node_to_q = {v: qids[i] for i, v in enumerate(nodes)}
    e = induced_edges(nodes, graph.adj)
    qedges = [(node_to_q[u], node_to_q[v]) for u, v in e]
    qtexts = {qids[i]: make_query_text(graph.texts[v], args.query_text_mode, args.keyword_topk) for i, v in enumerate(nodes)}
    pair_avg, centroid_min = semantic_cohesion(nodes, X)
    return QueryInstance(qids, nodes, qedges, qtexts, pair_avg, centroid_min)


def save_query(q: QueryInstance, graph: TextGraph, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    obj = {
        "qids": q.qids,
        "source_node_ids": [graph.node_ids[i] for i in q.source_vertices],
        "query_edges": q.query_edges,
        "query_texts": q.query_texts,
        "source_pair_avg_sim": q.source_pair_avg_sim,
        "source_centroid_min_sim": q.source_centroid_min_sim,
    }
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_query(path: str | Path, graph: TextGraph) -> QueryInstance:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    source_node_ids = obj["source_node_ids"]
    missing = [nid for nid in source_node_ids if nid not in graph.id_to_idx]
    if missing:
        raise ValueError(f"Loaded query refers to nodes not present in the current graph: {missing[:5]}")
    source_vertices = [graph.id_to_idx[nid] for nid in source_node_ids]
    return QueryInstance(
        qids=obj["qids"],
        source_vertices=source_vertices,
        query_edges=[tuple(x) for x in obj["query_edges"]],
        query_texts=obj["query_texts"],
        source_pair_avg_sim=float(obj.get("source_pair_avg_sim", 0.0)),
        source_centroid_min_sim=float(obj.get("source_centroid_min_sim", 0.0)),
    )


def query_edge_set(q: QueryInstance) -> Set[Tuple[str, str]]:
    return {tuple(sorted(e)) for e in q.query_edges}


def build_candidate_domains(q: QueryInstance, X: np.ndarray, graph: TextGraph, tau: float, source_mode: bool) -> Tuple[Dict[str, List[int]], Dict[Tuple[str, int], float]]:
    qvecs = []
    if source_mode:
        for v in q.source_vertices:
            qvecs.append(X[v])
        QX = np.vstack(qvecs)
    else:
        # If using query text vectors, encode separately using the same fitted encoder is nontrivial here.
        # For case-study ground truth, source mode is the intended setting.
        raise NotImplementedError("This script uses source query vectors for query-by-example ground truth.")

    qdeg = {qid: 0 for qid in q.qids}
    for a, b in q.query_edges:
        qdeg[a] += 1; qdeg[b] += 1

    sims = QX @ X.T
    domains: Dict[str, List[int]] = {}
    score: Dict[Tuple[str, int], float] = {}
    degs = np.array([len(a) for a in graph.adj])
    for i, qid in enumerate(q.qids):
        mask = (sims[i] >= tau) & (degs >= qdeg[qid])
        cand = np.where(mask)[0].tolist()
        # Stable order: high similarity first, then node index.
        cand.sort(key=lambda v: (-float(sims[i, v]), v))
        domains[qid] = cand
        for v in cand:
            score[(qid, v)] = float(sims[i, v])
    return domains, score


def enumerate_matches(
    q: QueryInstance,
    graph: TextGraph,
    domains: Dict[str, List[int]],
    score: Dict[Tuple[str, int], float],
    max_matches: int,
    source_vertices: Set[int],
    min_disjoint_similarity: float,
    stop_when_good_disjoint: bool,
) -> Tuple[List[Mapping], Optional[Mapping], int, bool]:
    qedges = query_edge_set(q)
    q_neighbors: Dict[str, Set[str]] = {qid: set() for qid in q.qids}
    for a, b in qedges:
        q_neighbors[a].add(b); q_neighbors[b].add(a)

    order = sorted(q.qids, key=lambda x: (len(domains[x]), -len(q_neighbors[x]), x))
    matches: List[Mapping] = []
    best_disjoint: Optional[Mapping] = None
    best_key = (-1.0, -1.0)
    nodes_visited = 0
    capped = False

    def feasible(qid: str, v: int, cur: Mapping, used: Set[int]) -> bool:
        if v in used:
            return False
        for q2, v2 in cur.items():
            if tuple(sorted((qid, q2))) in qedges and v2 not in graph.adj[v]:
                return False
        return True

    def update_best(m: Mapping) -> None:
        nonlocal best_disjoint, best_key
        img = set(m.values())
        if img.isdisjoint(source_vertices):
            vals = [score[(qid, v)] for qid, v in m.items()]
            mn = min(vals); sm = sum(vals)
            if mn >= min_disjoint_similarity and (mn, sm) > best_key:
                best_key = (mn, sm)
                best_disjoint = dict(m)

    def dfs(pos: int, cur: Mapping, used: Set[int]) -> None:
        nonlocal nodes_visited, capped
        if capped:
            return
        nodes_visited += 1
        if pos == len(order):
            m = dict(cur)
            matches.append(m)
            update_best(m)
            if len(matches) >= max_matches:
                capped = True
            if stop_when_good_disjoint and best_disjoint is not None and is_ground_truth_recovered(q, matches):
                capped = True
            return
        qid = order[pos]
        for v in domains[qid]:
            if feasible(qid, v, cur, used):
                cur[qid] = v; used.add(v)
                dfs(pos + 1, cur, used)
                used.remove(v); del cur[qid]
                if capped:
                    return

    dfs(0, {}, set())
    return matches, best_disjoint, nodes_visited, capped


def is_ground_truth_mapping(q: QueryInstance, m: Mapping) -> bool:
    return all(m.get(qid) == q.source_vertices[i] for i, qid in enumerate(q.qids))


def is_ground_truth_recovered(q: QueryInstance, matches: List[Mapping]) -> bool:
    return any(is_ground_truth_mapping(q, m) for m in matches)


def mapping_stats(m: Mapping, score: Dict[Tuple[str, int], float]) -> Tuple[float, float]:
    vals = [score[(qid, v)] for qid, v in m.items()]
    return min(vals), sum(vals)


def preserved_edges(q: QueryInstance, m: Mapping) -> List[Tuple[int, int]]:
    return [(m[a], m[b]) for a, b in q.query_edges]


def choose_query_case(graph: TextGraph, X: np.ndarray, args: argparse.Namespace) -> Tuple[QueryInstance, Dict[str, List[int]], Dict[Tuple[str, int], float], List[Mapping], Optional[Mapping], Dict[str, object]]:
    rng = random.Random(args.seed)
    best_record = None
    best_score_tuple = (-1.0, -1.0, -1.0)
    attempt_log = None
    if args.attempt_log:
        os.makedirs(os.path.dirname(args.attempt_log) or ".", exist_ok=True)
        attempt_log = open(args.attempt_log, "w", encoding="utf-8")

    try:
        for attempt in range(args.case_attempts):
            nodes = sample_connected_subgraph(graph.adj, args.query_size, rng, args.sample_tries)
            if nodes is None:
                continue
            e = induced_edges(nodes, graph.adj)
            if not (args.query_edge_min <= len(e) <= args.query_edge_max):
                continue
            pair_avg, cent_min = semantic_cohesion(nodes, X)
            if pair_avg < args.source_pair_avg_sim_min or cent_min < args.source_centroid_sim_min:
                continue
            q = build_query_from_nodes(nodes, graph, X, args)
            domains, simscore = build_candidate_domains(q, X, graph, args.tau, source_mode=True)
            cand_counts = [len(domains[qid]) for qid in q.qids]
            if min(cand_counts) < args.candidate_count_min or max(cand_counts) > args.candidate_count_max:
                continue
            matches, disjoint, visited, capped = enumerate_matches(
                q, graph, domains, simscore, args.max_matches, set(q.source_vertices),
                args.min_disjoint_similarity, args.stop_when_good_disjoint,
            )
            gt = is_ground_truth_recovered(q, matches)
            dis_mn = dis_sm = None
            if disjoint:
                dis_mn, dis_sm = mapping_stats(disjoint, simscore)
            rec = {
                "attempt": attempt,
                "source_node_ids": [graph.node_ids[v] for v in q.source_vertices],
                "edge_count": len(q.query_edges),
                "pair_avg": pair_avg,
                "centroid_min": cent_min,
                "candidate_counts": cand_counts,
                "match_count_seen": len(matches),
                "capped": capped,
                "ground_truth_recovered": gt,
                "disjoint_found": disjoint is not None,
                "disjoint_min_similarity": dis_mn,
                "disjoint_sum_similarity": dis_sm,
            }
            if attempt_log:
                attempt_log.write(json.dumps(rec) + "\n"); attempt_log.flush()
            quality = (dis_mn or -1.0, pair_avg, -max(cand_counts))
            if quality > best_score_tuple:
                best_score_tuple = quality
                best_record = (q, domains, simscore, matches, disjoint, rec)
                if args.save_best_query:
                    save_query(q, graph, args.save_best_query)
            if gt and (not args.require_disjoint or disjoint is not None):
                return q, domains, simscore, matches, disjoint, rec
        if best_record is not None and not args.fail_if_no_case:
            return best_record
        raise RuntimeError("No acceptable case found. Try increasing case-attempts/sample-tries, lowering tau, or relaxing candidate/cohesion constraints.")
    finally:
        if attempt_log:
            attempt_log.close()


def run_fixed_query(graph: TextGraph, X: np.ndarray, q: QueryInstance, args: argparse.Namespace):
    domains, simscore = build_candidate_domains(q, X, graph, args.tau, source_mode=True)
    matches, disjoint, visited, capped = enumerate_matches(
        q, graph, domains, simscore, args.max_matches, set(q.source_vertices),
        args.min_disjoint_similarity, args.stop_when_good_disjoint,
    )
    rec = {
        "edge_count": len(q.query_edges),
        "candidate_counts": [len(domains[qid]) for qid in q.qids],
        "match_count_seen": len(matches),
        "capped": capped,
        "ground_truth_recovered": is_ground_truth_recovered(q, matches),
        "disjoint_found": disjoint is not None,
    }
    return domains, simscore, matches, disjoint, rec


def write_outputs(outdir: str | Path, graph: TextGraph, X: np.ndarray, q: QueryInstance, domains, simscore, matches, disjoint, rec, args) -> None:
    outdir = Path(outdir); ensure_dir(outdir)
    gt = {qid: q.source_vertices[i] for i, qid in enumerate(q.qids)}
    gt_recovered = is_ground_truth_recovered(q, matches)
    total_edges = sum(len(a) for a in graph.adj) // 2 if args.match_undirected else sum(len(a) for a in graph.adj)

    result = {
        "dataset": graph.source_name,
        "num_vertices": len(graph.node_ids),
        "num_edges_used_for_matching": total_edges,
        "encoder": args.encoder,
        "tau": args.tau,
        "query": {
            "qids": q.qids,
            "source_node_ids": [graph.node_ids[v] for v in q.source_vertices],
            "query_edges": q.query_edges,
            "query_texts": q.query_texts,
            "source_pair_avg_sim": q.source_pair_avg_sim,
            "source_centroid_min_sim": q.source_centroid_min_sim,
        },
        "candidate_counts": {qid: len(domains[qid]) for qid in q.qids},
        "matches_seen": len(matches),
        "enumeration_capped": rec.get("capped", False),
        "ground_truth_recovered": gt_recovered,
        "additional_disjoint_found": disjoint is not None,
    }
    (outdir / "case_study.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    def node_label(v: int) -> str:
        parts = []
        if graph.labels is not None:
            parts.append(f"label={graph.labels[v]}")
        if graph.years is not None:
            parts.append(f"year={graph.years[v]}")
        return "; ".join(parts) if parts else ""

    def mapping_table(m: Mapping) -> str:
        lines = ["| Query vertex | Data vertex | Metadata | Similarity | Query text | Original title/text |", "|---|---|---|---:|---|---|"]
        for qid in q.qids:
            v = m[qid]
            orig = f"**{truncate(graph.titles[v], 120)}**. {truncate(graph.texts[v], 260)}" if graph.titles[v] else truncate(graph.texts[v], 360)
            lines.append(f"| `{qid}` | `{graph.node_ids[v]}` | {node_label(v)} | {simscore[(qid, v)]:.4f} | {truncate(q.query_texts[qid], 180)} | {orig} |")
        return "\n".join(lines)

    gt_mn, gt_sum = mapping_stats(gt, simscore)
    md = []
    md.append("# Real text-attributed graph case study\n")
    md.append("## Dataset and graph construction\n")
    md.append(f"Dataset: `{graph.source_name}`\n")
    md.append(f"Number of data vertices used for matching: `{len(graph.node_ids)}`\n")
    md.append(f"Number of {'undirected' if args.match_undirected else 'directed'} data edges used for matching: `{total_edges}`\n")
    md.append(f"Graph structure: real input edges, not text-kNN edges.\n")
    md.append(f"Text-to-vector encoder: `{args.encoder}`; vector dimension: `{X.shape[1]}`\n")
    md.append(f"Query vector mode: `source`; therefore the source/ground-truth vertex similarities are exactly 1 up to numerical precision.\n")
    md.append(f"Semantic threshold: `{args.tau:.4f}`\n")
    md.append(f"Total exact matches enumerated before stopping/cap: `{len(matches)}`\n")
    md.append(f"Enumeration capped: `{bool(rec.get('capped', False))}`\n")
    md.append(f"Ground-truth recovered: `{gt_recovered}`\n")
    md.append(f"Additional match disjoint from ground truth with min similarity >= `{args.min_disjoint_similarity:.4f}`: `{disjoint is not None}`\n")

    md.append("## Query sampled from the real graph\n")
    md.append(f"Query edge count: `{len(q.query_edges)}`\n")
    md.append(f"Source pairwise average similarity: `{q.source_pair_avg_sim:.4f}`\n")
    md.append(f"Source centroid minimum similarity: `{q.source_centroid_min_sim:.4f}`\n")
    md.append("\n| Query vertex | Ground-truth data vertex | Metadata | Candidate count | Query text | Source title/text |\n|---|---|---|---:|---|---|")
    for i, qid in enumerate(q.qids):
        v = q.source_vertices[i]
        orig = f"**{truncate(graph.titles[v], 120)}**. {truncate(graph.texts[v], 260)}" if graph.titles[v] else truncate(graph.texts[v], 360)
        md.append(f"| `{qid}` | `{graph.node_ids[v]}` | {node_label(v)} | {len(domains[qid])} | {truncate(q.query_texts[qid], 180)} | {orig} |")
    md.append("\nQuery edges with their ground-truth data edges:\n")
    for a, b in q.query_edges:
        md.append(f"- `({a},{b})` -> `({graph.node_ids[gt[a]]},{graph.node_ids[gt[b]]})`")

    md.append("\n## Recovered ground-truth match\n")
    md.append(f"Minimum vertex similarity: `{gt_mn:.4f}`; sum similarity: `{gt_sum:.4f}`\n")
    md.append(mapping_table(gt))
    md.append("\nPreserved data edges corresponding to query edges:\n")
    for u, v in preserved_edges(q, gt):
        md.append(f"- `({graph.node_ids[u]},{graph.node_ids[v]})`")

    if disjoint:
        mn, sm = mapping_stats(disjoint, simscore)
        md.append("\n## Additional vertex-disjoint match\n")
        md.append(f"Minimum vertex similarity: `{mn:.4f}`; sum similarity: `{sm:.4f}`\n")
        md.append(mapping_table(disjoint))
        md.append("\nPreserved data edges corresponding to query edges:\n")
        for u, v in preserved_edges(q, disjoint):
            md.append(f"- `({graph.node_ids[u]},{graph.node_ids[v]})`")
        md.append("\n## Disjointness check\n")
        gt_set = {graph.node_ids[v] for v in gt.values()}
        dj_set = {graph.node_ids[v] for v in disjoint.values()}
        md.append("Ground-truth image set:\n")
        md.append("`" + ", ".join(sorted(gt_set)) + "`\n")
        md.append("Additional-match image set:\n")
        md.append("`" + ", ".join(sorted(dj_set)) + "`\n")
        md.append("Intersection:\n")
        md.append("`" + ", ".join(sorted(gt_set & dj_set)) + "`\n")

    (outdir / "case_study.md").write_text("\n".join(md), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real text-attributed graph semantic subgraph case study")
    p.add_argument("--dataset", default="csv", choices=["csv", "ogbn-arxiv"])
    p.add_argument("--vertices-csv")
    p.add_argument("--edges-csv")
    p.add_argument("--id-col", default="id")
    p.add_argument("--src-col", default="src")
    p.add_argument("--dst-col", default="dst")
    p.add_argument("--text-cols", default=None, help="Comma-separated text columns, e.g., title,abstract")
    p.add_argument("--label-col", default="label")
    p.add_argument("--year-col", default="year")

    p.add_argument("--ogb-root", default="dataset")
    p.add_argument("--titleabs", default=None, help="Path to OGB titleabs.tsv.gz for ogbn-arxiv")
    p.add_argument("--nodeidx2paperid", default=None)
    p.add_argument("--label-filter", default=None, help="Comma-separated OGB class labels to keep, e.g., 0,1,2")
    p.add_argument("--max-nodes", type=int, default=50000, help="Random node subset before text filtering; use 0 for full graph")
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

    p.add_argument("--query-size", type=int, default=5)
    p.add_argument("--query-edge-min", type=int, default=4)
    p.add_argument("--query-edge-max", type=int, default=7)
    p.add_argument("--query-text-mode", choices=["original", "sentence", "keywords"], default="keywords")
    p.add_argument("--keyword-topk", type=int, default=12)
    p.add_argument("--source-pair-avg-sim-min", type=float, default=0.15)
    p.add_argument("--source-centroid-sim-min", type=float, default=0.20)
    p.add_argument("--sample-tries", type=int, default=2000)
    p.add_argument("--case-attempts", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-query", default=None)
    p.add_argument("--load-query", default=None)
    p.add_argument("--save-best-query", default=None)
    p.add_argument("--attempt-log", default=None)

    p.add_argument("--tau", type=float, default=0.70)
    p.add_argument("--min-disjoint-similarity", type=float, default=0.70)
    p.add_argument("--candidate-count-min", type=int, default=1)
    p.add_argument("--candidate-count-max", type=int, default=5000)
    p.add_argument("--max-matches", type=int, default=50000)
    p.add_argument("--require-disjoint", action="store_true")
    p.add_argument("--stop-when-good-disjoint", action="store_true", default=True)
    p.add_argument("--no-stop-when-good-disjoint", dest="stop_when_good_disjoint", action="store_false")
    p.add_argument("--fail-if-no-case", action="store_true")
    p.add_argument("--out", default="output_real_graph")
    args = p.parse_args()
    if args.max_nodes == 0:
        args.max_nodes = None
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "ogbn-arxiv":
        graph = load_ogbn_arxiv(args)
    else:
        graph = load_csv_graph(args)
    if len(graph.node_ids) == 0:
        raise RuntimeError("No text-bearing vertices loaded.")
    print(f"Loaded {graph.source_name}: {len(graph.node_ids)} vertices")
    print("Encoding text attributes into vectors...")
    X = encode_texts(graph.texts, args)
    print(f"Vector matrix: {X.shape}")

    if args.load_query:
        q = load_query(args.load_query, graph)
        domains, simscore, matches, disjoint, rec = run_fixed_query(graph, X, q, args)
    else:
        q, domains, simscore, matches, disjoint, rec = choose_query_case(graph, X, args)
        if args.save_query:
            save_query(q, graph, args.save_query)

    write_outputs(args.out, graph, X, q, domains, simscore, matches, disjoint, rec, args)
    print(f"Wrote outputs to {args.out}")
    print(f"Ground truth recovered: {is_ground_truth_recovered(q, matches)}")
    print(f"Additional disjoint match found: {disjoint is not None}")


if __name__ == "__main__":
    main()
