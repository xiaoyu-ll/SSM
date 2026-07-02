# Real graph + text attributes case study

This package implements the case-study pipeline you described:

1. Load a **real graph structure** dataset.
2. Use each node's **original text attribute** as the vertex text.
3. Encode node text into vectors to form a vector-attributed graph.
4. Sample a connected query subgraph from the data graph; the sampled source subgraph is the ground-truth match.
5. Run exact semantic subgraph matching under injectivity, query-edge preservation, and cosine-similarity threshold.
6. Map the matched data vertices back to their original texts.

The preferred dataset is `ogbn-arxiv`: it has real citation edges, paper title/abstract text, and a standard node index mapping. OGB documents `ogbn-arxiv` as a directed citation network where each node is a CS arXiv paper and each edge is a citation; OGB also links the mapping from MAG paper IDs to raw titles and abstracts.

## Install

```bash
pip install -r requirements.txt
```

## OGBN-Arxiv run

First prepare:

- OGB `ogbn-arxiv`, downloaded automatically by the `ogb` package.
- `titleabs.tsv.gz`, the raw title/abstract mapping linked on the OGB `ogbn-arxiv` page.

Then run:

```bash
python citation_text_case_study.py \
  --dataset ogbn-arxiv \
  --ogb-root dataset \
  --titleabs path/to/titleabs.tsv.gz \
  --max-nodes 50000 \
  --encoder tfidf-svd \
  --dim 128 \
  --query-size 5 \
  --query-edge-min 4 \
  --query-edge-max 7 \
  --source-pair-avg-sim-min 0.15 \
  --source-centroid-sim-min 0.20 \
  --tau 0.70 \
  --min-disjoint-similarity 0.70 \
  --candidate-count-max 5000 \
  --case-attempts 500 \
  --sample-tries 2000 \
  --require-disjoint \
  --attempt-log output_arxiv/attempts.jsonl \
  --save-query output_arxiv/fixed_query.json \
  --save-best-query output_arxiv/best_query.json \
  --out output_arxiv
```

Outputs:

```text
output_arxiv/case_study.md
output_arxiv/case_study.json
output_arxiv/fixed_query.json
output_arxiv/attempts.jsonl
```

To reproduce the same query:

```bash
python citation_text_case_study.py \
  --dataset ogbn-arxiv \
  --ogb-root dataset \
  --titleabs path/to/titleabs.tsv.gz \
  --max-nodes 50000 \
  --encoder tfidf-svd \
  --dim 128 \
  --tau 0.70 \
  --min-disjoint-similarity 0.70 \
  --load-query output_arxiv/fixed_query.json \
  --out output_arxiv_rerun
```

## Generic CSV run

Use this when you already have a real text-attributed graph, for example a citation graph:

`papers.csv`:

```csv
id,title,abstract,label,year
p1,Title text,Abstract text,cs.LG,2018
```

`citations.csv`:

```csv
src,dst
p1,p2
p1,p3
```

Run:

```bash
python citation_text_case_study.py \
  --dataset csv \
  --vertices-csv papers.csv \
  --edges-csv citations.csv \
  --id-col id \
  --src-col src \
  --dst-col dst \
  --text-cols title,abstract \
  --label-col label \
  --year-col year \
  --query-size 5 \
  --query-edge-min 4 \
  --query-edge-max 7 \
  --query-vector-mode source \
  --tau 0.86 \
  --min-disjoint-similarity 0.86 \
  --require-disjoint \
  --out output_csv
```

## Notes

- The matching graph uses **real input edges**. It does not build kNN text edges.
- Query vectors are copied from the sampled source vertices, so the source/ground-truth match has vertex similarity 1.0.
- Query text shown in the output is only for human interpretation; it does not change the ground-truth query vectors.
- The additional match is required to be vertex-disjoint from the ground-truth image set when `--require-disjoint` is used.
