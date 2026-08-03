# SSM: Semantic Subgraph Matching on Vector-Attributed Graphs

[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://isocpp.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg)](#requirements)

SSM is the reference implementation for **Semantic Subgraph Matching on
Vector-Attributed Graphs**. Given a query graph, a data graph, and a cosine
similarity threshold, SSM enumerates every injective mapping that:

1. preserves every query edge; and
2. maps every query vertex to a semantically compatible data vertex.

This combines the structural guarantees of exact subgraph matching with the
semantic expressiveness of vector search. Typical applications include
GraphRAG, citation analysis, knowledge-graph exploration, and scientific
evidence retrieval.

## Highlights

- **Exact semantics:** enumerates all mappings satisfying the structural and
  per-vertex similarity constraints.
- **Landmark-angle filtering:** eliminates infeasible vector pairs before full
  cosine evaluation.
- **Reusable structural support:** inverts overlapping candidate sets into
  data-vertex role sets and shares edge support across query roles.
- **Progressive execution:** `SSM-Plus+` combines progressive semantic
  evaluation with progressive multi-vertex joins.
- **Self-contained C++:** the matchers use only the C++ standard library and
  compile as standalone executables.

The repository contains three SSM variants, one adapted exact baseline, three
approximate baselines, benchmark query metadata, and a citation-text case
study.

## Repository layout

| Path | Description |
|---|---|
| `ssm_alg1.cpp` | **SSM-Basic**, the batch-enumeration framework. |
| `ssm_alg2.cpp` | **SSM-Plus**, with materialized role/support relations and support reduction. |
| `ssm_alg3.cpp` | **SSM-Plus+**, with progressive semantic evaluation and progressive joins; recommended. |
| `ceci_sem.cpp` | Semantic adaptation of the exact CECI baseline. |
| `nema.cpp` | NeMa-VEC approximate baseline. |
| `gfinder.cpp` | G-FINDER-VEC approximate baseline. |
| `magvec.cpp` | MAGE-VEC approximate baseline. |
| `query/` | Metadata for the benchmark query families. |
| `real_graph_text_case_study/` | End-to-end citation graph and text-attribute case study. |

## Requirements

- Linux or macOS
- A C++17 compiler (`g++` 9+ or a recent Apple Clang)
- Python 3.9+ only for the optional citation-text case study

No third-party C++ libraries are required.

## Build

Clone the repository and compile the standalone programs:

```bash
git clone https://github.com/xiaoyu-ll/SSM.git
cd SSM
mkdir -p build

g++ -O3 -std=c++17 ssm_alg1.cpp -o build/ssm-basic
g++ -O3 -std=c++17 ssm_alg2.cpp -o build/ssm-plus
g++ -O3 -std=c++17 ssm_alg3.cpp -o build/ssm-plus-plus

# Optional baselines
g++ -O3 -std=c++17 ceci_sem.cpp -o build/ceci-sem
g++ -O3 -std=c++17 nema.cpp     -o build/nema-vec
g++ -O3 -std=c++17 gfinder.cpp  -o build/gfinder-vec
g++ -O3 -std=c++17 magvec.cpp   -o build/mage-vec
```

Use `clang++` in place of `g++` on macOS if preferred.

## Quick start

Prepare a data graph and query graph as vertex and edge CSV files, then run the
recommended implementation:

```bash
./build/ssm-plus-plus \
  --graph-vertices data/graph_vertices.csv \
  --graph-edges data/graph_edges.csv \
  --query-vertices data/query_vertices.csv \
  --query-edges data/query_edges.csv \
  --tau 0.90 \
  --output results/matches.txt
```

Each output line is one complete mapping. The value at position `i` is the data
vertex matched to query vertex `i`:

```text
17 42 8 103
21 56 9 101
```

For large result sets, benchmark enumeration without materializing the matches:

```bash
./build/ssm-plus-plus \
  --graph-vertices data/graph_vertices.csv \
  --graph-edges data/graph_edges.csv \
  --query-vertices data/query_vertices.csv \
  --query-edges data/query_edges.csv \
  --tau 0.90 \
  --count-only
```

Run `./build/ssm-plus-plus --help` for all options. Frequently used options are:

| Option | Meaning | Default |
|---|---|---|
| `--tau FLOAT` | Per-vertex cosine-similarity threshold. | `0.97` |
| `--bmax INT` | Maximum batch size used during enumeration. | `4` |
| `--landmarks INT` | Number of landmarks; `0` uses one per query vertex. | `0` |
| `--max-matches INT` | Stop after this many matches; `0` means unlimited. | `0` |
| `--count-only` | Run enumeration without writing a match file. | off |
| `--no-normalize` | Treat input vectors as already normalized. | off |
| `--output PATH` | Match output file. | `matches_cpp.txt` |

If query paths are omitted, the program can sample a connected query from the
data graph using `--num-query-nodes` and `--seed`. For reproducible experiments,
we recommend supplying explicit query CSV files.

## Input format

### Vertex file

The first row is a header. Every following row contains a zero-based integer
vertex ID followed by a fixed-dimensional vector:

```csv
id,f0,f1,f2
0,0.12,0.44,0.81
1,0.70,0.18,0.32
2,0.09,0.63,0.55
```

Vertex IDs should be dense in `[0, n-1]`. Data and query vectors must have the
same dimensionality. By default, each vector is L2-normalized before matching;
pass `--no-normalize` only when the files already contain normalized vectors.

### Edge file

The first row is a header, followed by one endpoint pair per row:

```csv
src,dst
0,1
1,2
```

The current implementation treats graphs as **undirected**, removes duplicate
edges, and ignores self-loops.

## Benchmark data and queries

The experiments use eight public vector-attributed graphs:

- `ogbn-arxiv`
- `ogbn-mag`
- `ogbn-proteins`
- `ogbl-citation2`
- `ogbl-collab`
- `ogbl-ppa`
- `reddit_titles`
- `reddit_body`

The `query/` directory records the query-generation metadata used for nested,
topology, and density experiments. Large graph and vector CSV files are not
versioned in this repository; convert a dataset into the two CSV formats above
before running an experiment.

For fair comparisons, run all algorithms on the same normalized graph/query
files, similarity threshold, and stopping condition. The exact SSM variants
and CECI enumerate matches; the NeMa-VEC, G-FINDER-VEC, and MAGE-VEC programs
produce ranked approximate results and expose their own options through
`--help`.

## Citation-text case study

The optional Python pipeline builds a vector-attributed citation graph from
real edges and paper text, samples a connected query, runs exact semantic
subgraph matching, and maps matched vertices back to their titles and
abstracts.

```bash
cd real_graph_text_case_study
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See [`real_graph_text_case_study/README.md`](real_graph_text_case_study/README.md)
for the OGBN-Arxiv and generic CSV workflows.

## Results

In the accompanying manuscript, `SSM-Plus+` is the fastest evaluated exact
method on all eight datasets. It improves over the fastest adapted exact
baseline on each dataset by **1.5x--2.6x**, while retaining comparable memory
consumption.

## Citation

If you use this repository, please cite:

> Xiaoyu Leng, Hongchao Qin, Quanqing Xu, and Rong-Hua Li.  
> **Semantic Subgraph Matching on Vector-Attributed Graphs.**

The publication link and BibTeX entry will be added when they become available.

## Contact

For questions or bug reports, please open a
[GitHub issue](https://github.com/xiaoyu-ll/SSM/issues).
