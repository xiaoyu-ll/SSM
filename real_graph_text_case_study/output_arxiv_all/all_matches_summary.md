# Exhaustive fixed-query semantic subgraph matching

## Configuration

Dataset: `ogbn-arxiv (49993 text-bearing nodes)`
Data vertices: `49993`
Data edges used for matching: `99053`
Graph structure: real input edges, not text-kNN edges.
Text encoder: `tfidf-svd`; dimension: `128`
Semantic threshold: `0.8600`
Loaded fixed query: `output_arxiv1/fixed_query.json`
Optional match limit: `none`

## Query

Query vertices: `q1, q2, q3, q4, q5`
Query edge count: `5`
Query edges:
- `(q1,q2)`
- `(q1,q3)`
- `(q2,q3)`
- `(q3,q4)`
- `(q3,q5)`

Candidate counts:
- `q1`: `6`
- `q2`: `13`
- `q3`: `11`
- `q4`: `5`
- `q5`: `13`

## Exhaustive enumeration result

Total matches enumerated: `151`
Completed without optional limit: `True`
Search states visited: `245`
Elapsed time: `0.00` seconds
Ground-truth match count: `1`
First ground-truth match id: `1`
All matches JSONL: `all_matches.jsonl`
All matches CSV: `all_matches.csv`

## Ground-truth mapping

Minimum similarity: `1.0000`; sum similarity: `5.0000`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2953064494` | label=28 | 1.0000 | the capacity of the interference channel with a cognitive relay in very strong interference |
| `q3` | `2949179446` | label=28 | 1.0000 | outer bounds for the interference channel with a cognitive relay |
| `q4` | `2172248015` | label=28 | 1.0000 | capacity bounds for a class of interference relay channels |
| `q5` | `2952175429` | label=28 | 1.0000 | on the capacity of the interference channel with a cognitive relay |

## Top 20 matches by minimum similarity

### Top match 1: min similarity `1.0000`, sum similarity `5.0000`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2953064494` | label=28 | 1.0000 | the capacity of the interference channel with a cognitive relay in very strong interference |
| `q3` | `2949179446` | label=28 | 1.0000 | outer bounds for the interference channel with a cognitive relay |
| `q4` | `2172248015` | label=28 | 1.0000 | capacity bounds for a class of interference relay channels |
| `q5` | `2952175429` | label=28 | 1.0000 | on the capacity of the interference channel with a cognitive relay |
### Top match 2: min similarity `0.9006`, sum similarity `4.8013`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2953064494` | label=28 | 1.0000 | the capacity of the interference channel with a cognitive relay in very strong interference |
| `q3` | `2949179446` | label=28 | 1.0000 | outer bounds for the interference channel with a cognitive relay |
| `q4` | `2952175429` | label=28 | 0.9006 | on the capacity of the interference channel with a cognitive relay |
| `q5` | `2172248015` | label=28 | 0.9006 | capacity bounds for a class of interference relay channels |
### Top match 3: min similarity `0.8991`, sum similarity `4.8991`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2953064494` | label=28 | 1.0000 | the capacity of the interference channel with a cognitive relay in very strong interference |
| `q3` | `2949179446` | label=28 | 1.0000 | outer bounds for the interference channel with a cognitive relay |
| `q4` | `2950657450` | label=28 | 0.8991 | lattice coding and the generalized degrees of freedom of the interference channel with relay |
| `q5` | `2952175429` | label=28 | 1.0000 | on the capacity of the interference channel with a cognitive relay |
### Top match 4: min similarity `0.8991`, sum similarity `4.7998`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2953064494` | label=28 | 1.0000 | the capacity of the interference channel with a cognitive relay in very strong interference |
| `q3` | `2949179446` | label=28 | 1.0000 | outer bounds for the interference channel with a cognitive relay |
| `q4` | `2950657450` | label=28 | 0.8991 | lattice coding and the generalized degrees of freedom of the interference channel with relay |
| `q5` | `2172248015` | label=28 | 0.9006 | capacity bounds for a class of interference relay channels |
### Top match 5: min similarity `0.8940`, sum similarity `4.8940`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2953064494` | label=28 | 1.0000 | the capacity of the interference channel with a cognitive relay in very strong interference |
| `q3` | `2949179446` | label=28 | 1.0000 | outer bounds for the interference channel with a cognitive relay |
| `q4` | `2172248015` | label=28 | 1.0000 | capacity bounds for a class of interference relay channels |
| `q5` | `2950232678` | label=28 | 0.8940 | on the capacity of the cognitive interference channel with a common cognitive message |
### Top match 6: min similarity `0.8940`, sum similarity `4.7947`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2953064494` | label=28 | 1.0000 | the capacity of the interference channel with a cognitive relay in very strong interference |
| `q3` | `2949179446` | label=28 | 1.0000 | outer bounds for the interference channel with a cognitive relay |
| `q4` | `2952175429` | label=28 | 0.9006 | on the capacity of the interference channel with a cognitive relay |
| `q5` | `2950232678` | label=28 | 0.8940 | on the capacity of the cognitive interference channel with a common cognitive message |
### Top match 7: min similarity `0.8940`, sum similarity `4.7932`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2953064494` | label=28 | 1.0000 | the capacity of the interference channel with a cognitive relay in very strong interference |
| `q3` | `2949179446` | label=28 | 1.0000 | outer bounds for the interference channel with a cognitive relay |
| `q4` | `2950657450` | label=28 | 0.8991 | lattice coding and the generalized degrees of freedom of the interference channel with relay |
| `q5` | `2950232678` | label=28 | 0.8940 | on the capacity of the cognitive interference channel with a common cognitive message |
### Top match 8: min similarity `0.8802`, sum similarity `4.8802`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2953064494` | label=28 | 1.0000 | the capacity of the interference channel with a cognitive relay in very strong interference |
| `q3` | `2952937400` | label=28 | 0.8802 | gaussian interference channel capacity to within one bit |
| `q4` | `2172248015` | label=28 | 1.0000 | capacity bounds for a class of interference relay channels |
| `q5` | `2952175429` | label=28 | 1.0000 | on the capacity of the interference channel with a cognitive relay |
### Top match 9: min similarity `0.8802`, sum similarity `4.8418`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2949179446` | label=28 | 0.9615 | outer bounds for the interference channel with a cognitive relay |
| `q3` | `2952937400` | label=28 | 0.8802 | gaussian interference channel capacity to within one bit |
| `q4` | `2172248015` | label=28 | 1.0000 | capacity bounds for a class of interference relay channels |
| `q5` | `2952175429` | label=28 | 1.0000 | on the capacity of the interference channel with a cognitive relay |
### Top match 10: min similarity `0.8802`, sum similarity `4.8184`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2953064494` | label=28 | 1.0000 | the capacity of the interference channel with a cognitive relay in very strong interference |
| `q3` | `2952937400` | label=28 | 0.8802 | gaussian interference channel capacity to within one bit |
| `q4` | `2172248015` | label=28 | 1.0000 | capacity bounds for a class of interference relay channels |
| `q5` | `2949179446` | label=28 | 0.9381 | outer bounds for the interference channel with a cognitive relay |
### Top match 11: min similarity `0.8802`, sum similarity `4.7833`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2952988768` | label=28 | 0.9031 | the generalized degrees of freedom of the interference relay channel with strong interference |
| `q3` | `2952937400` | label=28 | 0.8802 | gaussian interference channel capacity to within one bit |
| `q4` | `2172248015` | label=28 | 1.0000 | capacity bounds for a class of interference relay channels |
| `q5` | `2952175429` | label=28 | 1.0000 | on the capacity of the interference channel with a cognitive relay |
### Top match 12: min similarity `0.8802`, sum similarity `4.7794`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2953064494` | label=28 | 1.0000 | the capacity of the interference channel with a cognitive relay in very strong interference |
| `q3` | `2952937400` | label=28 | 0.8802 | gaussian interference channel capacity to within one bit |
| `q4` | `2950657450` | label=28 | 0.8991 | lattice coding and the generalized degrees of freedom of the interference channel with relay |
| `q5` | `2952175429` | label=28 | 1.0000 | on the capacity of the interference channel with a cognitive relay |
### Top match 13: min similarity `0.8802`, sum similarity `4.7618`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2953064494` | label=28 | 1.0000 | the capacity of the interference channel with a cognitive relay in very strong interference |
| `q3` | `2952937400` | label=28 | 0.8802 | gaussian interference channel capacity to within one bit |
| `q4` | `2172248015` | label=28 | 1.0000 | capacity bounds for a class of interference relay channels |
| `q5` | `1846289915` | label=28 | 0.8816 | on the symmetric gaussian interference channel with partial unidirectional cooperation |
### Top match 14: min similarity `0.8802`, sum similarity `4.7479`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2949179446` | label=28 | 0.9615 | outer bounds for the interference channel with a cognitive relay |
| `q3` | `2952937400` | label=28 | 0.8802 | gaussian interference channel capacity to within one bit |
| `q4` | `2172248015` | label=28 | 1.0000 | capacity bounds for a class of interference relay channels |
| `q5` | `2953064494` | label=28 | 0.9061 | the capacity of the interference channel with a cognitive relay in very strong interference |
### Top match 15: min similarity `0.8802`, sum similarity `4.7409`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2949179446` | label=28 | 0.9615 | outer bounds for the interference channel with a cognitive relay |
| `q3` | `2952937400` | label=28 | 0.8802 | gaussian interference channel capacity to within one bit |
| `q4` | `2950657450` | label=28 | 0.8991 | lattice coding and the generalized degrees of freedom of the interference channel with relay |
| `q5` | `2952175429` | label=28 | 1.0000 | on the capacity of the interference channel with a cognitive relay |
### Top match 16: min similarity `0.8802`, sum similarity `4.7234`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2949179446` | label=28 | 0.9615 | outer bounds for the interference channel with a cognitive relay |
| `q3` | `2952937400` | label=28 | 0.8802 | gaussian interference channel capacity to within one bit |
| `q4` | `2172248015` | label=28 | 1.0000 | capacity bounds for a class of interference relay channels |
| `q5` | `1846289915` | label=28 | 0.8816 | on the symmetric gaussian interference channel with partial unidirectional cooperation |
### Top match 17: min similarity `0.8802`, sum similarity `4.7215`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2952988768` | label=28 | 0.9031 | the generalized degrees of freedom of the interference relay channel with strong interference |
| `q3` | `2952937400` | label=28 | 0.8802 | gaussian interference channel capacity to within one bit |
| `q4` | `2172248015` | label=28 | 1.0000 | capacity bounds for a class of interference relay channels |
| `q5` | `2949179446` | label=28 | 0.9381 | outer bounds for the interference channel with a cognitive relay |
### Top match 18: min similarity `0.8802`, sum similarity `4.7190`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2953064494` | label=28 | 1.0000 | the capacity of the interference channel with a cognitive relay in very strong interference |
| `q3` | `2952937400` | label=28 | 0.8802 | gaussian interference channel capacity to within one bit |
| `q4` | `2952175429` | label=28 | 0.9006 | on the capacity of the interference channel with a cognitive relay |
| `q5` | `2949179446` | label=28 | 0.9381 | outer bounds for the interference channel with a cognitive relay |
### Top match 19: min similarity `0.8802`, sum similarity `4.7175`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2953064494` | label=28 | 1.0000 | the capacity of the interference channel with a cognitive relay in very strong interference |
| `q3` | `2952937400` | label=28 | 0.8802 | gaussian interference channel capacity to within one bit |
| `q4` | `2950657450` | label=28 | 0.8991 | lattice coding and the generalized degrees of freedom of the interference channel with relay |
| `q5` | `2949179446` | label=28 | 0.9381 | outer bounds for the interference channel with a cognitive relay |
### Top match 20: min similarity `0.8802`, sum similarity `4.6895`

| Query vertex | Data vertex | Metadata | Similarity | Title |
|---|---|---|---:|---|
| `q1` | `2950391633` | label=28 | 1.0000 | cooperation for interference management a gdof perspective |
| `q2` | `2952988768` | label=28 | 0.9031 | the generalized degrees of freedom of the interference relay channel with strong interference |
| `q3` | `2952937400` | label=28 | 0.8802 | gaussian interference channel capacity to within one bit |
| `q4` | `2172248015` | label=28 | 1.0000 | capacity bounds for a class of interference relay channels |
| `q5` | `2953064494` | label=28 | 0.9061 | the capacity of the interference channel with a cognitive relay in very strong interference |