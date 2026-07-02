# Real text-attributed graph case study

## Dataset and graph construction

Dataset: `ogbn-arxiv (49993 text-bearing nodes)`

Number of data vertices used for matching: `49993`

Number of undirected data edges used for matching: `99053`

Graph structure: real input edges, not text-kNN edges.

Text-to-vector encoder: `tfidf-svd`; vector dimension: `128`

Query vector mode: `source`; therefore the source/ground-truth vertex similarities are exactly 1 up to numerical precision.

Semantic threshold: `0.8000`

Total exact matches enumerated before stopping/cap: `122`

Enumeration capped: `True`

Ground-truth recovered: `True`

Additional match disjoint from ground truth with min similarity >= `0.8000`: `True`

## Query sampled from the real graph

Query edge count: `4`

Source pairwise average similarity: `0.7610`

Source centroid minimum similarity: `0.8690`


| Query vertex | Ground-truth data vertex | Metadata | Candidate count | Query text | Source title/text |
|---|---|---|---:|---|---|
| `q1` | `2949640503` | label=28 | 51 | paper about decoding, sc-ldpc, code, ensembles, binary, both, codes, complexity, design, latency, over, thresholds | **design of spatially coupled ldpc codes over gf q for windowed decoding**. design of spatially coupled ldpc codes over gf q for windowed decoding. In this paper we consider the generalization of binary spatially coupled low-density parity-check (SC-LDPC) codes to finite fields GF$(q)$, $q\geq 2$, and develop design rules for $q$-a... |
| `q2` | `2950598277` | label=28 | 14 | paper about threshold, channels, over, transmission, also, binary, class, consider, coupling, ensemble, ensembles, memory | **threshold saturation on channels with memory via spatial coupling**. threshold saturation on channels with memory via spatial coupling. We consider spatially coupled code ensembles. A particular instance are convolutional LDPC ensembles. It was recently shown that, for transmission over the memoryless binary erasure channel,... |
| `q3` | `2125566184` | label=28 | 52 | paper about codes, analysis, channels, ldpc, binary, messages, approximation, arbitrary, coset, designed, gaussian, including | **design and analysis of nonbinary ldpc codes for arbitrary discrete memoryless channels**. design and analysis of nonbinary ldpc codes for arbitrary discrete memoryless channels. We present an analysis under the iterative decoding of coset low-density parity-check (LDPC) codes over GF(q), designed for use over arbitrary discrete-memoryless channe... |
| `q4` | `2951382382` | label=28 | 65 | paper about codes, ldpc, hybrid, condition, stability, advantages, analyse, analysis, asymptotic, asymptotically, been, behavior | **analysis of non binary hybrid ldpc codes**. analysis of non binary hybrid ldpc codes. In this paper, we analyse asymptotically a new class of LDPC codes called Non-binary Hybrid LDPC codes, which has been recently introduced. We use density evolution techniques to derive a stability condition for hyb... |
| `q5` | `2950093758` | label=28 | 11 | paper about over, posteriori, belief, binary, case, maximum, propagation, transmission, area, channel, erasure, general | **the generalized area theorem and some of its consequences**. the generalized area theorem and some of its consequences. There is a fundamental relationship between belief propagation and maximum a posteriori decoding. The case of transmission over the binary erasure channel was investigated in detail in a companion p... |

Query edges with their ground-truth data edges:

- `(q1,q2)` -> `(2949640503,2950598277)`
- `(q1,q3)` -> `(2949640503,2125566184)`
- `(q2,q5)` -> `(2950598277,2950093758)`
- `(q3,q4)` -> `(2125566184,2951382382)`

## Recovered ground-truth match

Minimum vertex similarity: `1.0000`; sum similarity: `5.0000`

| Query vertex | Data vertex | Metadata | Similarity | Query text | Original title/text |
|---|---|---|---:|---|---|
| `q1` | `2949640503` | label=28 | 1.0000 | paper about decoding, sc-ldpc, code, ensembles, binary, both, codes, complexity, design, latency, over, thresholds | **design of spatially coupled ldpc codes over gf q for windowed decoding**. design of spatially coupled ldpc codes over gf q for windowed decoding. In this paper we consider the generalization of binary spatially coupled low-density parity-check (SC-LDPC) codes to finite fields GF$(q)$, $q\geq 2$, and develop design rules for $q$-a... |
| `q2` | `2950598277` | label=28 | 1.0000 | paper about threshold, channels, over, transmission, also, binary, class, consider, coupling, ensemble, ensembles, memory | **threshold saturation on channels with memory via spatial coupling**. threshold saturation on channels with memory via spatial coupling. We consider spatially coupled code ensembles. A particular instance are convolutional LDPC ensembles. It was recently shown that, for transmission over the memoryless binary erasure channel,... |
| `q3` | `2125566184` | label=28 | 1.0000 | paper about codes, analysis, channels, ldpc, binary, messages, approximation, arbitrary, coset, designed, gaussian, including | **design and analysis of nonbinary ldpc codes for arbitrary discrete memoryless channels**. design and analysis of nonbinary ldpc codes for arbitrary discrete memoryless channels. We present an analysis under the iterative decoding of coset low-density parity-check (LDPC) codes over GF(q), designed for use over arbitrary discrete-memoryless channe... |
| `q4` | `2951382382` | label=28 | 1.0000 | paper about codes, ldpc, hybrid, condition, stability, advantages, analyse, analysis, asymptotic, asymptotically, been, behavior | **analysis of non binary hybrid ldpc codes**. analysis of non binary hybrid ldpc codes. In this paper, we analyse asymptotically a new class of LDPC codes called Non-binary Hybrid LDPC codes, which has been recently introduced. We use density evolution techniques to derive a stability condition for hyb... |
| `q5` | `2950093758` | label=28 | 1.0000 | paper about over, posteriori, belief, binary, case, maximum, propagation, transmission, area, channel, erasure, general | **the generalized area theorem and some of its consequences**. the generalized area theorem and some of its consequences. There is a fundamental relationship between belief propagation and maximum a posteriori decoding. The case of transmission over the binary erasure channel was investigated in detail in a companion p... |

Preserved data edges corresponding to query edges:

- `(2949640503,2950598277)`
- `(2949640503,2125566184)`
- `(2950598277,2950093758)`
- `(2125566184,2951382382)`

## Additional vertex-disjoint match

Minimum vertex similarity: `0.8249`; sum similarity: `4.4078`

| Query vertex | Data vertex | Metadata | Similarity | Query text | Original title/text |
|---|---|---|---:|---|---|
| `q1` | `2409452000` | label=28 | 0.9356 | paper about decoding, sc-ldpc, code, ensembles, binary, both, codes, complexity, design, latency, over, thresholds | **continuous transmission of spatially coupled ldpc code chains**. continuous transmission of spatially coupled ldpc code chains. We propose a novel encoding/transmission scheme called continuous chain (CC) transmission that is able to improve the finite-length performance of a system using spatially coupled low-density pa... |
| `q2` | `1706324650` | label=28 | 0.8290 | paper about threshold, channels, over, transmission, also, binary, class, consider, coupling, ensemble, ensembles, memory | **triggering wave like convergence of tail biting spatially coupled ldpc codes single and dual channel setup**. triggering wave like convergence of tail biting spatially coupled ldpc codes single and dual channel setup. Spatially coupled low-density parity-check (SC-LDPC) codes can achieve the channel capacity under low-complexity belief propagation (BP) decoding. Fo... |
| `q3` | `1864365761` | label=28 | 0.8921 | paper about codes, analysis, channels, ldpc, binary, messages, approximation, arbitrary, coset, designed, gaussian, including | **spatially coupled ldpc codes constructed from protographs**. spatially coupled ldpc codes constructed from protographs. In this paper, we construct protograph-based spatially coupled low-density parity-check (LDPC) codes by coupling together a series of $L$ disjoint, or uncoupled, LDPC code Tanner graphs into a singl... |
| `q4` | `2788036939` | label=28 | 0.9263 | paper about codes, ldpc, hybrid, condition, stability, advantages, analyse, analysis, asymptotic, asymptotically, been, behavior | **design of irregular sc ldpc codes with non uniform degree distributions by linear programing**. design of irregular sc ldpc codes with non uniform degree distributions by linear programing. In this paper, we propose a new design method of irregular spatially-coupled low-density paritycheck (SC-LDPC) codes with non-uniform degree distributions by linea... |
| `q5` | `2951577762` | label=28 | 0.8249 | paper about over, posteriori, belief, binary, case, maximum, propagation, transmission, area, channel, erasure, general | **threshold saturation on bms channels via spatial coupling**. threshold saturation on bms channels via spatial coupling. We consider spatially coupled code ensembles. A particular instance are convolutional LDPC ensembles. It was recently shown that, for transmission over the binary erasure channel, this coupling incr... |

Preserved data edges corresponding to query edges:

- `(2409452000,1706324650)`
- `(2409452000,1864365761)`
- `(1706324650,2951577762)`
- `(1864365761,2788036939)`

## Disjointness check

Ground-truth image set:

`2125566184, 2949640503, 2950093758, 2950598277, 2951382382`

Additional-match image set:

`1706324650, 1864365761, 2409452000, 2788036939, 2951577762`

Intersection:

``
