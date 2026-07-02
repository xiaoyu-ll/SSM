# Real text-attributed graph case study

## Dataset and graph construction

Dataset: `ogbn-arxiv (49993 text-bearing nodes)`

Number of data vertices used for matching: `49993`

Number of undirected data edges used for matching: `99053`

Graph structure: real input edges, not text-kNN edges.

Text-to-vector encoder: `tfidf-svd`; vector dimension: `128`

Query vector mode: `source`; therefore the source/ground-truth vertex similarities are exactly 1 up to numerical precision.

Semantic threshold: `0.8600`

Total exact matches enumerated before stopping/cap: `85`

Enumeration capped: `True`

Ground-truth recovered: `True`

Additional match disjoint from ground truth with min similarity >= `0.8600`: `True`

## Query sampled from the real graph

Query edge count: `5`

Source pairwise average similarity: `0.8668`

Source centroid minimum similarity: `0.9142`


| Query vertex | Ground-truth data vertex | Metadata | Candidate count | Query text | Source title/text |
|---|---|---|---:|---|---|
| `q1` | `2950391633` | label=28 | 6 | paper about gdof, interference, bounds, upper, regime, cooperation, gaussian, ld-irc, link, management, perspective, schemes | **cooperation for interference management a gdof perspective**. cooperation for interference management a gdof perspective. The impact of cooperation on interference management is investigated by studying an elemental wireless network, the so called symmetric interference relay channel (IRC), from a generalized degrees... |
| `q2` | `2953064494` | label=28 | 13 | paper about interference, channel, cognitive, capacity, relay, result, strong, very, both, bound, classical, messages | **the capacity of the interference channel with a cognitive relay in very strong interference**. the capacity of the interference channel with a cognitive relay in very strong interference. The interference channel with a cognitive relay consists of a classical interference channel with two sourcedestination pairs and with an additional cognitive relay... |
| `q3` | `2949179446` | label=28 | 11 | paper about channel, interference, bound, deterministic, outer, capacity, cognitive, relay, channels, special, approximation, certain | **outer bounds for the interference channel with a cognitive relay**. outer bounds for the interference channel with a cognitive relay. In this paper, we first present an outer bound for a general interference channel with a cognitive relay, i.e., a relay that has non-causal knowledge of both independent messages transmitted... |
| `q4` | `2172248015` | label=28 | 5 | paper about capacity, interference, relay, bounds, class, gaussian, bound, channels, coding, constant, cooperative, inner | **capacity bounds for a class of interference relay channels**. capacity bounds for a class of interference relay channels. The capacity of a class of interference relay channels (IRCs)—the injective semideterministic IRC where the relay can only observe one of the sources—is investigated. We first derive a novel outer... |
| `q5` | `2952175429` | label=28 | 13 | paper about channel, capacity, channels, cognitive, inner, interference, outer, relay, both, bounds, ifc-cr, messages | **on the capacity of the interference channel with a cognitive relay**. on the capacity of the interference channel with a cognitive relay. The InterFerence Channel with a Cognitive Relay (IFC-CR) consists of the classical interference channel with two independent source-destination pairs whose communication is aided by an addi... |

Query edges with their ground-truth data edges:

- `(q1,q2)` -> `(2950391633,2953064494)`
- `(q1,q3)` -> `(2950391633,2949179446)`
- `(q2,q3)` -> `(2953064494,2949179446)`
- `(q3,q4)` -> `(2949179446,2172248015)`
- `(q3,q5)` -> `(2949179446,2952175429)`

## Recovered ground-truth match

Minimum vertex similarity: `1.0000`; sum similarity: `5.0000`

| Query vertex | Data vertex | Metadata | Similarity | Query text | Original title/text |
|---|---|---|---:|---|---|
| `q1` | `2950391633` | label=28 | 1.0000 | paper about gdof, interference, bounds, upper, regime, cooperation, gaussian, ld-irc, link, management, perspective, schemes | **cooperation for interference management a gdof perspective**. cooperation for interference management a gdof perspective. The impact of cooperation on interference management is investigated by studying an elemental wireless network, the so called symmetric interference relay channel (IRC), from a generalized degrees... |
| `q2` | `2953064494` | label=28 | 1.0000 | paper about interference, channel, cognitive, capacity, relay, result, strong, very, both, bound, classical, messages | **the capacity of the interference channel with a cognitive relay in very strong interference**. the capacity of the interference channel with a cognitive relay in very strong interference. The interference channel with a cognitive relay consists of a classical interference channel with two sourcedestination pairs and with an additional cognitive relay... |
| `q3` | `2949179446` | label=28 | 1.0000 | paper about channel, interference, bound, deterministic, outer, capacity, cognitive, relay, channels, special, approximation, certain | **outer bounds for the interference channel with a cognitive relay**. outer bounds for the interference channel with a cognitive relay. In this paper, we first present an outer bound for a general interference channel with a cognitive relay, i.e., a relay that has non-causal knowledge of both independent messages transmitted... |
| `q4` | `2172248015` | label=28 | 1.0000 | paper about capacity, interference, relay, bounds, class, gaussian, bound, channels, coding, constant, cooperative, inner | **capacity bounds for a class of interference relay channels**. capacity bounds for a class of interference relay channels. The capacity of a class of interference relay channels (IRCs)—the injective semideterministic IRC where the relay can only observe one of the sources—is investigated. We first derive a novel outer... |
| `q5` | `2952175429` | label=28 | 1.0000 | paper about channel, capacity, channels, cognitive, inner, interference, outer, relay, both, bounds, ifc-cr, messages | **on the capacity of the interference channel with a cognitive relay**. on the capacity of the interference channel with a cognitive relay. The InterFerence Channel with a Cognitive Relay (IFC-CR) consists of the classical interference channel with two independent source-destination pairs whose communication is aided by an addi... |

Preserved data edges corresponding to query edges:

- `(2950391633,2953064494)`
- `(2950391633,2949179446)`
- `(2953064494,2949179446)`
- `(2949179446,2172248015)`
- `(2949179446,2952175429)`

## Additional vertex-disjoint match

Minimum vertex similarity: `0.8666`; sum similarity: `4.3989`

| Query vertex | Data vertex | Metadata | Similarity | Query text | Original title/text |
|---|---|---|---:|---|---|
| `q1` | `2949588692` | label=28 | 0.8666 | paper about gdof, interference, bounds, upper, regime, cooperation, gaussian, ld-irc, link, management, perspective, schemes | **sum capacity of the gaussian interference channel in the low interference regime**. sum capacity of the gaussian interference channel in the low interference regime. New upper bounds on the sum capacity of the two-user Gaussian interference channel are derived. Using these bounds, it is shown that treating interference as noise achieves th... |
| `q2` | `2125095995` | label=28 | 0.8713 | paper about interference, channel, cognitive, capacity, relay, result, strong, very, both, bound, classical, messages | **on the sum capacity of a class of cyclically symmetric deterministic interference channels**. on the sum capacity of a class of cyclically symmetric deterministic interference channels. Certain deterministic interference channels have been shown to accurately model Gaussian interference channels in the asymptotic low-noise regime. Motivated by this... |
| `q3` | `2952937400` | label=28 | 0.8802 | paper about channel, interference, bound, deterministic, outer, capacity, cognitive, relay, channels, special, approximation, certain | **gaussian interference channel capacity to within one bit**. gaussian interference channel capacity to within one bit. The capacity of the two-user Gaussian interference channel has been open for thirty years. The understanding on this problem has been limited. The best known achievable region is due to Han-Kobayashi... |
| `q4` | `2950657450` | label=28 | 0.8991 | paper about capacity, interference, relay, bounds, class, gaussian, bound, channels, coding, constant, cooperative, inner | **lattice coding and the generalized degrees of freedom of the interference channel with relay**. lattice coding and the generalized degrees of freedom of the interference channel with relay. The generalized degrees of freedom (GDoF) of the symmetric two-user Gaussian interference relay channel (IRC) is studied. While it is known that the relay does not... |
| `q5` | `1846289915` | label=28 | 0.8816 | paper about channel, capacity, channels, cognitive, inner, interference, outer, relay, both, bounds, ifc-cr, messages | **on the symmetric gaussian interference channel with partial unidirectional cooperation**. on the symmetric gaussian interference channel with partial unidirectional cooperation. A two-user symmetric Gaussian Interference Channel (IC) is considered in which a noiseless unidirectional link connects one encoder to the other. Having a constant capac... |

Preserved data edges corresponding to query edges:

- `(2949588692,2125095995)`
- `(2949588692,2952937400)`
- `(2125095995,2952937400)`
- `(2952937400,2950657450)`
- `(2952937400,1846289915)`

## Disjointness check

Ground-truth image set:

`2172248015, 2949179446, 2950391633, 2952175429, 2953064494`

Additional-match image set:

`1846289915, 2125095995, 2949588692, 2950657450, 2952937400`

Intersection:

``
