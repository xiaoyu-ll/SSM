#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <sys/resource.h>
#include <sys/stat.h>

#ifdef __APPLE__
#include <mach/mach.h>
#endif
#ifdef __linux__
#include <unistd.h>
#endif

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::unordered_map;
using std::unordered_set;

static inline double now_sec() {
    using clock = std::chrono::high_resolution_clock;
    static const auto t0 = clock::now();
    auto t = clock::now();
    return std::chrono::duration<double>(t - t0).count();
}

static double current_rss_mb() {
#ifdef __APPLE__
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) return -1.0;
    return double(info.resident_size) / 1024.0 / 1024.0;
#elif defined(__linux__)
    std::ifstream fin("/proc/self/statm");
    long pages_total = 0, pages_resident = 0;
    fin >> pages_total >> pages_resident;
    long page_size = sysconf(_SC_PAGESIZE);
    return double(pages_resident) * double(page_size) / 1024.0 / 1024.0;
#else
    return -1.0;
#endif
}

static double peak_rss_mb() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return -1.0;
#ifdef __APPLE__
    return double(usage.ru_maxrss) / 1024.0 / 1024.0;
#else
    return double(usage.ru_maxrss) / 1024.0;
#endif
}

static void print_memory(const string& tag) {
    cout << "[Memory] " << tag << ": current RSS=" << current_rss_mb()
         << " MB, peak RSS=" << peak_rss_mb() << " MB\n";
}

static vector<string> split_csv_line(const string& line) {
    vector<string> out;
    string cur;
    cur.reserve(32);
    for (char c : line) {
        if (c == ',') { out.push_back(cur); cur.clear(); }
        else if (c != '\r') cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

struct Graph {
    int n = 0;
    int dim = 0;
    vector<vector<float>> x;
    vector<vector<int>> adj;
    vector<int> deg;

    bool has_edge(int u, int v) const {
        const auto& a = adj[u];
        return std::binary_search(a.begin(), a.end(), v);
    }
};

static void normalize_features(Graph& G) {
    for (auto& row : G.x) {
        double s = 0.0;
        for (float z : row) s += double(z) * double(z);
        double norm = std::sqrt(s);
        if (norm <= 1e-30) continue;
        float inv = float(1.0 / norm);
        for (float& z : row) z *= inv;
    }
}

static Graph read_vertices_csv(const string& path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Cannot open vertices csv: " + path);
    string line;
    if (!std::getline(fin, line)) throw std::runtime_error("Empty vertices csv: " + path);

    vector<std::pair<int, vector<float>>> rows;
    int max_id = -1;
    int dim = -1;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto parts = split_csv_line(line);
        if (parts.size() < 2) continue;
        int id = std::stoi(parts[0]);
        vector<float> feat;
        feat.reserve(parts.size() - 1);
        for (size_t i = 1; i < parts.size(); ++i) feat.push_back(parts[i].empty() ? 0.0f : std::stof(parts[i]));
        if (dim < 0) dim = int(feat.size());
        if (int(feat.size()) != dim) throw std::runtime_error("Inconsistent feature dimension in " + path);
        max_id = std::max(max_id, id);
        rows.emplace_back(id, std::move(feat));
    }
    Graph G;
    G.n = max_id + 1;
    G.dim = dim < 0 ? 0 : dim;
    G.x.assign(G.n, vector<float>(G.dim, 0.0f));
    G.adj.assign(G.n, {});
    G.deg.assign(G.n, 0);
    for (auto& p : rows) G.x[p.first] = std::move(p.second);
    return G;
}

static void read_edges_csv_into(Graph& G, const string& path, bool undirected = true, bool skip_self = true) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Cannot open edges csv: " + path);
    string line;
    if (!std::getline(fin, line)) throw std::runtime_error("Empty edges csv: " + path);
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto parts = split_csv_line(line);
        if (parts.size() < 2) continue;
        int u = std::stoi(parts[0]);
        int v = std::stoi(parts[1]);
        if (u < 0 || v < 0 || u >= G.n || v >= G.n) throw std::runtime_error("Edge endpoint out of range in " + path);
        if (skip_self && u == v) continue;
        G.adj[u].push_back(v);
        if (undirected) G.adj[v].push_back(u);
    }
    for (int i = 0; i < G.n; ++i) {
        auto& a = G.adj[i];
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
        G.deg[i] = int(a.size());
    }
}

static Graph read_graph_csv(const string& vertices_path, const string& edges_path,
                            bool normalize = true, bool undirected = true) {
    Graph G = read_vertices_csv(vertices_path);
    read_edges_csv_into(G, edges_path, undirected, true);
    if (normalize) normalize_features(G);
    return G;
}

static inline float dot_product(const vector<float>& a, const vector<float>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += double(a[i]) * double(b[i]);
    return float(s);
}

static void ensure_parent_dir(const string& path) {
    auto pos = path.find_last_of("/\\");
    if (pos == string::npos) return;
    string dir = path.substr(0, pos);
    if (dir.empty()) return;
    string cmd = "mkdir -p \"" + dir + "\"";
    std::system(cmd.c_str());
}

static bool file_exists(const string& path) {
    std::ifstream fin(path);
    return bool(fin);
}

static bool is_directory_path(const string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static string path_basename_no_trailing_slash(string path) {
    while (!path.empty() && (path.back() == '/' || path.back() == '\\')) path.pop_back();
    if (path.empty()) return "dataset";
    size_t pos = path.find_last_of("/\\");
    return (pos == string::npos) ? path : path.substr(pos + 1);
}

static string parent_path(string path) {
    while (!path.empty() && (path.back() == '/' || path.back() == '\\')) path.pop_back();
    size_t pos = path.find_last_of("/\\");
    if (pos == string::npos) return "";
    return path.substr(0, pos);
}

static string dataset_name_from_graph_vertices(const string& graph_vertices_path) {
    string parent = parent_path(graph_vertices_path);
    string name = path_basename_no_trailing_slash(parent);
    return name.empty() ? string("dataset") : name;
}

static string default_query_prefix(const string& graph_vertices_path, int k, uint64_t seed) {
    string dataset = dataset_name_from_graph_vertices(graph_vertices_path);
    return string("query/") + dataset + "/q_k" + std::to_string(k) + "_seed" + std::to_string(seed);
}

static bool nonzero_feature(const vector<float>& x, double eps = 1e-12) {
    double s = 0.0;
    for (float v : x) s += double(v) * double(v);
    return s > eps * eps;
}

static void write_vertices_csv(const Graph& Q, const string& path) {
    ensure_parent_dir(path);
    std::ofstream fout(path, std::ios::out | std::ios::binary);
    if (!fout) throw std::runtime_error("Cannot write query vertices csv: " + path);
    fout << "id";
    for (int j = 0; j < Q.dim; ++j) fout << ",f" << j;
    fout << "\n";
    fout << std::setprecision(9);
    for (int i = 0; i < Q.n; ++i) {
        fout << i;
        for (int j = 0; j < Q.dim; ++j) fout << ',' << Q.x[i][j];
        fout << "\n";
    }
}

static void write_edges_csv(const Graph& Q, const string& path) {
    ensure_parent_dir(path);
    std::ofstream fout(path, std::ios::out | std::ios::binary);
    if (!fout) throw std::runtime_error("Cannot write query edges csv: " + path);
    fout << "src,dst\n";
    for (int u = 0; u < Q.n; ++u) {
        for (int v : Q.adj[u]) if (u < v) fout << u << ',' << v << "\n";
    }
}

static void write_query_csv(const Graph& Q, const string& vertices_path, const string& edges_path) {
    write_vertices_csv(Q, vertices_path);
    write_edges_csv(Q, edges_path);
}

static Graph sample_query_from_graph(const Graph& G, int k, uint64_t seed,
                                     vector<int>& gt_mapping,
                                     int max_tries = 100) {
    if (k <= 0) throw std::runtime_error("num_query_nodes must be positive.");
    vector<int> valid;
    valid.reserve(G.n);
    for (int i = 0; i < G.n; ++i) if (nonzero_feature(G.x[i])) valid.push_back(i);
    if (int(valid.size()) < k) throw std::runtime_error("Not enough non-zero-feature nodes to sample query.");

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, valid.size() - 1);

    for (int attempt = 0; attempt < max_tries; ++attempt) {
        int start = valid[pick(rng)];
        vector<int> chosen;
        chosen.reserve(k);
        vector<char> seen(G.n, 0);
        vector<int> queue;
        queue.reserve(k * 16);
        seen[start] = 1;
        queue.push_back(start);

        for (size_t head = 0; head < queue.size() && int(chosen.size()) < k; ++head) {
            int u = queue[head];
            if (!nonzero_feature(G.x[u])) continue;
            chosen.push_back(u);
            vector<int> nbrs = G.adj[u];
            std::shuffle(nbrs.begin(), nbrs.end(), rng);
            for (int v : nbrs) {
                if (!seen[v] && nonzero_feature(G.x[v])) {
                    seen[v] = 1;
                    queue.push_back(v);
                }
            }
        }
        if (int(chosen.size()) < k) continue;
        chosen.resize(k);

        unordered_map<int, int> old_to_new;
        old_to_new.reserve(k * 2);
        for (int i = 0; i < k; ++i) old_to_new[chosen[i]] = i;

        Graph Q;
        Q.n = k;
        Q.dim = G.dim;
        Q.x.assign(k, vector<float>(G.dim, 0.0f));
        Q.adj.assign(k, {});
        Q.deg.assign(k, 0);
        for (int i = 0; i < k; ++i) Q.x[i] = G.x[chosen[i]];

        for (int i = 0; i < k; ++i) {
            int old_u = chosen[i];
            for (int old_v : G.adj[old_u]) {
                auto it = old_to_new.find(old_v);
                if (it == old_to_new.end()) continue;
                int j = it->second;
                if (i != j) Q.adj[i].push_back(j);
            }
        }
        bool has_edge = false;
        for (int i = 0; i < k; ++i) {
            auto& a = Q.adj[i];
            std::sort(a.begin(), a.end());
            a.erase(std::unique(a.begin(), a.end()), a.end());
            Q.deg[i] = int(a.size());
            if (!a.empty()) has_edge = true;
        }
        if (!has_edge) continue;
        gt_mapping = chosen;
        return Q;
    }
    throw std::runtime_error("Failed to sample a connected query graph.");
}

static string resolve_file_or_dir_output(const string& output, const string& default_filename) {
    if (output.empty()) return default_filename;
    if (output.back() == '/' || output.back() == '\\') return output + default_filename;
    if (is_directory_path(output)) return output + "/" + default_filename;
    return output;
}

struct Args {
    string graph_vertices;
    string graph_edges;
    string query_vertices;
    string query_edges;
    string query_prefix;
    string output = "mage_vec_matches.txt";
    string scores_output;
    int num_query_nodes = 6;
    uint64_t seed = 42;
    float tau = 0.97f;
    uint64_t topk = 100;
    uint64_t seed_limit = 256;
    uint64_t beam_width = 1000;
    int local_topl = 200;
    int rwr_iters = 10;
    float rwr_alpha = 0.85f;
    int linker_depth = 2;
    float w_sem = 1.0f;
    float w_rwr = 1.0f;
    float w_edge = 1.0f;
    bool hard_semantic = true;
    bool require_edges = false;
    bool no_normalize = false;
    bool count_only = false;
    bool require_query = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        string k = argv[i];
        auto need = [&](const string& name) -> string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };
        if (k == "--graph-vertices") a.graph_vertices = need(k);
        else if (k == "--graph-edges") a.graph_edges = need(k);
        else if (k == "--query-vertices") a.query_vertices = need(k);
        else if (k == "--query-edges") a.query_edges = need(k);
        else if (k == "--query-prefix") a.query_prefix = need(k);
        else if (k == "--num-query-nodes") a.num_query_nodes = std::stoi(need(k));
        else if (k == "--seed") a.seed = std::stoull(need(k));
        else if (k == "--output") a.output = need(k);
        else if (k == "--scores-output") a.scores_output = need(k);
        else if (k == "--tau") a.tau = std::stof(need(k));
        else if (k == "--topk") a.topk = std::stoull(need(k));
        else if (k == "--seed-limit") a.seed_limit = std::stoull(need(k));
        else if (k == "--beam-width") a.beam_width = std::stoull(need(k));
        else if (k == "--local-topl") a.local_topl = std::stoi(need(k));
        else if (k == "--rwr-iters") a.rwr_iters = std::stoi(need(k));
        else if (k == "--rwr-alpha") a.rwr_alpha = std::stof(need(k));
        else if (k == "--linker-depth") a.linker_depth = std::stoi(need(k));
        else if (k == "--w-sem") a.w_sem = std::stof(need(k));
        else if (k == "--w-rwr") a.w_rwr = std::stof(need(k));
        else if (k == "--w-edge") a.w_edge = std::stof(need(k));
        else if (k == "--soft-semantic") a.hard_semantic = false;
        else if (k == "--hard-semantic") a.hard_semantic = true;
        else if (k == "--require-edges") a.require_edges = true;
        else if (k == "--no-normalize") a.no_normalize = true;
        else if (k == "--count-only") a.count_only = true;
        else if (k == "--require-query") a.require_query = true;
        else if (k == "--help") {
            cout << "Usage:\n"
                 << "  ./mage_vec --graph-vertices graph_vertices.csv --graph-edges graph_edges.csv \\\n"
                 << "      --query-vertices query_vertices.csv --query-edges query_edges.csv \\\n"
                 << "      --tau 0.97 --topk 100 --beam-width 1000 --output mage_matches.txt\n\n"
                 << "Options:\n"
                 << "  --query-prefix PREFIX      Query file prefix. If omitted, use query/<dataset>/q_k<num>_seed<seed>.\n"
                 << "  --num-query-nodes N        Query size used when sampling query; default 6.\n"
                 << "  --seed N                   Random seed used when sampling query; default 42.\n"
                 << "  --scores-output PATH        Write rank/score/exact-valid diagnostics.\n"
                 << "  --seed-limit N              Number of anchor seeds to expand.\n"
                 << "  --beam-width N              Beam width for partial mappings.\n"
                 << "  --local-topl N              Local candidates retained per expansion.\n"
                 << "  --rwr-iters N               RWR power iterations; default 10.\n"
                 << "  --rwr-alpha X               RWR continuation probability; default 0.85.\n"
                 << "  --linker-depth N            BFS depth for approximate edge linker; default 2.\n"
                 << "  --soft-semantic             Allow sim < tau with lower score.\n"
                 << "  --require-edges             Require query edges during construction.\n"
                 << "  --no-normalize              Use input vectors as already normalized.\n"
                 << "  --count-only                Do not write output files.\n"
                 << "  --require-query             Error if query CSV is not provided or found.\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown argument: " + k);
    }
    if (a.graph_vertices.empty() || a.graph_edges.empty()) {
        throw std::runtime_error("Missing graph csv paths. Use --help.");
    }
    if ((a.query_vertices.empty()) != (a.query_edges.empty())) {
        throw std::runtime_error("Provide both --query-vertices and --query-edges, or neither.");
    }
    if (a.require_query && (a.query_vertices.empty() || a.query_edges.empty())) {
        throw std::runtime_error("--require-query was set, but query CSV paths are missing.");
    }
    if (a.num_query_nodes <= 0) throw std::runtime_error("--num-query-nodes must be positive.");
    if (a.topk == 0) throw std::runtime_error("--topk must be positive.");
    if (a.beam_width == 0) throw std::runtime_error("--beam-width must be positive.");
    if (a.local_topl <= 0) throw std::runtime_error("--local-topl must be positive.");
    if (a.rwr_iters <= 0) throw std::runtime_error("--rwr-iters must be positive.");
    if (a.linker_depth < 0) throw std::runtime_error("--linker-depth must be non-negative.");
    return a;
}

struct Partial {
    vector<int> mapping;
    vector<int> used_nodes;
    double score = 0.0;
    double sem_score = 0.0;
    double prox_score = 0.0;
    double edge_score = 0.0;
    int missing_edges = 0;
};

struct Completed {
    vector<int> mapping;
    double score = 0.0;
    double sem_score = 0.0;
    double prox_score = 0.0;
    double edge_score = 0.0;
    int missing_edges = 0;
    bool exact_valid = false;
};

class MAGEVecMatcher {
public:
    const Graph& G;
    const Graph& Q;
    Args args;
    int qn;
    int gn;
    int center_q = -1;

    unordered_map<int, vector<float>> rwr_cache;
    uint64_t rwr_computes = 0;
    uint64_t sim_evals = 0;
    uint64_t local_search_calls = 0;
    uint64_t partial_expansions = 0;
    uint64_t completed_count = 0;
    uint64_t exact_hit_count = 0;

    MAGEVecMatcher(const Graph& data, const Graph& query, const Args& a)
        : G(data), Q(query), args(a), qn(query.n), gn(data.n) {
        if (G.dim != Q.dim) throw std::runtime_error("Feature dimensions do not match.");
        if (qn <= 0) throw std::runtime_error("Query graph is empty.");
        center_q = detect_candidate();
    }

    float sim_qv(int q, int v) {
        sim_evals++;
        return dot_product(Q.x[q], G.x[v]);
    }

    int detect_candidate() const {
        int best = 0;
        for (int u = 1; u < qn; ++u) {
            auto ku = std::make_tuple(-Q.deg[u], u);
            auto kb = std::make_tuple(-Q.deg[best], best);
            if (ku < kb) best = u;
        }
        cout << "[MAGE-VEC Detect-Candidate] query center = q" << best << ", deg=" << Q.deg[best] << "\n";
        return best;
    }

    bool used_contains(const vector<int>& used, int v) const {
        for (int z : used) if (z == v) return true;
        return false;
    }

    const vector<float>& rwr_from_seed(int seed) {
        auto it = rwr_cache.find(seed);
        if (it != rwr_cache.end()) return it->second;
        rwr_computes++;

        vector<float> r(gn, 0.0f), next(gn, 0.0f);
        r[seed] = 1.0f;
        const float restart = 1.0f - args.rwr_alpha;
        for (int itn = 0; itn < args.rwr_iters; ++itn) {
            std::fill(next.begin(), next.end(), 0.0f);
            next[seed] += restart;
            for (int u = 0; u < gn; ++u) {
                if (r[u] == 0.0f || G.deg[u] == 0) continue;
                float share = args.rwr_alpha * r[u] / float(G.deg[u]);
                for (int nb : G.adj[u]) next[nb] += share;
            }
            double norm1 = 0.0;
            for (float x : next) norm1 += x;
            if (norm1 > 1e-30) {
                float inv = float(1.0 / norm1);
                for (float& x : next) x *= inv;
            }
            r.swap(next);
        }
        auto inserted = rwr_cache.emplace(seed, std::move(r));
        return inserted.first->second;
    }

    int choose_next_query(const Partial& p) const {
        int best = -1;
        auto best_key = std::make_tuple(0, 0, 0, std::numeric_limits<int>::max());
        for (int u = 0; u < qn; ++u) {
            if (p.mapping[u] != -1) continue;
            int boundary = 0;
            int frontier_coupling = 0;
            for (int w : Q.adj[u]) {
                if (p.mapping[w] != -1) boundary++;
                else frontier_coupling++;
            }
            if (boundary == 0) continue;
            auto key = std::make_tuple(boundary, Q.deg[u], frontier_coupling, -u);
            if (best == -1 || key > best_key) {
                best = u;
                best_key = key;
            }
        }
        return best;
    }

    double linker_score_direct_or_short(int a, int b) const {
        if (G.has_edge(a, b)) return 1.0;
        if (args.linker_depth <= 0) return 0.0;
        vector<int> dist(gn, -1);
        std::queue<int> q;
        dist[a] = 0;
        q.push(a);
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            if (dist[u] >= args.linker_depth) continue;
            for (int nb : G.adj[u]) {
                if (dist[nb] != -1) continue;
                dist[nb] = dist[u] + 1;
                if (nb == b) return 1.0 / double(dist[nb]);
                q.push(nb);
            }
        }
        return 0.0;
    }

    struct LocalCand {
        int v;
        double score;
        double sem;
        double prox;
        double edge;
        int missing;
    };

    vector<LocalCand> local_search(int q_u, const Partial& p) {
        local_search_calls++;
        vector<int> mapped_neighbors;
        for (int w : Q.adj[q_u]) if (p.mapping[w] != -1) mapped_neighbors.push_back(w);

        // Cache RWR vectors for already-mapped query neighbors.
        vector<const vector<float>*> rwr_vecs;
        rwr_vecs.reserve(mapped_neighbors.size());
        for (int w : mapped_neighbors) rwr_vecs.push_back(&rwr_from_seed(p.mapping[w]));

        vector<LocalCand> cands;
        cands.reserve(std::min<int>(gn, args.local_topl * 4));
        for (int v = 0; v < gn; ++v) {
            if (used_contains(p.used_nodes, v)) continue;
            if (G.deg[v] < Q.deg[q_u]) continue;
            float sem = sim_qv(q_u, v);
            if (args.hard_semantic && sem < args.tau) continue;

            double prox = 0.0;
            for (const auto* rv : rwr_vecs) prox = std::max(prox, double((*rv)[v]));

            double edge = 0.0;
            int missing = 0;
            bool edge_fail = false;
            for (int w : mapped_neighbors) {
                int gw = p.mapping[w];
                double ls = linker_score_direct_or_short(v, gw);
                edge += ls;
                if (!G.has_edge(v, gw)) {
                    missing++;
                    if (args.require_edges) { edge_fail = true; break; }
                }
            }
            if (edge_fail) continue;

            double s = args.w_sem * double(sem) + args.w_rwr * prox + args.w_edge * edge;
            cands.push_back({v, s, double(sem), prox, edge, missing});
        }

        std::sort(cands.begin(), cands.end(), [](const LocalCand& a, const LocalCand& b) {
            return std::make_tuple(-a.score, a.missing, -a.sem, -a.prox, a.v)
                 < std::make_tuple(-b.score, b.missing, -b.sem, -b.prox, b.v);
        });
        if (int(cands.size()) > args.local_topl) cands.resize(args.local_topl);
        return cands;
    }

    bool exact_valid(const vector<int>& mapping) {
        vector<uint8_t> seen(gn, 0);
        for (int u = 0; u < qn; ++u) {
            int v = mapping[u];
            if (v < 0 || v >= gn) return false;
            if (seen[v]) return false;
            seen[v] = 1;
            if (G.deg[v] < Q.deg[u]) return false;
            if (sim_qv(u, v) < args.tau) return false;
        }
        for (int u = 0; u < qn; ++u) {
            for (int w : Q.adj[u]) {
                if (u < w && !G.has_edge(mapping[u], mapping[w])) return false;
            }
        }
        return true;
    }

    string key_mapping(const vector<int>& m) const {
        string s;
        s.reserve(size_t(qn) * 12);
        for (int i = 0; i < qn; ++i) {
            if (i) s.push_back(',');
            s += std::to_string(m[i]);
        }
        return s;
    }

    vector<int> build_anchor_seeds() {
        vector<LocalCand> seeds;
        seeds.reserve(gn);
        for (int v = 0; v < gn; ++v) {
            if (G.deg[v] < Q.deg[center_q]) continue;
            float sem = sim_qv(center_q, v);
            if (args.hard_semantic && sem < args.tau) continue;
            double score = args.w_sem * double(sem) + 1e-6 * std::log1p(double(G.deg[v]));
            seeds.push_back({v, score, double(sem), 0.0, 0.0, 0});
        }
        std::sort(seeds.begin(), seeds.end(), [](const LocalCand& a, const LocalCand& b) {
            return std::make_tuple(-a.score, -a.sem, a.v) < std::make_tuple(-b.score, -b.sem, b.v);
        });
        if (seeds.size() > args.seed_limit) seeds.resize(size_t(args.seed_limit));
        vector<int> out;
        out.reserve(seeds.size());
        for (const auto& c : seeds) out.push_back(c.v);
        cout << "[MAGE-VEC] anchor seed candidates expanded = " << out.size() << "\n";
        return out;
    }

    vector<Completed> run() {
        vector<int> seeds = build_anchor_seeds();
        vector<Partial> beam;
        beam.reserve(std::min<uint64_t>(args.beam_width, seeds.size() + 1));
        for (int v : seeds) {
            float sem = sim_qv(center_q, v);
            Partial p;
            p.mapping.assign(qn, -1);
            p.mapping[center_q] = v;
            p.used_nodes.push_back(v);
            p.sem_score = sem;
            p.score = args.w_sem * double(sem);
            beam.push_back(std::move(p));
        }

        vector<Completed> completed;
        unordered_set<string> completed_seen;

        for (int depth = 1; depth < qn && !beam.empty(); ++depth) {
            vector<Partial> next_beam;
            next_beam.reserve(std::min<uint64_t>(args.beam_width * 2, args.beam_width + 100));
            for (const Partial& p : beam) {
                int q_u = choose_next_query(p);
                if (q_u < 0) continue;
                auto cands = local_search(q_u, p);
                for (const auto& c : cands) {
                    partial_expansions++;
                    Partial np = p;
                    np.mapping[q_u] = c.v;
                    np.used_nodes.push_back(c.v);
                    np.sem_score += c.sem;
                    np.prox_score += c.prox;
                    np.edge_score += c.edge;
                    np.missing_edges += c.missing;
                    np.score += c.score;
                    next_beam.push_back(std::move(np));
                }
            }
            std::sort(next_beam.begin(), next_beam.end(), [](const Partial& a, const Partial& b) {
                return std::make_tuple(-a.score, a.missing_edges, -a.sem_score) <
                       std::make_tuple(-b.score, b.missing_edges, -b.sem_score);
            });
            if (next_beam.size() > args.beam_width) next_beam.resize(size_t(args.beam_width));
            beam.swap(next_beam);
            cout << "[MAGE-VEC] depth " << depth << " beam size = " << beam.size() << "\n";
        }

        for (const Partial& p : beam) {
            bool complete = true;
            for (int u = 0; u < qn; ++u) if (p.mapping[u] == -1) { complete = false; break; }
            if (!complete) continue;
            string key = key_mapping(p.mapping);
            if (!completed_seen.insert(key).second) continue;
            Completed c;
            c.mapping = p.mapping;
            c.score = p.score;
            c.sem_score = p.sem_score;
            c.prox_score = p.prox_score;
            c.edge_score = p.edge_score;
            c.missing_edges = p.missing_edges;
            c.exact_valid = exact_valid(c.mapping);
            completed_count++;
            if (c.exact_valid) exact_hit_count++;
            completed.push_back(std::move(c));
        }

        std::sort(completed.begin(), completed.end(), [](const Completed& a, const Completed& b) {
            return std::make_tuple(!a.exact_valid, -a.score, a.missing_edges, -a.sem_score) <
                   std::make_tuple(!b.exact_valid, -b.score, b.missing_edges, -b.sem_score);
        });
        if (completed.size() > args.topk) completed.resize(size_t(args.topk));
        return completed;
    }

    void print_profile() const {
        cout << "\n========== MAGE-VEC Profile ==========\n";
        cout << "query_center               : q" << center_q << "\n";
        cout << "rwr_computes               : " << rwr_computes << "\n";
        cout << "rwr_cache_size             : " << rwr_cache.size() << "\n";
        cout << "sim_evals                  : " << sim_evals << "\n";
        cout << "local_search_calls         : " << local_search_calls << "\n";
        cout << "partial_expansions         : " << partial_expansions << "\n";
        cout << "completed_unique           : " << completed_count << "\n";
        cout << "exact_hits_before_topk_cut : " << exact_hit_count << "\n";
        cout << "hard_semantic              : " << (args.hard_semantic ? "true" : "false") << "\n";
        cout << "require_edges              : " << (args.require_edges ? "true" : "false") << "\n";
        cout << "rwr_iters                  : " << args.rwr_iters << "\n";
        cout << "rwr_alpha                  : " << args.rwr_alpha << "\n";
        cout << "beam_width                 : " << args.beam_width << "\n";
        cout << "local_topl                 : " << args.local_topl << "\n";
    }
};

static void write_outputs(const string& match_path, const string& score_path,
                          const vector<Completed>& results, int qn) {
    ensure_parent_dir(match_path);
    ensure_parent_dir(score_path);
    std::ofstream fm(match_path, std::ios::out | std::ios::binary);
    std::ofstream fs(score_path, std::ios::out | std::ios::binary);
    if (!fm) throw std::runtime_error("Cannot open output: " + match_path);
    if (!fs) throw std::runtime_error("Cannot open scores output: " + score_path);
    fs << "rank,total_score,semantic_score,proximity_score,edge_score,missing_edges,exact_valid\n";
    fs << std::setprecision(9);
    for (size_t r = 0; r < results.size(); ++r) {
        const auto& c = results[r];
        for (int i = 0; i < qn; ++i) {
            if (i) fm << ' ';
            fm << c.mapping[i];
        }
        fm << '\n';
        fs << (r + 1) << ',' << c.score << ',' << c.sem_score << ',' << c.prox_score << ','
           << c.edge_score << ',' << c.missing_edges << ',' << (c.exact_valid ? 1 : 0) << '\n';
    }
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        const double t0 = now_sec();
        print_memory("program start");

        cout << "Loading graph CSV ...\n";
        Graph G = read_graph_csv(args.graph_vertices, args.graph_edges, !args.no_normalize, true);
        cout << "Data graph: n=" << G.n << " dim=" << G.dim << "\n";
        print_memory("after graph load");

        Graph Q;
        vector<int> gt_mapping;
        if (!args.query_vertices.empty() && !args.query_edges.empty()) {
            cout << "Loading query CSV ...\n";
            Q = read_graph_csv(args.query_vertices, args.query_edges, !args.no_normalize, true);
            cout << "Query graph: n=" << Q.n << " dim=" << Q.dim << "\n";
        } else {
            string prefix = args.query_prefix.empty()
                ? default_query_prefix(args.graph_vertices, args.num_query_nodes, args.seed)
                : args.query_prefix;
            string qv_path = prefix + "_vertices.csv";
            string qe_path = prefix + "_edges.csv";
            bool has_v = file_exists(qv_path);
            bool has_e = file_exists(qe_path);
            if (has_v != has_e) {
                throw std::runtime_error("Only one query CSV exists for prefix " + prefix +
                                         ". Expected both " + qv_path + " and " + qe_path);
            }
            if (has_v && has_e) {
                cout << "Loading sampled query CSV from prefix: " << prefix << "\n";
                Q = read_graph_csv(qv_path, qe_path, !args.no_normalize, true);
                cout << "Query graph: n=" << Q.n << " dim=" << Q.dim << "\n";
            } else {
                if (args.require_query) {
                    throw std::runtime_error("--require-query was set, but query CSV files do not exist for prefix " + prefix);
                }
                cout << "Sampling query from data graph ...\n";
                Q = sample_query_from_graph(G, args.num_query_nodes, args.seed, gt_mapping);
                cout << "Query graph: n=" << Q.n << " dim=" << Q.dim << "\n";
                cout << "Ground-truth mapping (query node -> data node):";
                for (int i = 0; i < int(gt_mapping.size()); ++i) cout << " q" << i << "->g" << gt_mapping[i];
                cout << "\n";
                write_query_csv(Q, qv_path, qe_path);
                cout << "Sampled query saved to:\n"
                     << "  " << qv_path << "\n"
                     << "  " << qe_path << "\n";
                cout << "Use the following arguments for all algorithms:\n"
                     << "  --query-vertices " << qv_path
                     << " --query-edges " << qe_path << "\n";
            }
        }
        print_memory("after query preparation");

        MAGEVecMatcher matcher(G, Q, args);
        cout << "Running MAGE-VEC baseline ...\n";
        vector<Completed> results = matcher.run();
        const string output_path = resolve_file_or_dir_output(args.output, "mage_vec_matches.txt");
        string scores_path;
        if (!args.scores_output.empty()) scores_path = resolve_file_or_dir_output(args.scores_output, "mage_vec_scores.txt");
        else {
            if (args.output.empty() || args.output.back() == '/' || args.output.back() == '\\' || is_directory_path(args.output)) {
                scores_path = resolve_file_or_dir_output(args.output, "mage_vec_scores.txt");
            } else {
                scores_path = output_path + ".scores.csv";
            }
        }
        if (!args.count_only) write_outputs(output_path, scores_path, results, Q.n);
        const double t1 = now_sec();

        matcher.print_profile();
        uint64_t exact_topk = 0;
        for (const auto& c : results) if (c.exact_valid) exact_topk++;
        cout << "\nReturned " << results.size() << " top-k mapping(s).\n";
        cout << "Exact-valid in returned top-k: " << exact_topk << "\n";
        if (args.count_only) {
            cout << "Output mode: count-only; no match files written.\n";
        } else {
            cout << "Matches written to: " << output_path << "\n";
            cout << "Scores written to : " << scores_path << "\n";
        }
        cout << "Total runtime     : " << (t1 - t0) << " s\n";
        cout << "Final current RSS : " << current_rss_mb() << " MB\n";
        cout << "Final peak RSS    : " << peak_rss_mb() << " MB\n";
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}
