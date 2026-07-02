#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
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
        if (u < 0 || v < 0 || u >= n || v >= n) return false;
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

static bool file_exists(const string& path) {
    std::ifstream fin(path);
    return bool(fin);
}

static void ensure_parent_dir(const string& path) {
    auto pos = path.find_last_of("/\\");
    if (pos == string::npos) return;
    string dir = path.substr(0, pos);
    if (dir.empty()) return;
    string cmd = "mkdir -p \"" + dir + "\"";
    std::system(cmd.c_str());
}

static bool is_directory_path(const string& path) {
    if (path.empty()) return false;
    if (path.back() == '/' || path.back() == '\\') return true;
    std::ifstream fin(path);
    if (fin.good()) return false;
    // If no extension and user likely passed an output directory, treat as directory.
    size_t slash = path.find_last_of("/\\");
    size_t dot = path.find_last_of('.');
    return dot == string::npos || (slash != string::npos && dot < slash);
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

static inline float dot_product(const vector<float>& a, const vector<float>& b) {
    double s = 0.0;
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) s += double(a[i]) * double(b[i]);
    return float(s);
}

static bool nonzero_feature(const vector<float>& x, double eps = 1e-12) {
    double s = 0.0;
    for (float v : x) s += double(v) * double(v);
    return s > eps * eps;
}

static Graph sample_query_from_graph(const Graph& G, int k, uint64_t seed,
                                     vector<int>& gt_mapping, int max_tries = 100) {
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
        std::unordered_map<int, int> old_to_new;
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

struct Args {
    string graph_vertices;
    string graph_edges;
    string query_vertices;
    string query_edges;
    string query_prefix;
    string output = "matches_output";
    string scores_output;
    float tau = 0.97f;
    int num_query_nodes = 6;
    uint64_t seed = 42;
    uint64_t topk = 100;
    int hop_t = 2;
    int miss_edges = 0;
    int beam_width = 10000;
    bool no_normalize = false;
    bool count_only = false;
    bool require_query = false;
    bool soft_semantic = false;
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
        else if (k == "--tau") a.tau = std::stof(need(k));
        else if (k == "--num-query-nodes") a.num_query_nodes = std::stoi(need(k));
        else if (k == "--seed") a.seed = std::stoull(need(k));
        else if (k == "--topk" || k == "--max-matches") a.topk = std::stoull(need(k));
        else if (k == "--hop-t") a.hop_t = std::stoi(need(k));
        else if (k == "--miss-edges") a.miss_edges = std::stoi(need(k));
        else if (k == "--beam-width") a.beam_width = std::stoi(need(k));
        else if (k == "--output") a.output = need(k);
        else if (k == "--scores-output") a.scores_output = need(k);
        else if (k == "--no-normalize") a.no_normalize = true;
        else if (k == "--count-only") a.count_only = true;
        else if (k == "--require-query") a.require_query = true;
        else if (k == "--soft-semantic") a.soft_semantic = true;
        else if (k == "--hard-semantic") a.soft_semantic = false;
        else if (k == "--help") {
            cout << "Usage:\n"
                 << "  ./gfinder --graph-vertices graph_vertices.csv --graph-edges graph_edges.csv \\\n"
                 << "      [--query-vertices q_vertices.csv --query-edges q_edges.csv] \\\n"
                 << "      [--num-query-nodes 6 --seed 42] --tau 0.9 --output matches_output\n\n"
                 << "Options:\n"
                 << "  --topk N                 Number of approximate mappings to output. Default 100.\n"
                 << "  --hop-t T                Neighbor expansion radius, G-FINDER-style. Default 2.\n"
                 << "  --miss-edges D           Missing mapped-neighbor edges tolerated during candidate generation. Default 0.\n"
                 << "  --beam-width B           Max partial states kept at each depth. Default 10000.\n"
                 << "  --soft-semantic          Allow sim < tau with score penalty. Default uses hard semantic threshold.\n"
                 << "  --count-only             Count only; do not write matches.\n"
                 << "  --require-query          Error if query CSV paths are missing.\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown argument: " + k);
    }
    if (a.graph_vertices.empty() || a.graph_edges.empty()) throw std::runtime_error("Missing graph csv paths. Use --help.");
    if ((a.query_vertices.empty()) != (a.query_edges.empty())) throw std::runtime_error("Provide both --query-vertices and --query-edges, or neither.");
    if (a.require_query && (a.query_vertices.empty() || a.query_edges.empty())) throw std::runtime_error("--require-query was set, but query CSV paths are missing.");
    if (a.num_query_nodes <= 0) throw std::runtime_error("--num-query-nodes must be positive.");
    if (a.hop_t < 1) throw std::runtime_error("--hop-t must be at least 1.");
    if (a.miss_edges < 0) throw std::runtime_error("--miss-edges must be non-negative.");
    if (a.beam_width <= 0) throw std::runtime_error("--beam-width must be positive.");
    return a;
}

static string output_matches_path(const string& output) {
    if (output.empty()) return "gfinder_matches.txt";
    if (is_directory_path(output)) {
        string dir = output;
        while (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) dir.pop_back();
        return dir + "/gfinder_matches.txt";
    }
    return output;
}

static string default_scores_path(const string& output, const string& explicit_path) {
    if (!explicit_path.empty()) return explicit_path;
    if (output.empty()) return "gfinder_scores.csv";
    if (is_directory_path(output)) {
        string dir = output;
        while (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) dir.pop_back();
        return dir + "/gfinder_scores.csv";
    }
    auto pos = output.find_last_of("/\\");
    string dir = pos == string::npos ? "" : output.substr(0, pos + 1);
    return dir + "gfinder_scores.csv";
}

class MatchWriter {
public:
    std::ofstream fout;
    int num_q;
    bool enabled;
    vector<string> buffer;
    size_t flush_lines;
    MatchWriter(const string& path, int n, bool enabled_, size_t flush_lines_ = 65536)
        : num_q(n), enabled(enabled_), flush_lines(flush_lines_) {
        if (enabled) {
            ensure_parent_dir(path);
            fout.open(path, std::ios::out | std::ios::binary);
            if (!fout) throw std::runtime_error("Cannot open output path: " + path);
            buffer.reserve(flush_lines);
        }
    }
    void write(const vector<int>& mapping) {
        if (!enabled) return;
        string line;
        line.reserve(size_t(num_q) * 12);
        for (int i = 0; i < num_q; ++i) {
            if (i) line.push_back(' ');
            line += std::to_string(mapping[i]);
        }
        line.push_back('\n');
        buffer.push_back(std::move(line));
        if (buffer.size() >= flush_lines) flush();
    }
    void flush() {
        if (!enabled || buffer.empty()) return;
        for (const auto& s : buffer) fout.write(s.data(), std::streamsize(s.size()));
        buffer.clear();
    }
    ~MatchWriter() { try { flush(); } catch (...) {} }
};

struct GFinderResult {
    vector<int> mapping;
    double loss = 0.0;
    double score = 0.0;
    int missing_nodes = 0;
    int missing_edges = 0;
    int intermediate_vertices = 0;
    bool exact_valid = false;
};

struct PartialState {
    vector<int> mapping;
    vector<int> used;
    double loss = 0.0;
    double score = 0.0;
    int missing_edges = 0;
    int intermediate_vertices = 0;
};

class GFinderVec {
public:
    const Graph& G;
    const Graph& Q;
    float tau;
    int hop_t;
    int miss_edges;
    uint64_t topk;
    int beam_width;
    bool soft_semantic;

    uint64_t candidate_expansions = 0;
    uint64_t generated_states = 0;
    uint64_t exact_valid_count = 0;
    uint64_t semantic_checks = 0;

    GFinderVec(const Graph& data, const Graph& query, float tau_, int hop_t_, int miss_edges_,
               uint64_t topk_, int beam_width_, bool soft_semantic_)
        : G(data), Q(query), tau(tau_), hop_t(hop_t_), miss_edges(miss_edges_), topk(topk_),
          beam_width(beam_width_), soft_semantic(soft_semantic_) {
        if (G.dim != Q.dim) throw std::runtime_error("Feature dimensions do not match.");
    }

    bool used_contains(const vector<int>& used, int v) const {
        for (int z : used) if (z == v) return true;
        return false;
    }

    int choose_root() const {
        int best = 0;
        double best_key0 = std::numeric_limits<double>::infinity();
        for (int u = 0; u < Q.n; ++u) {
            int cnt = 0;
            // Sample/scan all vertices for root select. This is intentionally simple and reproducible.
            for (int v = 0; v < G.n; ++v) {
                if (G.deg[v] < Q.deg[u]) continue;
                if (dot_product(Q.x[u], G.x[v]) >= tau) cnt++;
            }
            double key0 = double(cnt) / double(std::max(1, Q.deg[u]));
            if (key0 < best_key0 || (key0 == best_key0 && Q.deg[u] > Q.deg[best])) {
                best_key0 = key0;
                best = u;
            }
        }
        cout << "[G-FINDER Root] chosen q" << best << ", deg=" << Q.deg[best] << "\n";
        return best;
    }

    vector<int> query_order_from_root(int root) const {
        vector<int> order;
        order.reserve(Q.n);
        vector<uint8_t> seen(Q.n, 0);
        std::queue<int> q;
        seen[root] = 1;
        q.push(root);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            order.push_back(u);
            vector<int> nbrs = Q.adj[u];
            std::sort(nbrs.begin(), nbrs.end(), [&](int a, int b) {
                if (Q.deg[a] != Q.deg[b]) return Q.deg[a] > Q.deg[b];
                return a < b;
            });
            for (int w : nbrs) if (!seen[w]) { seen[w] = 1; q.push(w); }
        }
        for (int u = 0; u < Q.n; ++u) if (!seen[u]) order.push_back(u);
        return order;
    }

    vector<int> hop_expand_candidates(int src, int attr_q, vector<int>* dist_out = nullptr) {
        vector<int> out;
        vector<int> dist(G.n, -1);
        std::queue<int> q;
        dist[src] = 0;
        q.push(src);
        while (!q.empty()) {
            int v = q.front(); q.pop();
            if (dist[v] > 0) out.push_back(v);
            if (dist[v] >= hop_t) continue;
            for (int z : G.adj[v]) if (dist[z] == -1) {
                dist[z] = dist[v] + 1;
                q.push(z);
            }
        }
        if (dist_out) *dist_out = std::move(dist);
        (void)attr_q;
        return out;
    }

    int shortest_path_limited(int s, int t, int max_depth) const {
        if (s == t) return 0;
        if (G.has_edge(s, t)) return 1;
        vector<int> dist(G.n, -1);
        std::queue<int> q;
        dist[s] = 0;
        q.push(s);
        while (!q.empty()) {
            int v = q.front(); q.pop();
            if (dist[v] >= max_depth) continue;
            for (int z : G.adj[v]) {
                if (dist[z] != -1) continue;
                dist[z] = dist[v] + 1;
                if (z == t) return dist[z];
                q.push(z);
            }
        }
        return -1;
    }

    double node_vertex_similarity(int u, int v, const vector<int>& mapping) const {
        int denom = std::max(1, Q.deg[u]);
        int connected = 0;
        for (int w : Q.adj[u]) {
            int z = mapping[w];
            if (z != -1 && G.has_edge(v, z)) connected++;
        }
        return double(connected) / double(denom);
    }

    struct CandInfo {
        int v;
        double sim;
        int missing_edges;
        int intermediate_vertices;
        double node_sim;
        double loss_delta;
        double score_delta;
    };

    vector<CandInfo> build_candidates(int u, const PartialState& st) {
        candidate_expansions++;
        std::unordered_map<int, int> best_dist;
        best_dist.reserve(256);
        bool has_parent = false;
        for (int p : Q.adj[u]) {
            int gp = st.mapping[p];
            if (gp == -1) continue;
            has_parent = true;
            vector<int> dist;
            vector<int> exp = hop_expand_candidates(gp, u, &dist);
            for (int v : exp) {
                auto it = best_dist.find(v);
                if (it == best_dist.end() || dist[v] < it->second) best_dist[v] = dist[v];
            }
        }
        if (!has_parent) {
            for (int v = 0; v < G.n; ++v) best_dist[v] = 0;
        }

        vector<CandInfo> cands;
        cands.reserve(std::min<size_t>(best_dist.size(), 1024));
        for (const auto& kv : best_dist) {
            int v = kv.first;
            if (used_contains(st.used, v)) continue;
            if (G.deg[v] < std::max(0, Q.deg[u] - miss_edges)) continue;
            semantic_checks++;
            double sim = dot_product(Q.x[u], G.x[v]);
            if (!soft_semantic && sim < tau) continue;
            int me = 0, inter = 0, hard_connected = 0;
            for (int p : Q.adj[u]) {
                int gp = st.mapping[p];
                if (gp == -1) continue;
                if (G.has_edge(v, gp)) { hard_connected++; continue; }
                int sp = shortest_path_limited(v, gp, hop_t);
                if (sp >= 2) inter += sp - 1;
                else me++;
            }
            if (me > miss_edges) continue;
            double nvsim = node_vertex_similarity(u, v, st.mapping);
            double sem_penalty = sim >= tau ? 0.0 : (tau - sim);
            double loss_delta = double(me) + double(inter) + sem_penalty * 2.0;
            double score_delta = 2.0 * sim + 1.5 * nvsim - 1.0 * double(me) - 0.4 * double(inter);
            cands.push_back({v, sim, me, inter, nvsim, loss_delta, score_delta});
        }
        std::sort(cands.begin(), cands.end(), [](const CandInfo& a, const CandInfo& b) {
            return std::make_tuple(a.loss_delta, -a.score_delta, -a.sim, a.v) <
                   std::make_tuple(b.loss_delta, -b.score_delta, -b.sim, b.v);
        });
        const size_t local_cap = 512;
        if (cands.size() > local_cap) cands.resize(local_cap);
        return cands;
    }

    bool exact_valid(const vector<int>& mapping) const {
        vector<uint8_t> seen(G.n, 0);
        for (int u = 0; u < Q.n; ++u) {
            int v = mapping[u];
            if (v < 0 || v >= G.n) return false;
            if (seen[v]) return false;
            seen[v] = 1;
            if (G.deg[v] < Q.deg[u]) return false;
            if (dot_product(Q.x[u], G.x[v]) < tau) return false;
        }
        for (int u = 0; u < Q.n; ++u) {
            for (int w : Q.adj[u]) if (u < w) {
                if (!G.has_edge(mapping[u], mapping[w])) return false;
            }
        }
        return true;
    }

    vector<GFinderResult> run() {
        const int root = choose_root();
        vector<int> order = query_order_from_root(root);
        vector<CandInfo> root_cands;
        root_cands.reserve(G.n);
        for (int v = 0; v < G.n; ++v) {
            if (G.deg[v] < Q.deg[root]) continue;
            double sim = dot_product(Q.x[root], G.x[v]);
            if (!soft_semantic && sim < tau) continue;
            double sem_penalty = sim >= tau ? 0.0 : (tau - sim);
            root_cands.push_back({v, sim, 0, 0, 0.0, sem_penalty * 2.0, 2.0 * sim});
        }
        std::sort(root_cands.begin(), root_cands.end(), [](const CandInfo& a, const CandInfo& b) {
            return std::make_tuple(a.loss_delta, -a.sim, a.v) < std::make_tuple(b.loss_delta, -b.sim, b.v);
        });
        if (root_cands.size() > size_t(std::max<uint64_t>(topk * 20, 1000))) {
            root_cands.resize(size_t(std::max<uint64_t>(topk * 20, 1000)));
        }

        vector<PartialState> beam;
        beam.reserve(root_cands.size());
        for (const auto& c : root_cands) {
            PartialState st;
            st.mapping.assign(Q.n, -1);
            st.mapping[root] = c.v;
            st.used.push_back(c.v);
            st.loss = c.loss_delta;
            st.score = c.score_delta;
            beam.push_back(std::move(st));
        }

        for (int idx = 1; idx < int(order.size()); ++idx) {
            int u = order[idx];
            vector<PartialState> next;
            next.reserve(std::min<size_t>(size_t(beam_width) * 4, 100000));
            for (const auto& st : beam) {
                auto cands = build_candidates(u, st);
                if (cands.empty()) {
                    PartialState miss = st;
                    // Approximate mode: allow a missing query node if hard construction cannot find one.
                    miss.loss += 5.0;
                    miss.score -= 5.0;
                    next.push_back(std::move(miss));
                    continue;
                }
                for (const auto& c : cands) {
                    PartialState ns = st;
                    ns.mapping[u] = c.v;
                    ns.used.push_back(c.v);
                    ns.loss += c.loss_delta;
                    ns.score += c.score_delta;
                    ns.missing_edges += c.missing_edges;
                    ns.intermediate_vertices += c.intermediate_vertices;
                    next.push_back(std::move(ns));
                    generated_states++;
                }
            }
            std::sort(next.begin(), next.end(), [](const PartialState& a, const PartialState& b) {
                return std::make_tuple(a.loss, -a.score) < std::make_tuple(b.loss, -b.score);
            });
            if (int(next.size()) > beam_width) next.resize(beam_width);
            beam.swap(next);
            if (beam.empty()) break;
        }

        vector<GFinderResult> results;
        results.reserve(beam.size());
        std::unordered_set<string> seen;
        for (const auto& st : beam) {
            GFinderResult r;
            r.mapping = st.mapping;
            r.loss = st.loss;
            r.score = st.score;
            r.missing_edges = st.missing_edges;
            r.intermediate_vertices = st.intermediate_vertices;
            for (int u = 0; u < Q.n; ++u) if (r.mapping[u] < 0) r.missing_nodes++;
            if (r.missing_nodes == 0) r.exact_valid = exact_valid(r.mapping);
            if (r.exact_valid) exact_valid_count++;
            string key;
            for (int v : r.mapping) { key += std::to_string(v); key.push_back(','); }
            if (seen.insert(key).second) results.push_back(std::move(r));
        }
        std::sort(results.begin(), results.end(), [](const GFinderResult& a, const GFinderResult& b) {
            return std::make_tuple(a.loss, -a.score) < std::make_tuple(b.loss, -b.score);
        });
        if (results.size() > topk) results.resize(size_t(topk));
        return results;
    }

    void print_profile() const {
        cout << "\n========== G-FINDER-VEC Profile ==========" << "\n";
        cout << "candidate_expansions      : " << candidate_expansions << "\n";
        cout << "generated_states          : " << generated_states << "\n";
        cout << "semantic_checks           : " << semantic_checks << "\n";
        cout << "exact_valid_in_beam       : " << exact_valid_count << "\n";
        cout << "hop_t                     : " << hop_t << "\n";
        cout << "miss_edges                : " << miss_edges << "\n";
        cout << "beam_width                : " << beam_width << "\n";
        cout << "soft_semantic             : " << (soft_semantic ? 1 : 0) << "\n";
    }
};

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
                cout << "Sampling query from data graph ...\n";
                Q = sample_query_from_graph(G, args.num_query_nodes, args.seed, gt_mapping);
                cout << "Query graph: n=" << Q.n << " dim=" << Q.dim << "\n";
                cout << "Ground-truth mapping (query node -> data node):";
                for (int i = 0; i < int(gt_mapping.size()); ++i) cout << " q" << i << "->g" << gt_mapping[i];
                cout << "\n";
                write_query_csv(Q, qv_path, qe_path);
                cout << "Sampled query saved to:\n  " << qv_path << "\n  " << qe_path << "\n";
                cout << "Use the following arguments for all algorithms:\n  --query-vertices "
                     << qv_path << " --query-edges " << qe_path << "\n";
            }
        }
        const double t1 = now_sec();
        print_memory("after query preparation");

        cout << "Running G-FINDER-VEC approximate baseline ...\n";
        GFinderVec finder(G, Q, args.tau, args.hop_t, args.miss_edges, args.topk,
                          args.beam_width, args.soft_semantic);
        vector<GFinderResult> results = finder.run();

        const string matches_path = output_matches_path(args.output);
        const string scores_path = default_scores_path(args.output, args.scores_output);
        MatchWriter writer(matches_path, Q.n, !args.count_only);
        for (const auto& r : results) {
            if (r.missing_nodes == 0) writer.write(r.mapping);
        }
        writer.flush();

        if (!args.count_only) {
            ensure_parent_dir(scores_path);
            std::ofstream sf(scores_path, std::ios::out | std::ios::binary);
            if (!sf) throw std::runtime_error("Cannot open scores output path: " + scores_path);
            sf << "rank,loss,score,missing_nodes,missing_edges,intermediate_vertices,exact_valid\n";
            for (size_t i = 0; i < results.size(); ++i) {
                const auto& r = results[i];
                sf << (i + 1) << ',' << r.loss << ',' << r.score << ',' << r.missing_nodes << ','
                   << r.missing_edges << ',' << r.intermediate_vertices << ',' << (r.exact_valid ? 1 : 0) << "\n";
            }
        }

        const double t2 = now_sec();
        print_memory("after matching");
        finder.print_profile();

        uint64_t exact_topk = 0;
        for (const auto& r : results) if (r.exact_valid) exact_topk++;
        cout << "\nReturned " << results.size() << " approximate result(s).\n";
        cout << "Top-k exact-valid result(s): " << exact_topk << "\n";
        if (args.count_only) cout << "Output mode: count-only; no match file written.\n";
        else {
            cout << "Matches written to: " << matches_path << "\n";
            cout << "Scores written to : " << scores_path << "\n";
        }
        cout << "\n========== Runtime Summary ==========" << "\n";
        cout << "CSV loading + query : " << (t1 - t0) << " s\n";
        cout << "Matching + output   : " << (t2 - t1) << " s\n";
        cout << "Total runtime       : " << (t2 - t0) << " s\n";
        cout << "Final current RSS   : " << current_rss_mb() << " MB\n";
        cout << "Final peak RSS      : " << peak_rss_mb() << " MB\n";
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}
