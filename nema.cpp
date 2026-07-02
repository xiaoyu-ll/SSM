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
    return std::chrono::duration<double>(clock::now() - t0).count();
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

static inline float dot_product(const vector<float>& a, const vector<float>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += double(a[i]) * double(b[i]);
    return float(s);
}

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

static bool nonzero_feature(const vector<float>& x, double eps = 1e-12) {
    double s = 0.0;
    for (float v : x) s += double(v) * double(v);
    return s > eps * eps;
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

static Graph read_graph_csv(const string& vertices_path, const string& edges_path, bool normalize = true, bool undirected = true) {
    Graph G = read_vertices_csv(vertices_path);
    read_edges_csv_into(G, edges_path, undirected, true);
    if (normalize) normalize_features(G);
    return G;
}

static bool file_exists(const string& path) { std::ifstream fin(path); return bool(fin); }

static void ensure_parent_dir(const string& path) {
    auto pos = path.find_last_of("/\\");
    if (pos == string::npos) return;
    string dir = path.substr(0, pos);
    if (dir.empty()) return;
    string cmd = "mkdir -p \"" + dir + "\"";
    std::system(cmd.c_str());
}

static void write_vertices_csv(const Graph& Q, const string& path) {
    ensure_parent_dir(path);
    std::ofstream fout(path, std::ios::out | std::ios::binary);
    if (!fout) throw std::runtime_error("Cannot write query vertices csv: " + path);
    fout << "id";
    for (int j = 0; j < Q.dim; ++j) fout << ",f" << j;
    fout << "\n" << std::setprecision(9);
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

static Graph sample_query_from_graph(const Graph& G, int k, uint64_t seed, vector<int>& gt_mapping, int max_tries = 100) {
    if (k <= 0) throw std::runtime_error("num_query_nodes must be positive.");
    vector<int> valid;
    for (int i = 0; i < G.n; ++i) if (nonzero_feature(G.x[i])) valid.push_back(i);
    if (int(valid.size()) < k) throw std::runtime_error("Not enough non-zero-feature nodes to sample query.");

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, valid.size() - 1);
    for (int attempt = 0; attempt < max_tries; ++attempt) {
        int start = valid[pick(rng)];
        vector<int> chosen;
        vector<char> seen(G.n, 0);
        vector<int> q;
        seen[start] = 1;
        q.push_back(start);
        for (size_t head = 0; head < q.size() && int(chosen.size()) < k; ++head) {
            int u = q[head];
            if (!nonzero_feature(G.x[u])) continue;
            chosen.push_back(u);
            vector<int> nbrs = G.adj[u];
            std::shuffle(nbrs.begin(), nbrs.end(), rng);
            for (int v : nbrs) if (!seen[v] && nonzero_feature(G.x[v])) { seen[v] = 1; q.push_back(v); }
        }
        if (int(chosen.size()) < k) continue;
        chosen.resize(k);
        std::unordered_map<int, int> old_to_new;
        for (int i = 0; i < k; ++i) old_to_new[chosen[i]] = i;

        Graph Q;
        Q.n = k;
        Q.dim = G.dim;
        Q.x.assign(k, vector<float>(G.dim, 0.0f));
        Q.adj.assign(k, {});
        Q.deg.assign(k, 0);
        for (int i = 0; i < k; ++i) Q.x[i] = G.x[chosen[i]];
        for (int i = 0; i < k; ++i) {
            for (int ov : G.adj[chosen[i]]) {
                auto it = old_to_new.find(ov);
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
    string graph_vertices, graph_edges, query_vertices, query_edges, query_prefix;
    string output = "matches_output";
    string scores_output;
    float tau = 0.97f;
    float cand_tau = -1.0f;
    int num_query_nodes = 6;
    uint64_t seed = 42;
    int topk = 100;
    int candidate_limit = 200;
    int beam_width = 1000;
    int nema_iters = 4;
    int hops = 2;
    float alpha = 0.5f;
    float lambda = 0.4f;
    bool injective = true;
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
        else if (k == "--output") a.output = need(k);
        else if (k == "--scores-output") a.scores_output = need(k);
        else if (k == "--tau") a.tau = std::stof(need(k));
        else if (k == "--cand-tau") a.cand_tau = std::stof(need(k));
        else if (k == "--num-query-nodes") a.num_query_nodes = std::stoi(need(k));
        else if (k == "--seed") a.seed = std::stoull(need(k));
        else if (k == "--topk") a.topk = std::stoi(need(k));
        else if (k == "--max-matches") a.topk = std::stoi(need(k));
        else if (k == "--candidate-limit") a.candidate_limit = std::stoi(need(k));
        else if (k == "--beam-width") a.beam_width = std::stoi(need(k));
        else if (k == "--nema-iters") a.nema_iters = std::stoi(need(k));
        else if (k == "--hops") a.hops = std::stoi(need(k));
        else if (k == "--alpha") a.alpha = std::stof(need(k));
        else if (k == "--lambda") a.lambda = std::stof(need(k));
        else if (k == "--injective") a.injective = true;
        else if (k == "--non-injective") a.injective = false;
        else if (k == "--no-normalize") a.no_normalize = true;
        else if (k == "--count-only") a.count_only = true;
        else if (k == "--require-query") a.require_query = true;
        else if (k == "--help") {
            cout << "Usage:\n"
                 << "  ./nema_vec --graph-vertices graph_vertices.csv --graph-edges graph_edges.csv \\\n"
                 << "      [--query-vertices q_vertices.csv --query-edges q_edges.csv | --num-query-nodes 6] \\\n"
                 << "      --tau 0.9 --topk 100 --output matches_output\n\n"
                 << "Options:\n"
                 << "  --cand-tau X             Candidate threshold; default equals --tau.\n"
                 << "  --candidate-limit N      Max candidates per query node.\n"
                 << "  --beam-width N           Beam width for top-k refinement.\n"
                 << "  --nema-iters N           Iterative inference rounds.\n"
                 << "  --hops N                 NeMa h-hop neighborhood radius.\n"
                 << "  --alpha X                NeMa propagation factor.\n"
                 << "  --lambda X               Label/proximity tradeoff.\n"
                 << "  --injective / --non-injective\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown argument: " + k);
    }
    if (a.graph_vertices.empty() || a.graph_edges.empty()) throw std::runtime_error("Missing graph csv paths. Use --help.");
    if ((a.query_vertices.empty()) != (a.query_edges.empty())) throw std::runtime_error("Provide both --query-vertices and --query-edges, or neither.");
    if (a.require_query && (a.query_vertices.empty() || a.query_edges.empty())) throw std::runtime_error("--require-query was set, but query CSV paths are missing.");
    if (a.cand_tau < 0) a.cand_tau = a.tau;
    if (a.topk <= 0) throw std::runtime_error("--topk must be positive.");
    if (a.candidate_limit <= 0) throw std::runtime_error("--candidate-limit must be positive.");
    if (a.beam_width <= 0) throw std::runtime_error("--beam-width must be positive.");
    if (a.nema_iters < 0) throw std::runtime_error("--nema-iters must be non-negative.");
    if (a.hops <= 0) throw std::runtime_error("--hops must be positive.");
    return a;
}

static bool is_dir_like(const string& p) {
    return !p.empty() && (p.back() == '/' || p.back() == '\\' || p.find('.') == string::npos);
}
static string output_match_path(const string& output) {
    if (output.empty()) return "nema_matches.txt";
    if (is_dir_like(output)) return output + (output.back() == '/' || output.back() == '\\' ? "" : "/") + "nema_matches.txt";
    return output;
}
static string output_score_path(const string& output, const string& scores_output) {
    if (!scores_output.empty()) return scores_output;
    if (output.empty()) return "nema_scores.csv";
    if (is_dir_like(output)) return output + (output.back() == '/' || output.back() == '\\' ? "" : "/") + "nema_scores.csv";
    auto pos = output.find_last_of('.');
    if (pos == string::npos) return output + "_scores.csv";
    return output.substr(0, pos) + "_scores.csv";
}

class MatchWriter {
public:
    std::ofstream fout;
    int num_q;
    bool enabled;
    MatchWriter(const string& path, int n, bool enabled_) : num_q(n), enabled(enabled_) {
        if (enabled) {
            ensure_parent_dir(path);
            fout.open(path, std::ios::out | std::ios::binary);
            if (!fout) throw std::runtime_error("Cannot open output path: " + path);
        }
    }
    void write(const vector<int>& mapping) {
        if (!enabled) return;
        for (int i = 0; i < num_q; ++i) {
            if (i) fout << ' ';
            fout << mapping[i];
        }
        fout << '\n';
    }
};

struct Result {
    vector<int> mapping;
    double total_cost = 0.0;
    double label_cost = 0.0;
    double neighborhood_cost = 0.0;
    bool exact_valid = false;
    bool injective_valid = false;
    bool edge_valid = false;
};

class NeMaVec {
public:
    const Graph& G;
    const Graph& Q;
    Args args;
    int qn, gn;
    vector<vector<int>> cand;
    vector<vector<double>> label_cost;
    vector<vector<double>> U_prev, U_cur;
    vector<vector<int>> q_dist;
    vector<vector<double>> q_prox;
    vector<double> beta;
    uint64_t proximity_queries = 0;
    uint64_t beam_expansions = 0;

    NeMaVec(const Graph& data, const Graph& query, const Args& a) : G(data), Q(query), args(a), qn(query.n), gn(data.n) {
        if (G.dim != Q.dim) throw std::runtime_error("Feature dimensions do not match.");
        build_query_proximity();
        build_candidates();
        run_inference();
    }

    void build_query_proximity() {
        q_dist.assign(qn, vector<int>(qn, -1));
        q_prox.assign(qn, vector<double>(qn, 0.0));
        beta.assign(qn, 1.0);
        for (int s = 0; s < qn; ++s) {
            std::queue<int> qu;
            q_dist[s][s] = 0;
            qu.push(s);
            while (!qu.empty()) {
                int u = qu.front(); qu.pop();
                if (q_dist[s][u] >= args.hops) continue;
                for (int v : Q.adj[u]) if (q_dist[s][v] == -1) {
                    q_dist[s][v] = q_dist[s][u] + 1;
                    qu.push(v);
                }
            }
            double denom = 0.0;
            for (int t = 0; t < qn; ++t) {
                if (t == s) continue;
                if (q_dist[s][t] > 0 && q_dist[s][t] <= args.hops) {
                    q_prox[s][t] = std::pow(double(args.alpha), q_dist[s][t]);
                    denom += q_prox[s][t];
                }
            }
            beta[s] = denom > 0 ? 1.0 / denom : 1.0;
        }
    }

    void build_candidates() {
        cand.assign(qn, {});
        label_cost.assign(qn, {});
        for (int u = 0; u < qn; ++u) {
            vector<std::pair<double, int>> tmp;
            tmp.reserve(gn);
            for (int v = 0; v < gn; ++v) {
                double sim = dot_product(Q.x[u], G.x[v]);
                if (sim >= double(args.cand_tau)) tmp.emplace_back(1.0 - sim, v);
            }
            if (tmp.empty()) {
                // Approximate fallback: retain top candidates even if none pass cand_tau.
                for (int v = 0; v < gn; ++v) {
                    double sim = dot_product(Q.x[u], G.x[v]);
                    tmp.emplace_back(1.0 - sim, v);
                }
            }
            std::sort(tmp.begin(), tmp.end(), [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
            });
            if (int(tmp.size()) > args.candidate_limit) tmp.resize(args.candidate_limit);
            cand[u].reserve(tmp.size());
            label_cost[u].reserve(tmp.size());
            for (auto& p : tmp) {
                label_cost[u].push_back(std::max(0.0, p.first));
                cand[u].push_back(p.second);
            }
            if (cand[u].empty()) throw std::runtime_error("Candidate set unexpectedly empty.");
        }
    }

    double data_prox(int v, int z) {
        if (v == z) return 1.0;
        proximity_queries++;
        vector<int> dist(gn, -1);
        std::queue<int> qu;
        dist[v] = 0;
        qu.push(v);
        while (!qu.empty()) {
            int x = qu.front(); qu.pop();
            if (dist[x] >= args.hops) continue;
            for (int y : G.adj[x]) {
                if (dist[y] != -1) continue;
                dist[y] = dist[x] + 1;
                if (y == z) return std::pow(double(args.alpha), dist[y]);
                qu.push(y);
            }
        }
        return 0.0;
    }

    void run_inference() {
        U_prev = label_cost;
        U_cur = U_prev;
        for (int it = 0; it < args.nema_iters; ++it) {
            for (int u = 0; u < qn; ++u) {
                for (int i = 0; i < int(cand[u].size()); ++i) {
                    int v = cand[u][i];
                    double total = args.lambda * label_cost[u][i];
                    double neigh = 0.0;
                    for (int w = 0; w < qn; ++w) {
                        if (w == u || q_prox[u][w] <= 0) continue;
                        double best = std::numeric_limits<double>::infinity();
                        for (int j = 0; j < int(cand[w].size()); ++j) {
                            int z = cand[w][j];
                            double pg = data_prox(v, z);
                            double penalty = beta[u] * std::max(0.0, q_prox[u][w] - pg);
                            double val = (1.0 - args.lambda) * penalty + U_prev[w][j];
                            if (val < best) best = val;
                        }
                        if (std::isfinite(best)) neigh += best;
                    }
                    U_cur[u][i] = total + neigh;
                }
            }
            U_prev.swap(U_cur);
        }
    }

    int choose_anchor() const {
        int best = 0;
        for (int u = 1; u < qn; ++u) {
            auto ku = std::make_tuple(int(cand[u].size()), -Q.deg[u], u);
            auto kb = std::make_tuple(int(cand[best].size()), -Q.deg[best], best);
            if (ku < kb) best = u;
        }
        return best;
    }

    vector<int> query_order(int anchor) const {
        vector<int> order;
        order.reserve(qn);
        vector<char> seen(qn, 0);
        std::queue<int> qu;
        seen[anchor] = 1;
        qu.push(anchor);
        while (!qu.empty()) {
            int u = qu.front(); qu.pop();
            order.push_back(u);
            vector<int> nbrs = Q.adj[u];
            std::sort(nbrs.begin(), nbrs.end(), [&](int a, int b) {
                return std::make_tuple(int(cand[a].size()), -Q.deg[a], a) < std::make_tuple(int(cand[b].size()), -Q.deg[b], b);
            });
            for (int v : nbrs) if (!seen[v]) { seen[v] = 1; qu.push(v); }
        }
        for (int u = 0; u < qn; ++u) if (!seen[u]) order.push_back(u);
        return order;
    }

    int cand_pos(int u, int v) const {
        for (int i = 0; i < int(cand[u].size()); ++i) if (cand[u][i] == v) return i;
        return -1;
    }

    double incremental_cost(int u, int v, const vector<int>& mapping) {
        int pos = cand_pos(u, v);
        double lc = (pos >= 0 ? label_cost[u][pos] : 1.0 - dot_product(Q.x[u], G.x[v]));
        double cost = args.lambda * lc;
        for (int w = 0; w < qn; ++w) {
            if (mapping[w] == -1 || q_prox[u][w] <= 0) continue;
            double pg = data_prox(v, mapping[w]);
            double penalty = beta[u] * std::max(0.0, q_prox[u][w] - pg);
            cost += (1.0 - args.lambda) * penalty;
        }
        return cost;
    }

    bool used_contains(const vector<int>& used, int v) const {
        return std::find(used.begin(), used.end(), v) != used.end();
    }

    struct Partial {
        vector<int> mapping;
        vector<int> used;
        double cost = 0.0;
    };

    Result score_result(const vector<int>& mapping) {
        Result r;
        r.mapping = mapping;
        r.injective_valid = true;
        std::unordered_set<int> used;
        for (int v : mapping) {
            if (v < 0) { r.injective_valid = false; continue; }
            if (!used.insert(v).second) r.injective_valid = false;
        }
        r.edge_valid = true;
        for (int u = 0; u < qn; ++u) {
            for (int w : Q.adj[u]) if (u < w) {
                if (mapping[u] < 0 || mapping[w] < 0 || !G.has_edge(mapping[u], mapping[w])) r.edge_valid = false;
            }
        }
        bool sem_ok = true;
        double lc_sum = 0.0, nc_sum = 0.0;
        for (int u = 0; u < qn; ++u) {
            double sim = mapping[u] >= 0 ? dot_product(Q.x[u], G.x[mapping[u]]) : -1.0;
            if (sim < args.tau) sem_ok = false;
            lc_sum += std::max(0.0, 1.0 - sim);
            double denom = 0.0, acc = 0.0;
            for (int w = 0; w < qn; ++w) {
                if (w == u || q_prox[u][w] <= 0 || mapping[w] < 0) continue;
                denom += q_prox[u][w];
                double pg = data_prox(mapping[u], mapping[w]);
                acc += std::max(0.0, q_prox[u][w] - pg);
            }
            if (denom > 0) nc_sum += acc / denom;
        }
        r.label_cost = lc_sum;
        r.neighborhood_cost = nc_sum;
        r.total_cost = args.lambda * lc_sum + (1.0 - args.lambda) * nc_sum;
        r.exact_valid = sem_ok && r.edge_valid && r.injective_valid;
        return r;
    }

    vector<Result> run() {
        int anchor = choose_anchor();
        vector<int> order = query_order(anchor);
        vector<Partial> beam;
        vector<int> anchor_indices(cand[anchor].size());
        std::iota(anchor_indices.begin(), anchor_indices.end(), 0);
        std::sort(anchor_indices.begin(), anchor_indices.end(), [&](int a, int b) {
            return U_prev[anchor][a] < U_prev[anchor][b];
        });
        int seed_take = std::min<int>(args.beam_width, anchor_indices.size());
        for (int t = 0; t < seed_take; ++t) {
            int idx = anchor_indices[t];
            Partial p;
            p.mapping.assign(qn, -1);
            p.mapping[anchor] = cand[anchor][idx];
            p.used.push_back(cand[anchor][idx]);
            p.cost = U_prev[anchor][idx];
            beam.push_back(std::move(p));
        }

        for (int step = 1; step < int(order.size()); ++step) {
            int u = order[step];
            vector<Partial> next;
            next.reserve(size_t(args.beam_width) * 4);
            vector<int> cidx(cand[u].size());
            std::iota(cidx.begin(), cidx.end(), 0);
            std::sort(cidx.begin(), cidx.end(), [&](int a, int b) { return U_prev[u][a] < U_prev[u][b]; });
            int local_take = std::min<int>(int(cidx.size()), std::max(args.topk, std::min(args.candidate_limit, 200)));
            for (const Partial& p : beam) {
                for (int ii = 0; ii < local_take; ++ii) {
                    int idx = cidx[ii];
                    int v = cand[u][idx];
                    if (args.injective && used_contains(p.used, v)) continue;
                    Partial np = p;
                    np.mapping[u] = v;
                    np.used.push_back(v);
                    np.cost += incremental_cost(u, v, p.mapping);
                    next.push_back(std::move(np));
                    beam_expansions++;
                }
            }
            if (next.empty()) break;
            std::sort(next.begin(), next.end(), [](const Partial& a, const Partial& b) { return a.cost < b.cost; });
            if (int(next.size()) > args.beam_width) next.resize(args.beam_width);
            beam.swap(next);
        }

        vector<Result> results;
        results.reserve(beam.size());
        std::unordered_set<string> seen;
        for (auto& p : beam) {
            bool complete = true;
            for (int v : p.mapping) if (v < 0) complete = false;
            if (!complete) continue;
            string key;
            for (int v : p.mapping) { key += std::to_string(v); key.push_back(','); }
            if (!seen.insert(key).second) continue;
            Result r = score_result(p.mapping);
            r.total_cost = p.cost; // ranking cost from beam/inference.
            results.push_back(std::move(r));
        }
        std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) {
            if (a.total_cost != b.total_cost) return a.total_cost < b.total_cost;
            return a.mapping < b.mapping;
        });
        if (int(results.size()) > args.topk) results.resize(args.topk);
        return results;
    }

    void print_profile() const {
        cout << "\n========== NeMa-VEC Profile ==========" << "\n";
        cout << "query_nodes          : " << qn << "\n";
        cout << "data_nodes           : " << gn << "\n";
        cout << "topk                 : " << args.topk << "\n";
        cout << "candidate_limit      : " << args.candidate_limit << "\n";
        cout << "beam_width           : " << args.beam_width << "\n";
        cout << "nema_iters           : " << args.nema_iters << "\n";
        cout << "hops                 : " << args.hops << "\n";
        cout << "alpha                : " << args.alpha << "\n";
        cout << "lambda               : " << args.lambda << "\n";
        cout << "injective            : " << (args.injective ? 1 : 0) << "\n";
        cout << "proximity_queries    : " << proximity_queries << "\n";
        cout << "beam_expansions      : " << beam_expansions << "\n";
        double avg_c = 0.0;
        for (const auto& c : cand) avg_c += c.size();
        cout << "avg_candidates       : " << (avg_c / std::max(1, qn)) << "\n";
    }
};

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        double t0 = now_sec();
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
            string prefix = args.query_prefix.empty() ? default_query_prefix(args.graph_vertices, args.num_query_nodes, args.seed) : args.query_prefix;
            string qv_path = prefix + "_vertices.csv";
            string qe_path = prefix + "_edges.csv";
            bool has_v = file_exists(qv_path), has_e = file_exists(qe_path);
            if (has_v != has_e) throw std::runtime_error("Only one query CSV exists for prefix " + prefix + ". Expected both " + qv_path + " and " + qe_path);
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
                cout << "Use the following arguments for all algorithms:\n  --query-vertices " << qv_path << " --query-edges " << qe_path << "\n";
            }
        }

        double t1 = now_sec();
        print_memory("after query preparation");

        cout << "Initializing NeMa-VEC ...\n";
        NeMaVec matcher(G, Q, args);
        double t2 = now_sec();
        print_memory("after matcher initialization and inference");

        cout << "Running NeMa-VEC top-k refinement ...\n";
        vector<Result> results = matcher.run();
        double t3 = now_sec();

        string match_path = output_match_path(args.output);
        string score_path = output_score_path(args.output, args.scores_output);
        MatchWriter writer(match_path, Q.n, !args.count_only);
        for (const auto& r : results) writer.write(r.mapping);

        ensure_parent_dir(score_path);
        std::ofstream sf(score_path, std::ios::out | std::ios::binary);
        if (!sf) throw std::runtime_error("Cannot open scores output: " + score_path);
        sf << "rank,total_cost,label_cost,neighborhood_cost,exact_valid,injective_valid,edge_valid";
        for (int i = 0; i < Q.n; ++i) sf << ",q" << i;
        sf << "\n" << std::setprecision(10);
        for (int i = 0; i < int(results.size()); ++i) {
            const auto& r = results[i];
            sf << (i + 1) << ',' << r.total_cost << ',' << r.label_cost << ',' << r.neighborhood_cost
               << ',' << (r.exact_valid ? 1 : 0) << ',' << (r.injective_valid ? 1 : 0) << ',' << (r.edge_valid ? 1 : 0);
            for (int v : r.mapping) sf << ',' << v;
            sf << "\n";
        }

        print_memory("after matching");
        matcher.print_profile();
        uint64_t exact_hits = 0;
        for (const auto& r : results) if (r.exact_valid) exact_hits++;
        cout << "\nReturned " << results.size() << " approximate match(es).\n";
        cout << "Exact-valid hits under your definition: " << exact_hits << "\n";
        if (args.count_only) cout << "Output mode: count-only; no match file written.\n";
        else cout << "Matches written to: " << match_path << "\n";
        cout << "Scores written to: " << score_path << "\n";

        cout << "\n========== Runtime Summary ==========" << "\n";
        cout << "CSV loading + query : " << (t1 - t0) << " s\n";
        cout << "Inference init      : " << (t2 - t1) << " s\n";
        cout << "Top-k + output      : " << (t3 - t2) << " s\n";
        cout << "Total runtime       : " << (t3 - t0) << " s\n";
        cout << "Final current RSS   : " << current_rss_mb() << " MB\n";
        cout << "Final peak RSS      : " << peak_rss_mb() << " MB\n";
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}
