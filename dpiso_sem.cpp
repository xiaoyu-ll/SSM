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
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
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
    mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &n) != KERN_SUCCESS) return -1.0;
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
        for (size_t i = 1; i < parts.size(); ++i) {
            feat.push_back(parts[i].empty() ? 0.0f : std::stof(parts[i]));
        }
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

static void read_edges_csv_into(Graph& G, const string& path) {
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
        if (u < 0 || v < 0 || u >= G.n || v >= G.n) {
            throw std::runtime_error("Edge endpoint out of range in " + path);
        }
        if (u == v) continue;
        G.adj[u].push_back(v);
        G.adj[v].push_back(u);
    }
    for (int i = 0; i < G.n; ++i) {
        auto& a = G.adj[i];
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
        G.deg[i] = int(a.size());
    }
}

static Graph read_graph_csv(const string& vertices_path, const string& edges_path, bool do_normalize) {
    Graph G = read_vertices_csv(vertices_path);
    read_edges_csv_into(G, edges_path);
    if (do_normalize) normalize_features(G);
    return G;
}

static bool file_exists(const string& path) {
    std::ifstream fin(path);
    return bool(fin);
}

static string parent_path(string path) {
    while (!path.empty() && (path.back() == '/' || path.back() == '\\')) path.pop_back();
    size_t pos = path.find_last_of("/\\");
    if (pos == string::npos) return "";
    return path.substr(0, pos);
}

static string basename_no_slash(string path) {
    while (!path.empty() && (path.back() == '/' || path.back() == '\\')) path.pop_back();
    size_t pos = path.find_last_of("/\\");
    return pos == string::npos ? path : path.substr(pos + 1);
}

static string default_query_prefix(const string& graph_vertices_path, int k, uint64_t seed) {
    string dataset = basename_no_slash(parent_path(graph_vertices_path));
    if (dataset.empty()) dataset = "dataset";
    return string("query/") + dataset + "/q_k" + std::to_string(k) + "_seed" + std::to_string(seed);
}

static void ensure_parent_dir(const string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == string::npos) return;
    string dir = path.substr(0, pos);
    if (!dir.empty()) std::system(("mkdir -p \"" + dir + "\"").c_str());
}

static void write_query_csv(const Graph& Q, const string& vp, const string& ep) {
    ensure_parent_dir(vp);
    ensure_parent_dir(ep);
    std::ofstream fv(vp);
    if (!fv) throw std::runtime_error("Cannot write query vertices csv: " + vp);
    fv << "id";
    for (int j = 0; j < Q.dim; ++j) fv << ",f" << j;
    fv << "\n" << std::setprecision(9);
    for (int i = 0; i < Q.n; ++i) {
        fv << i;
        for (int j = 0; j < Q.dim; ++j) fv << ',' << Q.x[i][j];
        fv << "\n";
    }
    std::ofstream fe(ep);
    if (!fe) throw std::runtime_error("Cannot write query edges csv: " + ep);
    fe << "src,dst\n";
    for (int u = 0; u < Q.n; ++u) for (int v : Q.adj[u]) if (u < v) fe << u << ',' << v << "\n";
}

static bool nonzero_feature(const vector<float>& x) {
    double s = 0.0;
    for (float z : x) s += double(z) * double(z);
    return s > 1e-24;
}

static Graph sample_query_from_graph(const Graph& G, int k, uint64_t seed, vector<int>& gt, int max_tries = 100) {
    if (k <= 0) throw std::runtime_error("num-query-nodes must be positive.");
    vector<int> valid;
    for (int i = 0; i < G.n; ++i) if (nonzero_feature(G.x[i])) valid.push_back(i);
    if (int(valid.size()) < k) throw std::runtime_error("Not enough vertices with nonzero features.");

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, valid.size() - 1);
    for (int attempt = 0; attempt < max_tries; ++attempt) {
        int start = valid[pick(rng)];
        vector<int> chosen;
        vector<int> queue;
        vector<unsigned char> seen(G.n, 0);
        seen[start] = 1;
        queue.push_back(start);
        for (size_t head = 0; head < queue.size() && int(chosen.size()) < k; ++head) {
            int u = queue[head];
            if (nonzero_feature(G.x[u])) chosen.push_back(u);
            vector<int> nbrs = G.adj[u];
            std::shuffle(nbrs.begin(), nbrs.end(), rng);
            for (int v : nbrs) if (!seen[v] && nonzero_feature(G.x[v])) {
                seen[v] = 1;
                queue.push_back(v);
            }
        }
        if (int(chosen.size()) < k) continue;
        chosen.resize(k);
        std::unordered_map<int,int> idmap;
        for (int i = 0; i < k; ++i) idmap[chosen[i]] = i;
        Graph Q;
        Q.n = k;
        Q.dim = G.dim;
        Q.x.assign(k, vector<float>(G.dim, 0.0f));
        Q.adj.assign(k, {});
        Q.deg.assign(k, 0);
        for (int i = 0; i < k; ++i) Q.x[i] = G.x[chosen[i]];
        bool has_edge = false;
        for (int i = 0; i < k; ++i) {
            for (int old_v : G.adj[chosen[i]]) {
                auto it = idmap.find(old_v);
                if (it != idmap.end() && it->second != i) Q.adj[i].push_back(it->second);
            }
            auto& a = Q.adj[i];
            std::sort(a.begin(), a.end());
            a.erase(std::unique(a.begin(), a.end()), a.end());
            Q.deg[i] = int(a.size());
            if (!a.empty()) has_edge = true;
        }
        if (!has_edge) continue;
        gt = chosen;
        return Q;
    }
    throw std::runtime_error("Failed to sample a connected query graph.");
}

static inline double dot_product(const vector<float>& a, const vector<float>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += double(a[i]) * double(b[i]);
    return s;
}

struct DpisoStats {
    uint64_t complete_matches = 0;
};

class DpisoSemanticBaseline {
public:
    const Graph& G;
    const Graph& Q;
    double tau;
    int n;
    int qn;
    vector<vector<int>> C;
    vector<unsigned char> is_cand;
    DpisoStats stats;

    DpisoSemanticBaseline(const Graph& data, const Graph& query, double tau_)
        : G(data), Q(query), tau(tau_), n(data.n), qn(query.n) {
        if (G.dim != Q.dim) throw std::runtime_error("Feature dimensions do not match.");
        if (qn <= 0) throw std::runtime_error("Query graph is empty.");
        build_domains();
    }

    inline size_t pos(int u, int v) const { return size_t(u) * size_t(n) + size_t(v); }

    void build_domains() {
        // Degree-only structural domains.  This baseline intentionally does NOT
        // materialize semantic-threshold candidates and does NOT use semantic
        // membership during candidate generation. Semantic feasibility is
        // checked only after a structurally feasible partial extension is formed.
        C.assign(qn, {});
        is_cand.assign(size_t(qn) * size_t(n), 0);
        for (int u = 0; u < qn; ++u) {
            for (int v = 0; v < n; ++v) {
                if (G.deg[v] < Q.deg[u]) continue;
                C[u].push_back(v);
                is_cand[pos(u, v)] = 1;
            }
            std::sort(C[u].begin(), C[u].end(), [&](int a, int b) {
                if (G.deg[a] != G.deg[b]) return G.deg[a] < G.deg[b];
                return a < b;
            });
        }
    }

    bool semantic_feasible_pair(int u, int v) {
        if (dot_product(Q.x[u], G.x[v]) < tau) {
            return false;
        }
        return true;
    }

    vector<int> make_candidate_list(int u, const vector<int>& mapping) {
        int best_parent = -1;
        int best_degree = std::numeric_limits<int>::max();
        for (int p : Q.adj[u]) {
            int gp = mapping[p];
            if (gp >= 0 && G.deg[gp] < best_degree) {
                best_parent = p;
                best_degree = G.deg[gp];
            }
        }
        vector<int> out;
        if (best_parent < 0) {
            out = C[u];
        } else {
            int gp = mapping[best_parent];
            out.reserve(std::min<size_t>(G.adj[gp].size(), 64));
            for (int v : G.adj[gp]) if (is_cand[pos(u, v)]) out.push_back(v);
        }
        return out;
    }

    bool feasible(int u, int v, const vector<int>& mapping, const vector<unsigned char>& used) {
        if (used[v]) {
            return false;
        }
        for (int p : Q.adj[u]) {
            int gp = mapping[p];
            if (gp < 0) continue;
            if (!G.has_edge(v, gp)) {
                    return false;
            }
        }
        return true;
    }

    int choose_next_vertex(const vector<int>& mapping, int mapped_count) {
        int best = -1;
        size_t best_size = std::numeric_limits<size_t>::max();
        int best_mapped_neighbors = -1;
        for (int u = 0; u < qn; ++u) {
            if (mapping[u] >= 0) continue;
            int mapped_neighbors = 0;
            for (int p : Q.adj[u]) if (mapping[p] >= 0) mapped_neighbors++;
            if (mapped_count > 0 && mapped_neighbors == 0) continue;
            vector<int> cand = make_candidate_list(u, mapping);
            auto key = std::make_tuple(cand.size(), -mapped_neighbors, -Q.deg[u], u);
            auto best_key = std::make_tuple(best_size, -best_mapped_neighbors,
                                            best < 0 ? 0 : -Q.deg[best],
                                            best < 0 ? qn : best);
            if (best < 0 || key < best_key) {
                best = u;
                best_size = cand.size();
                best_mapped_neighbors = mapped_neighbors;
            }
        }
        if (best >= 0) return best;
        for (int u = 0; u < qn; ++u) if (mapping[u] < 0) return u;
        return -1;
    }

    template <class Emit>
    void dfs(vector<int>& mapping, vector<unsigned char>& used, int mapped_count,
             Emit&& emit, uint64_t max_matches) {
        if (max_matches && stats.complete_matches >= max_matches) return;
        if (mapped_count == qn) {
            stats.complete_matches++;
            emit(mapping);
            return;
        }
        int u = choose_next_vertex(mapping, mapped_count);
        if (u < 0) return;
        vector<int> cand = make_candidate_list(u, mapping);
        if (cand.empty()) {
            return;
        }
        for (int v : cand) {
            if (max_matches && stats.complete_matches >= max_matches) break;
            if (!feasible(u, v, mapping, used)) continue;
            // Semantic threshold is checked after the structural extension is feasible,
            // not during candidate generation.
            if (!semantic_feasible_pair(u, v)) continue;
            mapping[u] = v;
            used[v] = 1;
            dfs(mapping, used, mapped_count + 1, std::forward<Emit>(emit), max_matches);
            used[v] = 0;
            mapping[u] = -1;
        }
    }

    template <class Emit>
    uint64_t run(Emit&& emit, uint64_t max_matches) {
        for (int u = 0; u < qn; ++u) if (C[u].empty()) return 0;
        vector<int> mapping(qn, -1);
        vector<unsigned char> used(n, 0);
        dfs(mapping, used, 0, std::forward<Emit>(emit), max_matches);
        return stats.complete_matches;
    }

};

class MatchWriter {
public:
    std::ofstream fout;
    int qn;
    vector<string> buf;

    MatchWriter(const string& path, int qn_) : qn(qn_) {
        ensure_parent_dir(path);
        fout.open(path, std::ios::out | std::ios::binary);
        if (!fout) throw std::runtime_error("Cannot open output path: " + path);
    }

    void write(const vector<int>& mapping) {
        string line;
        for (int i = 0; i < qn; ++i) {
            if (i) line.push_back(' ');
            line += std::to_string(mapping[i]);
        }
        line.push_back('\n');
        buf.push_back(std::move(line));
        if (buf.size() >= 65536) flush();
    }

    void flush() {
        for (const auto& s : buf) fout << s;
        buf.clear();
    }

    ~MatchWriter() { flush(); }
};

struct Args {
    string graph_vertices;
    string graph_edges;
    string query_vertices;
    string query_edges;
    string query_prefix;
    string output = "matches_dpiso_sem_lateverify.txt";
    double tau = 0.97;
    int num_query_nodes = 6;
    uint64_t seed = 42;
    uint64_t max_matches = 0;
    bool no_normalize = false;
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
        else if (k == "--tau") a.tau = std::stod(need(k));
        else if (k == "--output") a.output = need(k);
        else if (k == "--num-query-nodes") a.num_query_nodes = std::stoi(need(k));
        else if (k == "--seed") a.seed = std::stoull(need(k));
        else if (k == "--max-matches") a.max_matches = std::stoull(need(k));
        else if (k == "--no-normalize") a.no_normalize = true;
        else if (k == "--require-query") a.require_query = true;
        else if (k == "--help") {
            cout << "Usage:\n"
                 << "  ./dpiso_sem_canonical_single --graph-vertices graph_vertices.csv --graph-edges graph_edges.csv \\\n"
                 << "      --query-vertices q_vertices.csv --query-edges q_edges.csv \\\n"
                 << "      --tau 0.97 --output matches_dpiso.txt\n\n"
                 << "This is a standalone DPiso-style partial semantic verification baseline. It always writes matches.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + k);
        }
    }
    if (a.graph_vertices.empty() || a.graph_edges.empty()) throw std::runtime_error("Missing graph csv paths. Use --help.");
    if ((a.query_vertices.empty()) != (a.query_edges.empty())) throw std::runtime_error("Provide both query csv paths, or neither.");
    if (a.require_query && (a.query_vertices.empty() || a.query_edges.empty())) throw std::runtime_error("Query csv paths are required.");
    return a;
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        double t0 = now_sec();
        print_memory("program start");

        cout << "Loading graph CSV ...\n";
        Graph G = read_graph_csv(args.graph_vertices, args.graph_edges, !args.no_normalize);
        cout << "Data graph: n=" << G.n << " dim=" << G.dim << "\n";
        print_memory("after graph load");

        Graph Q;
        vector<int> gt;
        if (!args.query_vertices.empty()) {
            cout << "Loading query CSV ...\n";
            Q = read_graph_csv(args.query_vertices, args.query_edges, !args.no_normalize);
            cout << "Query graph: n=" << Q.n << " dim=" << Q.dim << "\n";
        } else {
            string prefix = args.query_prefix.empty()
                ? default_query_prefix(args.graph_vertices, args.num_query_nodes, args.seed)
                : args.query_prefix;
            string qv = prefix + "_vertices.csv";
            string qe = prefix + "_edges.csv";
            if (file_exists(qv) && file_exists(qe)) {
                cout << "Loading sampled query CSV from prefix: " << prefix << "\n";
                Q = read_graph_csv(qv, qe, !args.no_normalize);
                cout << "Query graph: n=" << Q.n << " dim=" << Q.dim << "\n";
            } else {
                cout << "Sampling query from data graph ...\n";
                Q = sample_query_from_graph(G, args.num_query_nodes, args.seed, gt);
                cout << "Query graph: n=" << Q.n << " dim=" << Q.dim << "\n";
                cout << "Ground-truth mapping (query node -> data node):";
                for (int i = 0; i < int(gt.size()); ++i) cout << " q" << i << "->g" << gt[i];
                cout << "\n";
                write_query_csv(Q, qv, qe);
                cout << "Sampled query saved to:\n  " << qv << "\n  " << qe << "\n";
                cout << "Use: --query-vertices " << qv << " --query-edges " << qe << "\n";
            }
        }
        double t1 = now_sec();
        print_memory("after query preparation");

        cout << "Initializing DPiso partial-verify semantic baseline ...\n";
        DpisoSemanticBaseline matcher(G, Q, args.tau);
        double t2 = now_sec();
        print_memory("after matcher initialization");

        MatchWriter writer(args.output, Q.n);
        auto emit = [&](const vector<int>& mapping) { writer.write(mapping); };

        cout << "Running DPiso partial-verify semantic baseline ...\n";
        uint64_t total = matcher.run(emit, args.max_matches);
        writer.flush();
        double t3 = now_sec();
        print_memory("after matching");

        cout << "\nFound " << total << " match(es).\n";
        cout << "Matches written to: " << args.output << "\n";
        cout << "\n========== Runtime Summary ==========\n";
        cout << "CSV loading + query : " << (t1 - t0) << " s\n";
        cout << "Matcher init        : " << (t2 - t1) << " s\n";
        cout << "Matching + output   : " << (t3 - t2) << " s\n";
        cout << "Total runtime       : " << (t3 - t0) << " s\n";
        cout << "Final current RSS   : " << current_rss_mb() << " MB\n";
        cout << "Final peak RSS      : " << peak_rss_mb() << " MB\n";
        return 0;
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
