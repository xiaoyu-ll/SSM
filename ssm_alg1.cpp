#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <iomanip>
#include <limits>
#include <numeric>
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
    return double(usage.ru_maxrss) / 1024.0 / 1024.0; // bytes on macOS
#else
    return double(usage.ru_maxrss) / 1024.0;          // KiB on Linux
#endif
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

static inline float dot_product(const vector<float>& a, const vector<float>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += double(a[i]) * double(b[i]);
    return float(s);
}

static bool nonzero_feature(const vector<float>& x, double eps=1e-12) {
    double s = 0.0;
    for (float v : x) s += double(v) * double(v);
    return std::sqrt(s) > eps;
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

static Graph read_vertices_csv(const string& path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Cannot open vertices csv: " + path);
    string line;
    if (!std::getline(fin, line)) throw std::runtime_error("Empty vertices csv: " + path);

    vector<std::pair<int, vector<float>>> rows;
    rows.reserve(1024);
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

static void read_edges_csv_into(Graph& G, const string& path, bool undirected=true, bool skip_self=true) {
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

static Graph read_graph_csv(const string& vertices_path, const string& edges_path, bool normalize=true, bool undirected=true) {
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
        for (int v : Q.adj[u]) {
            if (u < v) fout << u << ',' << v << "\n";
        }
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

static Graph sample_query_fast_cpp(const Graph& G, int k, uint64_t seed, int num_hops, int max_tries, vector<int>& gt_mapping) {
    std::mt19937_64 rng(seed);
    vector<int> valid;
    valid.reserve(G.n);
    for (int i = 0; i < G.n; ++i) if (nonzero_feature(G.x[i])) valid.push_back(i);
    if (int(valid.size()) < k) throw std::runtime_error("Not enough non-zero-feature nodes to sample a query graph.");

    std::uniform_int_distribution<size_t> pick_valid(0, valid.size() - 1);
    for (int attempt = 0; attempt < max_tries; ++attempt) {
        int start = valid[pick_valid(rng)];
        unordered_set<int> subset_set;
        vector<int> frontier{start};
        subset_set.insert(start);
        for (int hop = 0; hop < num_hops; ++hop) {
            vector<int> next;
            for (int u : frontier) {
                for (int v : G.adj[u]) if (!subset_set.count(v)) { subset_set.insert(v); next.push_back(v); }
            }
            frontier.swap(next);
            if (frontier.empty()) break;
        }
        vector<int> subset;
        subset.reserve(subset_set.size());
        for (int u : subset_set) if (nonzero_feature(G.x[u])) subset.push_back(u);
        if (int(subset.size()) < k) continue;
        unordered_set<int> subset_filtered(subset.begin(), subset.end());
        if (!subset_filtered.count(start) || G.adj[start].empty()) continue;

        vector<int> chosen;
        chosen.reserve(k);
        unordered_set<int> visited;
        vector<int> q{start};
        visited.insert(start);
        while (!q.empty() && int(chosen.size()) < k) {
            int cur = q.front();
            q.erase(q.begin());
            chosen.push_back(cur);
            vector<int> nbrs;
            for (int nb : G.adj[cur]) if (subset_filtered.count(nb)) nbrs.push_back(nb);
            std::shuffle(nbrs.begin(), nbrs.end(), rng);
            for (int nb : nbrs) if (!visited.count(nb)) { visited.insert(nb); q.push_back(nb); }
        }
        if (int(chosen.size()) < k) continue;
        chosen.resize(k);
        unordered_map<int, int> old_to_new;
        for (int i = 0; i < k; ++i) old_to_new[chosen[i]] = i;
        Graph Q;
        Q.n = k; Q.dim = G.dim;
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


struct Stats {
    uint64_t complete_matches = 0;
};

class BasicBatchBFSMatcher {
public:
    const Graph& G;
    const Graph& Q;
    float tau;
    int num_g, num_q;
    bool enable_anchor_restrictive_pruning;
    int anchor_restrictive_prefilter_k;
    int anchor_restrictive_pilot_sample_size;

    int anchor_q = -1;
    int anchor_restrictive_query = -1;
    vector<int> anchor_candidates;
    vector<int> q_parent;
    vector<vector<int>> q_children;
    vector<int> q_depth;
    unordered_map<uint64_t, float> sim_cache;
    Stats stats;

    BasicBatchBFSMatcher(const Graph& data, const Graph& query, float tau_,
                         bool anchor_restrictive=true, int restrictive_k=1, int pilot_size=32)
        : G(data), Q(query), tau(tau_), num_g(data.n), num_q(query.n),
          enable_anchor_restrictive_pruning(anchor_restrictive),
          anchor_restrictive_prefilter_k(restrictive_k),
          anchor_restrictive_pilot_sample_size(pilot_size) {
        if (G.dim != Q.dim) throw std::runtime_error("Feature dimensions do not match.");
        anchor_q = choose_anchor();
        build_query_bfs_tree(anchor_q);
        build_anchor_candidates();
    }

    uint64_t key(int q, int v) const { return (uint64_t(uint32_t(q)) << 32) | uint32_t(v); }

    float sim(int q, int v) {
        uint64_t k = key(q, v);
        auto it = sim_cache.find(k);
        if (it != sim_cache.end()) return it->second;
        float s = dot_product(Q.x[q], G.x[v]);
        sim_cache[k] = s;
        return s;
    }

    int choose_anchor() const {
        vector<double> sim_sums(num_q, 0.0);
        for (int u = 0; u < num_q; ++u) {
            for (int w = 0; w < num_q; ++w) if (u != w) sim_sums[u] += dot_product(Q.x[u], Q.x[w]);
        }
        int best = 0;
        for (int u = 1; u < num_q; ++u) {
            auto ku = std::make_tuple(sim_sums[u], -Q.deg[u], u);
            auto kb = std::make_tuple(sim_sums[best], -Q.deg[best], best);
            if (ku < kb) best = u;
        }
        return best;
    }

    void build_query_bfs_tree(int root) {
        q_parent.assign(num_q, -1);
        q_depth.assign(num_q, -1);
        q_children.assign(num_q, {});
        vector<int> queue;
        vector<char> seen(num_q, 0);
        q_depth[root] = 0;
        seen[root] = 1;
        queue.push_back(root);
        size_t head = 0;
        while (head < queue.size()) {
            int cur = queue[head++];
            vector<int> nbrs = Q.adj[cur];
            std::sort(nbrs.begin(), nbrs.end(), [&](int a, int b){
                auto ka = std::make_tuple(dot_product(Q.x[cur], Q.x[a]), -Q.deg[a], a);
                auto kb = std::make_tuple(dot_product(Q.x[cur], Q.x[b]), -Q.deg[b], b);
                return ka < kb;
            });
            for (int nb : nbrs) {
                if (seen[nb]) continue;
                seen[nb] = 1;
                q_parent[nb] = cur;
                q_depth[nb] = q_depth[cur] + 1;
                q_children[cur].push_back(nb);
                queue.push_back(nb);
            }
        }
    }

    vector<int> prefilter_anchor_restrictive_neighbors() {
        vector<int> nbrs = Q.adj[anchor_q];
        std::sort(nbrs.begin(), nbrs.end(), [&](int a, int b){
            auto ka = std::make_tuple(dot_product(Q.x[anchor_q], Q.x[a]), -Q.deg[a], a);
            auto kb = std::make_tuple(dot_product(Q.x[anchor_q], Q.x[b]), -Q.deg[b], b);
            return ka < kb;
        });
        if (int(nbrs.size()) > anchor_restrictive_prefilter_k) nbrs.resize(anchor_restrictive_prefilter_k);
        return nbrs;
    }

    void choose_anchor_restrictive_neighbor(const vector<int>& survivors) {
        anchor_restrictive_query = -1;
        auto nbrs = prefilter_anchor_restrictive_neighbors();
        if (nbrs.empty() || survivors.empty()) return;
        vector<int> pilot = survivors;
        if (anchor_restrictive_pilot_sample_size > 0 && int(pilot.size()) > anchor_restrictive_pilot_sample_size) {
            std::mt19937_64 rng(12345);
            std::shuffle(pilot.begin(), pilot.end(), rng);
            pilot.resize(anchor_restrictive_pilot_sample_size);
        }
        int best_q = nbrs[0];
        int best_pass = std::numeric_limits<int>::max();
        for (int qr : nbrs) {
            int pass_count = 0;
            for (int ga : pilot) {
                bool ok = false;
                for (int nb : G.adj[ga]) {
                    if (G.deg[nb] < Q.deg[qr]) continue;
                    if (sim(qr, nb) >= tau) { ok = true; break; }
                }
                if (ok) pass_count++;
            }
            auto key_cur = std::make_tuple(pass_count, -Q.deg[qr], qr);
            auto key_best = std::make_tuple(best_pass, -Q.deg[best_q], best_q);
            if (key_cur < key_best) { best_q = qr; best_pass = pass_count; }
        }
        anchor_restrictive_query = best_q;
    }

    bool anchor_has_restrictive_support(int ga) {
        if (!enable_anchor_restrictive_pruning || anchor_restrictive_query < 0) return true;
        int qr = anchor_restrictive_query;
        for (int nb : G.adj[ga]) {
            if (G.deg[nb] < Q.deg[qr]) continue;
            if (sim(qr, nb) >= tau) return true;
        }
        return false;
    }

    void build_anchor_candidates() {
        vector<int> survivors;
        for (int v = 0; v < num_g; ++v) {
            if (G.deg[v] < Q.deg[anchor_q]) continue;
            if (sim(anchor_q, v) >= tau) survivors.push_back(v);
        }
        if (enable_anchor_restrictive_pruning && !survivors.empty()) choose_anchor_restrictive_neighbor(survivors);
        for (int v : survivors) if (anchor_has_restrictive_support(v)) anchor_candidates.push_back(v);
        std::sort(anchor_candidates.begin(), anchor_candidates.end(), [&](int a, int b){
            float sa = sim(anchor_q, a), sb = sim(anchor_q, b);
            if (sa != sb) return sa > sb;
            return a < b;
        });
    }

    bool used_contains(const vector<int>& used, int v) const {
        for (int z : used) if (z == v) return true;
        return false;
    }

    int select_next_batch_parent(const vector<int>& mapping, const vector<vector<int>>& remaining) const {
        int best = -1;
        for (int u = 0; u < num_q; ++u) {
            if (mapping[u] == -1 || remaining[u].empty()) continue;
            if (best < 0) { best = u; continue; }
            auto ku = std::make_tuple(q_depth[u], -int(remaining[u].size()), u);
            auto kb = std::make_tuple(q_depth[best], -int(remaining[best].size()), best);
            if (ku < kb) best = u;
        }
        return best;
    }

    vector<int> child_candidate_region(int q_child, int g_parent, const vector<int>& mapping, const vector<int>& used) {
        vector<int> out;
        for (int v : G.adj[g_parent]) {
            if (used_contains(used, v)) continue;
            if (G.deg[v] < Q.deg[q_child]) continue;
            if (sim(q_child, v) < tau) continue;
            bool feasible = true;
            for (int qw = 0; qw < num_q; ++qw) {
                int gw = mapping[qw];
                if (gw == -1) continue;
                if (Q.has_edge(q_child, qw) && !G.has_edge(v, gw)) { feasible = false; break; }
            }
            if (feasible) out.push_back(v);
        }
        std::sort(out.begin(), out.end(), [&](int a, int b){
            float sa = sim(q_child, a), sb = sim(q_child, b);
            if (sa != sb) return sa > sb;
            return a < b;
        });
        return out;
    }

    template<class Fn>
    void enumerate_batch(int, int g_parent, const vector<int>& child_list,
                         const vector<int>& mapping, const vector<int>& used, Fn&& fn) {
        if (child_list.empty()) { fn(vector<int>(num_q, -1)); return; }

        vector<vector<int>> C(num_q);
        for (int qc : child_list) {
            C[qc] = child_candidate_region(qc, g_parent, mapping, used);
            if (C[qc].empty()) { return; }
        }

        vector<int> ordered = child_list;
        std::sort(ordered.begin(), ordered.end(), [&](int a, int b){
            auto ka = std::make_tuple(int(C[a].size()), -Q.deg[a], a);
            auto kb = std::make_tuple(int(C[b].size()), -Q.deg[b], b);
            return ka < kb;
        });

        vector<int> local_map(num_q, -1), local_used;
        std::function<void(int)> bt = [&](int idx){
            if (idx == int(ordered.size())) { fn(local_map); return; }
            int u = ordered[idx];
            for (int v : C[u]) {
                bool used_local = false;
                for (int z : local_used) if (z == v) { used_local = true; break; }
                if (used_local) continue;
                bool ok = true;
                for (int w : ordered) {
                    int z = local_map[w];
                    if (z == -1) continue;
                    if (Q.has_edge(u, w) && !G.has_edge(v, z)) { ok = false; break; }
                }
                if (!ok) continue;
                local_map[u] = v;
                local_used.push_back(v);
                bt(idx + 1);
                local_used.pop_back();
                local_map[u] = -1;
            }
        };
        bt(0);
    }

    template<class EmitFn>
    void search(vector<int>& mapping, vector<int>& used, vector<vector<int>>& remaining, int matched_count,
                EmitFn&& emit, uint64_t max_matches=0) {
        if (max_matches && stats.complete_matches >= max_matches) return;
        if (matched_count == num_q) { stats.complete_matches++; emit(mapping); return; }

        int q_parent = select_next_batch_parent(mapping, remaining);
        if (q_parent < 0) return;
        int g_parent = mapping[q_parent];
        vector<int> child_list = remaining[q_parent];

        enumerate_batch(q_parent, g_parent, child_list, mapping, used, [&](const vector<int>& local_map){
            if (max_matches && stats.complete_matches >= max_matches) return;
            vector<int> added;
            added.reserve(child_list.size());
            for (int qc : child_list) {
                int gv = local_map[qc];
                if (gv != -1) { mapping[qc] = gv; used.push_back(gv); added.push_back(qc); }
            }

            vector<vector<int>> new_remaining = remaining;
            new_remaining[q_parent].clear();
            for (int qc : child_list) {
                new_remaining[qc].clear();
                for (int ch : q_children[qc]) if (mapping[ch] == -1) new_remaining[qc].push_back(ch);
            }
            search(mapping, used, new_remaining, matched_count + int(added.size()), emit, max_matches);

            for (int i = int(added.size()) - 1; i >= 0; --i) { mapping[added[i]] = -1; used.pop_back(); }
        });
    }

    template<class EmitFn>
    uint64_t run(EmitFn&& emit, uint64_t max_matches=0) {
        for (int va : anchor_candidates) {
            if (max_matches && stats.complete_matches >= max_matches) break;
            vector<int> mapping(num_q, -1), used;
            vector<vector<int>> remaining(num_q);
            mapping[anchor_q] = va;
            used.push_back(va);
            remaining[anchor_q] = q_children[anchor_q];
            search(mapping, used, remaining, 1, emit, max_matches);
        }
        return stats.complete_matches;
    }


};

class MatchWriter {
public:
    std::ofstream fout;
    int num_q;
    vector<string> buffer;
    size_t flush_lines;
    bool enabled;
    MatchWriter(const string& path, int n, bool enabled_, size_t flush_lines_=65536)
        : num_q(n), flush_lines(flush_lines_), enabled(enabled_) {
        if (enabled) {
            auto pos = path.find_last_of('/');
            if (pos != string::npos) {
                string dir = path.substr(0, pos);
                if (!dir.empty()) {
                    string cmd = "mkdir -p \"" + dir + "\"";
                    std::system(cmd.c_str());
                }
            }
            fout.open(path, std::ios::out | std::ios::binary);
            if (!fout) throw std::runtime_error("Cannot open output path: " + path);
        }
    }
    void write(const vector<int>& mapping) {
        if (!enabled) return;
        string line;
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
        for (const auto& s : buffer) fout << s;
        buffer.clear();
    }
    ~MatchWriter() { flush(); if (fout.is_open()) fout.close(); }
};

struct Args {
    string graph_vertices, graph_edges, query_vertices, query_edges, query_prefix, output = "matches_cpp.txt";
    float tau = 0.96f;
    int bmax = 4;
    int num_query_nodes = 6;
    uint64_t seed = 42;
    uint64_t max_matches = 0;
    bool no_normalize = false;
    bool count_only = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        string k = argv[i];
        auto need = [&](const string& name)->string{ if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name); return argv[++i]; };
        if (k == "--graph-vertices") a.graph_vertices = need(k);
        else if (k == "--graph-edges") a.graph_edges = need(k);
        else if (k == "--query-vertices") a.query_vertices = need(k);
        else if (k == "--query-edges") a.query_edges = need(k);
        else if (k == "--query-prefix") a.query_prefix = need(k);
        else if (k == "--tau") a.tau = std::stof(need(k));
        else if (k == "--bmax") a.bmax = std::stoi(need(k));
        else if (k == "--num-query-nodes") a.num_query_nodes = std::stoi(need(k));
        else if (k == "--seed") a.seed = std::stoull(need(k));
        else if (k == "--max-matches") a.max_matches = std::stoull(need(k));
        else if (k == "--output") a.output = need(k);
        else if (k == "--no-normalize") a.no_normalize = true;
        else if (k == "--count-only") a.count_only = true;
        else if (k == "--help") {
            cout << "Usage: ./basic_batch_bfs_csv_mem --graph-vertices graph_vertices.csv --graph-edges graph_edges.csv "
                    "[--query-vertices query_vertices.csv --query-edges query_edges.csv] "
                    "--tau 0.96 --output matches.txt [--count-only]\n"
                    "  --query-prefix PREFIX        Read or save sampled query as PREFIX_vertices.csv / PREFIX_edges.csv.\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown argument: " + k);
    }
    if (a.graph_vertices.empty() || a.graph_edges.empty()) throw std::runtime_error("Missing graph csv paths. Use --help.");
    if ((a.query_vertices.empty()) != (a.query_edges.empty())) throw std::runtime_error("Provide both --query-vertices and --query-edges, or neither.");
    return a;
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        double t0 = now_sec();
        cout << "Loading graph CSV ...\n";
        Graph G = read_graph_csv(args.graph_vertices, args.graph_edges, !args.no_normalize, true);
        cout << "Data graph: n=" << G.n << " dim=" << G.dim << "\n";

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
                cout << "Sampling query inside C++ ...\n";
                Q = sample_query_fast_cpp(G, args.num_query_nodes, args.seed, 2, 20, gt_mapping);
                cout << "Query graph: n=" << Q.n << " dim=" << Q.dim << "\n";
                cout << "Ground-truth mapping: ";
                for (int i = 0; i < int(gt_mapping.size()); ++i) {
                    if (i) cout << " ";
                    cout << i << "->" << gt_mapping[i];
                }
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

        double t_init0 = now_sec();
        BasicBatchBFSMatcher matcher(G, Q, args.tau);
        double t_init1 = now_sec();

        MatchWriter writer(args.output, Q.n, !args.count_only);
        vector<vector<int>> preview;
        preview.reserve(20);
        double t_search0 = now_sec();
        uint64_t cnt = matcher.run([&](const vector<int>& mapping){
            if (!args.count_only) writer.write(mapping);
            if (preview.size() < 20) preview.push_back(mapping);
        }, args.max_matches);
        writer.flush();
        double t_search1 = now_sec();

        cout << "\nFound " << cnt << " match(es).\n";
        if (!args.count_only) cout << "All matches streamed to: " << args.output << "\n";
        cout << "\nPreview matches:\n";
        for (size_t i = 0; i < preview.size(); ++i) {
            cout << "Match " << (i + 1) << ":";
            for (int v : preview[i]) cout << " " << v;
            cout << "\n";
        }
        cout << "\n========== Runtime Summary ==========\n";
        cout << "Matcher initialization : " << (t_init1 - t_init0) << " s\n";
        cout << "Matching search" << (args.count_only ? "" : " + output") << ": " << (t_search1 - t_search0) << " s\n";
        cout << "Total runtime          : " << now_sec() - t0 << " s\n";
        cout << "\n========== Process RSS Memory ==========\n";
        cout << "final current RSS      : " << current_rss_mb() << " MB\n";
        cout << "final peak RSS         : " << peak_rss_mb() << " MB\n";
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
