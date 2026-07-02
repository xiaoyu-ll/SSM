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

static Graph read_graph_csv(const string& vp, const string& ep, bool normalize=true) {
    Graph G = read_vertices_csv(vp);
    read_edges_csv_into(G, ep, true, true);
    if (normalize) normalize_features(G);
    return G;
}

static bool nonzero_feature(const vector<float>& x, double eps=1e-12) {
    double s = 0.0;
    for (float v : x) s += double(v) * double(v);
    return s > eps * eps;
}

static void ensure_parent_dir(const string& path) {
    auto pos = path.find_last_of("/\\");
    if (pos == string::npos) return;
    string dir = path.substr(0, pos);
    if (!dir.empty()) std::system(("mkdir -p \"" + dir + "\"").c_str());
}

static bool file_exists(const string& path) { std::ifstream fin(path); return bool(fin); }
static string parent_path(string path) { while (!path.empty() && (path.back()=='/'||path.back()=='\\')) path.pop_back(); auto p=path.find_last_of("/\\"); return p==string::npos?"":path.substr(0,p); }
static string basename_no_slash(string path) { while (!path.empty() && (path.back()=='/'||path.back()=='\\')) path.pop_back(); auto p=path.find_last_of("/\\"); return p==string::npos?path:path.substr(p+1); }
static string default_query_prefix(const string& graph_vertices, int k, uint64_t seed) {
    string name = basename_no_slash(parent_path(graph_vertices));
    if (name.empty()) name = "dataset";
    return "query/" + name + "/q_k" + std::to_string(k) + "_seed" + std::to_string(seed);
}

static void write_vertices_csv(const Graph& Q, const string& path) {
    ensure_parent_dir(path);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write query vertices: " + path);
    out << "id";
    for (int j=0;j<Q.dim;++j) out << ",f" << j;
    out << "\n" << std::setprecision(9);
    for (int i=0;i<Q.n;++i) { out << i; for (int j=0;j<Q.dim;++j) out << ',' << Q.x[i][j]; out << '\n'; }
}
static void write_edges_csv(const Graph& Q, const string& path) {
    ensure_parent_dir(path);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write query edges: " + path);
    out << "src,dst\n";
    for (int u=0;u<Q.n;++u) for (int v:Q.adj[u]) if (u < v) out << u << ',' << v << '\n';
}

static Graph sample_query_from_graph(const Graph& G, int k, uint64_t seed, vector<int>& gt, int max_tries=100) {
    if (k <= 0) throw std::runtime_error("num-query-nodes must be positive");
    vector<int> valid;
    for (int i=0;i<G.n;++i) if (nonzero_feature(G.x[i])) valid.push_back(i);
    if ((int)valid.size() < k) throw std::runtime_error("Not enough non-zero feature vertices");
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, valid.size()-1);
    for (int attempt=0; attempt<max_tries; ++attempt) {
        int start = valid[pick(rng)];
        vector<int> chosen, q;
        vector<uint8_t> seen(G.n, 0);
        seen[start] = 1; q.push_back(start);
        for (size_t h=0; h<q.size() && (int)chosen.size()<k; ++h) {
            int u=q[h];
            if (nonzero_feature(G.x[u])) chosen.push_back(u);
            vector<int> nbr=G.adj[u];
            std::shuffle(nbr.begin(), nbr.end(), rng);
            for (int v:nbr) if (!seen[v] && nonzero_feature(G.x[v])) { seen[v]=1; q.push_back(v); }
        }
        if ((int)chosen.size() < k) continue;
        chosen.resize(k);
        std::unordered_map<int,int> pos;
        for (int i=0;i<k;++i) pos[chosen[i]]=i;
        Graph Q; Q.n=k; Q.dim=G.dim; Q.x.assign(k, vector<float>(G.dim)); Q.adj.assign(k, {}); Q.deg.assign(k,0);
        for (int i=0;i<k;++i) Q.x[i]=G.x[chosen[i]];
        bool has_edge=false;
        for (int i=0;i<k;++i) {
            for (int ov:G.adj[chosen[i]]) {
                auto it=pos.find(ov); if (it==pos.end()) continue;
                int j=it->second; if (i!=j) Q.adj[i].push_back(j);
            }
            auto& a=Q.adj[i]; std::sort(a.begin(), a.end()); a.erase(std::unique(a.begin(), a.end()), a.end()); Q.deg[i]=int(a.size()); if (!a.empty()) has_edge=true;
        }
        if (!has_edge) continue;
        gt=chosen; return Q;
    }
    throw std::runtime_error("Failed to sample query");
}

static inline float dot_product(const vector<float>& a, const vector<float>& b) {
    double s = 0.0;
    for (size_t i=0;i<a.size();++i) s += double(a[i]) * double(b[i]);
    return float(s);
}

struct Stats {
    uint64_t complete_matches = 0;
    uint64_t recursive_calls = 0;
    uint64_t candidate_extensions = 0;
    uint64_t structural_prunes = 0;
    uint64_t injectivity_prunes = 0;
    uint64_t edge_checks = 0;
    uint64_t semantic_similarity_evals = 0;
    uint64_t semantic_partial_prunes = 0;
    uint64_t initial_degree_domain_values = 0;
    uint64_t after_topdown_domain_values = 0;
    uint64_t after_bottomup_domain_values = 0;
    uint64_t after_nte_domain_values = 0;
    uint64_t topdown_deletions = 0;
    uint64_t bottomup_deletions = 0;
    uint64_t nte_deletions = 0;
    uint64_t cpi_tree_edge_entries = 0;
    uint64_t cpi_tree_edge_lists = 0;
    uint64_t core_vertices = 0;
    uint64_t forest_vertices = 0;
    uint64_t leaf_vertices = 0;
};

class CFLMatchSemanticMat {
public:
    const Graph& G;
    const Graph& Q;
    float tau;
    int n, qn;

    int root = 0;
    vector<int> parent, level;
    vector<vector<int>> children;
    vector<int> bfs_order;
    vector<uint8_t> is_tree_edge_flat;
    vector<uint8_t> is_core, is_leaf, is_forest;

    vector<vector<int>> dom;
    vector<uint8_t> mark;
    vector<vector<int>> pos_in_dom;

    // cpi_child[child][parent-candidate-index] = child candidates adjacent to parent candidate.
    vector<vector<vector<int>>> cpi_child;
    vector<int> order;
    Stats stats;

    CFLMatchSemanticMat(const Graph& data, const Graph& query, float tau_)
        : G(data), Q(query), tau(tau_), n(data.n), qn(query.n) {
        if (G.dim != Q.dim) throw std::runtime_error("Feature dimensions do not match");
        if (qn <= 0) throw std::runtime_error("Query graph is empty");
        root = choose_root_by_degree_domain();
        build_bfs_tree();
        decompose_cfl();
        build_cpi_domains_semantic();
        cpi_top_down_refine();
        cpi_bottom_up_refine();
        cpi_non_tree_support_refine();
        rebuild_positions();
        build_cpi_adjacency_lists();
        build_cfl_matching_order();
    }

    inline size_t midx(int u, int v) const { return size_t(u) * size_t(n) + size_t(v); }
    inline size_t qedge_idx(int u, int v) const { return size_t(u) * size_t(qn) + size_t(v); }

    int semantic_domain_size(int u) const {
        int c = 0;
        for (int v=0; v<n; ++v) if (G.deg[v] >= Q.deg[u] && dot_product(Q.x[u], G.x[v]) >= tau) ++c;
        return c;
    }

    int choose_root_by_degree_domain() const {
        int best = 0;
        double best_score = std::numeric_limits<double>::infinity();
        for (int u=0; u<qn; ++u) {
            double score = double(semantic_domain_size(u)) / double(std::max(1, Q.deg[u]));
            if (score < best_score || (score == best_score && Q.deg[u] > Q.deg[best])) {
                best_score = score; best = u;
            }
        }
        return best;
    }

    void build_bfs_tree() {
        parent.assign(qn, -1);
        level.assign(qn, -1);
        children.assign(qn, {});
        bfs_order.clear();
        std::queue<int> qu;
        qu.push(root); level[root]=0; bfs_order.push_back(root);
        while (!qu.empty()) {
            int u=qu.front(); qu.pop();
            vector<int> nbr=Q.adj[u];
            std::sort(nbr.begin(), nbr.end(), [&](int a, int b){
                return std::make_tuple(semantic_domain_size(a), -Q.deg[a], a) <
                       std::make_tuple(semantic_domain_size(b), -Q.deg[b], b);
            });
            for (int v:nbr) if (level[v] < 0) {
                level[v] = level[u] + 1;
                parent[v] = u;
                children[u].push_back(v);
                qu.push(v);
                bfs_order.push_back(v);
            }
        }
        for (int u=0; u<qn; ++u) if (level[u] < 0) throw std::runtime_error("Query graph must be connected");
        is_tree_edge_flat.assign(size_t(qn)*size_t(qn), 0);
        for (int u=0; u<qn; ++u) if (parent[u] >= 0) {
            is_tree_edge_flat[qedge_idx(u,parent[u])] = 1;
            is_tree_edge_flat[qedge_idx(parent[u],u)] = 1;
        }
    }

    void decompose_cfl() {
        // Core is the 2-core. If the query is a tree, use the chosen root as the core entry.
        vector<int> d = Q.deg;
        vector<uint8_t> removed(qn,0);
        std::queue<int> q;
        for (int u=0; u<qn; ++u) if (d[u] <= 1) q.push(u);
        while (!q.empty()) {
            int u=q.front(); q.pop();
            if (removed[u]) continue;
            removed[u]=1;
            for (int v:Q.adj[u]) if (!removed[v]) {
                if (--d[v] == 1) q.push(v);
            }
        }
        is_core.assign(qn,0);
        int core_count = 0;
        for (int u=0; u<qn; ++u) if (!removed[u]) { is_core[u]=1; core_count++; }
        if (core_count == 0) { is_core[root]=1; core_count=1; }

        // Leaves are non-core vertices that have no non-core child in the BFS forest rooted at core side.
        is_leaf.assign(qn,0);
        is_forest.assign(qn,0);
        for (int u=0; u<qn; ++u) if (!is_core[u]) {
            bool has_noncore_child = false;
            for (int c:children[u]) if (!is_core[c]) { has_noncore_child = true; break; }
            if (!has_noncore_child) is_leaf[u] = 1;
            else is_forest[u] = 1;
        }
        stats.core_vertices = 0; stats.forest_vertices = 0; stats.leaf_vertices = 0;
        for (int u=0; u<qn; ++u) { stats.core_vertices += is_core[u]; stats.forest_vertices += is_forest[u]; stats.leaf_vertices += is_leaf[u]; }
    }

    bool semantic_pass_for_domain(int u, int v) const {
        return dot_product(Q.x[u], G.x[v]) >= tau;
    }

    void build_cpi_domains_semantic() {
        dom.assign(qn, {});
        mark.assign(size_t(qn)*size_t(n), 0);
        for (int u=0; u<qn; ++u) {
            dom[u].reserve(n);
            for (int v=0; v<n; ++v) if (G.deg[v] >= Q.deg[u] && semantic_pass_for_domain(u, v)) {
                dom[u].push_back(v);
                mark[midx(u,v)] = 1;
            }
            stats.initial_degree_domain_values += dom[u].size();
        }
    }

    uint64_t domain_values() const {
        uint64_t s=0; for (const auto& x:dom) s += x.size(); return s;
    }

    bool support_in_domain(int u, int data_v, int qnbr) const {
        (void)u;
        for (int z : G.adj[data_v]) if (mark[midx(qnbr, z)]) return true;
        return false;
    }

    void remove_from_domain(int u, int v) {
        if (mark[midx(u,v)]) mark[midx(u,v)] = 0;
    }

    void compact_domains() {
        for (int u=0; u<qn; ++u) {
            vector<int> kept;
            kept.reserve(dom[u].size());
            for (int v:dom[u]) if (mark[midx(u,v)]) kept.push_back(v);
            dom[u].swap(kept);
        }
    }

    void cpi_top_down_refine() {
        // Follow CFL top-down spirit: each non-root candidate must have support from its tree parent.
        for (int idx=1; idx<(int)bfs_order.size(); ++idx) {
            int u = bfs_order[idx];
            int p = parent[u];
            for (int v:dom[u]) {
                if (!mark[midx(u,v)]) continue;
                bool has_parent_support = false;
                for (int z:G.adj[v]) if (mark[midx(p,z)]) { has_parent_support = true; break; }
                if (!has_parent_support) { remove_from_domain(u,v); stats.topdown_deletions++; }
            }
            compact_domains();
        }
        stats.after_topdown_domain_values = domain_values();
    }

    void cpi_bottom_up_refine() {
        // One bottom-up pass: a parent candidate must support every tree child.
        for (int bi=(int)bfs_order.size()-1; bi>=0; --bi) {
            int u = bfs_order[bi];
            for (int v:dom[u]) {
                if (!mark[midx(u,v)]) continue;
                bool ok = true;
                for (int c:children[u]) {
                    if (!support_in_domain(u, v, c)) { ok = false; break; }
                }
                if (!ok) { remove_from_domain(u,v); stats.bottomup_deletions++; }
            }
            compact_domains();
        }
        stats.after_bottomup_domain_values = domain_values();
    }

    void cpi_non_tree_support_refine() {
        // Use non-tree edges for support pruning. Iterate to a fixed point over non-tree edges only.
        bool changed = true;
        while (changed) {
            changed = false;
            for (int u=0; u<qn; ++u) {
                for (int w:Q.adj[u]) {
                    if (u >= w) continue;
                    if (is_tree_edge_flat[qedge_idx(u,w)]) continue;
                    for (int v:dom[u]) {
                        if (!mark[midx(u,v)]) continue;
                        if (!support_in_domain(u, v, w)) { remove_from_domain(u,v); stats.nte_deletions++; changed = true; }
                    }
                    for (int z:dom[w]) {
                        if (!mark[midx(w,z)]) continue;
                        if (!support_in_domain(w, z, u)) { remove_from_domain(w,z); stats.nte_deletions++; changed = true; }
                    }
                    if (changed) compact_domains();
                }
            }
        }
        stats.after_nte_domain_values = domain_values();
    }

    void rebuild_positions() {
        pos_in_dom.assign(qn, vector<int>(n, -1));
        for (int u=0; u<qn; ++u) {
            std::sort(dom[u].begin(), dom[u].end(), [&](int a,int b){
                if (G.deg[a] != G.deg[b]) return G.deg[a] < G.deg[b];
                return a < b;
            });
            for (int i=0; i<(int)dom[u].size(); ++i) pos_in_dom[u][dom[u][i]] = i;
        }
    }

    void build_cpi_adjacency_lists() {
        cpi_child.assign(qn, {});
        for (int u=0; u<qn; ++u) if (parent[u] >= 0) {
            int p = parent[u];
            cpi_child[u].assign(dom[p].size(), {});
            stats.cpi_tree_edge_lists += dom[p].size();
            for (int pi=0; pi<(int)dom[p].size(); ++pi) {
                int pv = dom[p][pi];
                auto& list = cpi_child[u][pi];
                for (int cv : G.adj[pv]) {
                    if (cv >= 0 && cv < n && mark[midx(u,cv)]) list.push_back(cv);
                }
                stats.cpi_tree_edge_entries += list.size();
            }
        }
    }

    vector<int> sorted_subset_order(const vector<int>& vertices, const vector<uint8_t>& allowed) const {
        vector<int> result;
        vector<uint8_t> in_result(qn,0), remaining(qn,0);
        for (int u:vertices) remaining[u]=1;
        // Start with earliest BFS/lowest domain vertex inside the subset.
        while ((int)result.size() < (int)vertices.size()) {
            int best=-1;
            for (int u:vertices) if (remaining[u]) {
                bool connected = result.empty();
                if (!connected) for (int w:Q.adj[u]) if (in_result[w]) { connected=true; break; }
                if (!connected) continue;
                if (best < 0 || std::make_tuple(dom[u].size(), -Q.deg[u], level[u], u) <
                                std::make_tuple(dom[best].size(), -Q.deg[best], level[best], best)) best = u;
            }
            if (best < 0) {
                for (int u:vertices) if (remaining[u]) { best=u; break; }
            }
            result.push_back(best); in_result[best]=1; remaining[best]=0;
        }
        (void)allowed;
        return result;
    }

    void build_cfl_matching_order() {
        vector<int> core, forest, leaf;
        for (int u : bfs_order) if (is_core[u]) core.push_back(u);
        for (int u : bfs_order) if (is_forest[u]) forest.push_back(u);
        for (int u : bfs_order) if (is_leaf[u]) leaf.push_back(u);
        vector<uint8_t> dummy(qn,1);
        vector<int> oc = sorted_subset_order(core, dummy);
        vector<int> of = sorted_subset_order(forest, dummy);
        // Leaf last, smaller structural domain first to reduce Cartesian products.
        std::sort(leaf.begin(), leaf.end(), [&](int a,int b){
            return std::make_tuple(dom[a].size(), -Q.deg[a], a) < std::make_tuple(dom[b].size(), -Q.deg[b], b);
        });
        order.clear();
        order.insert(order.end(), oc.begin(), oc.end());
        order.insert(order.end(), of.begin(), of.end());
        order.insert(order.end(), leaf.begin(), leaf.end());
        // Repair connectivity: if any vertex appears before all its neighbors, move a BFS prefix order fallback.
        if ((int)order.size() != qn) throw std::runtime_error("Internal order size mismatch");
    }

    vector<int> candidates_from_cpi(int u, const vector<int>& mapping) {
        int p = parent[u];
        if (p >= 0 && mapping[p] >= 0) {
            int pi = pos_in_dom[p][mapping[p]];
            if (pi < 0 || pi >= (int)cpi_child[u].size()) return {};
            return cpi_child[u][pi];
        }
        // If the BFS parent is not yet mapped, use any mapped query neighbor's original adjacency list.
        int mapped_nbr = -1;
        for (int w:Q.adj[u]) if (mapping[w] >= 0) { mapped_nbr = w; break; }
        if (mapped_nbr >= 0) {
            vector<int> out;
            for (int v:G.adj[mapping[mapped_nbr]]) if (mark[midx(u,v)]) out.push_back(v);
            return out;
        }
        return dom[u];
    }

    bool structurally_feasible(int u, int v, const vector<int>& mapping, const vector<uint8_t>& used) {
        stats.candidate_extensions++;
        if (used[v]) { stats.injectivity_prunes++; return false; }
        for (int w:Q.adj[u]) {
            int z = mapping[w];
            if (z < 0) continue;
            stats.edge_checks++;
            if (!G.has_edge(v,z)) { stats.structural_prunes++; return false; }
        }
        return true;
    }

    bool semantic_pass(int u, int v) {
        stats.semantic_similarity_evals++;
        return dot_product(Q.x[u], G.x[v]) >= tau;
    }

    template<class Emit>
    void dfs(int pos, vector<int>& mapping, vector<uint8_t>& used, Emit&& emit, uint64_t max_matches) {
        stats.recursive_calls++;
        if (max_matches && stats.complete_matches >= max_matches) return;
        if (pos == qn) {
            stats.complete_matches++;
            emit(mapping);
            return;
        }
        int u = order[pos];
        vector<int> cand = candidates_from_cpi(u, mapping);
        for (int v:cand) {
            if (max_matches && stats.complete_matches >= max_matches) break;
            if (!mark[midx(u,v)]) continue;
            if (!structurally_feasible(u, v, mapping, used)) continue;
            // Semantic threshold has already been used in CPI/domain construction.
            // Keep this check only as a safety guard against implementation mistakes.
            if (!semantic_pass(u, v)) { stats.semantic_partial_prunes++; continue; }
            mapping[u] = v; used[v] = 1;
            dfs(pos+1, mapping, used, std::forward<Emit>(emit), max_matches);
            used[v] = 0; mapping[u] = -1;
        }
    }

    template<class Emit>
    uint64_t run(Emit&& emit, uint64_t max_matches=0) {
        for (int u=0; u<qn; ++u) if (dom[u].empty()) return 0;
        vector<int> mapping(qn, -1);
        vector<uint8_t> used(n, 0);
        dfs(0, mapping, used, std::forward<Emit>(emit), max_matches);
        return stats.complete_matches;
    }

    void print_profile() const {
        cout << "\n========== CFL-Match Semantic Materialized-CPI Profile ==========" << "\n";
        cout << "complete_matches              : " << stats.complete_matches << "\n";
        cout << "recursive_calls               : " << stats.recursive_calls << "\n";
        cout << "candidate_extensions          : " << stats.candidate_extensions << "\n";
        cout << "injectivity_prunes            : " << stats.injectivity_prunes << "\n";
        cout << "structural_prunes             : " << stats.structural_prunes << "\n";
        cout << "edge_checks                   : " << stats.edge_checks << "\n";
        cout << "semantic_similarity_evals     : " << stats.semantic_similarity_evals << "\n";
        cout << "semantic_partial_prunes       : " << stats.semantic_partial_prunes << "\n";
        cout << "core_vertices                 : " << stats.core_vertices << "\n";
        cout << "forest_vertices               : " << stats.forest_vertices << "\n";
        cout << "leaf_vertices                 : " << stats.leaf_vertices << "\n";
        cout << "root_query_vertex             : " << root << "\n";
        cout << "initial_semantic_domain_values: " << stats.initial_degree_domain_values << "\n";
        cout << "domain_values_after_topdown   : " << stats.after_topdown_domain_values << "\n";
        cout << "domain_values_after_bottomup  : " << stats.after_bottomup_domain_values << "\n";
        cout << "domain_values_after_nte       : " << stats.after_nte_domain_values << "\n";
        cout << "topdown_deletions             : " << stats.topdown_deletions << "\n";
        cout << "bottomup_deletions            : " << stats.bottomup_deletions << "\n";
        cout << "non_tree_edge_deletions       : " << stats.nte_deletions << "\n";
        cout << "cpi_tree_edge_lists           : " << stats.cpi_tree_edge_lists << "\n";
        cout << "cpi_tree_edge_entries         : " << stats.cpi_tree_edge_entries << "\n";
        cout << "matching_order                :";
        for (int u:order) cout << ' ' << u;
        cout << "\n";
    }
};

class MatchWriter {
public:
    std::ofstream out;
    int qn;
    vector<string> buf;
    MatchWriter(const string& path, int qn_) : qn(qn_) {
        ensure_parent_dir(path);
        out.open(path, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open output path: " + path);
    }
    void write(const vector<int>& mapping) {
        string line;
        for (int i=0;i<qn;++i) { if (i) line.push_back(' '); line += std::to_string(mapping[i]); }
        line.push_back('\n'); buf.push_back(std::move(line));
        if (buf.size() >= 65536) flush();
    }
    void flush() { for (const auto& s:buf) out << s; buf.clear(); }
    ~MatchWriter(){ flush(); }
};

struct Args {
    string graph_vertices, graph_edges, query_vertices, query_edges, query_prefix;
    string output = "matches_cfl_sem_partial_cpi_v2.txt";
    float tau = 0.97f;
    uint64_t max_matches = 0;
    int num_query_nodes = 6;
    uint64_t seed = 42;
    bool no_normalize = false;
    bool require_query = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i=1;i<argc;++i) {
        string k=argv[i];
        auto need=[&](const string& name)->string{ if (i+1>=argc) throw std::runtime_error("Missing value for "+name); return argv[++i]; };
        if (k=="--graph-vertices") a.graph_vertices=need(k);
        else if (k=="--graph-edges") a.graph_edges=need(k);
        else if (k=="--query-vertices") a.query_vertices=need(k);
        else if (k=="--query-edges") a.query_edges=need(k);
        else if (k=="--query-prefix") a.query_prefix=need(k);
        else if (k=="--tau") a.tau=std::stof(need(k));
        else if (k=="--output") a.output=need(k);
        else if (k=="--max-matches") a.max_matches=std::stoull(need(k));
        else if (k=="--num-query-nodes") a.num_query_nodes=std::stoi(need(k));
        else if (k=="--seed") a.seed=std::stoull(need(k));
        else if (k=="--no-normalize") a.no_normalize=true;
        else if (k=="--require-query") a.require_query=true;
        else if (k=="--help") {
            cout << "Usage:\n  ./cfl_match_sem_partial_cpi_v2_single --graph-vertices graph_vertices.csv --graph-edges graph_edges.csv --query-vertices q_vertices.csv --query-edges q_edges.csv --tau 0.97 --output matches.txt\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown argument: " + k);
    }
    if (a.graph_vertices.empty() || a.graph_edges.empty()) throw std::runtime_error("Missing graph paths");
    if ((a.query_vertices.empty()) != (a.query_edges.empty())) throw std::runtime_error("Provide both query files or neither");
    if (a.require_query && (a.query_vertices.empty() || a.query_edges.empty())) throw std::runtime_error("Missing query paths under --require-query");
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
        } else {
            string prefix = args.query_prefix.empty() ? default_query_prefix(args.graph_vertices, args.num_query_nodes, args.seed) : args.query_prefix;
            string qv = prefix + "_vertices.csv", qe = prefix + "_edges.csv";
            if (file_exists(qv) && file_exists(qe)) Q = read_graph_csv(qv, qe, !args.no_normalize);
            else {
                cout << "Sampling query from data graph ...\n";
                Q = sample_query_from_graph(G, args.num_query_nodes, args.seed, gt);
                write_vertices_csv(Q, qv); write_edges_csv(Q, qe);
                cout << "Sampled query saved to:\n  " << qv << "\n  " << qe << "\n";
            }
        }
        cout << "Query graph: n=" << Q.n << " dim=" << Q.dim << "\n";
        double t1 = now_sec();
        print_memory("after query preparation");

        cout << "Initializing CFL-Match Semantic Materialized-CPI baseline ...\n";
        CFLMatchSemanticMat matcher(G, Q, args.tau);
        double t2 = now_sec();
        print_memory("after CPI construction");

        MatchWriter writer(args.output, Q.n);
        auto emit = [&](const vector<int>& mapping){ writer.write(mapping); };
        cout << "Running matching ...\n";
        uint64_t count = matcher.run(emit, args.max_matches);
        writer.flush();
        double t3 = now_sec();
        print_memory("after matching");
        matcher.print_profile();
        cout << "\nFound " << count << " match(es).\n";
        cout << "Matches written to: " << args.output << "\n";
        cout << "\n========== Runtime Summary ==========\n";
        cout << "CSV loading + query : " << (t1-t0) << " s\n";
        cout << "CPI construction    : " << (t2-t1) << " s\n";
        cout << "Matching + output   : " << (t3-t2) << " s\n";
        cout << "Total runtime       : " << (t3-t0) << " s\n";
        cout << "Final current RSS   : " << current_rss_mb() << " MB\n";
        cout << "Final peak RSS      : " << peak_rss_mb() << " MB\n";
        return 0;
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
