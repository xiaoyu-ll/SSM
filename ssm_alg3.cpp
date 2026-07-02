#include <algorithm>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
        return -1.0;
    }
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

static void print_memory(const string&) {}

static vector<string> split_csv_line(const string& line) {
    vector<string> out;
    string cur;
    cur.reserve(32);
    for (char c : line) {
        if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else if (c != '\r') {
            cur.push_back(c);
        }
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
        for (size_t i = 1; i < parts.size(); ++i) {
            feat.push_back(parts[i].empty() ? 0.0f : std::stof(parts[i]));
        }

        if (dim < 0) dim = int(feat.size());
        if (int(feat.size()) != dim) {
            throw std::runtime_error("Inconsistent feature dimension in " + path);
        }
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
        if (u < 0 || v < 0 || u >= G.n || v >= G.n) {
            throw std::runtime_error("Edge endpoint out of range in " + path);
        }
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
    // Example: dataset/reddit_titles/graph_vertices.csv -> reddit_titles.
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
                                     vector<int>& gt_mapping,
                                     int max_tries = 100) {
    if (k <= 0) throw std::runtime_error("num_query_nodes must be positive.");

    vector<int> valid;
    valid.reserve(G.n);
    for (int i = 0; i < G.n; ++i) {
        if (nonzero_feature(G.x[i])) valid.push_back(i);
    }
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
                if (i == j) continue;
                Q.adj[i].push_back(j);
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







static bool is_connected_graph(const Graph& H) {
    if (H.n == 0) return true;
    vector<uint8_t> seen(H.n, 0);
    vector<int> q;
    q.reserve(H.n);
    seen[0] = 1;
    q.push_back(0);
    for (size_t head = 0; head < q.size(); ++head) {
        int u = q[head];
        for (int v : H.adj[u]) if (!seen[v]) {
            seen[v] = 1;
            q.push_back(v);
        }
    }
    return int(q.size()) == H.n;
}



struct IVFJoinStats {
    uint64_t recursive_calls = 0;
    uint64_t complete_matches = 0;

    uint64_t anchor_candidates_before_filter = 0;
    uint64_t anchor_candidates = 0;
    int anchor_restrictive_query = -1;
    uint64_t anchor_filter_pruned = 0;

    uint64_t ivf_cells = 0;
    uint64_t ivf_cell_tests = 0;
    uint64_t ivf_cell_prunes = 0;
    uint64_t ivf_points_scanned = 0;
    uint64_t ivf_exact_similarity_evals = 0;
    uint64_t progressive_pair_tests = 0;
    uint64_t progressive_blocks_scanned = 0;
    uint64_t progressive_early_rejects = 0;
    uint64_t progressive_full_evals = 0;
    uint64_t semantic_candidates_total = 0;
    uint64_t angular_cap_certified = 0;
    uint64_t angular_shells = 16;
    uint64_t angular_shell_entries = 0;
    uint64_t angular_shell_nonempty = 0;

    uint64_t shell_edge_bucket_directed_edges = 0;
    uint64_t shell_edge_bucket_tests = 0;
    uint64_t shell_edge_bucket_entries = 0;
    uint64_t shell_edge_bucket_nonempty = 0;
    uint64_t shell_edge_bucket_support_generated = 0;
    uint64_t shell_edge_bucket_pair_hits = 0;
    uint64_t ivf_shell_pair_blocks = 0;
    uint64_t ivf_shell_empty_blocks = 0;
    uint64_t ivf_shell_source_blocks = 0;
    uint64_t ivf_shell_target_blocks = 0;
    uint64_t ivf_shell_candidates_scanned = 0;

    uint64_t qscore_mask_vertices = 0;
    uint64_t qscore_mask_score_evals = 0;
    uint64_t maskpair_edge_scans = 0;
    uint64_t maskpair_edge_skips_untouched = 0;
    uint64_t maskpair_query_src_hits = 0;
    uint64_t maskpair_query_edge_hits = 0;
    uint64_t maskpair_support_generated = 0;

    uint64_t qsubspace_rank = 0;
    uint64_t qsubspace_score_evals = 0;
    uint64_t edge_range_directed_edges = 0;
    uint64_t edge_range_nodes = 0;
    uint64_t edge_range_node_tests = 0;
    uint64_t edge_range_node_prunes = 0;
    uint64_t edge_range_node_full_accepts = 0;
    uint64_t edge_range_leaf_tests = 0;
    uint64_t edge_range_support_generated = 0;

    uint64_t support_edge_build_scans = 0;
    uint64_t support_edges_total = 0;
    uint64_t ac_rounds = 0;
    uint64_t ac_support_tests = 0;
    uint64_t ac_vertices_removed = 0;
    uint64_t active_candidates_total = 0;

    uint64_t pivot_batch_calls = 0;
    uint64_t terminal_batch_calls = 0;
    uint64_t terminal_direct_emits = 0;
    uint64_t terminal_singleton_calls = 0;
    uint64_t terminal_singleton_direct_emits = 0;
    uint64_t support_domain_builds = 0;
    uint64_t support_base_entries_scanned = 0;
    uint64_t support_membership_tests = 0;
    uint64_t support_consistency_prunes = 0;
    uint64_t compact_active_total = 0;
    uint64_t compact_bitset_edges = 0;
    uint64_t compact_bitset_words = 0;
    uint64_t compact_domain_builds = 0;
    uint64_t compact_word_ands = 0;
    uint64_t compact_bits_enumerated = 0;
    uint64_t active_local_rows = 0;
    uint64_t active_local_row_lookups = 0;
    uint64_t active_local_empty_lookups = 0;
    uint64_t domain_empty_prunes = 0;

    uint64_t local_support_calls = 0;
    uint64_t local_support_removed = 0;
    uint64_t local_support_empty_prunes = 0;
};

class AngularShellIVFBucketMatcher {
public:
    const Graph& G;
    const Graph& Q;
    float tau;
    int num_g;
    int num_q;
    int dim;
    int bmax;
    int ivf_cells_req;
    int anchor_pilot_size;
    int anchor_q = -1;

    struct AngularLandmarkIndex {
        vector<float> landmark_vec;
        vector<float> data_angle;
        vector<std::pair<float,int>> sorted_by_angle;
    };

    vector<AngularLandmarkIndex> angular_index;
    vector<vector<float>> query_angle;
    float theta_tau = 0.0f;

    // Progressive similarity filtering over vector blocks.  For a prefix of
    // processed blocks we maintain the accumulated inner product and use
    // Cauchy--Schwarz on the remaining blocks as a safe upper bound.
    int sim_blocks = 8;
    vector<int> sim_block_start;
    vector<int> sim_block_end;
    vector<vector<float>> q_rem_norm; // q_rem_norm[u][t] = norm of blocks t..ell-1
    vector<vector<float>> g_rem_norm; // g_rem_norm[v][t] = norm of blocks t..ell-1
    vector<vector<int>> sem_cands;       // exact semantic candidates after IVF verification, then AC-refined.
    vector<uint8_t> sem_member;          // num_q * num_g, active after AC.
    vector<int> sem_shell;               // num_q * num_g -> angular shell id in exact cap, -1 if not candidate.
    vector<vector<vector<int>>> sem_shell_cands; // sem_shell_cands[u][sid] is the in-cap IVF inverted list.

    vector<std::pair<int,int>> dir_qedges;
    vector<int> edge_id;                 // num_q * num_q -> directed edge id, -1 if absent.

    // Sparse support relation.  The previous implementation stored
    // support[eid][v] for every data vertex v, which creates many empty
    // vectors and noticeably increases RSS.  Here support_row_id[eid][v]
    // maps a source vertex to a compact row id, and only non-empty rows are
    // materialized.
    vector<vector<int>> support_row_id;
    vector<vector<int>> support_sources;
    vector<vector<vector<int>>> support_lists;

    // Compact active-candidate ids after AC. active_id[u][v] gives the index
    // of data vertex v in the active candidate list of query vertex u, or -1.
    vector<vector<int>> active_id;
    vector<int> active_words;

    // Sparse bitset support rows.  support_bits_row_id[eid][v] gives the row
    // in support_bits_rows[eid], or -1 if v has no active target support.
    vector<vector<int>> support_bits_row_id; // legacy; not used by ActiveLocal executor
    vector<vector<int>> support_bits_active_row_id; // [eid][active_id(source_q, from)] -> row id
    vector<vector<vector<uint64_t>>> support_bits_rows;
    vector<vector<vector<int>>> support_bits_row_words; // [eid][row id] -> nonzero word ids for sparse terminal domains
    vector<vector<int>> support_bits_row_popcnt;   // [eid][row id] -> active target count; avoids repeated popcount in pivot selection.

    IVFJoinStats stats;
    vector<vector<uint64_t>> terminal_domain_scratch;
    vector<vector<int>> terminal_word_scratch;     // nonzero-word lists for terminal bit domains
    vector<vector<uint64_t>> domain_scratch;        // reusable domain buffers for non-terminal candidate construction
    vector<vector<uint64_t>> local_mask_scratch;    // reusable C[u] masks for local support pruning

    // Scratch stamping array used by batch_union_feasible().  This removes
    // a high-frequency unordered_set allocation from the DFS hot path while
    // preserving the exact injectivity feasibility test.
    vector<int> union_seen_stamp;
    int union_seen_token = 1;

    AngularShellIVFBucketMatcher(const Graph& data, const Graph& query, float tau_, int bmax_ = 4,
                          int landmarks_ = 0, int anchor_pilot_size_ = 64)
        : G(data), Q(query), tau(tau_), num_g(data.n), num_q(query.n), dim(data.dim),
          bmax(bmax_), ivf_cells_req(landmarks_), anchor_pilot_size(anchor_pilot_size_) {
        if (G.dim != Q.dim) throw std::runtime_error("Feature dimensions do not match.");
        if (num_q <= 0) throw std::runtime_error("Query graph is empty.");
        if (!is_connected_graph(Q)) {
            throw std::runtime_error("IVF Support-Join matching requires a connected query graph.");
        }
        if (bmax <= 0) throw std::runtime_error("bmax must be positive.");
        if (anchor_pilot_size <= 0) throw std::runtime_error("anchor pilot size must be positive.");
        if (ivf_cells_req < 0) ivf_cells_req = 0;

        build_progressive_similarity_blocks();
        build_semantic_candidates_by_progressive_filter();
        build_query_edge_support();
        arc_consistency();
        build_compact_support_bitsets();
        terminal_domain_scratch.assign(num_q, {});
        terminal_word_scratch.assign(num_q, {});
        domain_scratch.assign(num_q, {});
        local_mask_scratch.assign(num_q, {});
        for (int u = 0; u < num_q; ++u) {
            terminal_domain_scratch[u].reserve(active_words[u]);
            terminal_word_scratch[u].reserve(active_words[u]);
            domain_scratch[u].reserve(active_words[u]);
            local_mask_scratch[u].reserve(active_words[u]);
        }
        release_raw_support_lists();
        union_seen_stamp.assign(num_g, 0);
        choose_anchor_after_index();
    }

    inline size_t qv_index(int q, int v) const {
        return size_t(q) * size_t(num_g) + size_t(v);
    }
    inline int qedge_index(int u, int w) const {
        return u * num_q + w;
    }

    float exact_sim(int u, int v) {
        stats.ivf_exact_similarity_evals++;
        return dot_product(Q.x[u], G.x[v]);
    }

    static inline float clamp_unit(float x) {
        if (x > 1.0f) return 1.0f;
        if (x < -1.0f) return -1.0f;
        return x;
    }

    static inline float angle_from_dot(float x) {
        return std::acos(clamp_unit(x));
    }


    void build_progressive_similarity_blocks() {
        theta_tau = angle_from_dot(tau);
        sim_blocks = std::min(8, std::max(1, dim));
        sim_block_start.assign(sim_blocks, 0);
        sim_block_end.assign(sim_blocks, 0);
        for (int b = 0; b < sim_blocks; ++b) {
            sim_block_start[b] = int((int64_t(b) * dim) / sim_blocks);
            sim_block_end[b] = int((int64_t(b + 1) * dim) / sim_blocks);
        }

        q_rem_norm.assign(num_q, vector<float>(sim_blocks + 1, 0.0f));
        for (int u = 0; u < num_q; ++u) {
            q_rem_norm[u][sim_blocks] = 0.0f;
            for (int b = sim_blocks - 1; b >= 0; --b) {
                double ss = 0.0;
                for (int j = sim_block_start[b]; j < sim_block_end[b]; ++j) {
                    ss += double(Q.x[u][j]) * double(Q.x[u][j]);
                }
                double prev = double(q_rem_norm[u][b + 1]);
                q_rem_norm[u][b] = float(std::sqrt(ss + prev * prev));
            }
        }

        g_rem_norm.assign(num_g, vector<float>(sim_blocks + 1, 0.0f));
        for (int v = 0; v < num_g; ++v) {
            g_rem_norm[v][sim_blocks] = 0.0f;
            for (int b = sim_blocks - 1; b >= 0; --b) {
                double ss = 0.0;
                for (int j = sim_block_start[b]; j < sim_block_end[b]; ++j) {
                    ss += double(G.x[v][j]) * double(G.x[v][j]);
                }
                double prev = double(g_rem_norm[v][b + 1]);
                g_rem_norm[v][b] = float(std::sqrt(ss + prev * prev));
            }
        }
        stats.ivf_cells = uint64_t(sim_blocks);
    }

    inline bool progressive_similarity_pass(int u, int v, float* final_score = nullptr) {
        stats.progressive_pair_tests++;
        double s = 0.0;
        for (int b = 0; b < sim_blocks; ++b) {
            const int lo = sim_block_start[b];
            const int hi = sim_block_end[b];
            double part = 0.0;
            const auto& qx = Q.x[u];
            const auto& gx = G.x[v];
            for (int j = lo; j < hi; ++j) part += double(qx[j]) * double(gx[j]);
            s += part;
            stats.progressive_blocks_scanned++;

            if (b + 1 < sim_blocks) {
                double ub = s + double(q_rem_norm[u][b + 1]) * double(g_rem_norm[v][b + 1]);
                if (ub < double(tau)) {
                    stats.progressive_early_rejects++;
                    if (final_score) *final_score = float(s);
                    return false;
                }
            }
        }
        stats.progressive_full_evals++;
        if (final_score) *final_score = float(s);
        return s >= double(tau);
    }

    void build_semantic_candidates_by_progressive_filter() {
        sem_cands.assign(num_q, {});
        sem_member.assign(size_t(num_q) * size_t(num_g), 0);
        sem_shell.assign(size_t(num_q) * size_t(num_g), -1);
        const int shell_count = int(stats.angular_shells);
        sem_shell_cands.assign(num_q, vector<vector<int>>(shell_count));

        for (int u = 0; u < num_q; ++u) {
            auto& out = sem_cands[u];
            out.reserve(1024);
            vector<vector<int>> shells(shell_count);

            for (int v = 0; v < num_g; ++v) {
                stats.ivf_points_scanned++;
                if (G.deg[v] < Q.deg[u]) continue;

                float score = 0.0f;
                if (!progressive_similarity_pass(u, v, &score)) continue;

                // The progressive filter computes the exact full inner product for
                // all non-rejected pairs.  Accepted pairs therefore satisfy the
                // original predicate x_u^T x_v >= tau.  Shells are used only as a
                // stable in-cap ordering for downstream processing.
                int sid = 0;
                if (tau < 1.0f - 1e-9f) {
                    double pos = (double(score) - double(tau)) / std::max(1e-12, 1.0 - double(tau));
                    sid = int(pos * double(shell_count));
                    if (sid < 0) sid = 0;
                    if (sid >= shell_count) sid = shell_count - 1;
                }
                shells[sid].push_back(v);
                stats.angular_cap_certified++;
            }

            for (int sid = 0; sid < shell_count; ++sid) {
                if (!shells[sid].empty()) stats.angular_shell_nonempty++;
                std::sort(shells[sid].begin(), shells[sid].end());
                for (int v : shells[sid]) {
                    out.push_back(v);
                    sem_member[qv_index(u, v)] = 1;
                    sem_shell[qv_index(u, v)] = sid;
                    sem_shell_cands[u][sid].push_back(v);
                    stats.angular_shell_entries++;
                }
            }
            std::sort(out.begin(), out.end());
            stats.semantic_candidates_total += out.size();
        }
    }

    // Query-adaptive angular range index. The first |V_q| landmarks are exactly
    // the query vectors. For query vertex u, the landmark x_u gives the exact
    // angular range [0, arccos(tau)] before full cosine verification; additional
    // landmarks add safe triangle-inequality filters.
    void build_angular_range_index() {
        theta_tau = angle_from_dot(tau);
        int L = (ivf_cells_req <= 0) ? num_q : std::max(num_q, ivf_cells_req);
        L = std::max(1, L);
        vector<vector<float>> landmarks;
        landmarks.reserve(L);
        for (int u = 0; u < num_q && int(landmarks.size()) < L; ++u) landmarks.push_back(Q.x[u]);
        for (int i = 0; int(landmarks.size()) < L && i < num_g; ++i) {
            int v = int((uint64_t(i) * 2654435761ull) % uint64_t(std::max(1, num_g)));
            landmarks.push_back(G.x[v]);
        }

        angular_index.assign(landmarks.size(), AngularLandmarkIndex{});
        for (int l = 0; l < int(landmarks.size()); ++l) {
            auto& idx = angular_index[l];
            idx.landmark_vec = std::move(landmarks[l]);
            idx.data_angle.resize(num_g);
            idx.sorted_by_angle.reserve(num_g);
            for (int v = 0; v < num_g; ++v) {
                float a = angle_from_dot(dot_product(idx.landmark_vec, G.x[v]));
                idx.data_angle[v] = a;
                idx.sorted_by_angle.emplace_back(a, v);
            }
            std::sort(idx.sorted_by_angle.begin(), idx.sorted_by_angle.end(),
                      [](const auto& a, const auto& b) {
                          if (a.first != b.first) return a.first < b.first;
                          return a.second < b.second;
                      });
        }

        query_angle.assign(num_q, vector<float>(angular_index.size(), 0.0f));
        for (int u = 0; u < num_q; ++u) {
            for (int l = 0; l < int(angular_index.size()); ++l) {
                query_angle[u][l] = angle_from_dot(dot_product(Q.x[u], angular_index[l].landmark_vec));
            }
        }
        stats.ivf_cells = angular_index.size();
    }

    void build_semantic_candidates_by_angular_range() {
        sem_cands.assign(num_q, {});
        sem_member.assign(size_t(num_q) * size_t(num_g), 0);
        sem_shell.assign(size_t(num_q) * size_t(num_g), -1);
        const int shell_count = int(stats.angular_shells);
        sem_shell_cands.assign(num_q, vector<vector<int>>(shell_count));

        // Angular-cap first, IVF-like shells second.
        // The first |V_q| landmarks are exactly query vectors.  For query u,
        // landmark u gives an exact spherical-cap test: angle(x_u,x_v) <= theta_tau
        // iff x_u^T x_v >= tau.  Thus vertices in this interval are semantically
        // certified without an additional dot-product verification.  We then place
        // the certified vertices into angular shells inside the cap.  These shells
        // are query-adaptive IVF-like cells, but they are built only inside the exact
        // angular range rather than by global k-means/IVF partitioning.
        for (int u = 0; u < num_q; ++u) {
            auto& out = sem_cands[u];
            out.reserve(1024);

            // Still evaluate landmark intervals for profiling and for the optional
            // multi-landmark necessary filters, but use the query-vector cap as the
            // primary exact access path rather than a global IVF cell assignment.
            vector<std::pair<size_t,size_t>> intervals(angular_index.size());
            for (int l = 0; l < int(angular_index.size()); ++l) {
                stats.ivf_cell_tests++;
                const auto& arr = angular_index[l].sorted_by_angle;
                float qa = query_angle[u][l];
                float lo_val = std::max(0.0f, qa - theta_tau);
                float hi_val = std::min(float(M_PI), qa + theta_tau);
                auto lo_it = std::lower_bound(arr.begin(), arr.end(), std::make_pair(lo_val, -1),
                    [](const auto& a, const auto& b){
                        if (a.first != b.first) return a.first < b.first;
                        return a.second < b.second;
                    });
                auto hi_it = std::upper_bound(arr.begin(), arr.end(), std::make_pair(hi_val, std::numeric_limits<int>::max()),
                    [](const auto& a, const auto& b){
                        if (a.first != b.first) return a.first < b.first;
                        return a.second < b.second;
                    });
                if (hi_it == lo_it) stats.ivf_cell_prunes++;
                intervals[l] = {size_t(lo_it - arr.begin()), size_t(hi_it - arr.begin())};
            }

            // Primary angular range: exact cap around x_u.
            int primary_l = u; // guaranteed by build_angular_range_index()
            const auto& primary = angular_index[primary_l].sorted_by_angle;
            size_t lo = intervals[primary_l].first;
            size_t hi = intervals[primary_l].second;

            vector<vector<int>> shells(shell_count);
            for (size_t ii = lo; ii < hi; ++ii) {
                int v = primary[ii].second;
                float av = primary[ii].first;
                stats.ivf_points_scanned++;
                if (G.deg[v] < Q.deg[u]) continue;

                // Additional landmarks are safe necessary filters.  They are not IVF
                // cells; they refine the exact cap boundary using triangle inequality.
                bool cone_ok = true;
                for (int l = 0; l < int(angular_index.size()); ++l) {
                    if (l == primary_l) continue;
                    if (std::fabs(query_angle[u][l] - angular_index[l].data_angle[v]) > theta_tau + 1e-6f) {
                        cone_ok = false;
                        break;
                    }
                }
                if (!cone_ok) continue;

                // The primary angular interval is used as a tight access path, but we still
                // verify the original dot-product predicate to keep the algorithm exactly
                // identical to the problem definition.  This protects against acos/float
                // boundary effects and any non-perfect normalization in the input vectors.
                // Interior points of the primary query-vector cap are certified by
                // the angular access path.  Only boundary points are verified by
                // the original dot-product predicate, preserving exactness under
                // floating-point error while avoiding redundant similarity checks.
                constexpr float ANG_BOUNDARY_EPS = 1e-6f;
                if (av > theta_tau - ANG_BOUNDARY_EPS) {
                    stats.ivf_exact_similarity_evals++;
                    if (dot_product(Q.x[u], G.x[v]) < tau) continue;
                }

                int sid = 0;
                if (theta_tau > 1e-9f) {
                    sid = int((double(av) / double(theta_tau)) * double(shell_count));
                    if (sid < 0) sid = 0;
                    if (sid >= shell_count) sid = shell_count - 1;
                }
                shells[sid].push_back(v);
                stats.angular_cap_certified++;
            }

            // IVF-like shell output order: tight cap center first.  This improves
            // downstream anchor/order decisions without changing the exact candidate set.
            for (int sid = 0; sid < shell_count; ++sid) {
                if (!shells[sid].empty()) stats.angular_shell_nonempty++;
                std::sort(shells[sid].begin(), shells[sid].end());
                for (int v : shells[sid]) {
                    out.push_back(v);
                    sem_member[qv_index(u, v)] = 1;
                    sem_shell[qv_index(u, v)] = sid;
                    sem_shell_cands[u][sid].push_back(v);
                    stats.angular_shell_entries++;
                }
            }
            std::sort(out.begin(), out.end());
            stats.semantic_candidates_total += out.size();
        }
    }

    void build_query_edge_support() {
        edge_id.assign(num_q * num_q, -1);
        dir_qedges.clear();
        for (int u = 0; u < num_q; ++u) {
            for (int w : Q.adj[u]) {
                int id = int(dir_qedges.size());
                edge_id[qedge_index(u, w)] = id;
                dir_qedges.emplace_back(u, w);
            }
        }

        support_row_id.assign(dir_qedges.size(), vector<int>(num_g, -1));
        support_sources.assign(dir_qedges.size(), {});
        support_lists.assign(dir_qedges.size(), {});

        auto add_support_edge = [&](int eid, int from, int to) {
            int rid = support_row_id[eid][from];
            if (rid < 0) {
                rid = int(support_lists[eid].size());
                support_row_id[eid][from] = rid;
                support_sources[eid].push_back(from);
                support_lists[eid].push_back({});
            }
            support_lists[eid][rid].push_back(to);
        };

        // QSubspace halfspace-code edge retrieval.
        // For each touched data vertex v, the semantic relation to the whole query
        // is encoded as the exact query-score halfspace code
        //     M(v) = {u : x_u^T x_v >= tau and v passes the candidate filters of u}.
        // This code is filled from the already verified angular-cap candidate lists,
        // so no additional score dot products are required.  A directed data edge
        // (a,b) belongs to a unique product code (M(a), M(b)).  A directed query edge
        // (u,w) retrieves exactly those buckets whose product code contains u on the
        // source side and w on the target side.  This is an exact vector-range index
        // over the query-score subspace, without the heavy range-tree node bounds.
        stats.qsubspace_rank = uint64_t(num_q);

        vector<uint8_t> touched(num_g, 0);
        vector<uint64_t> qmask(num_g, 0ull);
        vector<int> touched_vertices;
        touched_vertices.reserve(stats.semantic_candidates_total);

        if (num_q <= 63) {
            for (int u = 0; u < num_q; ++u) {
                uint64_t bit = (1ull << u);
                for (int v : sem_cands[u]) {
                    qmask[v] |= bit;
                    if (!touched[v]) {
                        touched[v] = 1;
                        touched_vertices.push_back(v);
                    }
                }
            }
        } else {
            // Very large queries cannot be represented in a single 64-bit code.
            // Keep the old membership-safe construction path by using sem_member
            // checks below.  Typical semantic-subgraph queries are small; this branch
            // is only a correctness fallback and introduces no user-visible parameter.
            for (int u = 0; u < num_q; ++u) {
                for (int v : sem_cands[u]) {
                    if (!touched[v]) {
                        touched[v] = 1;
                        touched_vertices.push_back(v);
                    }
                }
            }
        }

        struct DEdge { int a; int b; };
        uint64_t touched_edge_scans = 0;
        uint64_t skipped_untouched_dst = 0;
        uint64_t induced_edges = 0;

        const bool use_code_table = (num_q > 0 && num_q <= 12);
        if (use_code_table) {
            const int total_bits = 2 * num_q;
            const size_t table_size = size_t(1) << total_bits;
            vector<vector<DEdge>> buckets(table_size);
            vector<uint8_t> bucket_nonempty(table_size, 0);
            vector<size_t> nonempty_codes;
            nonempty_codes.reserve(1024);

            // Build the query-subspace halfspace-code edge-object index.  This is
            // still an edge-level vector index: each directed edge object is keyed by
            // the pair of endpoint query-halfspace codes (M(a),M(b)).  The scan below
            // is the one-time index construction pass, not query-edge neighborhood
            // expansion.
            for (int a : touched_vertices) {
                touched_edge_scans += uint64_t(G.deg[a]);
                uint64_t ma = qmask[a];
                if (ma == 0) continue;
                for (int b : G.adj[a]) {
                    if (!touched[b]) { skipped_untouched_dst++; continue; }
                    uint64_t mb = qmask[b];
                    if (mb == 0) continue;
                    uint64_t code = ma | (mb << num_q);
                    size_t ci = size_t(code);
                    if (!bucket_nonempty[ci]) {
                        bucket_nonempty[ci] = 1;
                        nonempty_codes.push_back(ci);
                    }
                    buckets[ci].push_back({a, b});
                    induced_edges++;
                }
            }

            // Bucket-driven direct support-row materialization.  Each halfspace-code
            // bucket is tested once.  If its endpoint code supports one or more directed
            // query edges, the whole edge-object bucket is streamed directly into the
            // corresponding sparse support rows.  This avoids the BatchRow variant's
            // intermediate qe_edges[eid] buffers and their per-query-edge sorting pass.
            // Since every directed data edge belongs to exactly one product code bucket,
            // and each code bucket is routed at most once to a directed query edge, this
            // produces the same exact support relation without duplicate generation.
            uint64_t code_bucket_tests = 0;
            uint64_t routed_code_buckets = 0;
            uint64_t routed_bucket_copies = 0;

            for (size_t code : nonempty_codes) {
                code_bucket_tests++;
                uint64_t src_mask = uint64_t(code) & ((1ull << num_q) - 1ull);
                uint64_t dst_mask = uint64_t(code) >> num_q;
                uint64_t qe_mask = 0ull;
                for (int eid = 0; eid < int(dir_qedges.size()); ++eid) {
                    int u = dir_qedges[eid].first;
                    int w = dir_qedges[eid].second;
                    if ((src_mask & (1ull << u)) && (dst_mask & (1ull << w))) {
                        qe_mask |= (1ull << eid);
                    }
                }
                if (qe_mask == 0) continue;
                routed_code_buckets++;
                const auto& bucket = buckets[code];
                while (qe_mask) {
                    int eid = __builtin_ctzll(qe_mask);
                    qe_mask &= (qe_mask - 1ull);
                    // Endpoint candidate validity is implied by the endpoint codes,
                    // which are filled only from sem_cands.  The support row stores raw
                    // data ids because AC still runs before active-local bitsets exist.
                    for (const DEdge& e : bucket) add_support_edge(eid, e.a, e.b);
                    routed_bucket_copies++;
                    stats.maskpair_query_edge_hits += bucket.size();
                    stats.maskpair_support_generated += bucket.size();
                    stats.edge_range_support_generated += bucket.size();
                    stats.shell_edge_bucket_entries += bucket.size();
                    stats.shell_edge_bucket_support_generated += bucket.size();
                }
            }

            stats.edge_range_node_tests = code_bucket_tests;
            stats.edge_range_node_full_accepts = routed_code_buckets;
            stats.edge_range_leaf_tests = stats.edge_range_support_generated;
            stats.edge_range_node_prunes = code_bucket_tests - routed_code_buckets;
            stats.shell_edge_bucket_nonempty = routed_bucket_copies;
        } else {
            // Correctness fallback for larger query sizes: scan each induced edge once
            // and enumerate compatible query-edge bits by sem_member.  This branch is
            // not expected for the small-query workloads used in the experiments.
            for (int a : touched_vertices) {
                touched_edge_scans += uint64_t(G.deg[a]);
                for (int b : G.adj[a]) {
                    if (!touched[b]) { skipped_untouched_dst++; continue; }
                    induced_edges++;
                    for (int eid = 0; eid < int(dir_qedges.size()); ++eid) {
                        int u = dir_qedges[eid].first;
                        int w = dir_qedges[eid].second;
                        stats.edge_range_leaf_tests++;
                        if (!sem_member[qv_index(u, a)] || !sem_member[qv_index(w, b)]) continue;
                        add_support_edge(eid, a, b);
                        stats.maskpair_query_edge_hits++;
                        stats.maskpair_support_generated++;
                        stats.edge_range_support_generated++;
                        stats.shell_edge_bucket_entries++;
                        stats.shell_edge_bucket_support_generated++;
                    }
                }
            }
        }

        stats.edge_range_directed_edges = induced_edges;
        stats.shell_edge_bucket_directed_edges = touched_edge_scans;
        stats.maskpair_edge_scans = touched_edge_scans;
        stats.maskpair_edge_skips_untouched = skipped_untouched_dst;
        stats.qscore_mask_vertices = touched_vertices.size();
        stats.qscore_mask_score_evals = 0;
        stats.qsubspace_score_evals = 0;
        stats.edge_range_nodes = 0;
        stats.shell_edge_bucket_tests = stats.edge_range_node_tests + stats.edge_range_leaf_tests;
        stats.shell_edge_bucket_nonempty = stats.edge_range_node_full_accepts;
        stats.shell_edge_bucket_pair_hits = stats.edge_range_support_generated;
        stats.ivf_shell_pair_blocks = 0;
        stats.ivf_shell_empty_blocks = 0;
        stats.ivf_shell_source_blocks = 0;
        stats.ivf_shell_target_blocks = 0;
        stats.ivf_shell_candidates_scanned = 0;

        // The code-bucket path routes every directed edge object at most once to a
        // directed query edge, so support rows do not require a separate sort/unique
        // materialization pass.  This keeps the vector-retrieval output in direct row
        // form for AC and compact-bitset construction.
        stats.support_edges_total = 0;
        for (int eid = 0; eid < int(dir_qedges.size()); ++eid) {
            for (const auto& list : support_lists[eid]) stats.support_edges_total += list.size();
        }
    }



    bool has_active_support(int eid, int target_q, int data_v) {
        stats.ac_support_tests++;
        if (eid < 0) return false;
        int rid = support_row_id[eid][data_v];
        if (rid < 0) return false;
        const auto& list = support_lists[eid][rid];
        for (int z : list) {
            if (sem_member[qv_index(target_q, z)]) return true;
        }
        return false;
    }

    void arc_consistency() {
        bool changed = true;
        while (changed) {
            changed = false;
            stats.ac_rounds++;
            for (int eid = 0; eid < int(dir_qedges.size()); ++eid) {
                int u = dir_qedges[eid].first;
                int w = dir_qedges[eid].second;
                auto& cand = sem_cands[u];
                for (int v : cand) {
                    if (!sem_member[qv_index(u, v)]) continue;
                    if (!has_active_support(eid, w, v)) {
                        sem_member[qv_index(u, v)] = 0;
                        stats.ac_vertices_removed++;
                        changed = true;
                    }
                }
            }
        }
        for (int u = 0; u < num_q; ++u) {
            vector<int> kept;
            kept.reserve(sem_cands[u].size());
            for (int v : sem_cands[u]) {
                if (sem_member[qv_index(u, v)]) kept.push_back(v);
            }
            sem_cands[u].swap(kept);
            stats.active_candidates_total += sem_cands[u].size();
        }
    }


    void build_compact_support_bitsets() {
        active_id.assign(num_q, vector<int>(num_g, -1));
        active_words.assign(num_q, 0);
        for (int u = 0; u < num_q; ++u) {
            for (int i = 0; i < int(sem_cands[u].size()); ++i) {
                active_id[u][sem_cands[u][i]] = i;
            }
            active_words[u] = (int(sem_cands[u].size()) + 63) >> 6;
            stats.compact_active_total += sem_cands[u].size();
        }

        // Active-local support materialization.  The previous compact executor
        // still indexed support rows by the original data vertex id.  Here every
        // source row is addressed by the AC-local id of the source candidate, and
        // every target bit is an AC-local target id.  No extra parameter is used;
        // the representation is determined entirely by the AC survivor sets.
        support_bits_row_id.clear();
        support_bits_active_row_id.assign(dir_qedges.size(), {});
        support_bits_rows.assign(dir_qedges.size(), {});
        support_bits_row_words.assign(dir_qedges.size(), {});
        support_bits_row_popcnt.assign(dir_qedges.size(), {});
        for (int eid = 0; eid < int(dir_qedges.size()); ++eid) {
            int u = dir_qedges[eid].first;
            int w = dir_qedges[eid].second;
            int U = int(sem_cands[u].size());
            int W = active_words[w];
            support_bits_active_row_id[eid].assign(U, -1);
            if (U == 0 || W == 0) continue;
            support_bits_rows[eid].reserve(support_lists[eid].size());
            support_bits_row_words[eid].reserve(support_lists[eid].size());
            support_bits_row_popcnt[eid].reserve(support_lists[eid].size());
            for (int rid = 0; rid < int(support_lists[eid].size()); ++rid) {
                const int from = support_sources[eid][rid];
                int from_id = active_id[u][from];
                if (from_id < 0) continue;
                const auto& lst = support_lists[eid][rid];
                if (lst.empty()) continue;
                vector<uint64_t> bits(W, 0ull);
                for (int z : lst) {
                    int id = active_id[w][z];
                    if (id >= 0) bits[id >> 6] |= (1ull << (id & 63));
                }
                int pcnt = 0;
                vector<int> nz_words;
                nz_words.reserve(8);
                for (int wi = 0; wi < W; ++wi) {
                    uint64_t x = bits[wi];
                    if (x) {
                        nz_words.push_back(wi);
                        pcnt += __builtin_popcountll(x);
                    }
                }
                if (pcnt > 0) {
                    int brid = int(support_bits_rows[eid].size());
                    support_bits_active_row_id[eid][from_id] = brid;
                    support_bits_rows[eid].push_back(std::move(bits));
                    support_bits_row_words[eid].push_back(std::move(nz_words));
                    support_bits_row_popcnt[eid].push_back(pcnt);
                    stats.compact_bitset_edges++;
                    stats.compact_bitset_words += W;
                    stats.active_local_rows++;
                }
            }
        }
    }

    void release_raw_support_lists() {
        // After AC and compact bitset construction, the raw target lists are no
        // longer required: ordering, support membership, anchor filtering, and
        // domain-size estimation all use sparse bitsets.  Releasing them lowers
        // the steady-state RSS while preserving exactness.
        vector<vector<vector<int>>>().swap(support_lists);
        vector<vector<int>>().swap(support_sources);
        vector<vector<int>>().swap(support_row_id);
    }

    inline int active_local_row(int eid, int source_q, int from_data) {
        stats.active_local_row_lookups++;
        if (eid < 0) { stats.active_local_empty_lookups++; return -1; }
        int from_id = active_id[source_q][from_data];
        if (from_id < 0) { stats.active_local_empty_lookups++; return -1; }
        const auto& rows = support_bits_active_row_id[eid];
        if (from_id >= int(rows.size())) { stats.active_local_empty_lookups++; return -1; }
        int brid = rows[from_id];
        if (brid < 0) stats.active_local_empty_lookups++;
        return brid;
    }

    inline bool compact_support_contains(int eid, int from_data, int to_data) {
        stats.support_membership_tests++;
        if (eid < 0) return false;
        int source_q = dir_qedges[eid].first;
        int target_q = dir_qedges[eid].second;
        int id = active_id[target_q][to_data];
        if (id < 0) return false;
        int brid = active_local_row(eid, source_q, from_data);
        if (brid < 0) return false;
        const auto& bits = support_bits_rows[eid][brid];
        return (bits[id >> 6] >> (id & 63)) & 1ull;
    }

    inline int pop_lsb_index(uint64_t& x) const {
        int b = __builtin_ctzll(x);
        x &= x - 1;
        return b;
    }

    void choose_anchor_after_index() {
        int best = 0;
        std::tuple<int,int,int> best_key(int(sem_cands[0].size()), -Q.deg[0], 0);
        for (int u = 1; u < num_q; ++u) {
            auto key = std::make_tuple(int(sem_cands[u].size()), -Q.deg[u], u);
            if (key < best_key) {
                best_key = key;
                best = u;
            }
        }
        anchor_q = best;
        cout << "[AngularShell-IVF Anchor] chosen anchor = q" << best
             << ", deg=" << Q.deg[best]
             << ", active_semantic_candidates=" << sem_cands[best].size() << "\n";
    }

    int residual_degree(int u, const vector<int>& mapping) const {
        int c = 0;
        for (int w : Q.adj[u]) if (mapping[w] == -1) ++c;
        return c;
    }

    int matched_boundary(int u, const vector<int>& mapping) const {
        int c = 0;
        for (int w : Q.adj[u]) if (mapping[w] != -1) ++c;
        return c;
    }

    inline int get_eid(int u, int w) const {
        return edge_id[qedge_index(u, w)];
    }

    inline bool support_contains(int eid, int from_data, int to_data) {
        return compact_support_contains(eid, from_data, to_data);
    }

    inline int active_row_id_no_stats(int eid, int from_data) const {
        if (eid < 0) return -1;
        int source_q = dir_qedges[eid].first;
        int from_id = active_id[source_q][from_data];
        if (from_id < 0 || from_id >= int(support_bits_active_row_id[eid].size())) return -1;
        return support_bits_active_row_id[eid][from_id];
    }

    int active_support_size(int eid, int target_q, int from_data) const {
        (void)target_q;
        int brid = active_row_id_no_stats(eid, from_data);
        if (brid < 0) return 0;
        return support_bits_row_popcnt[eid][brid];
    }

    bool has_active_support_no_stats(int eid, int target_q, int from_data) const {
        (void)target_q;
        int brid = active_row_id_no_stats(eid, from_data);
        return brid >= 0 && support_bits_row_popcnt[eid][brid] > 0;
    }

    bool has_anchor_neighbor_support(int q_neighbor, int anchor_image) const {
        int eid = get_eid(anchor_q, q_neighbor);
        return has_active_support_no_stats(eid, q_neighbor, anchor_image);
    }

    int choose_restrictive_anchor_neighbor(const vector<int>& anchor_cands) {
        if (Q.adj[anchor_q].empty() || anchor_cands.empty()) return -1;
        const int sample_n = std::min<int>(anchor_pilot_size, anchor_cands.size());
        int best_q = -1;
        int best_pass = std::numeric_limits<int>::max();
        for (int qr : Q.adj[anchor_q]) {
            int pass = 0;
            for (int i = 0; i < sample_n; ++i) {
                size_t pos = (sample_n == 1) ? 0 :
                    size_t((uint64_t(i) * uint64_t(anchor_cands.size() - 1)) / uint64_t(sample_n - 1));
                if (has_anchor_neighbor_support(qr, anchor_cands[pos])) ++pass;
            }
            if (best_q == -1 || std::make_tuple(pass, -Q.deg[qr], qr) <
                                std::make_tuple(best_pass, -Q.deg[best_q], best_q)) {
                best_pass = pass;
                best_q = qr;
            }
        }
        stats.anchor_restrictive_query = best_q;
        return best_q;
    }

    vector<int> restrictive_anchor_filter(const vector<int>& anchor_cands, int qr) {
        if (qr < 0) return anchor_cands;
        vector<int> kept;
        kept.reserve(anchor_cands.size());
        for (int va : anchor_cands) {
            if (has_anchor_neighbor_support(qr, va)) kept.push_back(va);
            else stats.anchor_filter_pruned++;
        }
        return kept;
    }

    vector<int> initial_anchor_candidates() {
        vector<int> cands = sem_cands[anchor_q];
        std::sort(cands.begin(), cands.end(), [&](int a, int b) {
            if (G.deg[a] != G.deg[b]) return G.deg[a] < G.deg[b];
            return a < b;
        });
        stats.anchor_candidates_before_filter = cands.size();
        int qr = choose_restrictive_anchor_neighbor(cands);
        cands = restrictive_anchor_filter(cands, qr);
        stats.anchor_candidates = cands.size();
        return cands;
    }

    int estimate_domain_from_pivot(int p, int u, int gp) const {
        int eid = get_eid(p, u);
        if (eid < 0) return 0;
        return active_support_size(eid, u, gp);
    }

    int select_pivot(const vector<int>& mapping) {
        int best = -1;
        int best_est = std::numeric_limits<int>::max();
        int best_batch = -1;
        int best_residual_edges = -1;
        int best_scan = std::numeric_limits<int>::max();

        for (int p = 0; p < num_q; ++p) {
            int gp = mapping[p];
            if (gp == -1) continue;
            vector<int> children;
            int residual_edges = 0;
            for (int u : Q.adj[p]) {
                if (mapping[u] == -1) {
                    children.push_back(u);
                    residual_edges += residual_degree(u, mapping);
                }
            }
            if (children.empty()) continue;
            std::sort(children.begin(), children.end(), [&](int a, int b) {
                int ea = estimate_domain_from_pivot(p, a, gp);
                int eb = estimate_domain_from_pivot(p, b, gp);
                return std::make_tuple(ea, -matched_boundary(a, mapping), -Q.deg[a], a) <
                       std::make_tuple(eb, -matched_boundary(b, mapping), -Q.deg[b], b);
            });
            if (int(children.size()) > bmax) children.resize(bmax);
            int est = 0;
            for (int u : children) est += estimate_domain_from_pivot(p, u, gp);
            int batch_size = int(children.size());
            int scan = G.deg[gp];
            auto key = std::make_tuple(est, scan, -batch_size, -residual_edges, p);
            auto best_key = std::make_tuple(best_est, best_scan, -best_batch, -best_residual_edges, best);
            if (best == -1 || key < best_key) {
                best = p;
                best_est = est;
                best_batch = batch_size;
                best_residual_edges = residual_edges;
                best_scan = scan;
            }
        }
        return best;
    }

    vector<int> build_pivot_batch(int pstar, const vector<int>& mapping) const {
        vector<int> B;
        int gp = mapping[pstar];
        for (int u : Q.adj[pstar]) if (mapping[u] == -1) B.push_back(u);
        std::sort(B.begin(), B.end(), [&](int a, int b) {
            int ea = estimate_domain_from_pivot(pstar, a, gp);
            int eb = estimate_domain_from_pivot(pstar, b, gp);
            return std::make_tuple(ea, -matched_boundary(a, mapping), -Q.deg[a], a) <
                   std::make_tuple(eb, -matched_boundary(b, mapping), -Q.deg[b], b);
        });
        if (int(B.size()) > bmax) B.resize(bmax);
        return B;
    }

    void build_support_join_candidates(int pstar,
                                       const vector<int>& B,
                                       const vector<int>& mapping,
                                       const vector<uint8_t>& used_g,
                                       vector<vector<int>>& C) {
        for (int u : B) C[u].clear();
        int gp = mapping[pstar];

        for (int u : B) {
            stats.support_domain_builds++;
            stats.compact_domain_builds++;
            int eid0 = get_eid(pstar, u);
            if (eid0 < 0) continue;
            int source0_q = dir_qedges[eid0].first;
            int gp_id = active_id[source0_q][gp];
            int brid0 = (gp_id >= 0 && gp_id < int(support_bits_active_row_id[eid0].size())) ? support_bits_active_row_id[eid0][gp_id] : -1;
            int W = active_words[u];
            if (W == 0 || brid0 < 0) {
                stats.domain_empty_prunes++;
                continue;
            }
            const auto& base_bits = support_bits_rows[eid0][brid0];
            if (base_bits.empty()) {
                stats.domain_empty_prunes++;
                continue;
            }

            vector<uint64_t>& dom = domain_scratch[u];
            dom.assign(base_bits.begin(), base_bits.end());
            stats.compact_word_ands += dom.size();

            for (int p : Q.adj[u]) {
                int z = mapping[p];
                if (z == -1 || p == pstar) continue;
                int eid = get_eid(p, u);
                int source_q = dir_qedges[eid].first;
                int z_id = active_id[source_q][z];
                int brid = (z_id >= 0 && z_id < int(support_bits_active_row_id[eid].size())) ? support_bits_active_row_id[eid][z_id] : -1;
                if (brid < 0) {
                    std::fill(dom.begin(), dom.end(), 0ull);
                    break;
                }
                const auto& sb = support_bits_rows[eid][brid];
                bool any = false;
                for (int i = 0; i < W; ++i) {
                    dom[i] &= sb[i];
                    any |= (dom[i] != 0ull);
                }
                stats.compact_word_ands += W;
                if (!any) break;
            }

            const auto& vals = sem_cands[u];
            for (int wi = 0; wi < W; ++wi) {
                uint64_t word = dom[wi];
                while (word) {
                    int b = pop_lsb_index(word);
                    int id = (wi << 6) + b;
                    if (id >= int(vals.size())) continue;
                    int v = vals[id];
                    stats.compact_bits_enumerated++;
                    // Keep Algorithm 3 output/fairness aligned with Algorithm 2:
                    // non-terminal candidate domains must exclude globally mapped
                    // data vertices before batch enumeration.
                    if (used_g[v]) continue;
                    C[u].push_back(v);
                }
            }
            stats.support_base_entries_scanned += C[u].size();
            if (C[u].empty()) stats.domain_empty_prunes++;
        }
    }


    bool build_terminal_domain_bits(int pstar,
                                    int u,
                                    const vector<int>& mapping,
                                    vector<uint64_t>& dom,
                                    vector<int>* nonzero_words = nullptr) {
        if (nonzero_words) nonzero_words->clear();
        stats.support_domain_builds++;
        stats.compact_domain_builds++;
        int gp = mapping[pstar];
        int eid0 = get_eid(pstar, u);
        if (eid0 < 0) { stats.domain_empty_prunes++; return false; }
        int source0_q = dir_qedges[eid0].first;
        int gp_id = active_id[source0_q][gp];
        int brid0 = (gp_id >= 0 && gp_id < int(support_bits_active_row_id[eid0].size())) ? support_bits_active_row_id[eid0][gp_id] : -1;
        int W = active_words[u];
        if (W == 0 || brid0 < 0) { stats.domain_empty_prunes++; return false; }
        const auto& base_bits = support_bits_rows[eid0][brid0];
        if (base_bits.empty()) { stats.domain_empty_prunes++; return false; }

        dom.assign(base_bits.begin(), base_bits.end());
        stats.compact_word_ands += dom.size();
        for (int p : Q.adj[u]) {
            int z = mapping[p];
            if (z == -1 || p == pstar) continue;
            int eid = get_eid(p, u);
            int source_q = dir_qedges[eid].first;
            int z_id = active_id[source_q][z];
            int brid = (z_id >= 0 && z_id < int(support_bits_active_row_id[eid].size())) ? support_bits_active_row_id[eid][z_id] : -1;
            if (brid < 0) { std::fill(dom.begin(), dom.end(), 0ull); stats.domain_empty_prunes++; return false; }
            const auto& sb = support_bits_rows[eid][brid];
            for (int i = 0; i < W; ++i) dom[i] &= sb[i];
            stats.compact_word_ands += W;
        }
        for (int q = 0; q < num_q; ++q) {
            int z = mapping[q];
            if (z < 0) continue;
            int id = active_id[u][z];
            if (id >= 0 && (id >> 6) < int(dom.size())) dom[id >> 6] &= ~(1ull << (id & 63));
        }
        if (nonzero_words) {
            nonzero_words->reserve(dom.size());
            for (int i = 0; i < int(dom.size()); ++i) {
                if (dom[i]) nonzero_words->push_back(i);
            }
            if (!nonzero_words->empty()) return true;
        } else {
            for (uint64_t x : dom) if (x) return true;
        }
        stats.domain_empty_prunes++;
        return false;
    }

    bool build_terminal_domain_sparse(int pstar,
                                      int u,
                                      const vector<int>& mapping,
                                      vector<int>& words,
                                      vector<uint64_t>& vals) {
        words.clear();
        vals.clear();
        stats.support_domain_builds++;
        stats.compact_domain_builds++;

        int gp = mapping[pstar];
        int eid0 = get_eid(pstar, u);
        if (eid0 < 0) { stats.domain_empty_prunes++; return false; }
        int source0_q = dir_qedges[eid0].first;
        int gp_id = active_id[source0_q][gp];
        int brid0 = (gp_id >= 0 && gp_id < int(support_bits_active_row_id[eid0].size())) ? support_bits_active_row_id[eid0][gp_id] : -1;
        if (active_words[u] == 0 || brid0 < 0) { stats.domain_empty_prunes++; return false; }

        const auto& base_row = support_bits_rows[eid0][brid0];
        const auto& base_words = support_bits_row_words[eid0][brid0];
        if (base_words.empty()) { stats.domain_empty_prunes++; return false; }
        words.reserve(base_words.size());
        vals.reserve(base_words.size());
        for (int wi : base_words) {
            uint64_t x = base_row[wi];
            if (x) { words.push_back(wi); vals.push_back(x); }
        }
        stats.compact_word_ands += words.size();

        for (int p : Q.adj[u]) {
            int z = mapping[p];
            if (z == -1 || p == pstar) continue;
            int eid = get_eid(p, u);
            int source_q = dir_qedges[eid].first;
            int z_id = active_id[source_q][z];
            int brid = (z_id >= 0 && z_id < int(support_bits_active_row_id[eid].size())) ? support_bits_active_row_id[eid][z_id] : -1;
            if (brid < 0) { stats.domain_empty_prunes++; return false; }
            const auto& sb = support_bits_rows[eid][brid];
            int out = 0;
            const int oldn = int(words.size());
            for (int i = 0; i < oldn; ++i) {
                uint64_t x = vals[i] & sb[words[i]];
                if (x) {
                    words[out] = words[i];
                    vals[out] = x;
                    ++out;
                }
            }
            stats.compact_word_ands += oldn;
            words.resize(out);
            vals.resize(out);
            if (out == 0) { stats.domain_empty_prunes++; return false; }
        }

        for (int q = 0; q < num_q; ++q) {
            int z = mapping[q];
            if (z < 0) continue;
            int id = active_id[u][z];
            if (id < 0) continue;
            int wi = id >> 6;
            auto it = std::lower_bound(words.begin(), words.end(), wi);
            if (it == words.end() || *it != wi) continue;
            int pos = int(it - words.begin());
            vals[pos] &= ~(1ull << (id & 63));
            if (vals[pos] == 0) {
                words.erase(words.begin() + pos);
                vals.erase(vals.begin() + pos);
            }
        }
        if (words.empty()) { stats.domain_empty_prunes++; return false; }
        return true;
    }


    inline bool terminal_has_extra_mapped_neighbor(int pstar,
                                                   int u,
                                                   const vector<int>& mapping) const {
        for (int p : Q.adj[u]) {
            if (p == pstar) continue;
            if (mapping[p] != -1) return true;
        }
        return false;
    }

    bool build_terminal_base_domain_view(int pstar,
                                         int u,
                                         const vector<int>& mapping,
                                         const vector<int>*& words_view,
                                         const vector<uint64_t>*& row_view) {
        words_view = nullptr;
        row_view = nullptr;
        stats.support_domain_builds++;
        stats.compact_domain_builds++;

        int gp = mapping[pstar];
        int eid0 = get_eid(pstar, u);
        if (eid0 < 0) { stats.domain_empty_prunes++; return false; }
        int source0_q = dir_qedges[eid0].first;
        int gp_id = active_id[source0_q][gp];
        int brid0 = (gp_id >= 0 && gp_id < int(support_bits_active_row_id[eid0].size())) ? support_bits_active_row_id[eid0][gp_id] : -1;
        if (active_words[u] == 0 || brid0 < 0) { stats.domain_empty_prunes++; return false; }

        const auto& base_words = support_bits_row_words[eid0][brid0];
        if (base_words.empty()) { stats.domain_empty_prunes++; return false; }
        words_view = &base_words;
        row_view = &support_bits_rows[eid0][brid0];
        stats.compact_word_ands += base_words.size();
        return true;
    }

    template <class EmitFn>
    bool emit_terminal_pair_fused(int pstar,
                                  const vector<int>& B,
                                  vector<int>& mapping,
                                  const vector<uint8_t>& used_g,
                                  EmitFn&& emit,
                                  uint64_t max_matches) {
        if (B.size() != 2) return false;
        stats.pivot_batch_calls++;
        stats.terminal_batch_calls++;
        int u = B[0], w = B[1];
        int eid = get_eid(u, w);
        if (eid < 0) {
            int reid = get_eid(w, u);
            if (reid >= 0) { std::swap(u, w); eid = reid; }
        }

        vector<int>& Du_words_buf = terminal_word_scratch[u];
        vector<int>& Dw_words_buf = terminal_word_scratch[w];
        vector<uint64_t>& Du_vals_buf = terminal_domain_scratch[u];
        vector<uint64_t>& Dw_vals = terminal_domain_scratch[w];

        const vector<int>* Du_words_view = nullptr;
        const vector<uint64_t>* Du_row_view = nullptr;
        bool Du_base_view = !terminal_has_extra_mapped_neighbor(pstar, u, mapping);
        if (Du_base_view) {
            if (!build_terminal_base_domain_view(pstar, u, mapping, Du_words_view, Du_row_view)) return true;
        } else {
            if (!build_terminal_domain_sparse(pstar, u, mapping, Du_words_buf, Du_vals_buf)) return true;
            Du_words_view = &Du_words_buf;
        }
        if (!build_terminal_domain_sparse(pstar, w, mapping, Dw_words_buf, Dw_vals)) return true;
        const vector<int>& Du_words = *Du_words_view;
        const vector<int>& Dw_words = Dw_words_buf;
        const auto& vals_u = sem_cands[u];
        const auto& vals_w = sem_cands[w];

        if (max_matches == 0) {
            uint64_t emitted = 0;
            uint64_t bits_enum = 0;
            const int* vals_u_data = vals_u.data();
            const int* vals_w_data = vals_w.data();
            if (eid >= 0) {
                const auto& row_id = support_bits_active_row_id[eid];
                for (int di = 0; di < int(Du_words.size()); ++di) {
                    int wi = Du_words[di];
                    uint64_t word = Du_base_view ? (*Du_row_view)[wi] : Du_vals_buf[di];
                    while (word) {
                        int b = pop_lsb_index(word);
                        int uid = (wi << 6) + b;
                        int v = vals_u_data[uid];
                        if (Du_base_view && used_g[v]) continue;
                        ++bits_enum;
                        int brid = row_id[uid];
                        if (brid < 0) continue;
                        const auto& row = support_bits_rows[eid][brid];
                        mapping[u] = v;
                        for (int zj_i = 0; zj_i < int(Dw_words.size()); ++zj_i) {
                            int zj = Dw_words[zj_i];
                            uint64_t zw = Dw_vals[zj_i] & row[zj];
                            bits_enum += uint64_t(__builtin_popcountll(zw));
                            while (zw) {
                                int bb = pop_lsb_index(zw);
                                int zid = (zj << 6) + bb;
                                mapping[w] = vals_w_data[zid];
                                emit(mapping);
                                ++emitted;
                            }
                        }
                        mapping[w] = -1;
                        mapping[u] = -1;
                    }
                }
            } else {
                for (int di = 0; di < int(Du_words.size()); ++di) {
                    int wi = Du_words[di];
                    uint64_t word = Du_base_view ? (*Du_row_view)[wi] : Du_vals_buf[di];
                    while (word) {
                        int b = pop_lsb_index(word);
                        int uid = (wi << 6) + b;
                        int v = vals_u_data[uid];
                        if (Du_base_view && used_g[v]) continue;
                        ++bits_enum;
                        mapping[u] = v;
                        for (int zj_i = 0; zj_i < int(Dw_words.size()); ++zj_i) {
                            int zj = Dw_words[zj_i];
                            uint64_t zw = Dw_vals[zj_i];
                            bits_enum += uint64_t(__builtin_popcountll(zw));
                            while (zw) {
                                int bb = pop_lsb_index(zw);
                                int zid = (zj << 6) + bb;
                                int z = vals_w_data[zid];
                                if (z == v) continue;
                                mapping[w] = z;
                                emit(mapping);
                                ++emitted;
                            }
                        }
                        mapping[w] = -1;
                        mapping[u] = -1;
                    }
                }
            }
            stats.compact_bits_enumerated += bits_enum;
            stats.complete_matches += emitted;
            stats.terminal_direct_emits += emitted;
            return true;
        }

        if (eid >= 0) {
            const auto& row_id = support_bits_active_row_id[eid];
            for (int di = 0; di < int(Du_words.size()); ++di) {
                int wi = Du_words[di];
                uint64_t word = Du_base_view ? (*Du_row_view)[wi] : Du_vals_buf[di];
                while (word) {
                    int b = pop_lsb_index(word);
                    int uid = (wi << 6) + b;
                    int v = vals_u[uid];
                    if (Du_base_view && used_g[v]) continue;
                    stats.compact_bits_enumerated++;
                    int brid = row_id[uid];
                    if (brid < 0) continue;
                    const auto& row = support_bits_rows[eid][brid];
                    mapping[u] = v;
                    for (int zj_i = 0; zj_i < int(Dw_words.size()); ++zj_i) {
                        int zj = Dw_words[zj_i];
                        uint64_t zw = Dw_vals[zj_i] & row[zj];
                        while (zw) {
                            int bb = pop_lsb_index(zw);
                            int zid = (zj << 6) + bb;
                            int z = vals_w[zid];
                            stats.compact_bits_enumerated++;
                            if (z == v) continue;
                            if (stats.complete_matches >= max_matches) { mapping[u] = -1; return true; }
                            mapping[w] = z;
                            stats.complete_matches++;
                            stats.terminal_direct_emits++;
                            emit(mapping);
                            mapping[w] = -1;
                        }
                    }
                    mapping[w] = -1;
                    mapping[u] = -1;
                }
            }
            return true;
        }
        for (int di = 0; di < int(Du_words.size()); ++di) {
            int wi = Du_words[di];
            uint64_t word = Du_base_view ? (*Du_row_view)[wi] : Du_vals_buf[di];
            while (word) {
                int b = pop_lsb_index(word);
                int uid = (wi << 6) + b;
                int v = vals_u[uid];
                if (Du_base_view && used_g[v]) continue;
                stats.compact_bits_enumerated++;
                mapping[u] = v;
                for (int zj_i = 0; zj_i < int(Dw_words.size()); ++zj_i) {
                    int zj = Dw_words[zj_i];
                    uint64_t zw = Dw_vals[zj_i];
                    while (zw) {
                        int bb = pop_lsb_index(zw);
                        int zid = (zj << 6) + bb;
                        int z = vals_w[zid];
                        stats.compact_bits_enumerated++;
                        if (z == v) continue;
                        if (stats.complete_matches >= max_matches) { mapping[u] = -1; return true; }
                        mapping[w] = z;
                        stats.complete_matches++;
                        stats.terminal_direct_emits++;
                        emit(mapping);
                        mapping[w] = -1;
                    }
                }
                mapping[w] = -1;
                mapping[u] = -1;
            }
        }
        return true;
    }

    double average_degree(const vector<int>& cand) const {
        if (cand.empty()) return std::numeric_limits<double>::infinity();
        double sum = 0.0;
        for (int v : cand) sum += G.deg[v];
        return sum / double(cand.size());
    }

    void rebuild_local_candidate_mask(int u, const vector<int>& cand) {
        const int W = active_words[u];
        vector<uint64_t>& mask = local_mask_scratch[u];
        mask.assign(W, 0ull);
        for (int v : cand) {
            int id = active_id[u][v];
            if (id >= 0) mask[id >> 6] |= (1ull << (id & 63));
        }
    }

    inline bool row_intersects_candidate_mask_excluding_self(int eid,
                                                            int from_data,
                                                            int target_q,
                                                            int same_data,
                                                            const vector<uint64_t>& mask) const {
        int brid = active_row_id_no_stats(eid, from_data);
        if (brid < 0) return false;
        const auto& row = support_bits_rows[eid][brid];
        const int W = std::min<int>(row.size(), mask.size());
        int self_id = active_id[target_q][same_data];
        const int self_word = self_id >= 0 ? (self_id >> 6) : -1;
        const uint64_t self_bit = self_id >= 0 ? (1ull << (self_id & 63)) : 0ull;
        for (int i = 0; i < W; ++i) {
            uint64_t x = row[i] & mask[i];
            if (i == self_word) x &= ~self_bit;
            if (x) return true;
        }
        return false;
    }

    bool local_support_prune(const vector<int>& B, vector<vector<int>>& C) {
        bool has_internal_edge = false;
        for (size_t i = 0; i < B.size(); ++i) {
            for (size_t j = i + 1; j < B.size(); ++j) {
                if (get_eid(B[i], B[j]) >= 0) { has_internal_edge = true; break; }
            }
            if (has_internal_edge) break;
        }
        if (!has_internal_edge) return true;

        stats.local_support_calls++;
        for (int u : B) rebuild_local_candidate_mask(u, C[u]);

        bool changed = true;
        while (changed) {
            changed = false;
            for (int u : B) {
                vector<int> kept;
                kept.reserve(C[u].size());
                for (int v : C[u]) {
                    bool valid = true;
                    for (int w : B) {
                        if (u == w) continue;
                        int eid = get_eid(u, w);
                        if (eid < 0) continue;
                        if (!row_intersects_candidate_mask_excluding_self(eid, v, w, v, local_mask_scratch[w])) {
                            valid = false;
                            break;
                        }
                    }
                    if (valid) kept.push_back(v);
                    else { stats.local_support_removed++; changed = true; }
                }
                if (kept.size() != C[u].size()) {
                    C[u].swap(kept);
                    rebuild_local_candidate_mask(u, C[u]);
                }
                if (C[u].empty()) {
                    stats.local_support_empty_prunes++;
                    return false;
                }
            }
        }
        return true;
    }

    vector<int> order_batch_vertices(const vector<int>& B, const vector<vector<int>>& C) const {
        vector<int> ordered = B;
        vector<uint8_t> in_batch(num_q, 0);
        vector<double> avg_deg(num_q, std::numeric_limits<double>::infinity());
        for (int u : B) { in_batch[u] = 1; avg_deg[u] = average_degree(C[u]); }
        std::sort(ordered.begin(), ordered.end(), [&](int a, int b) {
            int ia = 0, ib = 0;
            for (int w : Q.adj[a]) if (in_batch[w]) ia++;
            for (int w : Q.adj[b]) if (in_batch[w]) ib++;
            return std::make_tuple(int(C[a].size()), avg_deg[a], -ia, -Q.deg[a], a) <
                   std::make_tuple(int(C[b].size()), avg_deg[b], -ib, -Q.deg[b], b);
        });
        return ordered;
    }

    bool batch_union_feasible(const vector<int>& B, const vector<vector<int>>& C) {
        if (B.size() <= 1) return !B.empty() && !C[B[0]].empty();
        if (++union_seen_token == std::numeric_limits<int>::max()) {
            std::fill(union_seen_stamp.begin(), union_seen_stamp.end(), 0);
            union_seen_token = 1;
        }
        int uniq = 0;
        const int need = int(B.size());
        for (int u : B) {
            if (C[u].empty()) return false;
            for (int v : C[u]) {
                if (union_seen_stamp[v] != union_seen_token) {
                    union_seen_stamp[v] = union_seen_token;
                    if (++uniq >= need) return true;
                }
            }
        }
        return false;
    }

    template <class Fn>
    void enumerate_batch_rec(const vector<int>& ordered, const vector<vector<int>>& C,
                             int idx, vector<int>& local_map, vector<int>& local_used, Fn& fn) {
        if (idx == int(ordered.size())) { fn(local_map); return; }
        int u = ordered[idx];
        for (int v : C[u]) {
            bool already_used = false;
            for (int z : local_used) if (z == v) { already_used = true; break; }
            if (already_used) continue;

            bool ok = true;
            for (int j = 0; j < idx; ++j) {
                int w = ordered[j];
                int z = local_map[w];
                if (z == -1) continue;
                int eid = get_eid(u, w);
                if (eid >= 0) {
                    if (!support_contains(eid, v, z)) { ok = false; break; }
                }
            }
            if (!ok) continue;
            local_map[u] = v;
            local_used.push_back(v);
            enumerate_batch_rec(ordered, C, idx + 1, local_map, local_used, fn);
            local_used.pop_back();
            local_map[u] = -1;
        }
    }

    template <class Fn>
    void enumerate_batch_assignments(const vector<int>& B, const vector<vector<int>>& C, Fn&& fn) {
        stats.pivot_batch_calls++;
        if (B.empty() || !batch_union_feasible(B, C)) return;

        vector<int> ordered = order_batch_vertices(B, C);
        vector<int> local_map(num_q, -1);
        vector<int> local_used;
        local_used.reserve(B.size());
        enumerate_batch_rec(ordered, C, 0, local_map, local_used, fn);
    }

    // Full blockwise-progressive expansion for non-terminal pivot batches.
    // Compared with enumerate_batch_assignments(), this routine does not wait
    // until a complete batch assignment is formed to check all block-internal
    // constraints.  After assigning one query vertex u -> v inside the batch,
    // it immediately contracts the residual domains of the remaining batch
    // vertices by compact support rows S_{u,w}(v).  Thus Algorithm 3 is no
    // longer only a terminal fast path; every selected pivot batch is consumed
    // progressively.  The terminal fused executors above are kept as special
    // fast paths when the current batch already completes the query.
    template <class Fn>
    void blockwise_progressive_rec(const vector<int>& ordered,
                                   int idx,
                                   vector<vector<int>>& D,
                                   vector<int>& local_map,
                                   vector<uint8_t>& used_g,
                                   Fn& fn) {
        if (idx == int(ordered.size())) { fn(local_map); return; }

        const int u = ordered[idx];
        const vector<int> cur_domain = D[u];

        for (int v : cur_domain) {
            if (used_g[v]) continue;

            // Edges to already assigned vertices in this block should already
            // have been enforced by previous progressive contractions.  The
            // explicit check below keeps the routine robust when a previous
            // non-edge contraction did not touch this domain.
            bool ok = true;
            for (int j = 0; j < idx; ++j) {
                int w = ordered[j];
                int z = local_map[w];
                if (z < 0) continue;
                int eid = get_eid(u, w);
                if (eid >= 0 && !support_contains(eid, v, z)) { ok = false; break; }
                if (eid < 0) {
                    int reid = get_eid(w, u);
                    if (reid >= 0 && !support_contains(reid, z, v)) { ok = false; break; }
                }
            }
            if (!ok) continue;

            local_map[u] = v;
            used_g[v] = 1;

            // Save and contract future domains.  b_max is small, so copying
            // only the affected future domains is cheap and keeps rollback safe.
            vector<std::pair<int, vector<int>>> saved;
            saved.reserve(ordered.size() - idx - 1);
            bool nonempty = true;

            for (int k = idx + 1; k < int(ordered.size()); ++k) {
                int w = ordered[k];
                saved.emplace_back(w, D[w]);

                int eid = get_eid(u, w);
                int reid = eid < 0 ? get_eid(w, u) : -1;
                vector<int> kept;
                kept.reserve(D[w].size());

                if (eid >= 0) {
                    for (int z : D[w]) {
                        if (used_g[z]) continue;
                        if (support_contains(eid, v, z)) kept.push_back(z);
                    }
                } else if (reid >= 0) {
                    for (int z : D[w]) {
                        if (used_g[z]) continue;
                        if (support_contains(reid, z, v)) kept.push_back(z);
                    }
                } else {
                    for (int z : D[w]) {
                        if (!used_g[z]) kept.push_back(z);
                    }
                }

                D[w].swap(kept);
                if (D[w].empty()) { nonempty = false; break; }
            }

            if (nonempty) {
                blockwise_progressive_rec(ordered, idx + 1, D, local_map, used_g, fn);
            }

            for (auto it = saved.rbegin(); it != saved.rend(); ++it) {
                D[it->first].swap(it->second);
            }
            used_g[v] = 0;
            local_map[u] = -1;
        }
    }

    template <class Fn>
    void blockwise_progressive_assignments(const vector<int>& B,
                                           const vector<vector<int>>& C,
                                           vector<uint8_t>& used_g,
                                           Fn&& fn) {
        stats.pivot_batch_calls++;
        if (B.empty() || !batch_union_feasible(B, C)) return;

        vector<int> ordered = order_batch_vertices(B, C);
        vector<vector<int>> D(num_q);
        for (int u : B) D[u] = C[u];
        vector<int> local_map(num_q, -1);
        blockwise_progressive_rec(ordered, 0, D, local_map, used_g, fn);
    }

    template <class EmitFn>
    inline void terminal_bt_fast(int idx,
                                 const vector<int>& ordered,
                                 const vector<vector<int>>& C,
                                 const vector<vector<int>>& prev_edge_pos,
                                 vector<int>& mapping,
                                 vector<uint8_t>& used_g,
                                 EmitFn&& emit,
                                 uint64_t max_matches) {
        if (max_matches && stats.complete_matches >= max_matches) return;
        if (idx == int(ordered.size())) {
            stats.complete_matches++;
            stats.terminal_direct_emits++;
            emit(mapping);
            return;
        }
        const int u = ordered[idx];
        for (int v : C[u]) {
            if (max_matches && stats.complete_matches >= max_matches) break;
            if (used_g[v]) continue;
            bool ok = true;
            for (int pos : prev_edge_pos[idx]) {
                const int w = ordered[pos];
                const int z = mapping[w];
                if (z < 0) { ok = false; break; }
                const int eid = get_eid(u, w);
                if (eid >= 0 && !support_contains(eid, v, z)) { ok = false; break; }
            }
            if (!ok) continue;
            mapping[u] = v;
            used_g[v] = 1;
            terminal_bt_fast(idx + 1, ordered, C, prev_edge_pos,
                             mapping, used_g, std::forward<EmitFn>(emit), max_matches);
            used_g[v] = 0;
            mapping[u] = -1;
        }
    }


    template <class EmitFn>
    void emit_terminal_pair_direct(const vector<int>& B,
                                   const vector<vector<int>>& C,
                                   vector<int>& mapping,
                                   vector<uint8_t>& used_g,
                                   EmitFn&& emit,
                                   uint64_t max_matches) {
        stats.pivot_batch_calls++;
        stats.terminal_batch_calls++;
        if (B.size() != 2) return;

        // Blockwise-progressive terminal executor for the dominant 2-vertex
        // residual block.  Unlike the previous pair terminal version, this does
        // not allocate a target bit-mask and does not re-sort the two-vertex
        // block.  build_pivot_batch() has already ordered B by cached support
        // cardinality (PivotFast), so we consume the block in that order and
        // emit complete mappings directly.  This keeps Algorithm 3 as a pure
        // continuation of Algorithm 2: same code-bucket support rows, same
        // PivotFast ordering, but no extra DFS state for terminal blocks.
        int u = B[0];
        int w = B[1];
        int eid = get_eid(u, w);

        // If only the reverse directed support row is present, flip the block.
        // For undirected query graphs both directions normally exist, but this
        // keeps the executor robust and allows it to choose a valid directed row.
        if (eid < 0) {
            int reid = get_eid(w, u);
            if (reid >= 0) {
                std::swap(u, w);
                eid = reid;
            }
        }

        if (eid >= 0) {
            for (int v : C[u]) {
                if (max_matches && stats.complete_matches >= max_matches) break;
                mapping[u] = v;
                used_g[v] = 1;
                for (int z : C[w]) {
                    if (max_matches && stats.complete_matches >= max_matches) break;
                    if (used_g[z]) continue;
                    if (!support_contains(eid, v, z)) continue;
                    mapping[w] = z;
                    used_g[z] = 1;
                    stats.complete_matches++;
                    stats.terminal_direct_emits++;
                    emit(mapping);
                    used_g[z] = 0;
                    mapping[w] = -1;
                }
                used_g[v] = 0;
                mapping[u] = -1;
            }
            return;
        }

        // Non-adjacent terminal pair: direct Cartesian product with injectivity.
        // No union-feasibility hash/stamp test is needed here; injectivity is
        // enforced by used_g and equality is naturally skipped.
        u = B[0];
        w = B[1];
        for (int v : C[u]) {
            if (max_matches && stats.complete_matches >= max_matches) break;
            if (used_g[v]) continue;
            mapping[u] = v;
            used_g[v] = 1;
            for (int z : C[w]) {
                if (max_matches && stats.complete_matches >= max_matches) break;
                if (used_g[z]) continue;
                mapping[w] = z;
                used_g[z] = 1;
                stats.complete_matches++;
                stats.terminal_direct_emits++;
                emit(mapping);
                used_g[z] = 0;
                mapping[w] = -1;
            }
            used_g[v] = 0;
            mapping[u] = -1;
        }
    }

    template <class EmitFn>
    void enumerate_terminal_batch_direct(const vector<int>& B,
                                         const vector<vector<int>>& C,
                                         vector<int>& mapping,
                                         vector<uint8_t>& used_g,
                                         EmitFn&& emit,
                                         uint64_t max_matches) {
        stats.pivot_batch_calls++;
        stats.terminal_batch_calls++;
        if (B.empty()) return;

        // Progressive terminal blocks are already produced by build_pivot_batch(),
        // which uses PivotFast's cached support cardinalities.  Avoid another
        // terminal-time ordering pass unless the block has more than two vertices.
        vector<int> ordered = B.size() <= 2 ? B : order_batch_vertices(B, C);
        vector<vector<int>> prev_edge_pos(ordered.size());
        for (int i = 0; i < int(ordered.size()); ++i) {
            for (int j = 0; j < i; ++j) {
                if (get_eid(ordered[i], ordered[j]) >= 0) prev_edge_pos[i].push_back(j);
            }
        }
        terminal_bt_fast(0, ordered, C, prev_edge_pos,
                         mapping, used_g, std::forward<EmitFn>(emit), max_matches);
    }

    template <class EmitFn>
    void emit_terminal_singleton_direct(const vector<int>& cand,
                                        int u,
                                        vector<int>& mapping,
                                        vector<uint8_t>& used_g,
                                        EmitFn&& emit,
                                        uint64_t max_matches) {
        stats.pivot_batch_calls++;
        stats.terminal_batch_calls++;
        stats.terminal_singleton_calls++;
        for (int v : cand) {
            if (max_matches && stats.complete_matches >= max_matches) break;
            if (used_g[v]) continue;
            mapping[u] = v;
            used_g[v] = 1;
            stats.complete_matches++;
            stats.terminal_direct_emits++;
            stats.terminal_singleton_direct_emits++;
            emit(mapping);
            used_g[v] = 0;
            mapping[u] = -1;
        }
    }

    template <class EmitFn>
    void dfs(vector<int>& mapping,
             vector<uint8_t>& used_g,
             int mapped_count,
             vector<vector<int>>& C,
             EmitFn&& emit,
             uint64_t max_matches) {
        stats.recursive_calls++;
        if (max_matches && stats.complete_matches >= max_matches) return;
        if (mapped_count == num_q) {
            stats.complete_matches++;
            emit(mapping);
            return;
        }

        int pstar = select_pivot(mapping);
        if (pstar < 0) return;
        vector<int> B = build_pivot_batch(pstar, mapping);
        if (B.empty()) return;

        if (mapped_count + int(B.size()) == num_q && B.size() == 2) {
            if (emit_terminal_pair_fused(pstar, B, mapping, used_g, emit, max_matches)) return;
        }

        build_support_join_candidates(pstar, B, mapping, used_g, C);
        for (int u : B) if (C[u].empty()) return;
        if (!local_support_prune(B, C)) return;

        // Algorithm 3 extension: when the current pivot batch completes the
        // residual query, enumerate the batch in place and emit matches directly.
        // This preserves the same qsubspace code-bucket support relation and
        // PivotFast row-cardinality cache, but avoids creating one recursive DFS
        // state per completed match.
        if (mapped_count + int(B.size()) == num_q) {
            if (B.size() == 1) {
                emit_terminal_singleton_direct(C[B[0]], B[0], mapping, used_g, emit, max_matches);
            } else if (B.size() == 2) {
                emit_terminal_pair_direct(B, C, mapping, used_g, emit, max_matches);
            } else {
                enumerate_terminal_batch_direct(B, C, mapping, used_g, emit, max_matches);
            }
            return;
        }

        if (B.size() == 1) {
            stats.pivot_batch_calls++;
            int u = B[0];
            for (int v : C[u]) {
                if (max_matches && stats.complete_matches >= max_matches) break;
                mapping[u] = v;
                used_g[v] = 1;
                dfs(mapping, used_g, mapped_count + 1, C, emit, max_matches);
                used_g[v] = 0;
                mapping[u] = -1;
            }
            return;
        }

        blockwise_progressive_assignments(B, C, used_g, [&](const vector<int>& local_map) {
            if (max_matches && stats.complete_matches >= max_matches) return;
            vector<int> added;
            added.reserve(B.size());
            for (int u : B) {
                int v = local_map[u];
                if (v == -1) return;
                mapping[u] = v;
                // The progressive block routine has already marked these
                // vertices during block enumeration.  Keep the assignment in
                // mapping and call the next DFS level directly.
                added.push_back(u);
            }
            dfs(mapping, used_g, mapped_count + int(added.size()), C, emit, max_matches);
            for (int u : added) {
                mapping[u] = -1;
            }
        });
    }

    template <class EmitFn>
    uint64_t run(EmitFn&& emit, uint64_t max_matches = 0) {
        vector<int> anchor_cands = initial_anchor_candidates();
        if (anchor_cands.empty()) return 0;

        vector<int> mapping(num_q, -1);
        vector<uint8_t> used_g(num_g, 0);
        vector<vector<int>> C(num_q);
        for (int u = 0; u < num_q; ++u) C[u].reserve(32);

        for (int va : anchor_cands) {
            if (max_matches && stats.complete_matches >= max_matches) break;
            mapping[anchor_q] = va;
            used_g[va] = 1;
            dfs(mapping, used_g, 1, C, emit, max_matches);
            used_g[va] = 0;
            mapping[anchor_q] = -1;
        }
        return stats.complete_matches;
    }

    void print_profile() const {
        cout << "\n========== Algorithm 3 QSubspace Code-Bucket PivotFast ProgressiveFilter PSBM Profile ==========\n";
        cout << "complete_matches           : " << stats.complete_matches << "\n";
        cout << "recursive_calls            : " << stats.recursive_calls << "\n";
        cout << "anchor_before_raf          : " << stats.anchor_candidates_before_filter << "\n";
        cout << "anchor_after_raf           : " << stats.anchor_candidates << "\n";
        cout << "anchor_pruned_by_raf       : " << stats.anchor_filter_pruned << "\n";
        cout << "restrictive_neighbor       : " << stats.anchor_restrictive_query << "\n";
        cout << "progressive_blocks         : " << stats.ivf_cells << "\n";
        cout << "progressive_pair_tests     : " << stats.progressive_pair_tests << "\n";
        cout << "progressive_blocks_scanned : " << stats.progressive_blocks_scanned << "\n";
        cout << "progressive_early_rejects  : " << stats.progressive_early_rejects << "\n";
        cout << "progressive_full_evals     : " << stats.progressive_full_evals << "\n";
        cout << "semantic_candidates_total  : " << stats.semantic_candidates_total << "\n";
        cout << "angular_cap_certified      : " << stats.angular_cap_certified << "\n";
        cout << "angular_shells             : " << stats.angular_shells << "\n";
        cout << "angular_shell_entries      : " << stats.angular_shell_entries << "\n";
        cout << "angular_shell_nonempty     : " << stats.angular_shell_nonempty << "\n";
        cout << "shell_edge_bucket_directed_edges: " << stats.shell_edge_bucket_directed_edges << "\n";
        cout << "shell_edge_bucket_tests    : " << stats.shell_edge_bucket_tests << "\n";
        cout << "shell_edge_bucket_entries  : " << stats.shell_edge_bucket_entries << "\n";
        cout << "shell_edge_bucket_nonempty : " << stats.shell_edge_bucket_nonempty << "\n";
        cout << "shell_edge_bucket_support_generated: " << stats.shell_edge_bucket_support_generated << "\n";
        cout << "shell_edge_bucket_pair_hits: " << stats.shell_edge_bucket_pair_hits << "\n";
        cout << "qscore_mask_vertices     : " << stats.qscore_mask_vertices << "\n";
        cout << "maskpair_edge_scans            : " << stats.maskpair_edge_scans << "\n";
        cout << "maskpair_edge_skips_untouched  : " << stats.maskpair_edge_skips_untouched << "\n";
        cout << "maskpair_query_edge_hits       : " << stats.maskpair_query_edge_hits << "\n";
        cout << "maskpair_support_generated     : " << stats.maskpair_support_generated << "\n";
        cout << "qsubspace_rank             : " << stats.qsubspace_rank << "\n";
        cout << "edge_range_directed_edges  : " << stats.edge_range_directed_edges << "\n";
        cout << "edge_range_node_tests      : " << stats.edge_range_node_tests << "\n";
        cout << "edge_range_node_prunes     : " << stats.edge_range_node_prunes << "\n";
        cout << "edge_range_node_full_accepts: " << stats.edge_range_node_full_accepts << "\n";
        cout << "edge_range_leaf_tests      : " << stats.edge_range_leaf_tests << "\n";
        cout << "edge_range_support_generated: " << stats.edge_range_support_generated << "\n";
        cout << "support_edges_total        : " << stats.support_edges_total << "\n";
        cout << "ac_rounds                  : " << stats.ac_rounds << "\n";
        cout << "ac_support_tests           : " << stats.ac_support_tests << "\n";
        cout << "ac_vertices_removed        : " << stats.ac_vertices_removed << "\n";
        cout << "active_candidates_total    : " << stats.active_candidates_total << "\n";
        cout << "pivot_batch_calls          : " << stats.pivot_batch_calls << "\n";
        cout << "terminal_batch_calls       : " << stats.terminal_batch_calls << "\n";
        cout << "terminal_direct_emits      : " << stats.terminal_direct_emits << "\n";
        cout << "support_domain_builds      : " << stats.support_domain_builds << "\n";
        cout << "support_base_entries_scanned: " << stats.support_base_entries_scanned << "\n";
        cout << "support_membership_tests   : " << stats.support_membership_tests << "\n";
        cout << "compact_active_total       : " << stats.compact_active_total << "\n";
        cout << "compact_bitset_edges       : " << stats.compact_bitset_edges << "\n";
        cout << "compact_bitset_words       : " << stats.compact_bitset_words << "\n";
        cout << "compact_domain_builds      : " << stats.compact_domain_builds << "\n";
        cout << "compact_word_ands         : " << stats.compact_word_ands << "\n";
        cout << "compact_bits_enumerated   : " << stats.compact_bits_enumerated << "\n";
        cout << "active_local_rows          : " << stats.active_local_rows << "\n";
        cout << "active_local_row_lookups   : " << stats.active_local_row_lookups << "\n";
        cout << "active_local_empty_lookups : " << stats.active_local_empty_lookups << "\n";
        cout << "domain_empty_prunes        : " << stats.domain_empty_prunes << "\n";
        cout << "local_support_calls        : " << stats.local_support_calls << "\n";
        cout << "local_support_removed      : " << stats.local_support_removed << "\n";
        cout << "local_support_empty_prunes : " << stats.local_support_empty_prunes << "\n";
    }

};

class MatchWriter {
public:
    std::ofstream fout;
    int num_q;
    vector<string> buffer;
    string flush_buffer;
    size_t flush_lines;
    size_t buffered = 0;
    bool enabled;
    size_t line_reserve;
    MatchWriter(const string& path, int n, bool enabled_, size_t flush_lines_=65536)
        : num_q(n), flush_lines(flush_lines_), enabled(enabled_),
          line_reserve(size_t(std::max(16, num_q * 12))) {
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
            // Keep the original vector<string> buffered writer, but reuse the
            // same string objects across flushes. This avoids per-match string
            // allocation while preserving the same output format and flush model.
            buffer.resize(flush_lines);
            for (auto& s : buffer) s.reserve(line_reserve);
            flush_buffer.reserve(flush_lines * line_reserve);
        }
    }
    static inline void append_int(string& line, int x) {
        char tmp[32];
        auto res = std::to_chars(tmp, tmp + sizeof(tmp), x);
        line.append(tmp, res.ptr);
    }
    void write(const vector<int>& mapping) {
        if (!enabled) return;
        string& line = buffer[buffered++];
        line.clear();
        for (int i = 0; i < num_q; ++i) {
            if (i) line.push_back(' ');
            append_int(line, mapping[i]);
        }
        line.push_back('\n');
        if (buffered >= flush_lines) flush();
    }
    void flush() {
        if (!enabled || buffered == 0) return;
        // Preserve the vector<string> buffered writer, but batch a flush into
        // one contiguous write.  This keeps the line format unchanged while
        // avoiding one ostream call per emitted match.
        flush_buffer.clear();
        for (size_t i = 0; i < buffered; ++i) {
            flush_buffer.append(buffer[i]);
        }
        fout.write(flush_buffer.data(), std::streamsize(flush_buffer.size()));
        buffered = 0;
    }
    ~MatchWriter() { flush(); if (fout.is_open()) fout.close(); }
};

struct Args {
    string graph_vertices;
    string graph_edges;
    string query_vertices;
    string query_edges;
    string query_prefix;
    string output = "matches_cpp.txt";
    float tau = 0.97f;
    int bmax = 4;
    int anchor_pilot_size = 32;
    int ivf_cells = 512;
    int landmarks = 0;
    int num_query_nodes = 6;
    uint64_t seed = 42;
    uint64_t max_matches = 0;
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
        else if (k == "--tau") a.tau = std::stof(need(k));
        else if (k == "--bmax") a.bmax = std::stoi(need(k));
        else if (k == "--anchor-pilot-size") a.anchor_pilot_size = std::stoi(need(k));
        else if (k == "--ivf-cells") a.ivf_cells = std::stoi(need(k));
        else if (k == "--landmarks") a.landmarks = std::stoi(need(k));
        else if (k == "--num-query-nodes") a.num_query_nodes = std::stoi(need(k));
        else if (k == "--seed") a.seed = std::stoull(need(k));
        else if (k == "--max-matches") a.max_matches = std::stoull(need(k));
        else if (k == "--output") a.output = need(k);
        else if (k == "--no-normalize") a.no_normalize = true;
        else if (k == "--count-only") a.count_only = true;
        else if (k == "--require-query") a.require_query = true;
        else if (k == "--help") {
            cout << "Usage:\n"
                 << "  ./vsm_pbsb_angular_range_join --graph-vertices graph_vertices.csv --graph-edges graph_edges.csv \\\n"
                 << "      --query-vertices query_vertices.csv --query-edges query_edges.csv \\\n"
                 << "      --tau 0.97 --bmax 4 --landmarks 0 --output matches_cpp.txt\n\n"
                 << "Options:\n"
                 << "  --ivf-cells K              Number of angular landmarks; 0 means one landmark per query vertex.\n"
                 << "  --query-prefix PREFIX      Query file prefix if query CSV is omitted.\n"
                 << "  --anchor-pilot-size N      Pilot anchor candidates for restrictive neighbor; default 32.\n"
                 << "  --count-only               Count matches without writing output.\n"
                 << "  --max-matches N            Stop after N matches. 0 means no limit.\n"
                 << "  --no-normalize             Use input vectors as already normalized.\n"
                 << "  --require-query            Error if query CSV is not provided.\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown argument: " + k);
    }
    if (a.graph_vertices.empty() || a.graph_edges.empty()) throw std::runtime_error("Missing graph csv paths. Use --help.");
    if ((a.query_vertices.empty()) != (a.query_edges.empty())) throw std::runtime_error("Provide both --query-vertices and --query-edges, or neither.");
    if (a.require_query && (a.query_vertices.empty() || a.query_edges.empty())) throw std::runtime_error("--require-query was set, but query CSV paths are missing.");
    if (a.bmax <= 0) throw std::runtime_error("--bmax must be positive.");
    if (a.anchor_pilot_size <= 0) throw std::runtime_error("--anchor-pilot-size must be positive.");
    if (a.ivf_cells <= 0) throw std::runtime_error("--ivf-cells must be positive.");
    if (a.landmarks < 0) throw std::runtime_error("--landmarks must be nonnegative.");
    return a;
}

static string resolve_output_path(const string& output) {
    if (output.empty()) return "matches_cpp.txt";
    if (output.back() == '/' || output.back() == '\\') return output + "matches_cpp.txt";
    return output;
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        const double t0 = now_sec();
        print_memory("program start");

        Graph G = read_graph_csv(args.graph_vertices, args.graph_edges, !args.no_normalize, true);
        print_memory("after graph load");

        Graph Q;
        vector<int> gt_mapping;
        if (!args.query_vertices.empty() && !args.query_edges.empty()) {
                Q = read_graph_csv(args.query_vertices, args.query_edges, !args.no_normalize, true);
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
                Q = read_graph_csv(qv_path, qe_path, !args.no_normalize, true);
                } else {
                Q = sample_query_from_graph(G, args.num_query_nodes, args.seed, gt_mapping);
                    write_query_csv(Q, qv_path, qe_path);
            }
        }

        const double t1 = now_sec();
        print_memory("after query preparation");

        AngularShellIVFBucketMatcher matcher(G, Q, args.tau, args.bmax, args.landmarks, args.anchor_pilot_size);
        const double t2 = now_sec();
        print_memory("after matcher initialization");

        const string output_path = resolve_output_path(args.output);
        MatchWriter writer(output_path, Q.n, !args.count_only);
        auto emit = [&](const vector<int>& mapping) {
            if (!args.count_only) writer.write(mapping);
        };

        uint64_t count = matcher.run(emit, args.max_matches);
        writer.flush();
        matcher.print_profile();

        const double t3 = now_sec();
        print_memory("after matching");
        (void)count;

        cout << "\n========== Runtime Summary ==========" << "\n";
        cout << "CSV loading + query : " << (t1 - t0) << " s\n";
        cout << "Matcher init        : " << (t2 - t1) << " s\n";
        cout << "Matching + output   : " << (t3 - t2) << " s\n";
        cout << "Total runtime       : " << (t3 - t0) << " s\n";
        cout << "Final current RSS   : " << current_rss_mb() << " MB\n";
        cout << "Final peak RSS      : " << peak_rss_mb() << " MB\n";
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}
