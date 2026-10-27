// mmstress: stress probe for the Phase 3 small-M NVFP4 kernel. For a MUL_MAT NVFP4 [K, R] x [K, n] it
//  (1) recomputes the same graph REPEAT times (a large scratch op between runs dirties registers / shared memory)
//      and reports every run whose output is not bitwise identical to run 0 (race or uninitialised read), and
//  (2) over SEEDS random inputs compares the CUDA result with the CPU backend (NMSE) and reports outliers
//      (data-dependent bug).
// Build: bench/opinv/build.sh p3 mmstress     Run: build-p3/bin/mmstress [--k 256] [--r 32] [--n 8] [--repeat 500] [--seeds 50]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int K = 256, R = 32, N = 8, REPEAT = 500, SEEDS = 50, PREC_Q8 = 0, DUMP = 0, ISOLATE = 0, UNIT = 0;

struct problem { std::vector<uint8_t> wq; std::vector<float> x; };

static problem make_problem(int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<float> w((size_t) K * R);
    for (auto & v : w) v = dis(gen);
    problem p;
    p.x.resize((size_t) K * N);
    for (auto & v : p.x) v = dis(gen);
    p.wq.resize(ggml_row_size(GGML_TYPE_NVFP4, K) * R);
    ggml_quantize_chunk(GGML_TYPE_NVFP4, w.data(), p.wq.data(), 0, R, K, nullptr);
    return p;
}

struct graph {
    ggml_context * ctx; ggml_backend_buffer_t buf; ggml_cgraph * gf; ggml_tensor * W, * X, * O, * S, * SO;
};

static graph build(ggml_backend_t be, const problem & p, bool with_scratch) {
    graph g;
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    g.W = ggml_new_tensor_2d(g.ctx, GGML_TYPE_NVFP4, K, R);
    g.X = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, N);
    g.O = ggml_mul_mat(g.ctx, g.W, g.X);
    if (PREC_Q8) ggml_prec_set_src(g.O, GGML_PREC_Q8, 1);
    g.S = g.SO = nullptr;
    if (with_scratch) { // a big elementwise op whose blocks leave non-zero registers and shared memory behind
        g.S = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 1 << 22);
        g.SO = ggml_mul(g.ctx, g.S, g.S);
    }
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
    ggml_backend_tensor_set(g.W, p.wq.data(), 0, p.wq.size());
    ggml_backend_tensor_set(g.X, p.x.data(), 0, p.x.size() * 4);
    if (g.S) { std::vector<float> s(1 << 22); for (size_t i = 0; i < s.size(); ++i) s[i] = 1.0f + (i % 97) * 1e-3f; ggml_backend_tensor_set(g.S, s.data(), 0, s.size() * 4); }
    g.gf = ggml_new_graph(g.ctx);
    if (g.SO) ggml_build_forward_expand(g.gf, g.SO);
    ggml_build_forward_expand(g.gf, g.O);
    return g;
}
static std::vector<float> run(ggml_backend_t be, graph & g) {
    ggml_backend_graph_compute(be, g.gf);
    std::vector<float> o((size_t) R * N);
    ggml_backend_tensor_get(g.O, o.data(), 0, o.size() * 4);
    return o;
}
static void destroy(graph & g) { ggml_backend_buffer_free(g.buf); ggml_free(g.ctx); }

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - b[i]) * (double) (a[i] - b[i]); den += b[i] * (double) b[i]; }
    return den > 0 ? num / den : num;
}


// Host reference: dequantized NVFP4 weights x per-32 q8 activations (d = amax/127), summed in double.
static std::vector<double> host_ref(const problem & p) {
    const auto * tt = ggml_get_type_traits(GGML_TYPE_NVFP4);
    std::vector<float> w((size_t) K * R);
    const size_t rs = ggml_row_size(GGML_TYPE_NVFP4, K);
    for (int r = 0; r < R; ++r) tt->to_float(p.wq.data() + r * rs, w.data() + (size_t) r * K, K);
    std::vector<float> xq((size_t) K * N);
    for (int c = 0; c < N; ++c) for (int b = 0; b < K / 32; ++b) {
        float amax = 0; for (int k = 0; k < 32; ++k) amax = std::max(amax, fabsf(p.x[(size_t) c * K + b * 32 + k]));
        const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        for (int k = 0; k < 32; ++k) xq[(size_t) c * K + b * 32 + k] = d * roundf(p.x[(size_t) c * K + b * 32 + k] * id);
    }
    std::vector<double> o((size_t) R * N);
    for (int c = 0; c < N; ++c) for (int r = 0; r < R; ++r) {
        double acc = 0; for (int k = 0; k < K; ++k) acc += (double) w[(size_t) r * K + k] * xq[(size_t) c * K + k];
        o[(size_t) c * R + r] = acc;
    }
    return o;
}

static void dump_pattern(const std::vector<float> & oc, const std::vector<float> & oh, const std::vector<double> & ref) {
    printf("  pattern (cuda vs host ref), ref vs cpu NMSE %.3g:\n", [&]{ double n=0,d=0; for (size_t i=0;i<oh.size();++i){n+=(oh[i]-ref[i])*(oh[i]-ref[i]); d+=ref[i]*ref[i];} return d>0?n/d:n; }());
    std::vector<double> num_col(N, 0), den_col(N, 0), num_r16(16, 0), den_r16(16, 0);
    int nbad = 0, shown = 0;
    for (int c = 0; c < N; ++c) for (int r = 0; r < R; ++r) {
        const size_t i = (size_t) c * R + r;
        const double e = (oc[i] - ref[i]) * (oc[i] - ref[i]);
        num_col[c] += e; den_col[c] += ref[i] * ref[i];
        num_r16[r % 16] += e; den_r16[r % 16] += ref[i] * ref[i];
        if (fabs(oc[i] - ref[i]) > 1e-3 * (1 + fabs(ref[i]))) {
            nbad++;
            if (shown++ < 12) printf("    bad r=%d (r%%16=%d, blk %d) c=%d: cuda %.6f  ref %.6f  cpu %.6f  ratio %.4f\n", r, r % 16, r / 16, c, oc[i], ref[i], oh[i], ref[i] != 0 ? oc[i] / ref[i] : 0.0);
        }
    }
    printf("    %d/%zu elements bad\n", nbad, oc.size());
    printf("    per-col NMSE:"); for (int c = 0; c < N; ++c) printf(" %.2g", den_col[c] > 0 ? num_col[c] / den_col[c] : 0); printf("\n");
    printf("    per-(row%%16) NMSE:"); for (int r = 0; r < 16; ++r) printf(" %.2g", den_r16[r] > 0 ? num_r16[r] / den_r16[r] : 0); printf("\n");
}

// Isolate: keep only one 32-wide activation block of column 0 non-zero and see which blocks the CUDA path gets wrong.
static void isolate(ggml_backend_t cuda, const problem & p0) {
    printf("  isolate by 32-block of column 0 (NMSE cuda vs host ref):");
    for (int b = 0; b < K / 32; ++b) {
        problem p = p0;
        for (int c = 0; c < N; ++c) for (int k = 0; k < K; ++k) if (c != 0 || k / 32 != b) p.x[(size_t) c * K + k] = 0.0f;
        graph gc = build(cuda, p, false); std::vector<float> oc = run(cuda, gc); destroy(gc);
        std::vector<double> ref = host_ref(p);
        double n = 0, d = 0; for (size_t i = 0; i < oc.size(); ++i) { n += (oc[i] - ref[i]) * (oc[i] - ref[i]); d += ref[i] * ref[i]; }
        printf(" b%d:%.2g", b, d > 0 ? n / d : n);
    }
    printf("\n");
    printf("  isolate by 64-block of weights (NMSE cuda vs host ref):");
    for (int b = 0; b < K / 64; ++b) {
        problem p = p0;
        std::mt19937 gen(1000); std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        std::vector<float> w((size_t) K * R);
        for (auto & v : w) v = dis(gen);
        for (int r = 0; r < R; ++r) for (int k = 0; k < K; ++k) if (k / 64 != b) w[(size_t) r * K + k] = 0.0f;
        ggml_quantize_chunk(GGML_TYPE_NVFP4, w.data(), p.wq.data(), 0, R, K, nullptr);
        graph gc = build(cuda, p, false); std::vector<float> oc = run(cuda, gc); destroy(gc);
        std::vector<double> ref = host_ref(p);
        double n = 0, d = 0; for (size_t i = 0; i < oc.size(); ++i) { n += (oc[i] - ref[i]) * (oc[i] - ref[i]); d += ref[i] * ref[i]; }
        printf(" b%d:%.2g", b, d > 0 ? n / d : n);
    }
    printf("\n");
}

// Unit test: x = e_k (exactly representable in q8), so out[r] = Wdq[r][k]. Reports k positions the CUDA path gets wrong
// and, when the wrong output matches another weight column k2 in every row, the pairing k -> k2.
static void unit_map(ggml_backend_t cuda, const problem & p0) {
    const auto * tt = ggml_get_type_traits(GGML_TYPE_NVFP4);
    std::vector<float> w((size_t) K * R);
    const size_t rs = ggml_row_size(GGML_TYPE_NVFP4, K);
    for (int r = 0; r < R; ++r) tt->to_float(p0.wq.data() + r * rs, w.data() + (size_t) r * K, K);
    int nbad = 0;
    printf("  unit map (column 0 only):\n");
    for (int k = 0; k < K; ++k) {
        problem p = p0;
        std::fill(p.x.begin(), p.x.end(), 0.0f);
        p.x[k] = 1.0f;
        graph gc = build(cuda, p, false); std::vector<float> oc = run(cuda, gc); destroy(gc);
        bool ok = true;
        for (int r = 0; r < R; ++r) if (fabsf(oc[r] - w[(size_t) r * K + k]) > 1e-4f) ok = false;
        if (ok) continue;
        nbad++;
        int k2 = -1;
        for (int kk = 0; kk < K && k2 < 0; ++kk) {
            bool m = true;
            for (int r = 0; r < R; ++r) if (fabsf(oc[r] - w[(size_t) r * K + kk]) > 1e-4f) { m = false; break; }
            if (m) k2 = kk;
        }
        if (nbad <= 40) {
            printf("    k=%3d (blk %d sub %d e %2d) wrong: ", k, k / 64, (k % 64) / 16, k % 16);
            if (k2 >= 0) printf("matches weight k=%d (blk %d sub %d e %2d)\n", k2, k2 / 64, (k2 % 64) / 16, k2 % 16);
            else printf("row0 cuda %.5f want %.5f row1 cuda %.5f want %.5f\n", oc[0], w[k], oc[1], w[(size_t) K + k]);
        }
    }
    printf("    %d/%d unit positions wrong\n", nbad, K);
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--k" && i + 1 < argc) K = atoi(argv[++i]);
        else if (a == "--r" && i + 1 < argc) R = atoi(argv[++i]);
        else if (a == "--n" && i + 1 < argc) N = atoi(argv[++i]);
        else if (a == "--repeat" && i + 1 < argc) REPEAT = atoi(argv[++i]);
        else if (a == "--seeds" && i + 1 < argc) SEEDS = atoi(argv[++i]);
        else if (a == "--q8") PREC_Q8 = 1;
        else if (a == "--dump") DUMP = 1;
        else if (a == "--isolate") ISOLATE = 1;
        else if (a == "--unit") UNIT = 1;
        else { fprintf(stderr, "usage: mmstress [--k K] [--r R] [--n N] [--repeat N] [--seeds N] [--q8]\n"); return 1; }
    }
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    ggml_backend_t cpu  = ggml_backend_cpu_init();
    if (!cuda || !cpu) { fprintf(stderr, "backend init failed\n"); return 1; }
    ggml_backend_cpu_set_n_threads(cpu, 8);
    printf("mmstress: NVFP4 K=%d R=%d n=%d repeat=%d seeds=%d prec_q8=%d  FP4_UNLOCK_KERNEL=%s\n", K, R, N, REPEAT, SEEDS, PREC_Q8,
           getenv("FP4_UNLOCK_KERNEL") ? getenv("FP4_UNLOCK_KERNEL") : "(unset)");

    // (1) repeat the same graph
    {
        problem p = make_problem(1);
        graph g = build(cuda, p, true);
        std::vector<float> ref = run(cuda, g);
        int ndiff_runs = 0; double worst = 0;
        for (int it = 1; it < REPEAT; ++it) {
            std::vector<float> o = run(cuda, g);
            if (memcmp(o.data(), ref.data(), o.size() * 4) != 0) {
                ndiff_runs++;
                double m = 0; int nd = 0;
                for (size_t i = 0; i < o.size(); ++i) if (o[i] != ref[i]) { nd++; m = std::max(m, (double) fabsf(o[i] - ref[i])); }
                worst = std::max(worst, m);
                if (ndiff_runs <= 5) printf("  run %d differs from run 0: %d/%zu values, max |delta| %.3g\n", it, nd, o.size(), m);
            }
        }
        printf("repeat: %d/%d runs differ bitwise from run 0 (worst |delta| %.3g)\n", ndiff_runs, REPEAT - 1, worst);
        destroy(g);
    }
    // (2) random inputs vs CPU
    {
        int bad = 0; double worst = 0;
        for (int s = 0; s < SEEDS; ++s) {
            problem p = make_problem(1000 + s);
            graph gc = build(cuda, p, false); std::vector<float> oc = run(cuda, gc); destroy(gc);
            graph gh = build(cpu,  p, false); std::vector<float> oh = run(cpu,  gh); destroy(gh);
            double e = nmse(oc, oh);
            worst = std::max(worst, e);
            if (e > 5e-4) { bad++; if (bad <= 5) printf("  seed %d: NMSE vs CPU %.3g\n", 1000 + s, e);
                if (DUMP && bad == 1) dump_pattern(oc, oh, host_ref(p));
                if (ISOLATE && bad == 1) isolate(cuda, p);
                if (UNIT && bad == 1) unit_map(cuda, p); }
        }
        printf("seeds: %d/%d inputs with NMSE vs CPU > 5e-4 (worst %.3g)\n", bad, SEEDS, worst);
    }
    ggml_backend_free(cuda); ggml_backend_free(cpu);
    return 0;
}
