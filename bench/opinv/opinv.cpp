// opinv: op-level batch-invariance probe for the CUDA backend.
//
// For MUL_MAT (NVFP4 / MXFP4 / Q8_0 / Q4_K / Q5_K / Q6_K weights) and FLASH_ATTN_EXT (f16 / q8_0 KV), computes the op with
// n = 1 .. NMAX query columns where column 0 is identical in every run, and reports whether the output for
// column 0 is bitwise identical to the n = 1 result. This isolates which op's reduction order depends on the
// batch size M (the speculative verify batch is M = n_max + 1).
//
// Build: bench/opinv/build.sh [before|after|apoze]     Run: build-<tree>/bin/opinv [--nmax 16] [--kv 8192] [--seed 7]
// Env read by the backend applies: FP4_UNLOCK_SMALLM, GGML_CUDA_MMQ_PREC.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int nmax = 16, kv = 8192, seed = 7;

struct result { int n; bool same; double max_abs; int ndiff; };

static result compare(const std::vector<float> & ref, const std::vector<float> & got, int n) {
    result r{n, true, 0.0, 0};
    for (size_t i = 0; i < ref.size(); ++i) {
        if (memcmp(&ref[i], &got[i], 4) != 0) {
            r.same = false; r.ndiff++;
            r.max_abs = std::max(r.max_abs, (double) fabsf(ref[i] - got[i]));
        }
    }
    return r;
}

static void print(const char * label, const std::vector<result> & rs) {
    printf("%-48s", label);
    for (auto & r : rs) printf(" n=%d:%s", r.n, r.same ? "same" : "DIFF");
    printf("\n");
    for (auto & r : rs) if (!r.same) printf("    n=%d: %d values differ, max |delta| %.3g\n", r.n, r.ndiff, r.max_abs);
}

// y = W^T x : W [K, R] quantized, x [K, n] f32 -> out [R, n]
static std::vector<result> probe_mul_mat(ggml_backend_t be, ggml_type wtype, int K, int R) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<float> w((size_t) K * R);
    for (auto & v : w) v = dis(gen);
    std::vector<float> x((size_t) K * nmax);                 // column j is the same in every run
    for (auto & v : x) v = dis(gen);

    const size_t row_bytes = ggml_row_size(wtype, K);
    std::vector<uint8_t> wq(row_bytes * R);
    ggml_quantize_chunk(wtype, w.data(), wq.data(), 0, R, K, nullptr);

    std::vector<float> ref;
    std::vector<result> rs;
    for (int n = 1; n <= nmax; ++n) {
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * W = ggml_new_tensor_2d(ctx, wtype, K, R);
        ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n);
        ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(W, wq.data(), 0, wq.size());
        ggml_backend_tensor_set(X, x.data(), 0, (size_t) K * n * 4);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, Y);
        ggml_backend_graph_compute(be, gf);
        std::vector<float> col0(R);
        ggml_backend_tensor_get(Y, col0.data(), 0, (size_t) R * 4);
        if (n == 1) ref = col0; else rs.push_back(compare(ref, col0, n));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    return rs;
}

// flash attention, decode shape: q [D, n, nh, 1], k/v [D, kv, nh_kv, 1], mask [kv, n, 1, 1]
// query 0 attends kv positions [0, kv0); query j attends [0, kv0 + j). Output for query 0 (all heads) is compared.
static std::vector<result> probe_fattn(ggml_backend_t be, ggml_type kvtype, int D, int nh, int nh_kv) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<float> q((size_t) D * nmax * nh);
    for (auto & v : q) v = dis(gen);
    std::vector<float> kf((size_t) D * kv * nh_kv), vf((size_t) D * kv * nh_kv);
    for (auto & v : kf) v = dis(gen);
    for (auto & v : vf) v = dis(gen);
    const size_t kv_row = ggml_row_size(kvtype, D);
    std::vector<uint8_t> kq(kv_row * kv * nh_kv), vq(kv_row * kv * nh_kv);
    ggml_quantize_chunk(kvtype, kf.data(), kq.data(), 0, (int64_t) kv * nh_kv, D, nullptr);
    ggml_quantize_chunk(kvtype, vf.data(), vq.data(), 0, (int64_t) kv * nh_kv, D, nullptr);
    const int kv0 = kv - nmax - 1;

    std::vector<float> ref;
    std::vector<result> rs;
    for (int n = 1; n <= nmax; ++n) {
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n, nh, 1);
        ggml_tensor * Kt = ggml_new_tensor_4d(ctx, kvtype, D, kv, nh_kv, 1);
        ggml_tensor * Vt = ggml_new_tensor_4d(ctx, kvtype, D, kv, nh_kv, 1);
        ggml_tensor * M = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, n, 1, 1);
        ggml_tensor * O = ggml_flash_attn_ext(ctx, Q, Kt, Vt, M, 1.0f / sqrtf((float) D), 0.0f, 0.0f);
        ggml_prec_set_acc(O, GGML_PREC_F32);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        // q layout [D, n, nh]: element (d, j, h) at ((h*n)+j)*D + d; source q is [D, nmax, nh]
        std::vector<float> qn((size_t) D * n * nh);
        for (int h = 0; h < nh; ++h) for (int j = 0; j < n; ++j)
            memcpy(&qn[((size_t) h * n + j) * D], &q[((size_t) h * nmax + j) * D], D * 4);
        ggml_backend_tensor_set(Q, qn.data(), 0, qn.size() * 4);
        ggml_backend_tensor_set(Kt, kq.data(), 0, kq.size());
        ggml_backend_tensor_set(Vt, vq.data(), 0, vq.size());
        std::vector<ggml_fp16_t> m((size_t) kv * n);
        for (int j = 0; j < n; ++j) for (int p = 0; p < kv; ++p)
            m[(size_t) j * kv + p] = ggml_fp32_to_fp16(p < kv0 + j ? 0.0f : -INFINITY);
        ggml_backend_tensor_set(M, m.data(), 0, m.size() * 2);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, O);
        ggml_backend_graph_compute(be, gf);
        // out [D, nh, n, 1]: query 0 = first D*nh floats
        std::vector<float> o0((size_t) D * nh);
        ggml_backend_tensor_get(O, o0.data(), 0, o0.size() * 4);
        if (n == 1) ref = o0; else rs.push_back(compare(ref, o0, n));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    return rs;
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--nmax" && i + 1 < argc) nmax = atoi(argv[++i]);
        else if (a == "--kv" && i + 1 < argc) kv = atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = atoi(argv[++i]);
        else { fprintf(stderr, "usage: opinv [--nmax N] [--kv KV] [--seed S]\n"); return 1; }
    }
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { fprintf(stderr, "no CUDA backend\n"); return 1; }
    printf("opinv: nmax=%d kv=%d seed=%d  env FP4_UNLOCK_SMALLM=%s GGML_CUDA_MMQ_PREC=%s\n", nmax, kv, seed,
           getenv("FP4_UNLOCK_SMALLM") ? getenv("FP4_UNLOCK_SMALLM") : "(unset)",
           getenv("GGML_CUDA_MMQ_PREC") ? getenv("GGML_CUDA_MMQ_PREC") : "(unset)");
    printf("column 0 of the n-column result vs the n=1 result, bitwise\n\n");

    print("MUL_MAT NVFP4 K=5120 R=17408 (ffn_up)",   probe_mul_mat(be, GGML_TYPE_NVFP4, 5120, 17408));
    print("MUL_MAT NVFP4 K=17408 R=5120 (ffn_down)", probe_mul_mat(be, GGML_TYPE_NVFP4, 17408, 5120));
    print("MUL_MAT MXFP4 K=5120 R=17408",            probe_mul_mat(be, GGML_TYPE_MXFP4, 5120, 17408));
    print("MUL_MAT Q8_0  K=5120 R=17408",            probe_mul_mat(be, GGML_TYPE_Q8_0,  5120, 17408));
    print("MUL_MAT Q5_K  K=5120 R=6144 (attn_q)",     probe_mul_mat(be, GGML_TYPE_Q5_K,  5120, 6144));
    print("MUL_MAT Q5_K  K=6144 R=5120 (attn_out)",   probe_mul_mat(be, GGML_TYPE_Q5_K,  6144, 5120));
    print("MUL_MAT Q4_K  K=5120 R=6144",              probe_mul_mat(be, GGML_TYPE_Q4_K,  5120, 6144));
    print("MUL_MAT Q6_K  K=5120 R=6144",              probe_mul_mat(be, GGML_TYPE_Q6_K,  5120, 6144));
    print("FLASH_ATTN D=256 nh=24 nh_kv=4 f16 KV",   probe_fattn(be, GGML_TYPE_F16,  256, 24, 4));
    print("FLASH_ATTN D=256 nh=24 nh_kv=4 q8_0 KV",  probe_fattn(be, GGML_TYPE_Q8_0, 256, 24, 4));

    ggml_backend_free(be);
    return 0;
}
