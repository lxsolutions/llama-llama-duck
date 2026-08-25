// Quantized MUL_MAT throughput, with and without llama.cpp's CPU_REPACK path.
//
// Answers: for a given quant format and matrix shape, how much does the blocked
// AVX-512/VNNI repack layout actually buy on this CPU?
//
// Note IQ2_XS / IQ3_XXS are NOT repack-eligible -- they are absent from
// ggml_repack_get_optimal_repack_type(). Placing one in the repack buffer
// SEGFAULTS (null traits -> null extra -> deref), so they are gated off below.
//
// build:
//   g++ -O2 -std=c++17 -I <llama.cpp>/ggml/include -o kbench kbench.cpp \
//       -L <build>/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,<build>/bin
// usage:
//   KDIM=4096 ONLY=MXFP4 OMP_PROC_BIND=close OMP_PLACES=cores \
//     numactl --cpunodebind=0 --membind=0 ./kbench <threads> <M>
//   env: KDIM sets K (default 6144), ONLY restricts to one type name.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <string>

// exported by libggml-cpu.so (nm -D: _Z35ggml_backend_cpu_repack_buffer_typev)
ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

static double now(){ timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

struct Res { double ms; bool repacked; };

static Res bench(ggml_type type, int K, int N, int M, int nthr, bool try_repack,
                 const std::vector<float>& src) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, nthr);

    ggml_backend_buffer_type_t def = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_type_t buft = def;
    bool repacked = false;
    if (try_repack) {
        ggml_backend_buffer_type_t rb = ggml_backend_cpu_repack_buffer_type();
        if (rb) { buft = rb; }
    }

    ggml_init_params wp = { (size_t)2*1024*1024, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_tensor * a = ggml_new_tensor_2d(ctx_w, type, K, N);
    ggml_set_name(a, "a");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    if (!wbuf) { ggml_free(ctx_w); ggml_backend_free(backend); return {-1,false}; }

    // quantize host-side then upload (repack happens inside set_tensor)
    size_t nbytes = ggml_row_size(type, K) * N;
    std::vector<char> q(nbytes);
    std::vector<float> imat(K, 1.0f);
    ggml_quantize_chunk(type, src.data(), q.data(), 0, N, K, imat.data());
    ggml_backend_tensor_set(a, q.data(), 0, nbytes);
    repacked = try_repack && (ggml_backend_buffer_get_type(wbuf) != def);

    ggml_init_params cp = { (size_t)16*1024*1024, nullptr, true };
    ggml_context * ctx_c = ggml_init(cp);
    ggml_tensor * b = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, K, M);
    ggml_tensor * c = ggml_mul_mat(ctx_c, a, b);
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_c, def);
    std::vector<float> bv((size_t)K*M);
    for (size_t i=0;i<bv.size();i++) bv[i] = (float)((i*2654435761u)%1000)/1000.0f - 0.5f;
    ggml_backend_tensor_set(b, bv.data(), 0, bv.size()*sizeof(float));

    ggml_cgraph * gf = ggml_new_graph(ctx_c);
    ggml_build_forward_expand(gf, c);

    ggml_backend_graph_compute(backend, gf);           // warm
    double best = 1e30;
    for (int r=0;r<5;r++){
        double t0=now();
        for (int it=0; it<4; it++) ggml_backend_graph_compute(backend, gf);
        double dt=(now()-t0)/4.0;
        if (dt<best) best=dt;
    }
    ggml_free(ctx_c); ggml_free(ctx_w);
    ggml_backend_buffer_free(cbuf); ggml_backend_buffer_free(wbuf);
    ggml_backend_free(backend);
    return { best*1000.0, repacked };
}

int main(int argc,char**argv){
    setvbuf(stdout,nullptr,_IONBF,0);
    int K=getenv("KDIM")?atoi(getenv("KDIM")):6144, N=2048, nthr = argc>1?atoi(argv[1]):16, M = argc>2?atoi(argv[2]):1;
    std::vector<float> src((size_t)K*N);
    for (size_t i=0;i<src.size();i++) src[i] = ((float)((i*2246822519u)%2000)/1000.0f - 1.0f)*0.05f;

    struct E { ggml_type t; const char* n; bool rp; };
    E types[] = {
        {GGML_TYPE_IQ2_XS,"IQ2_XS",false}, {GGML_TYPE_IQ3_XXS,"IQ3_XXS",false},
        {GGML_TYPE_Q2_K,"Q2_K",true}, {GGML_TYPE_Q4_K,"Q4_K",true},
        {GGML_TYPE_Q5_K,"Q5_K",true}, {GGML_TYPE_Q4_0,"Q4_0",true}, {GGML_TYPE_Q8_0,"Q8_0",true}, {GGML_TYPE_MXFP4,"MXFP4",true},
    };
    printf("expert shape [K=%d x N=%d], M=%d, threads=%d\n\n", K,N,M,nthr);
    printf("%-9s %6s %11s %11s %9s %10s\n","type","bpw","default ms","repack ms","speedup","GB/s(rp)");
    const char* only = getenv("ONLY");
    for (auto&e : types) {
        if (only && strcmp(only,e.n)!=0) continue;
        size_t rs = ggml_row_size(e.t, K);
        double bpw = (double)rs*8.0/K;
        double gb  = (double)rs*N/1e9;
        fprintf(stderr,"[%s] default...\n", e.n); fflush(stderr);
        Res d = bench(e.t,K,N,M,nthr,false,src);
        Res r = e.rp ? bench(e.t,K,N,M,nthr,true,src) : Res{-1,false};
        if (d.ms<0){ printf("%-9s   n/a\n", e.n); continue; }
        if (r.ms<0) printf("%-9s %6.2f %11.3f %11s %9s %10.1f  LUT-only\n", e.n,bpw,d.ms,"n/a","-",gb/(d.ms/1000.0));
        else printf("%-9s %6.2f %11.3f %11.3f %8.2fx %10.1f\n", e.n,bpw,d.ms,r.ms,d.ms/r.ms,gb/(r.ms/1000.0));
    }
    return 0;
}
