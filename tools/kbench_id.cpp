// THE KEY TEST: does CPU_REPACK accelerate 3D MUL_MAT_ID (MoE experts), or only
// 2D MUL_MAT (dense)?
//
// Result on a 4-socket Xeon Gold 6242, [K=4096 x N=2048], 16 threads:
//   MXFP4  MUL_MAT(2D)     0.525 -> 0.078 ms   6.73x
//   MXFP4  MUL_MAT_ID(3D)  0.415 -> 0.445 ms   0.93x   (slower!)
//   Q4_K   MUL_MAT(2D)     0.363 -> 0.146 ms   2.48x
//   Q4_K   MUL_MAT_ID(3D)  0.533 -> 0.424 ms   1.26x
//
// The tensors ARE repacked in both cases ("repacked: YES"). The MoE path does
// not benefit, because at batch 1 each expert receives a single row and
// forward_mul_mat_id takes the gemv branch:
//   "If there are more than three rows in src1, use gemm; otherwise, use gemv."
// The blocked layout amortizes across rows, so one row gains nothing.
//
// Implication: for MoE models, repack-eligibility of the expert quant is nearly
// irrelevant at decode time. Dense models get the full 2D speedup.
//
// Shapes for ggml_mul_mat_id: as=[K,N,n_expert], b=[K,n_used,n_tokens],
// ids=[n_used,n_tokens]. Getting b's last two dims backwards trips
// GGML_ASSERT(ids->ne[1] == b->ne[2]).
//
// build:
//   g++ -O2 -std=c++17 -I <llama.cpp>/ggml/include -o kbench_id kbench_id.cpp \
//       -L <build>/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,<build>/bin
// usage: ./kbench_id <K> <N> <threads>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);
static double now(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

// mode 0 = 2D MUL_MAT, mode 1 = 3D MUL_MAT_ID
static double bench(ggml_type type,int K,int N,int NEXP,int NUSED,int nthr,bool rp,int mode,bool*used_rp){
    ggml_backend_t be=ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(be,nthr);
    ggml_backend_buffer_type_t def=ggml_backend_get_default_buffer_type(be);
    ggml_backend_buffer_type_t buft=def;
    if(rp){ auto r=ggml_backend_cpu_repack_buffer_type(); if(r) buft=r; }
    ggml_init_params wp={(size_t)8*1024*1024,nullptr,true}; ggml_context*cw=ggml_init(wp);
    ggml_tensor*a = (mode==0)? ggml_new_tensor_2d(cw,type,K,N)
                             : ggml_new_tensor_3d(cw,type,K,N,NEXP);
    ggml_backend_buffer_t wb=ggml_backend_alloc_ctx_tensors_from_buft(cw,buft);
    if(!wb){ggml_free(cw);ggml_backend_free(be);*used_rp=false;return -1;}
    size_t nb=ggml_nbytes(a);
    std::vector<char> q(nb);
    size_t nel=(size_t)K*N*((mode==0)?1:NEXP);
    std::vector<float> src(nel), imat(K,1.0f);
    for(size_t i=0;i<nel;i++) src[i]=((float)((i*2246822519u)%2000)/1000.0f-1.0f)*0.05f;
    ggml_quantize_chunk(type,src.data(),q.data(),0,N*((mode==0)?1:NEXP),K,imat.data());
    ggml_backend_tensor_set(a,q.data(),0,nb);
    *used_rp = rp && (ggml_backend_buffer_get_type(wb)!=def);

    ggml_init_params cp={(size_t)64*1024*1024,nullptr,true}; ggml_context*cc=ggml_init(cp);
    ggml_tensor*b,*c,*ids=nullptr;
    if(mode==0){ b=ggml_new_tensor_2d(cc,GGML_TYPE_F32,K,1); c=ggml_mul_mat(cc,a,b); }
    else {
        b   = ggml_new_tensor_3d(cc,GGML_TYPE_F32,K,NUSED,1);
        ids = ggml_new_tensor_2d(cc,GGML_TYPE_I32,NUSED,1);
        c   = ggml_mul_mat_id(cc,a,b,ids);
    }
    ggml_backend_buffer_t cb=ggml_backend_alloc_ctx_tensors_from_buft(cc,def);
    std::vector<float> bv((size_t)K*(mode==0?1:NUSED),0.3f);
    ggml_backend_tensor_set(b,bv.data(),0,bv.size()*4);
    if(ids){ std::vector<int32_t> iv(NUSED); for(int i=0;i<NUSED;i++) iv[i]=(i*37)%NEXP;
             ggml_backend_tensor_set(ids,iv.data(),0,iv.size()*4); }
    ggml_cgraph*gf=ggml_new_graph(cc); ggml_build_forward_expand(gf,c);
    ggml_backend_graph_compute(be,gf);
    double best=1e30;
    for(int r=0;r<5;r++){ double t0=now(); for(int it=0;it<4;it++) ggml_backend_graph_compute(be,gf);
        double dt=(now()-t0)/4.0; if(dt<best)best=dt; }
    ggml_free(cc); ggml_free(cw); ggml_backend_buffer_free(cb); ggml_backend_buffer_free(wb); ggml_backend_free(be);
    return best*1000.0;
}
int main(int argc,char**argv){
    setvbuf(stdout,nullptr,_IONBF,0);
    int K=argc>1?atoi(argv[1]):4096, N=argc>2?atoi(argv[2]):2048, nthr=argc>3?atoi(argv[3]):16;
    int NEXP=32, NUSED=6;
    struct E{ggml_type t;const char*n;} types[]={{GGML_TYPE_MXFP4,"MXFP4"},{GGML_TYPE_Q4_K,"Q4_K"}};
    printf("shape [K=%d x N=%d], %d threads, MUL_MAT_ID: %d experts / %d used\n\n",K,N,nthr,NEXP,NUSED);
    printf("%-7s %-12s %10s %10s %9s %8s\n","type","op","default ms","repack ms","speedup","repacked");
    for(auto&e:types){
        for(int mode=0;mode<2;mode++){
            bool r1=false,r2=false;
            double d=bench(e.t,K,N,NEXP,NUSED,nthr,false,mode,&r1);
            double r=bench(e.t,K,N,NEXP,NUSED,nthr,true ,mode,&r2);
            const char*opn = mode==0?"MUL_MAT(2D)":"MUL_MAT_ID(3D)";
            if(d<0||r<0){ printf("%-7s %-12s   n/a\n",e.n,opn); continue; }
            printf("%-7s %-12s %10.3f %10.3f %8.2fx %8s\n",e.n,opn,d,r,d/r,r2?"YES":"NO");
        }
    }
    return 0;
}
