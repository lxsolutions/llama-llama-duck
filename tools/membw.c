// Memory read bandwidth, AVX-512, independent accumulators.
//
// WHY NOT A SIMPLE SUM: `s += a[i]` is a serial FP dependency chain that the
// compiler will not vectorize without -ffast-math. On a 2.9 GHz box with 16
// threads that reports ~93 GB/s, which is exactly 2.9e9/4cycles*8B*16thr --
// close enough to the true figure to be mistaken for it. Use independent
// accumulators and non-temporal loads instead.
//
// build: gcc -O3 -march=native -mavx512f -fopenmp -o membw membw.c
// usage: membw <GiB> <threads>
//   single socket : OMP_PROC_BIND=close OMP_PLACES=cores numactl --cpunodebind=0 --membind=0 ./membw 8 16
//   all sockets   : for n in 0 1 2 3; do (numactl --cpunodebind=$n --membind=$n ./membw 8 16 &) ; done; wait
//                   (sum the per-node results for true aggregate local bandwidth)
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
    size_t GB=argc>1?atol(argv[1]):8; int nthr=argc>2?atoi(argv[2]):16;
    size_t N=GB*(1ull<<30)/64;               /* number of 64B cachelines */
    char *a=aligned_alloc(64,N*64);
    if(!a){printf("alloc fail\n");return 1;}
    #pragma omp parallel for num_threads(nthr) schedule(static)
    for(size_t i=0;i<N;i++) a[i*64]=(char)i;
    double best=0;
    for(int rep=0;rep<4;rep++){
        double t0=now();
        #pragma omp parallel num_threads(nthr)
        {
            /* 8 independent AVX512 accumulators -> no dependency chain */
            __m512i s0=_mm512_setzero_si512(),s1=s0,s2=s0,s3=s0,s4=s0,s5=s0,s6=s0,s7=s0;
            #pragma omp for schedule(static)
            for(size_t i=0;i<N;i+=8){
                const char*p=a+i*64;
                s0=_mm512_add_epi64(s0,_mm512_stream_load_si512((void*)(p+0)));
                s1=_mm512_add_epi64(s1,_mm512_stream_load_si512((void*)(p+64)));
                s2=_mm512_add_epi64(s2,_mm512_stream_load_si512((void*)(p+128)));
                s3=_mm512_add_epi64(s3,_mm512_stream_load_si512((void*)(p+192)));
                s4=_mm512_add_epi64(s4,_mm512_stream_load_si512((void*)(p+256)));
                s5=_mm512_add_epi64(s5,_mm512_stream_load_si512((void*)(p+320)));
                s6=_mm512_add_epi64(s6,_mm512_stream_load_si512((void*)(p+384)));
                s7=_mm512_add_epi64(s7,_mm512_stream_load_si512((void*)(p+448)));
            }
            __m512i s=_mm512_add_epi64(_mm512_add_epi64(_mm512_add_epi64(s0,s1),_mm512_add_epi64(s2,s3)),
                                       _mm512_add_epi64(_mm512_add_epi64(s4,s5),_mm512_add_epi64(s6,s7)));
            if(_mm512_reduce_add_epi64(s)==0x7fffffff) printf(" ");
        }
        double dt=now()-t0, bw=(double)N*64/dt/1e9;
        if(bw>best)best=bw;
    }
    printf("  read: %7.1f GB/s  (%d threads, %zu GB)\n",best,nthr,GB);
    free(a); return 0;
}
