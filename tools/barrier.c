// OpenMP barrier cost vs thread count and socket span.
//
// Tests whether cross-socket synchronization explains slow batch-1 decode.
// On a 4-socket Xeon this measures 1.06 / 2.10 / 4.78 us for 1 / 2 / 4 sockets
// -- 4.5x worse across sockets, but at ~1000 barriers/token that is ~1% of
// decode time, so it is usually NOT the limiter.
//
// build: gcc -O2 -fopenmp -o barrier barrier.c
// usage: OMP_PROC_BIND=spread OMP_PLACES=cores numactl --cpunodebind=0 ./barrier 16 20000
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
    int nthr = argc>1?atoi(argv[1]):16;
    int iters = argc>2?atoi(argv[2]):20000;
    volatile double sink=0;
    // warm
    #pragma omp parallel num_threads(nthr)
    { for(int i=0;i<1000;i++){ 
        #pragma omp barrier
      } }
    double best=1e30;
    for(int rep=0;rep<3;rep++){
        double t0=now();
        #pragma omp parallel num_threads(nthr)
        {
            for(int i=0;i<iters;i++){
                #pragma omp barrier
            }
        }
        double dt=now()-t0;
        if(dt<best)best=dt;
    }
    printf("  %2d threads : %7.2f us/barrier\n", nthr, best/iters*1e6);
    if(sink<0)printf(" ");
    return 0;
}
