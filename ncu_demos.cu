// Four kernels whose Nsight Compute reports are each misleading in a specific way.
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA ERROR %s:%d %s -> %s\n",__FILE__,__LINE__,#x, \
    cudaGetErrorString(e_)); exit(1);} } while(0)

// ---------------------------------------------------------------------------
// PAIR A: same total FMA work, different occupancy.
// The point: occupancy differs a lot, speed goes the other way.
// ---------------------------------------------------------------------------

// One dependent chain per thread. Every FMA waits on the previous one, so each
// warp can only ever have one instruction in flight. Needs many warps to hide
// that. High occupancy, latency bound on the FMA pipe itself.
__global__ void fma_dependent(float *out, float a, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = a + i * 1e-6f;
    for (int k = 0; k < iters; k++) x = fmaf(x, 0.9999f, 1e-4f);
    out[i] = x;
}

// Eight independent chains per thread. Eight FMAs in flight per thread with no
// dependency between them, so one warp does the work eight warps used to.
// More registers per thread, so lower occupancy.
__global__ void fma_ilp(float *out, float a, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x0=a+i*1e-6f, x1=x0+1, x2=x0+2, x3=x0+3,
          x4=x0+4,      x5=x0+5, x6=x0+6, x7=x0+7;
    for (int k = 0; k < iters; k++) {
        x0 = fmaf(x0, 0.9999f, 1e-4f);  x1 = fmaf(x1, 0.9999f, 1e-4f);
        x2 = fmaf(x2, 0.9999f, 1e-4f);  x3 = fmaf(x3, 0.9999f, 1e-4f);
        x4 = fmaf(x4, 0.9999f, 1e-4f);  x5 = fmaf(x5, 0.9999f, 1e-4f);
        x6 = fmaf(x6, 0.9999f, 1e-4f);  x7 = fmaf(x7, 0.9999f, 1e-4f);
    }
    out[i] = x0+x1+x2+x3+x4+x5+x6+x7;
}

// ---------------------------------------------------------------------------
// KERNEL C: long scoreboard stalls that are fully covered.
// Plenty of warps, coalesced loads. The stall breakdown looks alarming and
// the issue slots are busy anyway.
// ---------------------------------------------------------------------------
__global__ void mem_covered(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.f;
    #pragma unroll
    for (int k = 0; k < 8; k++) s += in[i + (size_t)k * n];
    out[i] = s;
}

// ---------------------------------------------------------------------------
// PAIR D: identical useful work, different access pattern.
// stride 1 is fully coalesced. stride 32 gives every lane its own sector.
// ---------------------------------------------------------------------------
__global__ void gather(float *out, const float *in, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t src = ((size_t)i * stride) & (n - 1);   // n is a power of two
    out[i] = in[src];
}

int main(int argc, char **argv) {
    const int N       = 1 << 22;     // 4M elements, power of two for the mask
    const int THREADS = 256;
    const int ITERS   = 4096;

    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("%s (sm_%d%d), %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);

    float *dOut, *dIn;
    CK(cudaMalloc(&dOut, (size_t)N * 4));
    CK(cudaMalloc(&dIn,  (size_t)N * 8 * 4));
    CK(cudaMemset(dIn, 0, (size_t)N * 8 * 4));

    // Pair A. Total FMAs are equal: dependent does 8x the threads at 1 chain,
    // ilp does 1x the threads at 8 chains.
    int ilpThreads = N / 8;
    fma_dependent<<<N / THREADS, THREADS>>>(dOut, 1.0f, ITERS);
    CK(cudaGetLastError());
    fma_ilp<<<ilpThreads / THREADS, THREADS>>>(dOut, 1.0f, ITERS);
    CK(cudaGetLastError());

    // Kernel C.
    mem_covered<<<N / THREADS, THREADS>>>(dOut, dIn, N);
    CK(cudaGetLastError());

    // Pair D.
    gather<<<N / THREADS, THREADS>>>(dOut, dIn, N, 1);
    CK(cudaGetLastError());
    gather<<<N / THREADS, THREADS>>>(dOut, dIn, N, 32);
    CK(cudaGetLastError());

    CK(cudaDeviceSynchronize());
    CK(cudaFree(dOut)); CK(cudaFree(dIn));
    printf("all kernels launched\n");
    return 0;
}
