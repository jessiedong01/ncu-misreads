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

// Sixteen independent chains per thread, launched on far fewer warps.
// Occupancy is low by construction (one block per SM), and the ILP keeps the
// FMA pipe fed anyway. Total FMA count matches fma_dependent exactly.
#define ILP 16
__global__ void fma_ilp(float *out, float a, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x[ILP];
    #pragma unroll
    for (int j = 0; j < ILP; j++) x[j] = a + i * 1e-6f + j;
    for (int k = 0; k < iters; k++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) x[j] = fmaf(x[j], 0.9999f, 1e-4f);
    }
    float s = 0.f;
    #pragma unroll
    for (int j = 0; j < ILP; j++) s += x[j];
    out[i] = s;
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
    const int THREADS = 256;          // 8 warps per block

    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("%s (sm_%d%d), %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);

    // Occupancy has to differ by construction, not by hoping the register
    // allocator cooperates. Saturate the device for the dependent version and
    // give the ILP version one block per SM, then match the total FMA count.
    int SMs       = p.multiProcessorCount;
    int blocksDep = SMs * 32;
    int blocksIlp = SMs * 1;
    const int ITERS_DEP = 2048;
    // blocksDep*THREADS*ITERS_DEP*1  ==  blocksIlp*THREADS*ITERS_ILP*ILP
    int ITERS_ILP = (int)((long long)blocksDep * ITERS_DEP / ((long long)blocksIlp * ILP));
    printf("dependent: %d blocks x %d iters x 1 chain\n", blocksDep, ITERS_DEP);
    printf("ilp      : %d blocks x %d iters x %d chains\n", blocksIlp, ITERS_ILP, ILP);
    printf("total fma per thread-lane is equal in both\n");

    float *dOut, *dIn;
    CK(cudaMalloc(&dOut, (size_t)N * 4));
    CK(cudaMalloc(&dIn,  (size_t)N * 8 * 4));
    CK(cudaMemset(dIn, 0, (size_t)N * 8 * 4));

    // Pair A.
    fma_dependent<<<blocksDep, THREADS>>>(dOut, 1.0f, ITERS_DEP);
    CK(cudaGetLastError());
    fma_ilp<<<blocksIlp, THREADS>>>(dOut, 1.0f, ITERS_ILP);
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
