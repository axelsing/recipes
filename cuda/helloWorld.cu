#include <iostream>

__device__ __inline__ uint32_t get_smid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}
__device__ __inline__ uint32_t get_warpid() {
    uint32_t warpid;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
    return warpid;
}
__device__ __inline__ uint32_t get_laneid() {
    uint32_t laneid;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return laneid;
}

__global__ void hello() {
    int gThreadIdxX = blockDim.x * blockIdx.x + threadIdx.x;
    int gThreadIdxY = blockDim.y * blockIdx.y + threadIdx.y;

    uint32_t smid = get_smid();
    auto warpid = get_warpid();
    auto laneid = get_laneid();
    
    printf("Hello from GPU globalThrIdx:%d*%d, gridDim:%d*%d blockDim:%d*%d"
            " blockIdx:%d*%d threadIdx:%d:%d warpSize:%d!"
            "\tsmid:%d warpid:%d laneid:%d\n", 
            gThreadIdxX, gThreadIdxY, gridDim.x, gridDim.y, blockDim.x, blockDim.y,
            blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, warpSize,
            smid, warpid, laneid);
}

int main() {
    if (1) {
        hello<<<3, 2>>>();
        cudaDeviceSynchronize();
    }
    printf("---------------------------------------\n");

    if (1) {
        //const dim3 block_sz(2, 4);
        const dim3 block_sz(32, 16);
        const dim3 grid_sz(3, 2);
        hello<<<grid_sz, block_sz>>>();
        cudaDeviceSynchronize();
    }

    return 0;
}

// nvcc -v -arch=native -g -G -O0 ./helloWorld.cu -o helloWorld
