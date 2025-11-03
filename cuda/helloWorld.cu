#include <iostream>
__global__ void hello() {
    int idx = threadIdx.x;
    printf("Hello from GPU, thread:%d!\n", idx);
}
int main() {
    hello<<<1, 2>>>();
    cudaDeviceSynchronize();
    return 0;
}
