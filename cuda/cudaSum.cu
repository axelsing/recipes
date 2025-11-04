#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
//#include <sys/time.h>
#include <chrono>

uint64_t now() {
    auto p = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(p.time_since_epoch()).count();
}

#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA错误: " << cudaGetErrorString(err) << " (行号: " << __LINE__ << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void sumKernel(const double* arr, size_t n, double* result) {
    extern __shared__ double sdata[];

    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_idx = threadIdx.x;

    //std::cout << " global_idx:" << global_idx
    //    << " g.x:" << gridDim.x << " bIdx.x" << blockIdx.x
    //    << " b.x:" << blockDim.x << " tIdx.x:" << threadIdx.x
    //    << " w.s:" << warpSize << std::endl;

    // 加载数据到共享内存（超出n的部分填0）
    sdata[local_idx] = (global_idx < n) ? arr[global_idx] : 0.0f;
    __syncthreads();  // 等待块内所有线程加载完成

    // 块内归约（折叠求和）
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if (local_idx % (2 * stride) == 0) {  // 步长为2*stride的线程参与计算
            sdata[local_idx] += sdata[local_idx + stride];
        }
        __syncthreads();  // 同步确保当前步完成
    }

    // 块内第一个线程将块内结果写入全局内存的临时数组
    if (local_idx == 0) {
        // 使用原子操作累加所有块的结果（避免竞争）
        atomicAdd(result, sdata[0]);
    }
}

uint64_t gts = 0;
double cudaSum(const double* h_arr, size_t n) {
    if (n == 0) return 0.0f;

    // 1. 配置核函数参数
    const unsigned int blockSize = 256;  // 线程块大小（2的幂次，优化性能）
    const unsigned int gridSize = (n + blockSize - 1) / blockSize;  // 网格大小（向上取整）
    std::cout << "block:#" << gridSize << " thread:#" << blockSize << std::endl;

    // 2. 分配设备内存
    double* d_arr = nullptr;
    double* d_result = nullptr;
    CHECK(cudaMalloc(&d_arr, n * sizeof(double)));
    CHECK(cudaMalloc(&d_result, sizeof(double)));
    CHECK(cudaMemset(d_result, 0, sizeof(double)));  // 结果初始化为0

    // 3. 主机→设备数据传输
    CHECK(cudaMemcpy(d_arr, h_arr, n * sizeof(double), cudaMemcpyHostToDevice));

    gts = now();
    // 4. 启动核函数：共享内存大小 = 块大小 * double字节数
    sumKernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(d_arr, n, d_result);
    CHECK(cudaGetLastError());  // 检查核函数启动错误
    CHECK(cudaDeviceSynchronize());  // 等待核函数执行完成
    gts = now() - gts;

    // 5. 设备→主机传输结果
    double h_result;
    CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));

    // 6. 释放设备内存
    CHECK(cudaFree(d_arr));
    CHECK(cudaFree(d_result));

    return h_result;
}

__global__ void sumKernelFixed(const double* arr, size_t n, double* result, size_t m) {
    extern __shared__ double sdata[];

    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_idx = threadIdx.x;
    //m = (n + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);

    // 1. 线程内局部累加（每个线程处理 m 个元素）
    double thread_sum = 0.0f;
    for (size_t i = 0; i < m; ++i) {
        size_t data_idx = global_idx * m + i;
        if (data_idx < n) {  // 避免越界
            thread_sum += arr[data_idx];
        }
    }
    sdata[local_idx] = thread_sum;  // 写入共享内存
    __syncthreads();

    // 2. 块内归约（折叠求和，与之前相同）
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if (local_idx % (2 * stride) == 0) {
            sdata[local_idx] += sdata[local_idx + stride];
        }
        __syncthreads();
    }

    // 3. 块内结果原子累加至全局结果
    if (local_idx == 0) {
        atomicAdd(result, sdata[0]);
    }
}
__global__ void sumKernelFixed2(const double* arr, size_t n, double* result, size_t m) {
    extern __shared__ double sdata[];

    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_idx = threadIdx.x;
    //m = (n + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);

    // 1. 线程内局部累加（每个线程处理 m 个元素）
    double thread_sum = 0.0f;
    for (size_t i = 0; i < m; ++i) {
        size_t data_idx = global_idx * m + i;
        if (data_idx < n) {  // 避免越界
            thread_sum += arr[data_idx];
        }
    }
    sdata[local_idx] = thread_sum;  // 写入共享内存
    __syncthreads();

    // 2. 块内求和
    thread_sum = sdata[0];
    if (local_idx == 0) {
        for (size_t i = 1; i < blockDim.x; ++i) {
            thread_sum += sdata[i];
        }
    }

    // 3. 块内结果原子累加至全局结果
    if (local_idx == 0) {
        atomicAdd(result, thread_sum);
    }
}

// 主机端调用
uint64_t gts1 = 0;
double cudaSumFixed(const double* h_arr, size_t n) {
    const unsigned int blockSize = 256;
    const unsigned int gridSize = 1024;  // 固定 grid 大小
    const size_t total_threads = blockSize * gridSize;
    const size_t m = (n + total_threads - 1) / total_threads;  // 每个线程处理的元素数
    std::cout << "block:#" << gridSize << " thread:#" << blockSize << " m:" << m << std::endl;

    double *d_arr, *d_result;
    CHECK(cudaMalloc(&d_arr, n * sizeof(double)));
    CHECK(cudaMalloc(&d_result, sizeof(double)));
    CHECK(cudaMemset(d_result, 0, sizeof(double)));
    CHECK(cudaMemcpy(d_arr, h_arr, n * sizeof(double), cudaMemcpyHostToDevice));

    // 启动核函数（共享内存大小 = blockSize * sizeof(double)）
    gts1 = now();
    sumKernelFixed2<<<gridSize, blockSize, blockSize * sizeof(double)>>>(d_arr, n, d_result, m);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    gts1 = now() - gts1;

    double h_result;
    CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_arr));
    CHECK(cudaFree(d_result));
    return h_result;
}

__global__ void blockPartialSumKernel(const double* arr, size_t n, double* d_partial_sums, size_t m) {
    extern __shared__ double sdata[];

    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_idx = threadIdx.x;

    // 1. 线程内局部累加（每个线程处理 m 个元素）
    double thread_sum = 0.0f;
    for (size_t i = 0; i < m; ++i) {
        size_t data_idx = global_idx * m + i;
        if (data_idx < n) {
            thread_sum += arr[data_idx];
        }
    }
    sdata[local_idx] = thread_sum;  // 写入共享内存
    __syncthreads();

    // 2. 块内归约（共享内存中完成，无全局操作）
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if (local_idx % (2 * stride) == 0) {
            sdata[local_idx] += sdata[local_idx + stride];
        }
        __syncthreads();
    }

    // 3. 块内第一个线程将块内总和写入全局临时数组（无原子操作，索引唯一）
    if (local_idx == 0) {
        d_partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void sumPartialSumsKernel(const double* d_partial_sums, size_t num_blocks, double* d_result) {
    extern __shared__ double sdata[];

    unsigned int local_idx = threadIdx.x;

    // 1. 加载临时数组数据到共享内存（仅处理有效索引）
    sdata[local_idx] = (local_idx < num_blocks) ? d_partial_sums[local_idx] : 0.0f;
    __syncthreads();

    // 2. 块内归约（共享内存中完成最终累加）
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if (local_idx % (2 * stride) == 0) {
            sdata[local_idx] += sdata[local_idx + stride];
        }
        __syncthreads();
    }

    // 3. 第一个线程将最终结果写入全局变量
    if (local_idx == 0) {
        *d_result = sdata[0];
    }
}

uint64_t gts2 = 0;
double cudaSumShared(const double* h_arr, size_t n) {
    if (n == 0) return 0.0f;

    // 1. 配置参数（固定grid和block大小）
    const unsigned int blockSize = 256;
    const unsigned int gridSize = 1024;  // 总线程数 = 1024*256 = 262144
    const size_t total_threads = blockSize * gridSize;
    const size_t m = (n + total_threads - 1) / total_threads;  // 每个线程处理的元素数

    // 2. 分配设备内存：输入数组 + 临时部分和数组 + 最终结果
    double *d_arr, *d_partial_sums, *d_result;
    CHECK(cudaMalloc(&d_arr, n * sizeof(double)));
    CHECK(cudaMalloc(&d_partial_sums, gridSize * sizeof(double)));  // 存储每个块的部分和
    CHECK(cudaMalloc(&d_result, sizeof(double)));

    // 3. 主机→设备数据传输
    CHECK(cudaMemcpy(d_arr, h_arr, n * sizeof(double), cudaMemcpyHostToDevice));

	
    gts2 = now();
    // 4. 第一步：计算每个块的部分和（共享内存归约）
    // 共享内存大小 = blockSize * sizeof(double)
    blockPartialSumKernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(d_arr, n, d_partial_sums, m);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());  // 等待所有块完成，确保临时数组有效

    // 5. 第二步：汇总所有部分和（仅用一个块）
    // 块大小取能覆盖 gridSize 的最小2的幂次（如gridSize=1024，块大小=1024）
    const unsigned int sum_blockSize = (gridSize <= 1024) ? gridSize : 1024;
    sumPartialSumsKernel<<<1, sum_blockSize, sum_blockSize * sizeof(double)>>>(d_partial_sums, gridSize, d_result);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    gts2 = now() - gts2;

    // 6. 设备→主机传输结果
    double h_result;
    CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));

    // 7. 释放内存
    CHECK(cudaFree(d_arr));
    CHECK(cudaFree(d_partial_sums));
    CHECK(cudaFree(d_result));

    return h_result;
}

int main() {
    // 生成测试数据
    const size_t n = 1000000000;
    std::vector<double> h_arr(n);
    for (size_t i = 0; i < n; ++i) {
        h_arr[i] = static_cast<double>(i % 100);  // 简单数据：0~99的循环
    }

    auto cts = now();
    // CPU计算总和（验证基准）
    double cpu_sum = 0.0f;
    for (double val : h_arr) {
        cpu_sum += val;
    }
    cts = now() - cts;

    //auto gts = now();
    // CUDA计算总和
    double gpu_sum = cudaSum(h_arr.data(), n);
    //gts = now() - gts;

    double gpu_sum1 = cudaSumFixed(h_arr.data(), n);
    double gpu_sum2 = cudaSumShared(h_arr.data(), n);

    // 输出并验证（允许浮点误差）
    std::cout << "CPU 总和: " << cpu_sum << " cost:" << cts << std::endl;
    std::cout << "CUDA总和: " << gpu_sum << " cost:" << gts << std::endl;
    std::cout << "CUDA总和: " << gpu_sum1 << " cost:" << gts1 << std::endl;
    std::cout << "CUDA总和: " << gpu_sum2 << " cost:" << gts2 << std::endl;
    assert(fabs(gpu_sum - cpu_sum) < 1e-3f && "not OK!");
    std::cout << "pass, perfect!" << std::endl;

    return 0;
}

// float -> double
// nvcc -v -arch=native -g -G -O0 ./cudaSum.cu -o cudaSum

// L20 env
//block:#3906250 thread:#256
//block:#1024 thread:#256 m:3815
//CPU 总和: 4.95e+10 cost:5379
//CUDA总和: 4.95e+10 cost:63
//CUDA总和: 4.95e+10 cost:37
//验证通过：结果一致。
