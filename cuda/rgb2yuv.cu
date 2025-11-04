#include <iostream>
#include <chrono>
#include <algorithm>

uint64_t now() {
    const auto p = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(p.time_since_epoch()).count();
}

inline __device__ __host__ int iDivUp(int a, int b) { return (a + b - 1) / b; }
inline __device__ __host__ int clamp(float v, float l, float r) {
    return ::fmin(::fmax(r, 0.f), 255.f);
}
inline __device__ void rgb_clip(float & r, float & g, float & b, uint8_t & r_u, uint8_t & g_u, uint8_t & b_u) {
    r = clamp(r, 0.f, 255.f);
    r_u = static_cast<uint8_t>(floor(r + 0.5));

    g = clamp(g, 0.f, 255.f);
    g_u = static_cast<uint8_t>(floor(g + 0.5));

    b = clamp(b, 0.f, 255.f);
    b_u = static_cast<uint8_t>(floor(b + 0.5));
}
inline __device__ void rgb_to_y(const uint8_t r, const uint8_t g, const uint8_t b, uint8_t& y) {
  // full range
  // y = static_cast<uint8_t>(floor(0.299 * r + 0.587 * g + 0.114 * b + 0.5));

  // tv range
  y = static_cast<uint8_t>(floor(0.2568 * r + 0.5041 * g + 0.0979 * b + 16 + 0.5));
}

inline __device__ void rgb_to_yuv(const uint8_t r, const uint8_t g, const uint8_t b, uint8_t& y, uint8_t& u,
                                  uint8_t& v) {
  rgb_to_y(r, g, b, y);
  // full range
  // u = static_cast<uint8_t>(floor(- 0.1687 * r - 0.3313 * g + 0.5 * b + 128 + 0.5));
  // v = static_cast<uint8_t>(floor(0.5 * r - 0.4187 * g - 0.0813 * b + 128 + 0.5));

  // tv range
  u = static_cast<uint8_t>(floor(-0.1479 * r - 0.2896 * g + 0.4375 * b + 128 + 0.5));
  v = static_cast<uint8_t>(floor(0.4375 * r - 0.3666 * g - 0.0709 * b + 128 + 0.5));
}

__global__ void d_rgb2yuv_transform(const float * rgb_f, uint8_t * yuv, int dstWidth, int dstHeight) {
    const int x_idx = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
    const int y_idx = (blockDim.y * blockIdx.y + threadIdx.y) * 2;
    const int x_idx1 = x_idx + 1;
    const int y_idx1 = y_idx + 1;

    if (x_idx1 >= dstWidth || y_idx1 >= dstHeight) {
        return ;
    }

    const int rgbPlaneSize = dstWidth * dstHeight;
    const float * r_plane = rgb_f;
    const float * g_plane = r_plane + rgbPlaneSize;
    const float * b_plane = g_plane + rgbPlaneSize;

    float r1, r2, r3, r4;
    float g1, g2, g3, g4;
    float b1, b2, b3, b4;
    r1 = r_plane[y_idx * dstWidth + x_idx] * 255.f;
    r2 = r_plane[y_idx * dstWidth + x_idx1] * 255.f;
    r3 = r_plane[y_idx1 * dstWidth + x_idx] * 255.f;
    r4 = r_plane[y_idx1 * dstWidth + x_idx1] * 255.f;
    g1 = g_plane[y_idx * dstWidth + x_idx] * 255.f;
    g2 = g_plane[y_idx * dstWidth + x_idx1] * 255.f;
    g3 = g_plane[y_idx1 * dstWidth + x_idx] * 255.f;
    g4 = g_plane[y_idx1 * dstWidth + x_idx1] * 255.f;
    b1 = b_plane[y_idx * dstWidth + x_idx] * 255.f;
    b2 = b_plane[y_idx * dstWidth + x_idx1] * 255.f;
    b3 = b_plane[y_idx1 * dstWidth + x_idx] * 255.f;
    b4 = b_plane[y_idx1 * dstWidth + x_idx1] * 255.f;

    uint8_t r_u, g_u, b_u;
    uint8_t y1_u, y2_u, y3_u, y4_u, u_u, v_u;

    rgb_clip(r1, g1, b1, r_u, g_u, b_u);
    rgb_to_yuv(r_u, g_u, b_u, y1_u, u_u, v_u);

    rgb_clip(r2, g2, b2, r_u, g_u, b_u);
    rgb_to_y(r_u, g_u, b_u, y2_u);
    
    rgb_clip(r3, g3, b3, r_u, g_u, b_u);
    rgb_to_y(r_u, g_u, b_u, y3_u);

    rgb_clip(r4, g4, b4, r_u, g_u, b_u);
    rgb_to_y(r_u, g_u, b_u, y4_u);

    const int yPlaneSize = dstWidth * dstHeight;
    uint8_t * y_plane = yuv;
    uint8_t * u_plane = y_plane + yPlaneSize;
    uint8_t * v_plane = u_plane + yPlaneSize / 4;

    y_plane[y_idx * dstWidth + x_idx]  = y1_u;
    y_plane[y_idx * dstWidth + x_idx1] = y2_u;
    y_plane[y_idx1 * dstWidth + x_idx]  = y3_u;
    y_plane[y_idx1 * dstWidth + x_idx1]  = y4_u;
    
    const int uvWidth = dstWidth / 2;
    const int uv_idx = (y_idx / 2) * uvWidth + x_idx / 2;

    u_plane[uv_idx] = u_u;
    v_plane[uv_idx] = v_u;
}

bool cuda_rgbf2yuv(const float * rgb, uint8_t * yuv, int dstWidth, int dstHeight) {
    const dim3 block_trans(32, 16);
    const dim3 grid_trans(iDivUp(dstWidth, block_trans.x / 2), iDivUp(dstHeight, block_trans.y / 2));

    d_rgb2yuv_transform<<<grid_trans, block_trans>>>(rgb, yuv, dstWidth, dstHeight);
    cudaError_t retval = cudaGetLastError();
    return retval == cudaSuccess;
}

#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA错误: " << cudaGetErrorString(err) << " (行号: " << __LINE__ << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    int dstWidth = 1920;
    int dstHeight = 1080;
    int bitWidth = 1;

    int rgbSize = dstWidth * dstHeight * 3;
    float * rgb = nullptr;
    {
        CHECK(cudaMallocHost(&rgb, rgbSize * sizeof(float) * bitWidth));
    }
    printf("rgb:%p size:#%d\n", rgb, rgbSize);

    int yuvSize = dstWidth * dstHeight * 3 / 2;
    uint8_t * yuv = nullptr;
    {
        CHECK(cudaMalloc(&yuv, yuvSize * bitWidth));
    }
    printf("yuv:%p size:#%d\n", yuv, yuvSize);

    auto ts = now();
    auto res = cuda_rgbf2yuv(rgb, yuv, dstWidth, dstHeight);
    cudaDeviceSynchronize();
    printf("cuda_rgbf2yuv cost:%lu\n", now() - ts);

    cudaFreeHost(rgb);
    cudaFree(yuv);

    return 0;
}

// nvcc -v -arch=native -g -G -O0 -std=c++17 ./rgb2yuv.cu -o rgb2yuv

