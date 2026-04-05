#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void vecAdd(float *a, float *b, float *c, int n) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 1 << 30;
    size_t bytes = N * sizeof(float);
    std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N, 0.0f);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
    vecAdd<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (c[i] != 3.0f) {
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "Result correct!" : "Result wrong!") << std::endl;
    return 0;
}
