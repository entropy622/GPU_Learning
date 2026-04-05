#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void matMulNaive(const float* A, const float* B, float* C,
                            int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}

void checkCuda(cudaError_t result, const char* step) {
    if (result != cudaSuccess) {
        std::cerr << step << " failed: " << cudaGetErrorString(result) << "\n";
        std::exit(1);
    }
}

bool runSmallCorrectnessCheck() {
    int M = 2, K = 3, N = 2;

    std::vector<float> A = {
        1, 2, 3,
        4, 5, 6
    };

    std::vector<float> B = {
        7, 8,
        9, 10,
        11, 12
    };

    std::vector<float> C(M * N, 0.0f);
    std::vector<float> expected = {58, 64, 139, 154};

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

    checkCuda(cudaMalloc(&d_A, bytesA), "cudaMalloc d_A");
    checkCuda(cudaMalloc(&d_B, bytesB), "cudaMalloc d_B");
    checkCuda(cudaMalloc(&d_C, bytesC), "cudaMalloc d_C");

    checkCuda(cudaMemcpy(d_A, A.data(), bytesA, cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(d_B, B.data(), bytesB, cudaMemcpyHostToDevice), "copy B");

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaMemcpy(C.data(), d_C, bytesC, cudaMemcpyDeviceToHost), "copy C");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    bool ok = true;
    for (int i = 0; i < M * N; i++) {
        if (std::fabs(C[i] - expected[i]) > 1e-5f) {
            ok = false;
            break;
        }
    }

    std::cout << "Small correctness check: " << (ok ? "PASS" : "FAIL") << "\n";
    if (ok) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << C[i * N + j] << " ";
            }
            std::cout << "\n";
        }
    }
    std::cout << "\n";

    return ok;
}

void runBenchmark(int n, int repeats) {
    int M = n, K = n, N = n;

    size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);

    for (int i = 0; i < M * K; i++) {
        A[i] = static_cast<float>(i % 7) * 0.1f + 1.0f;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = static_cast<float>(i % 11) * 0.1f + 1.0f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCuda(cudaMalloc(&d_A, bytesA), "cudaMalloc d_A");
    checkCuda(cudaMalloc(&d_B, bytesB), "cudaMalloc d_B");
    checkCuda(cudaMalloc(&d_C, bytesC), "cudaMalloc d_C");

    checkCuda(cudaMemcpy(d_A, A.data(), bytesA, cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(d_B, B.data(), bytesB, cudaMemcpyHostToDevice), "copy B");

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

    matMulNaive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    checkCuda(cudaGetLastError(), "warmup launch");
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    checkCuda(cudaEventRecord(start), "record start");
    for (int i = 0; i < repeats; i++) {
        matMulNaive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }
    checkCuda(cudaEventRecord(stop), "record stop");
    checkCuda(cudaEventSynchronize(stop), "sync stop");

    float totalMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&totalMs, start, stop), "elapsed time");
    float avgMs = totalMs / repeats;
    double gflops = (2.0 * M * N * K) / (avgMs * 1e6);

    std::cout << "N=" << std::setw(4) << n
              << "  avg kernel time=" << std::fixed << std::setprecision(3) << avgMs << " ms"
              << "  throughput=" << std::setprecision(2) << gflops << " GFLOP/s"
              << "  repeats=" << repeats << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    if (!runSmallCorrectnessCheck()) {
        return 1;
    }

    std::cout << "Naive matmul benchmark (square matrices)\n";
    runBenchmark(128, 20);
    runBenchmark(256, 20);
    runBenchmark(512, 10);
    runBenchmark(1024, 5);

    return 0;
}
