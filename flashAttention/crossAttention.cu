#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr int kQueryLen = 64;
constexpr int kKeyLen = 64;
constexpr int kHeadDim = 64;
constexpr int kValueDim = 64;
constexpr int kThreadsPerBlock = 128;
constexpr int kProfileRepeats = 50;

void checkCuda(cudaError_t result, const char* step) {
    if (result != cudaSuccess) {
        std::cerr << step << " failed: " << cudaGetErrorString(result) << "\n";
        std::exit(1);
    }
}

void fillTensor(std::vector<float>& tensor, float scale) {
    for (size_t i = 0; i < tensor.size(); ++i) {
        int centered = static_cast<int>((i * 17 + 13) % 29) - 14;
        tensor[i] = scale * static_cast<float>(centered);
    }
}

void cpuCrossAttention(const std::vector<float>& q,
                       const std::vector<float>& k,
                       const std::vector<float>& v,
                       std::vector<float>& out,
                       int qLen,
                       int kLen,
                       int headDim,
                       int valueDim) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    std::vector<float> scores(kLen, 0.0f);
    std::vector<float> probs(kLen, 0.0f);

    for (int qi = 0; qi < qLen; ++qi) {
        float rowMax = -std::numeric_limits<float>::infinity();
        for (int kj = 0; kj < kLen; ++kj) {
            float score = 0.0f;
            for (int d = 0; d < headDim; ++d) {
                score += q[qi * headDim + d] * k[kj * headDim + d];
            }
            score *= scale;
            scores[kj] = score;
            rowMax = std::max(rowMax, score);
        }

        float rowSum = 0.0f;
        for (int kj = 0; kj < kLen; ++kj) {
            probs[kj] = std::exp(scores[kj] - rowMax);
            rowSum += probs[kj];
        }

        for (int vd = 0; vd < valueDim; ++vd) {
            float acc = 0.0f;
            for (int kj = 0; kj < kLen; ++kj) {
                acc += (probs[kj] / rowSum) * v[kj * valueDim + vd];
            }
            out[qi * valueDim + vd] = acc;
        }
    }
}

__global__ void crossAttentionKernel(const float* q,
                                     const float* k,
                                     const float* v,
                                     float* out,
                                     int qLen,
                                     int kLen,
                                     int headDim,
                                     int valueDim,
                                     float scale) {
    int queryIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (queryIdx >= qLen) {
        return;
    }

    extern __shared__ float shared[];
    float* scores = shared + threadIdx.x * kLen;

    float rowMax = -FLT_MAX;
    for (int kj = 0; kj < kLen; ++kj) {
        float score = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            score += q[queryIdx * headDim + d] * k[kj * headDim + d];
        }
        score *= scale;
        scores[kj] = score;
        rowMax = fmaxf(rowMax, score);
    }

    float rowSum = 0.0f;
    for (int kj = 0; kj < kLen; ++kj) {
        float prob = __expf(scores[kj] - rowMax);
        scores[kj] = prob;
        rowSum += prob;
    }

    for (int vd = 0; vd < valueDim; ++vd) {
        float acc = 0.0f;
        for (int kj = 0; kj < kLen; ++kj) {
            acc += (scores[kj] / rowSum) * v[kj * valueDim + vd];
        }
        out[queryIdx * valueDim + vd] = acc;
    }
}

bool validate(const std::vector<float>& ref, const std::vector<float>& got) {
    constexpr float kTolerance = 1e-3f;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::fabs(ref[i] - got[i]) > kTolerance) {
            std::cerr << "Mismatch at " << i << ": ref=" << ref[i]
                      << " got=" << got[i] << "\n";
            return false;
        }
    }
    return true;
}

float profileKernel(const float* dQ,
                    const float* dK,
                    const float* dV,
                    float* dOut,
                    int qLen,
                    int kLen,
                    int headDim,
                    int valueDim,
                    int repeats) {
    dim3 block(kThreadsPerBlock);
    dim3 grid((qLen + block.x - 1) / block.x);
    size_t sharedBytes = static_cast<size_t>(block.x) * kLen * sizeof(float);
    float scale = 1.0f / std::sqrt(static_cast<float>(headDim));

    cudaEvent_t start;
    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

    crossAttentionKernel<<<grid, block, sharedBytes>>>(dQ, dK, dV, dOut, qLen, kLen, headDim, valueDim, scale);
    checkCuda(cudaGetLastError(), "warmup launch");
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    checkCuda(cudaEventRecord(start), "record start");
    for (int i = 0; i < repeats; ++i) {
        crossAttentionKernel<<<grid, block, sharedBytes>>>(dQ, dK, dV, dOut, qLen, kLen, headDim, valueDim, scale);
    }
    checkCuda(cudaGetLastError(), "profile launch");
    checkCuda(cudaEventRecord(stop), "record stop");
    checkCuda(cudaEventSynchronize(stop), "sync stop");

    float totalMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&totalMs, start, stop), "elapsed time");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return totalMs / repeats;
}

void printPreview(const std::vector<float>& out, int qLen, int valueDim) {
    int rows = std::min(qLen, 2);
    int cols = std::min(valueDim, 8);
    std::cout << "Output preview:\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << out[i * valueDim + j] << " ";
        }
        std::cout << "\n";
    }
}

}  // namespace

int main() {
    const int qLen = kQueryLen;
    const int kLen = kKeyLen;
    const int headDim = kHeadDim;
    const int valueDim = kValueDim;

    std::vector<float> hQ(qLen * headDim);
    std::vector<float> hK(kLen * headDim);
    std::vector<float> hV(kLen * valueDim);
    std::vector<float> hOutCpu(qLen * valueDim, 0.0f);
    std::vector<float> hOutGpu(qLen * valueDim, 0.0f);

    fillTensor(hQ, 0.025f);
    fillTensor(hK, 0.020f);
    fillTensor(hV, 0.030f);

    cpuCrossAttention(hQ, hK, hV, hOutCpu, qLen, kLen, headDim, valueDim);

    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dOut = nullptr;
    size_t bytesQ = hQ.size() * sizeof(float);
    size_t bytesK = hK.size() * sizeof(float);
    size_t bytesV = hV.size() * sizeof(float);
    size_t bytesOut = hOutGpu.size() * sizeof(float);

    checkCuda(cudaMalloc(&dQ, bytesQ), "cudaMalloc dQ");
    checkCuda(cudaMalloc(&dK, bytesK), "cudaMalloc dK");
    checkCuda(cudaMalloc(&dV, bytesV), "cudaMalloc dV");
    checkCuda(cudaMalloc(&dOut, bytesOut), "cudaMalloc dOut");

    checkCuda(cudaMemcpy(dQ, hQ.data(), bytesQ, cudaMemcpyHostToDevice), "copy Q");
    checkCuda(cudaMemcpy(dK, hK.data(), bytesK, cudaMemcpyHostToDevice), "copy K");
    checkCuda(cudaMemcpy(dV, hV.data(), bytesV, cudaMemcpyHostToDevice), "copy V");

    float avgKernelMs = profileKernel(dQ, dK, dV, dOut, qLen, kLen, headDim, valueDim, kProfileRepeats);
    checkCuda(cudaMemcpy(hOutGpu.data(), dOut, bytesOut, cudaMemcpyDeviceToHost), "copy output");

    bool ok = validate(hOutCpu, hOutGpu);
    std::cout << "Cross-attention correctness: " << (ok ? "PASS" : "FAIL") << "\n";
    printPreview(hOutGpu, qLen, valueDim);

    double scoreFlops = 2.0 * qLen * kLen * headDim;
    double valueFlops = 2.0 * qLen * kLen * valueDim;
    double totalGflops = (scoreFlops + valueFlops) / (avgKernelMs * 1e6);

    std::cout << "Profile summary:\n";
    std::cout << "  qLen=" << qLen
              << " kLen=" << kLen
              << " headDim=" << headDim
              << " valueDim=" << valueDim << "\n";
    std::cout << "  avg kernel time=" << std::fixed << std::setprecision(4) << avgKernelMs << " ms\n";
    std::cout << "  estimated throughput=" << std::setprecision(2) << totalGflops << " GFLOP/s\n";

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dOut);
    return ok ? 0 : 1;
}
