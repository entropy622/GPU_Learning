#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

void checkCuda(cudaError_t result, const char* step) {
    if (result != cudaSuccess) {
        std::cerr << step << " failed: " << cudaGetErrorString(result) << "\n";
        std::exit(1);
    }
}

struct AttentionConfig {
    int queryLen = 64;
    int keyLen = 64;
    int headDim = 64;
    int valueDim = 64;
    int threadsPerBlock = 128;
    int profileRepeats = 50;

    float scale() const {
        return 1.0f / std::sqrt(static_cast<float>(headDim));
    }
};

struct AttentionTensors {
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> cpuOut;
    std::vector<float> gpuOut;

    explicit AttentionTensors(const AttentionConfig& config)
        : q(config.queryLen * config.headDim),
          k(config.keyLen * config.headDim),
          v(config.keyLen * config.valueDim),
          cpuOut(config.queryLen * config.valueDim, 0.0f),
          gpuOut(config.queryLen * config.valueDim, 0.0f) {}
};

class TensorInitializer {
public:
    static void fill(std::vector<float>& tensor, float scale) {
        for (size_t i = 0; i < tensor.size(); ++i) {
            int centered = static_cast<int>((i * 17 + 13) % 29) - 14;
            tensor[i] = scale * static_cast<float>(centered);
        }
    }

    static void fillInputs(AttentionTensors& tensors) {
        fill(tensors.q, 0.025f);
        fill(tensors.k, 0.020f);
        fill(tensors.v, 0.030f);
    }
};

class CpuCrossAttention {
public:
    explicit CpuCrossAttention(AttentionConfig config) : config_(config) {}

    void run(const AttentionTensors& tensors, std::vector<float>& out) const {
        std::vector<float> scores(config_.keyLen, 0.0f);
        std::vector<float> probs(config_.keyLen, 0.0f);

        for (int qi = 0; qi < config_.queryLen; ++qi) {
            float rowMax = -std::numeric_limits<float>::infinity();
            for (int kj = 0; kj < config_.keyLen; ++kj) {
                float score = 0.0f;
                for (int d = 0; d < config_.headDim; ++d) {
                    score += tensors.q[qi * config_.headDim + d] *
                             tensors.k[kj * config_.headDim + d];
                }
                score *= config_.scale();
                scores[kj] = score;
                rowMax = std::max(rowMax, score);
            }

            float rowSum = 0.0f;
            for (int kj = 0; kj < config_.keyLen; ++kj) {
                probs[kj] = std::exp(scores[kj] - rowMax);
                rowSum += probs[kj];
            }

            for (int vd = 0; vd < config_.valueDim; ++vd) {
                float acc = 0.0f;
                for (int kj = 0; kj < config_.keyLen; ++kj) {
                    acc += (probs[kj] / rowSum) * tensors.v[kj * config_.valueDim + vd];
                }
                out[qi * config_.valueDim + vd] = acc;
            }
        }
    }

private:
    AttentionConfig config_;
};

__global__ void crossAttentionKernel(const float* q,
                                     const float* k,
                                     const float* v,
                                     float* out,
                                     int queryLen,
                                     int keyLen,
                                     int headDim,
                                     int valueDim,
                                     float scale) {
    int queryIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (queryIdx >= queryLen) {
        return;
    }

    extern __shared__ float shared[];
    float* scores = shared + threadIdx.x * keyLen;

    float rowMax = -FLT_MAX;
    for (int keyIdx = 0; keyIdx < keyLen; ++keyIdx) {
        float score = 0.0f;
        for (int dim = 0; dim < headDim; ++dim) {
            score += q[queryIdx * headDim + dim] * k[keyIdx * headDim + dim];
        }
        score *= scale;
        scores[keyIdx] = score;
        rowMax = fmaxf(rowMax, score);
    }

    float rowSum = 0.0f;
    for (int keyIdx = 0; keyIdx < keyLen; ++keyIdx) {
        float prob = __expf(scores[keyIdx] - rowMax);
        scores[keyIdx] = prob;
        rowSum += prob;
    }

    for (int valueIdx = 0; valueIdx < valueDim; ++valueIdx) {
        float acc = 0.0f;
        for (int keyIdx = 0; keyIdx < keyLen; ++keyIdx) {
            acc += (scores[keyIdx] / rowSum) * v[keyIdx * valueDim + valueIdx];
        }
        out[queryIdx * valueDim + valueIdx] = acc;
    }
}

class DeviceAttentionBuffers {
public:
    explicit DeviceAttentionBuffers(const AttentionConfig& config)
        : bytesQ_(static_cast<size_t>(config.queryLen) * config.headDim * sizeof(float)),
          bytesK_(static_cast<size_t>(config.keyLen) * config.headDim * sizeof(float)),
          bytesV_(static_cast<size_t>(config.keyLen) * config.valueDim * sizeof(float)),
          bytesOut_(static_cast<size_t>(config.queryLen) * config.valueDim * sizeof(float)) {
        checkCuda(cudaMalloc(&q_, bytesQ_), "cudaMalloc q");
        checkCuda(cudaMalloc(&k_, bytesK_), "cudaMalloc k");
        checkCuda(cudaMalloc(&v_, bytesV_), "cudaMalloc v");
        checkCuda(cudaMalloc(&out_, bytesOut_), "cudaMalloc out");
    }

    ~DeviceAttentionBuffers() {
        cudaFree(q_);
        cudaFree(k_);
        cudaFree(v_);
        cudaFree(out_);
    }

    void copyInputsFromHost(const AttentionTensors& tensors) const {
        checkCuda(cudaMemcpy(q_, tensors.q.data(), bytesQ_, cudaMemcpyHostToDevice), "copy q");
        checkCuda(cudaMemcpy(k_, tensors.k.data(), bytesK_, cudaMemcpyHostToDevice), "copy k");
        checkCuda(cudaMemcpy(v_, tensors.v.data(), bytesV_, cudaMemcpyHostToDevice), "copy v");
    }

    void copyOutputToHost(std::vector<float>& hostOut) const {
        checkCuda(cudaMemcpy(hostOut.data(), out_, bytesOut_, cudaMemcpyDeviceToHost), "copy out");
    }

    const float* q() const { return q_; }
    const float* k() const { return k_; }
    const float* v() const { return v_; }
    float* out() const { return out_; }

private:
    float* q_ = nullptr;
    float* k_ = nullptr;
    float* v_ = nullptr;
    float* out_ = nullptr;
    size_t bytesQ_ = 0;
    size_t bytesK_ = 0;
    size_t bytesV_ = 0;
    size_t bytesOut_ = 0;
};

struct ProfileResult {
    float avgKernelMs = 0.0f;
    double estimatedGflops = 0.0;
};

class CudaCrossAttentionRunner {
public:
    explicit CudaCrossAttentionRunner(AttentionConfig config) : config_(config) {}

    ProfileResult runAndProfile(const DeviceAttentionBuffers& buffers) const {
        dim3 block(config_.threadsPerBlock);
        dim3 grid((config_.queryLen + block.x - 1) / block.x);
        size_t sharedBytes = static_cast<size_t>(block.x) * config_.keyLen * sizeof(float);

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        checkCuda(cudaEventCreate(&start), "create start event");
        checkCuda(cudaEventCreate(&stop), "create stop event");

        launchKernel(buffers, grid, block, sharedBytes);
        checkCuda(cudaGetLastError(), "warmup launch");
        checkCuda(cudaDeviceSynchronize(), "warmup sync");

        checkCuda(cudaEventRecord(start), "record start");
        for (int i = 0; i < config_.profileRepeats; ++i) {
            launchKernel(buffers, grid, block, sharedBytes);
        }
        checkCuda(cudaGetLastError(), "profile launch");
        checkCuda(cudaEventRecord(stop), "record stop");
        checkCuda(cudaEventSynchronize(stop), "sync stop");

        float totalMs = 0.0f;
        checkCuda(cudaEventElapsedTime(&totalMs, start, stop), "elapsed time");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        ProfileResult result;
        result.avgKernelMs = totalMs / config_.profileRepeats;

        double scoreFlops = 2.0 * config_.queryLen * config_.keyLen * config_.headDim;
        double valueFlops = 2.0 * config_.queryLen * config_.keyLen * config_.valueDim;
        result.estimatedGflops = (scoreFlops + valueFlops) / (result.avgKernelMs * 1e6);
        return result;
    }

private:
    void launchKernel(const DeviceAttentionBuffers& buffers,
                      dim3 grid,
                      dim3 block,
                      size_t sharedBytes) const {
        crossAttentionKernel<<<grid, block, sharedBytes>>>(
            buffers.q(),
            buffers.k(),
            buffers.v(),
            buffers.out(),
            config_.queryLen,
            config_.keyLen,
            config_.headDim,
            config_.valueDim,
            config_.scale());
    }

    AttentionConfig config_;
};

class AttentionValidator {
public:
    static bool compare(const std::vector<float>& reference,
                        const std::vector<float>& actual,
                        float tolerance = 1e-3f) {
        for (size_t i = 0; i < reference.size(); ++i) {
            if (std::fabs(reference[i] - actual[i]) > tolerance) {
                std::cerr << "Mismatch at " << i
                          << ": ref=" << reference[i]
                          << " got=" << actual[i] << "\n";
                return false;
            }
        }
        return true;
    }
};

class AttentionReporter {
public:
    static void printPreview(const std::vector<float>& out, const AttentionConfig& config) {
        int rows = std::min(config.queryLen, 2);
        int cols = std::min(config.valueDim, 8);
        std::cout << "Output preview:\n";
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                std::cout << std::fixed << std::setprecision(4)
                          << out[row * config.valueDim + col] << " ";
            }
            std::cout << "\n";
        }
    }

    static void printSummary(const AttentionConfig& config,
                             const ProfileResult& profile,
                             bool ok) {
        std::cout << "Cross-attention correctness: " << (ok ? "PASS" : "FAIL") << "\n";
        std::cout << "Profile summary:\n";
        std::cout << "  queryLen=" << config.queryLen
                  << " keyLen=" << config.keyLen
                  << " headDim=" << config.headDim
                  << " valueDim=" << config.valueDim << "\n";
        std::cout << "  threadsPerBlock=" << config.threadsPerBlock
                  << " repeats=" << config.profileRepeats << "\n";
        std::cout << "  avg kernel time=" << std::fixed << std::setprecision(4)
                  << profile.avgKernelMs << " ms\n";
        std::cout << "  estimated throughput=" << std::setprecision(2)
                  << profile.estimatedGflops << " GFLOP/s\n";
    }
};

class CrossAttentionDemoApp {
public:
    explicit CrossAttentionDemoApp(AttentionConfig config)
        : config_(config),
          tensors_(config),
          cpuReference_(config),
          gpuRunner_(config) {}

    int run() {
        TensorInitializer::fillInputs(tensors_);
        cpuReference_.run(tensors_, tensors_.cpuOut);

        DeviceAttentionBuffers deviceBuffers(config_);
        deviceBuffers.copyInputsFromHost(tensors_);

        ProfileResult profile = gpuRunner_.runAndProfile(deviceBuffers);
        deviceBuffers.copyOutputToHost(tensors_.gpuOut);

        bool ok = AttentionValidator::compare(tensors_.cpuOut, tensors_.gpuOut);
        AttentionReporter::printPreview(tensors_.gpuOut, config_);
        AttentionReporter::printSummary(config_, profile, ok);
        return ok ? 0 : 1;
    }

private:
    AttentionConfig config_;
    AttentionTensors tensors_;
    CpuCrossAttention cpuReference_;
    CudaCrossAttentionRunner gpuRunner_;
};

}  // namespace

int main() {
    AttentionConfig config;
    CrossAttentionDemoApp app(config);
    return app.run();
}
