#include "cuda_cross_attention.h"

#include <cfloat>

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

DeviceAttentionBuffers::DeviceAttentionBuffers(const AttentionConfig& config)
    : bytesQ_(static_cast<size_t>(config.queryLen) * config.headDim * sizeof(float)),
      bytesK_(static_cast<size_t>(config.keyLen) * config.headDim * sizeof(float)),
      bytesV_(static_cast<size_t>(config.keyLen) * config.valueDim * sizeof(float)),
      bytesOut_(static_cast<size_t>(config.queryLen) * config.valueDim * sizeof(float)) {
    checkCuda(cudaMalloc(&q_, bytesQ_), "cudaMalloc q");
    checkCuda(cudaMalloc(&k_, bytesK_), "cudaMalloc k");
    checkCuda(cudaMalloc(&v_, bytesV_), "cudaMalloc v");
    checkCuda(cudaMalloc(&out_, bytesOut_), "cudaMalloc out");
}

DeviceAttentionBuffers::~DeviceAttentionBuffers() {
    cudaFree(q_);
    cudaFree(k_);
    cudaFree(v_);
    cudaFree(out_);
}

void DeviceAttentionBuffers::copyInputsFromHost(const AttentionTensors& tensors) const {
    checkCuda(cudaMemcpy(q_, tensors.q.data(), bytesQ_, cudaMemcpyHostToDevice), "copy q");
    checkCuda(cudaMemcpy(k_, tensors.k.data(), bytesK_, cudaMemcpyHostToDevice), "copy k");
    checkCuda(cudaMemcpy(v_, tensors.v.data(), bytesV_, cudaMemcpyHostToDevice), "copy v");
}

void DeviceAttentionBuffers::copyOutputToHost(std::vector<float>& hostOut) const {
    checkCuda(cudaMemcpy(hostOut.data(), out_, bytesOut_, cudaMemcpyDeviceToHost), "copy out");
}

CudaCrossAttentionRunner::CudaCrossAttentionRunner(AttentionConfig config) : config_(config) {}

ProfileResult CudaCrossAttentionRunner::runAndProfile(const DeviceAttentionBuffers& buffers) const {
    dim3 block(config_.threadsPerBlock);
    dim3 grid((config_.queryLen + block.x - 1) / block.x);
    size_t sharedBytes = static_cast<size_t>(block.x) * config_.keyLen * sizeof(float);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

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
    checkCuda(cudaGetLastError(), "warmup launch");
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    checkCuda(cudaEventRecord(start), "record start");
    for (int i = 0; i < config_.profileRepeats; ++i) {
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
