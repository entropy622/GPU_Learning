#pragma once

#include "attention_common.h"

#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

inline void checkCuda(cudaError_t result, const char* step) {
    if (result != cudaSuccess) {
        std::cerr << step << " failed: " << cudaGetErrorString(result) << "\n";
        std::exit(1);
    }
}

class DeviceAttentionBuffers {
public:
    explicit DeviceAttentionBuffers(const AttentionConfig& config);
    ~DeviceAttentionBuffers();

    DeviceAttentionBuffers(const DeviceAttentionBuffers&) = delete;
    DeviceAttentionBuffers& operator=(const DeviceAttentionBuffers&) = delete;

    void copyInputsFromHost(const AttentionTensors& tensors) const;
    void copyOutputToHost(std::vector<float>& hostOut) const;

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

class CudaCrossAttentionRunner {
public:
    explicit CudaCrossAttentionRunner(AttentionConfig config);
    ProfileResult runAndProfile(const DeviceAttentionBuffers& buffers) const;

private:
    AttentionConfig config_;
};
