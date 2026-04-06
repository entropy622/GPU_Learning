#pragma once

#include <cmath>
#include <vector>

struct AttentionConfig {
    int queryLen = 64;
    int keyLen = 64;
    int headDim = 64;
    int valueDim = 64;
    int threadsPerBlock = 128;
    int profileRepeats = 50;
    int queryTileRows = 16;
    int keyTileCols = 16;

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

struct ProfileResult {
    float avgKernelMs = 0.0f;
    double estimatedGflops = 0.0;
};
