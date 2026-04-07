#include "attention_common.h"
#include "attention_utils.h"
#include "blockwise_cross_attention.h"
#include "cpu_cross_attention.h"
#include "cuda_cross_attention.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool canRunNaive(const AttentionConfig& config) {
    int maxSharedBytes = 0;
    checkCuda(cudaDeviceGetAttribute(&maxSharedBytes, cudaDevAttrMaxSharedMemoryPerBlock, 0),
              "query max shared memory per block");
    size_t naiveSharedBytes = static_cast<size_t>(config.threadsPerBlock) * config.keyLen * sizeof(float);
    return naiveSharedBytes <= static_cast<size_t>(maxSharedBytes);
}

int repeatsForLength(int seqLen) {
    if (seqLen <= 1024) {
        return 20;
    }
    if (seqLen <= 2048) {
        return 10;
    }
    if (seqLen <= 4096) {
        return 4;
    }
    return 2;
}

bool envFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

int envIntOrDefault(const char* name, int defaultValue) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return defaultValue;
    }
    return std::atoi(value);
}

int runSingleBenchmark(AttentionConfig config, bool validateAgainstCpu) {
    AttentionTensors tensors(config);
    TensorInitializer::fillInputs(tensors);

    if (validateAgainstCpu) {
        CpuCrossAttention cpuReference(config);
        cpuReference.run(tensors, tensors.cpuOut);
    }

    DeviceAttentionBuffers deviceBuffers(config);
    deviceBuffers.copyInputsFromHost(tensors);

    bool allOk = true;
    bool ranNaive = false;
    ProfileResult naiveProfile;
    if (canRunNaive(config)) {
        CudaCrossAttentionRunner naiveRunner(config);
        naiveProfile = naiveRunner.runAndProfile(deviceBuffers);
        deviceBuffers.copyOutputToHost(tensors.gpuOut);
        ranNaive = true;

        bool naiveOk = validateAgainstCpu ? AttentionValidator::compare(tensors.cpuOut, tensors.gpuOut) : true;
        allOk = allOk && naiveOk;

        std::cout << "Naive CUDA result\n";
        if (validateAgainstCpu) {
            AttentionReporter::printPreview(tensors.gpuOut, config);
        }
        AttentionReporter::printSummary(config, naiveProfile, naiveOk);
    } else {
        std::cout << "Naive CUDA result\n";
        std::cout << "  skipped: dynamic shared memory requirement exceeds device limit\n";
    }

    BlockwiseCrossAttentionRunner blockwiseRunner(config);
    ProfileResult blockwiseProfile = blockwiseRunner.runAndProfileSkeleton(deviceBuffers);
    deviceBuffers.copyOutputToHost(tensors.gpuOut);
    bool blockwiseOk = validateAgainstCpu ? AttentionValidator::compare(tensors.cpuOut, tensors.gpuOut) : true;
    allOk = allOk && blockwiseOk;

    std::cout << "Blockwise result\n";
    if (validateAgainstCpu) {
        AttentionReporter::printPreview(tensors.gpuOut, config);
    }
    std::cout << "Blockwise summary:\n";
    std::cout << "  queryLen=" << config.queryLen
              << " keyLen=" << config.keyLen
              << " queryTileRows=" << config.queryTileRows
              << " keyTileCols=" << config.keyTileCols << "\n";
    std::cout << "  correctness=" << (blockwiseOk ? "PASS" : "SKIPPED/FAIL") << "\n";
    std::cout << "  avg kernel time=" << std::fixed << std::setprecision(4)
              << blockwiseProfile.avgKernelMs << " ms\n";
    if (ranNaive && blockwiseProfile.avgKernelMs > 0.0f) {
        std::cout << "  speedup vs naive=" << std::setprecision(2)
                  << naiveProfile.avgKernelMs / blockwiseProfile.avgKernelMs << "x\n";
    }

    return allOk ? 0 : 1;
}

int runProfileOnly(AttentionConfig config) {
    AttentionTensors tensors(config);
    TensorInitializer::fillInputs(tensors);

    DeviceAttentionBuffers deviceBuffers(config);
    deviceBuffers.copyInputsFromHost(tensors);

    BlockwiseCrossAttentionRunner blockwiseRunner(config);
    ProfileResult blockwiseProfile = blockwiseRunner.runAndProfileSkeleton(deviceBuffers);

    std::cout << "Profile-only blockwise run\n";
    std::cout << "  queryLen=" << config.queryLen
              << " keyLen=" << config.keyLen
              << " queryTileRows=" << config.queryTileRows
              << " keyTileCols=" << config.keyTileCols << "\n";
    std::cout << "  avg kernel time=" << std::fixed << std::setprecision(4)
              << blockwiseProfile.avgKernelMs << " ms\n";
    return 0;
}

}  // namespace

int main() {
    if (envFlagEnabled("FLASH_PROFILE_ONLY")) {
        AttentionConfig config;
        config.queryLen = envIntOrDefault("FLASH_QUERY_LEN", 8192);
        config.keyLen = envIntOrDefault("FLASH_KEY_LEN", config.queryLen);
        config.profileRepeats = envIntOrDefault("FLASH_REPEATS", repeatsForLength(config.queryLen));
        config.queryTileRows = envIntOrDefault("FLASH_QUERY_TILE_ROWS", config.queryTileRows);
        config.keyTileCols = envIntOrDefault("FLASH_KEY_TILE_COLS", config.keyTileCols);
        return runProfileOnly(config);
    }

    std::vector<int> sequenceLengths = {1024, 2048, 4096, 8192};
    int exitCode = 0;

    for (size_t i = 0; i < sequenceLengths.size(); ++i) {
        AttentionConfig config;
        config.queryLen = sequenceLengths[i];
        config.keyLen = sequenceLengths[i];
        config.profileRepeats = repeatsForLength(sequenceLengths[i]);

        std::cout << "\n=== Benchmark N=" << sequenceLengths[i] << " ===\n";
        bool validateAgainstCpu = (sequenceLengths[i] <= 1024);
        int rc = runSingleBenchmark(config, validateAgainstCpu);
        if (rc != 0) {
            exitCode = rc;
        }
    }

    return exitCode;
}
