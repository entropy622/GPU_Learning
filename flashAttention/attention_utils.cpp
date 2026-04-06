#include "attention_utils.h"

#include <cmath>
#include <iomanip>
#include <iostream>

void TensorInitializer::fill(std::vector<float>& tensor, float scale) {
    for (size_t i = 0; i < tensor.size(); ++i) {
        int centered = static_cast<int>((i * 17 + 13) % 29) - 14;
        tensor[i] = scale * static_cast<float>(centered);
    }
}

void TensorInitializer::fillInputs(AttentionTensors& tensors) {
    fill(tensors.q, 0.025f);
    fill(tensors.k, 0.020f);
    fill(tensors.v, 0.030f);
}

bool AttentionValidator::compare(const std::vector<float>& reference,
                                 const std::vector<float>& actual,
                                 float tolerance) {
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

void AttentionReporter::printPreview(const std::vector<float>& out, const AttentionConfig& config) {
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

void AttentionReporter::printSummary(const AttentionConfig& config,
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
