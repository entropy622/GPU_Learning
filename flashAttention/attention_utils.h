#pragma once

#include "attention_common.h"

class TensorInitializer {
public:
    static void fill(std::vector<float>& tensor, float scale);
    static void fillInputs(AttentionTensors& tensors);
};

class AttentionValidator {
public:
    static bool compare(const std::vector<float>& reference,
                        const std::vector<float>& actual,
                        float tolerance = 1e-3f);
};

class AttentionReporter {
public:
    static void printPreview(const std::vector<float>& out, const AttentionConfig& config);
    static void printSummary(const AttentionConfig& config,
                             const ProfileResult& profile,
                             bool ok);
};
