#pragma once

#include "attention_common.h"

class CpuCrossAttention {
public:
    explicit CpuCrossAttention(AttentionConfig config);
    void run(const AttentionTensors& tensors, std::vector<float>& out) const;

private:
    AttentionConfig config_;
};
