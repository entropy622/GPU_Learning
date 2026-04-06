#include "cpu_cross_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>

CpuCrossAttention::CpuCrossAttention(AttentionConfig config) : config_(config) {}

void CpuCrossAttention::run(const AttentionTensors& tensors, std::vector<float>& out) const {
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
