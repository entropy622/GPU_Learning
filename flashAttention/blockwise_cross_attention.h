#pragma once

#include "attention_common.h"
#include "cuda_cross_attention.h"

class BlockwiseCrossAttentionRunner {
public:
    explicit BlockwiseCrossAttentionRunner(AttentionConfig config);

    // This skeleton is the staging area for a FlashAttention-like implementation.
    // For now it only exposes a launch path so we can incrementally fill in:
    // 1. tiled score computation
    // 2. tiled softmax statistics
    // 3. online softmax state update
    // 4. tiled V accumulation
    ProfileResult runAndProfileSkeleton(const DeviceAttentionBuffers& buffers) const;

private:
    AttentionConfig config_;
};
