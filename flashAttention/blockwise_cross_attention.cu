#include "blockwise_cross_attention.h"

namespace {

// Teaching skeleton for a FlashAttention-like kernel.
//
// What this file already establishes:
// - one block owns one tile of query rows
// - the kernel scans K/V in tiles
// - each K/V tile is staged in shared memory
// - tile loading is cooperative across the whole block
//
// What is still missing:
// - tile-local score computation that feeds real attention math
// - online softmax state update
// - output accumulation with V tiles
__global__ void blockwiseAttentionSkeletonKernel(const float* q,
                                                 const float* k,
                                                 const float* v,
                                                 float* out,
                                                 int queryLen,
                                                 int keyLen,
                                                 int headDim,
                                                 int valueDim,
                                                 int queryTileRows,
                                                 int keyTileCols,
                                                 float scale) {
    // Current simple mapping:
    // one thread handles one query row inside the current query tile.
    int localQueryRow = threadIdx.x;
    int queryTileStart = blockIdx.x * queryTileRows;
    int queryIdx = queryTileStart + localQueryRow;

    // Guard against partial tiles at the tail.
    if (localQueryRow >= queryTileRows || queryIdx >= queryLen) {
        return;
    }

    // Shared memory layout for the current K/V tiles:
    // keyTile   = [keyTileCols][headDim]
    // valueTile = [keyTileCols][valueDim]
    extern __shared__ float shared[];
    float* keyTile = shared;
    float* valueTile = keyTile + keyTileCols * headDim;

    // Scan the full K/V sequence tile by tile.
    for (int keyTileStart = 0; keyTileStart < keyLen; keyTileStart += keyTileCols) {
        // Last tile may be smaller than keyTileCols.
        int tileCols = min(keyTileCols, keyLen - keyTileStart);

        // Step 1: cooperative load for K tile.
        //
        // Flatten [tileCols][headDim] into a 1D range and let the block
        // cover it with a strided loop:
        //   linear = threadIdx.x, threadIdx.x + blockDim.x, ...
        int keyTileElements = tileCols * headDim;
        for (int linear = threadIdx.x; linear < keyTileElements; linear += blockDim.x) {
            int tileCol = linear / headDim;
            int dim = linear % headDim;
            keyTile[linear] = k[(keyTileStart + tileCol) * headDim + dim];
        }

        // Step 1: cooperative load for V tile.
        int valueTileElements = tileCols * valueDim;
        for (int linear = threadIdx.x; linear < valueTileElements; linear += blockDim.x) {
            int tileCol = linear / valueDim;
            int dim = linear % valueDim;
            valueTile[linear] = v[(keyTileStart + tileCol) * valueDim + dim];
        }

        // Wait until the full K/V tile is visible to the whole block.
        __syncthreads();

        // Step 2 placeholder:
        // consume the loaded keyTile so we can verify the staged data path.
        //
        // This is not real attention output. It is only a deterministic
        // placeholder while we build the blockwise algorithm step by step.
        float debugAccumulator = 0.0f;
        for (int tileCol = 0; tileCol < tileCols; ++tileCol) {
            float score = 0.0f;
            for (int dim = 0; dim < headDim; ++dim) {
                score += q[queryIdx * headDim + dim] * keyTile[tileCol * headDim + dim];
            }
            debugAccumulator += score * scale;
        }

        // Temporary writeback so the skeleton has a stable output buffer.
        if (keyTileStart == 0) {
            for (int dim = 0; dim < valueDim; ++dim) {
                out[queryIdx * valueDim + dim] = 0.0f;
            }
            if (valueDim > 0) {
                out[queryIdx * valueDim] = debugAccumulator;
            }
        }

        // Wait until every thread is done reading the current tile before
        // the next iteration overwrites shared memory.
        __syncthreads();
    }
}

}  // namespace

BlockwiseCrossAttentionRunner::BlockwiseCrossAttentionRunner(AttentionConfig config) : config_(config) {}

ProfileResult BlockwiseCrossAttentionRunner::runAndProfileSkeleton(const DeviceAttentionBuffers& buffers) const {
    // One block handles one query tile. Block size currently matches
    // queryTileRows so each thread maps to one query row.
    dim3 block(config_.queryTileRows);
    dim3 grid((config_.queryLen + config_.queryTileRows - 1) / config_.queryTileRows);

    // Shared memory holds one K tile and one V tile at the same time.
    size_t sharedBytes =
        static_cast<size_t>(config_.keyTileCols) * (config_.headDim + config_.valueDim) * sizeof(float);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "create blockwise start event");
    checkCuda(cudaEventCreate(&stop), "create blockwise stop event");

    blockwiseAttentionSkeletonKernel<<<grid, block, sharedBytes>>>(
        buffers.q(),
        buffers.k(),
        buffers.v(),
        buffers.out(),
        config_.queryLen,
        config_.keyLen,
        config_.headDim,
        config_.valueDim,
        config_.queryTileRows,
        config_.keyTileCols,
        config_.scale());
    checkCuda(cudaGetLastError(), "blockwise warmup launch");
    checkCuda(cudaDeviceSynchronize(), "blockwise warmup sync");

    checkCuda(cudaEventRecord(start), "record blockwise start");
    for (int i = 0; i < config_.profileRepeats; ++i) {
        blockwiseAttentionSkeletonKernel<<<grid, block, sharedBytes>>>(
            buffers.q(),
            buffers.k(),
            buffers.v(),
            buffers.out(),
            config_.queryLen,
            config_.keyLen,
            config_.headDim,
            config_.valueDim,
            config_.queryTileRows,
            config_.keyTileCols,
            config_.scale());
    }
    checkCuda(cudaGetLastError(), "blockwise profile launch");
    checkCuda(cudaEventRecord(stop), "record blockwise stop");
    checkCuda(cudaEventSynchronize(stop), "sync blockwise stop");

    float totalMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&totalMs, start, stop), "blockwise elapsed time");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Skeleton only reports time for now.
    ProfileResult result;
    result.avgKernelMs = totalMs / config_.profileRepeats;
    result.estimatedGflops = 0.0;
    return result;
}
