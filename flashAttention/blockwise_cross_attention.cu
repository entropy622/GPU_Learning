#include "blockwise_cross_attention.h"

namespace {

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
    int localQueryRow = threadIdx.x;
    int queryTileStart = blockIdx.x * queryTileRows;
    int queryIdx = queryTileStart + localQueryRow;

    if (localQueryRow >= queryTileRows || queryIdx >= queryLen) {
        return;
    }

    extern __shared__ float shared[];
    float* keyTile = shared;                                   // [keyTileCols * headDim]
    float* valueTile = keyTile + keyTileCols * headDim;        // [keyTileCols * valueDim]

    // Stage 1: iterate through K/V tiles.
    for (int keyTileStart = 0; keyTileStart < keyLen; keyTileStart += keyTileCols) {
        int tileCols = min(keyTileCols, keyLen - keyTileStart);

        // Step 1: cooperatively load a K tile and a V tile into shared memory.
        // We flatten each tile and let the block cover it in a strided pattern.
        int keyTileElements = tileCols * headDim;
        for (int linear = threadIdx.x; linear < keyTileElements; linear += blockDim.x) {
            int tileCol = linear / headDim;
            int dim = linear % headDim;
            keyTile[linear] = k[(keyTileStart + tileCol) * headDim + dim];
        }

        int valueTileElements = tileCols * valueDim;
        for (int linear = threadIdx.x; linear < valueTileElements; linear += blockDim.x) {
            int tileCol = linear / valueDim;
            int dim = linear % valueDim;
            valueTile[linear] = v[(keyTileStart + tileCol) * valueDim + dim];
        }


        __syncthreads();

        // TODO(step 2): compute q * k_tile^T for this query row.
        // TODO(step 3): update running row max / row sum for online softmax.
        // TODO(step 4): update output accumulation against valueTile.
        //
        // For now we only touch the loaded data so this file stays compileable and gives
        // us a concrete place to implement the algorithm step by step.
        float debugAccumulator = 0.0f;
        for (int tileCol = 0; tileCol < tileCols; ++tileCol) {
            float score = 0.0f;
            for (int dim = 0; dim < headDim; ++dim) {
                score += q[queryIdx * headDim + dim] * keyTile[tileCol * headDim + dim];
            }
            debugAccumulator += score * scale;
        }

        // Keep writes deterministic so the skeleton can be profiled if needed.
        if (keyTileStart == 0) {
            for (int dim = 0; dim < valueDim; ++dim) {
                out[queryIdx * valueDim + dim] = 0.0f;
            }
            if (valueDim > 0) {
                out[queryIdx * valueDim] = debugAccumulator;
            }
        }

        __syncthreads();
    }
}

}  // namespace

BlockwiseCrossAttentionRunner::BlockwiseCrossAttentionRunner(AttentionConfig config) : config_(config) {}

ProfileResult BlockwiseCrossAttentionRunner::runAndProfileSkeleton(const DeviceAttentionBuffers& buffers) const {
    dim3 block(config_.queryTileRows);
    dim3 grid((config_.queryLen + config_.queryTileRows - 1) / config_.queryTileRows);
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

    ProfileResult result;
    result.avgKernelMs = totalMs / config_.profileRepeats;
    result.estimatedGflops = 0.0;
    return result;
}
