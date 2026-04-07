#include "attention_common.h"
#include "attention_utils.h"
#include "blockwise_cross_attention.h"
#include "cpu_cross_attention.h"
#include "cuda_cross_attention.h"

#include <iomanip>
#include <iostream>

class CrossAttentionDemoApp {
public:
    explicit CrossAttentionDemoApp(AttentionConfig config)
        : config_(config),
          tensors_(config),
          cpuReference_(config),
          gpuRunner_(config),
          blockwiseSkeleton_(config) {}

    int run() {
        TensorInitializer::fillInputs(tensors_);
        cpuReference_.run(tensors_, tensors_.cpuOut);

        DeviceAttentionBuffers deviceBuffers(config_);
        deviceBuffers.copyInputsFromHost(tensors_);

        ProfileResult naiveProfile = gpuRunner_.runAndProfile(deviceBuffers);
        deviceBuffers.copyOutputToHost(tensors_.gpuOut);

        bool naiveOk = AttentionValidator::compare(tensors_.cpuOut, tensors_.gpuOut);
        std::cout << "Naive CUDA result\n";
        AttentionReporter::printPreview(tensors_.gpuOut, config_);
        AttentionReporter::printSummary(config_, naiveProfile, naiveOk);

        ProfileResult blockwiseProfile = blockwiseSkeleton_.runAndProfileSkeleton(deviceBuffers);
        deviceBuffers.copyOutputToHost(tensors_.gpuOut);
        bool blockwiseOk = AttentionValidator::compare(tensors_.cpuOut, tensors_.gpuOut);

        std::cout << "Blockwise result\n";
        AttentionReporter::printPreview(tensors_.gpuOut, config_);
        std::cout << "Blockwise summary:\n";
        std::cout << "  queryTileRows=" << config_.queryTileRows
                  << " keyTileCols=" << config_.keyTileCols << "\n";
        std::cout << "  correctness=" << (blockwiseOk ? "PASS" : "FAIL") << "\n";
        std::cout << "  avg kernel time=" << std::fixed << std::setprecision(4)
                  << blockwiseProfile.avgKernelMs << " ms\n";
        if (blockwiseProfile.avgKernelMs > 0.0f) {
            std::cout << "  speedup vs naive=" << std::setprecision(2)
                      << naiveProfile.avgKernelMs / blockwiseProfile.avgKernelMs << "x\n";
        }
        return (naiveOk && blockwiseOk) ? 0 : 1;
    }

private:
    AttentionConfig config_;
    AttentionTensors tensors_;
    CpuCrossAttention cpuReference_;
    CudaCrossAttentionRunner gpuRunner_;
    BlockwiseCrossAttentionRunner blockwiseSkeleton_;
};

int main() {
    AttentionConfig config;
    CrossAttentionDemoApp app(config);
    return app.run();
}
