#include "attention_common.h"
#include "attention_utils.h"
#include "cpu_cross_attention.h"
#include "cuda_cross_attention.h"

class CrossAttentionDemoApp {
public:
    explicit CrossAttentionDemoApp(AttentionConfig config)
        : config_(config),
          tensors_(config),
          cpuReference_(config),
          gpuRunner_(config) {}

    int run() {
        TensorInitializer::fillInputs(tensors_);
        cpuReference_.run(tensors_, tensors_.cpuOut);

        DeviceAttentionBuffers deviceBuffers(config_);
        deviceBuffers.copyInputsFromHost(tensors_);

        ProfileResult profile = gpuRunner_.runAndProfile(deviceBuffers);
        deviceBuffers.copyOutputToHost(tensors_.gpuOut);

        bool ok = AttentionValidator::compare(tensors_.cpuOut, tensors_.gpuOut);
        AttentionReporter::printPreview(tensors_.gpuOut, config_);
        AttentionReporter::printSummary(config_, profile, ok);
        return ok ? 0 : 1;
    }

private:
    AttentionConfig config_;
    AttentionTensors tensors_;
    CpuCrossAttention cpuReference_;
    CudaCrossAttentionRunner gpuRunner_;
};

int main() {
    AttentionConfig config;
    CrossAttentionDemoApp app(config);
    return app.run();
}
