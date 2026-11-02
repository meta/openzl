// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "contrib/gpu/src/decompress/gpu_decompress.hpp"
#include "contrib/gpu/src/device/device.hpp"
#include "openzl/cpp/poly/Span.hpp"

namespace openzl::gpu::testkit {

/**
 * One frame in host memory, staged for planning on the host memory device as
 * `decompressChunks` stages it: its chunks found, their headers copied, and one
 * planner view for each chunk from `viewOf`.
 *
 * The views point into this object, so it can be neither copied nor moved.
 * The constructor throws `std::runtime_error` if staging fails.
 */
class StagedHostFrame {
   public:
    explicit StagedHostFrame(std::string frame);

    StagedHostFrame(const StagedHostFrame&)            = delete;
    StagedHostFrame& operator=(const StagedHostFrame&) = delete;
    StagedHostFrame(StagedHostFrame&&)                 = delete;
    StagedHostFrame& operator=(StagedHostFrame&&)      = delete;
    ~StagedHostFrame();

    poly::span<const PreparedGpuChunkView> views() const
    {
        return views_;
    }

   private:
    std::string frame_;
    std::unique_ptr<Device> device_;
    std::vector<GPUChunk> chunks_;
    std::vector<GPUFrameHeaderForChunks> frameHeaders_;
    std::vector<PreparedGPUChunk> prepared_;
    std::vector<PreparedGpuChunkView> views_;
};

} // namespace openzl::gpu::testkit
