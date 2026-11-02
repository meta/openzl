// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/testkit/staged_host_frame.h"

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "contrib/gpu/testkit/host_memory_device.h"

namespace openzl::gpu::testkit {

StagedHostFrame::StagedHostFrame(std::string frame)
        : frame_(std::move(frame)), device_(makeHostMemoryDevice())
{
    const poly::span<const std::byte> frame_d{
        reinterpret_cast<const std::byte*>(frame_.data()), frame_.size()
    };
    if (ZL_isError(collectGPUChunks(
                device_->defaultStream(), frame_d, chunks_, frameHeaders_))) {
        throw std::runtime_error("cannot collect chunks");
    }
    Result<std::vector<PreparedGPUChunk>> prepared = prepareChunksForPlanning(
            device_->defaultStream(), frameHeaders_, chunks_);
    if (prepared.isError()) {
        throw std::runtime_error("cannot prepare chunks for planning");
    }
    prepared_ = std::move(prepared).value();
    views_.reserve(prepared_.size());
    for (PreparedGPUChunk& chunk : prepared_) {
        views_.push_back(viewOf(chunk));
    }
}

StagedHostFrame::~StagedHostFrame() = default;

} // namespace openzl::gpu::testkit
