// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "contrib/gpu/src/device/device.hpp"

namespace openzl::gpu {

/** Settings of a CUDA device. They fix its kernel table at creation. */
struct CudaDeviceOptions : DeviceOptions {
    /**
     * Lists the nvcomp kernels, such as `zstd.nvcomp` after `zstd.cuda`, if
     * nvcomp 5 is usable. The order stays the same, so this does not make
     * nvcomp preferred.
     *
     * The first device that enables nvcomp loads it, unless `registerNvcomp`
     * registered it. Later devices reuse that result, so they do not load
     * nvcomp again or retry a load that failed.
     */
    bool enableNvcomp = false;
};

/**
 * Creates a device that runs on the CUDA device that is current on the calling
 * thread when each operation starts.
 *
 * The caller makes the right device current and passes streams that belong to
 * it. Operations never change the current device.
 *
 * Without `enableNvcomp`, creation does not access the CUDA runtime. With it,
 * the first creation in the process can load nvcomp and query it, and nvcomp
 * can load its own CUDA runtime library.
 *
 * The CUDA runtime and driver must be version 12.8 or newer.
 *
 * @returns The device, or nullptr if allocation fails.
 */
std::unique_ptr<Device> makeCurrentCudaDevice(
        const CudaDeviceOptions& options = {});

} // namespace openzl::gpu
