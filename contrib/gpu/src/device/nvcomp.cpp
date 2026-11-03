// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/src/device/nvcomp.hpp"

#include <dlfcn.h>

#include <atomic>
#include <mutex>

namespace openzl::gpu {
namespace {

/// The process's nvcomp when no nvcomp is usable, so that this answer is
/// cached too, and a missing library is searched for only once.
const NvcompFunctions kNoNvcomp{};

/// The process's nvcomp, or null before the first load or registration. Each
/// published object stays alive until the process exits, so readers need no
/// lock.
std::atomic<const NvcompFunctions*> gPublished{ nullptr };
std::mutex gPublishMutex;

template <typename Function>
bool lookUp(void* handle, const char* symbol, Function& function)
{
    function = reinterpret_cast<Function>(dlsym(handle, symbol));
    return function != nullptr;
}

std::optional<NvcompZstdFunctions> lookUpZstd(void* handle)
{
    NvcompZstdFunctions zstd;
    if (lookUp(handle,
               "nvcompBatchedZstdDecompressGetTempSizeAsync",
               zstd.decompressGetTempSizeAsync)
        && lookUp(
                handle,
                "nvcompBatchedZstdDecompressAsync",
                zstd.decompressAsync)
        && lookUp(
                handle,
                "nvcompBatchedZstdDecompressGetRequiredAlignments",
                zstd.decompressGetRequiredAlignments)) {
        return zstd;
    }
    return std::nullopt;
}

bool isComplete(const NvcompZstdFunctions& zstd)
{
    return zstd.decompressGetTempSizeAsync != nullptr
            && zstd.decompressAsync != nullptr
            && zstd.decompressGetRequiredAlignments != nullptr;
}

} // namespace

std::optional<NvcompFunctions> loadNvcomp(const char* library)
{
    void* const handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        return std::nullopt;
    }
    NvcompFunctions functions;
    functions.zstd = lookUpZstd(handle);
    if (!functions.zstd.has_value()) {
        (void)dlclose(handle);
        return std::nullopt;
    }
    // Kernels call these functions until the process exits.
    return functions;
}

void registerNvcomp(const NvcompFunctions& functions)
{
    NvcompFunctions complete = functions;
    if (complete.zstd.has_value() && !isComplete(*complete.zstd)) {
        complete.zstd.reset();
    }
    const NvcompFunctions* const copy = new NvcompFunctions(complete);
    const std::lock_guard<std::mutex> lock(gPublishMutex);
    gPublished.store(copy, std::memory_order_release);
}

const NvcompFunctions& nvcompFunctions()
{
    if (const NvcompFunctions* const published =
                gPublished.load(std::memory_order_acquire)) {
        return *published;
    }
    const std::lock_guard<std::mutex> lock(gPublishMutex);
    // Another thread may have published while this one waited for the lock.
    if (const NvcompFunctions* const published =
                gPublished.load(std::memory_order_relaxed)) {
        return *published;
    }
    const std::optional<NvcompFunctions> loaded = loadNvcomp("libnvcomp.so.5");
    const NvcompFunctions* const published =
            loaded.has_value() ? new NvcompFunctions(*loaded) : &kNoNvcomp;
    gPublished.store(published, std::memory_order_release);
    return *published;
}

} // namespace openzl::gpu
