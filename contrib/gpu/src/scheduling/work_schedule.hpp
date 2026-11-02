// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <vector>

#include "contrib/gpu/src/common/result.hpp"
#include "contrib/gpu/src/planning/decode_plan.hpp"

namespace openzl::gpu {

/**
 * Codec runs that may execute after every earlier wave has completed.
 *
 * The runs of one wave are independent of each other, so the executor may
 * launch them in any grouping. It must complete the entire wave before it
 * launches the next one. Waves are never empty.
 */
struct WorkWave {
    /// Indices of the @c DecodePlan::codecRuns in this wave, in plan order.
    std::vector<PlannedCodecRunIndex> codecRunIndices;
};

/**
 * Execution order derived from a decode plan's dependency graph.
 *
 * Each codec run's wave exceeds every producer codec run it depends on.
 * Aliasing codec runs, whose outputs are all backed by existing storage, are
 * the exception: they launch no work, appear in no wave, and add no dependency
 * depth.
 */
struct WorkSchedule {
    /// Topological execution buckets in dependency order.
    std::vector<WorkWave> waves;
    /// Aliasing codec runs, in decode-plan order.
    std::vector<PlannedCodecRunIndex> aliasingCodecRunIndices;
};

/**
 * Assigns each codec run to a topological wave.
 *
 * It reads only each codec run's dependencies and the storage facts of its
 * output streams. It does not read input streams, transforms, codec headers,
 * or sizes.
 *
 * @param plan Decode graph whose codec runs precede their dependents.
 * @returns The work schedule, @c corruption if a range or stream index in
 * @p plan is out of bounds or a dependency does not precede its dependent codec
 * run, or @c allocation if an allocation fails.
 */
Result<WorkSchedule> buildDecodeWorkSchedule(const DecodePlan& plan);

} // namespace openzl::gpu
