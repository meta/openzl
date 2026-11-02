// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/src/scheduling/work_schedule.hpp"

#include <algorithm>
#include <cstddef>
#include <new>
#include <vector>

namespace openzl::gpu {
namespace {

/**
 * Checks the plan ranges and stream indices that scheduling reads.
 *
 * @c DecodePlan::dependenciesOf and @c DecodePlan::outputsOf do not check
 * bounds, so a malformed plan must return @c corruption before they run.
 */
ZL_Report checkScheduledRanges(const DecodePlan& plan)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    for (const PlannedCodecRun& codecRun : plan.codecRuns) {
        ZL_ERR_IF_NOT(
                codecRun.dependencies.fitsIn(plan.codecRunIndices.size()),
                corruption,
                "WorkSchedule: codec run dependency range is invalid");
        ZL_ERR_IF_NOT(
                codecRun.outputs.fitsIn(plan.streamIndices.size()),
                corruption,
                "WorkSchedule: codec run output range is invalid");
        for (const PlannedStreamIndex output : plan.outputsOf(codecRun)) {
            ZL_ERR_IF_GE(
                    output,
                    plan.streams.size(),
                    corruption,
                    "WorkSchedule: output references an unknown stream");
        }
    }
    return ZL_returnSuccess();
}

/**
 * Reports whether a codec run materializes no storage of its own.
 *
 * Every output must be backed by an existing stream's data region and carry no
 * separate string-length table, because an alias covers only the primary data
 * region.
 */
bool materializesNothing(
        const DecodePlan& plan,
        const PlannedCodecRun& codecRun)
{
    if (codecRun.outputs.count == 0) {
        return false;
    }
    for (const PlannedStreamIndex output : plan.outputsOf(codecRun)) {
        const PlannedStream& stream = plan.streams.at(output);
        if (stream.storageClass != StorageClass::StreamRef
            || stream.storageSize.stringLengthsBytes != 0) {
            return false;
        }
    }
    return true;
}

/**
 * Computes the first wave in which each codec run's outputs are ready.
 *
 * An aliasing codec run's outputs alias storage that its dependencies produce,
 * so they are ready in the same wave as those dependencies.
 *
 * One forward pass suffices because codec runs are recorded in frame execution
 * order, so a dependency always precedes its dependent. A dependency at or
 * after its dependent means the graph is not in the topological order the plan
 * claims.
 */
Result<std::vector<size_t>> assignWavesToCodecRuns(const DecodePlan& plan)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    std::vector<size_t> waveByCodecRun(plan.codecRuns.size(), 0);
    for (size_t codecRunIndex = 0; codecRunIndex < plan.codecRuns.size();
         ++codecRunIndex) {
        const PlannedCodecRun& codecRun = plan.codecRuns.at(codecRunIndex);
        size_t wave                     = 0;
        for (const PlannedCodecRunIndex dependencyCodecRun :
             plan.dependenciesOf(codecRun)) {
            ZL_ERR_IF_GE(
                    dependencyCodecRun,
                    codecRunIndex,
                    corruption,
                    "WorkSchedule: dependency must precede its dependent "
                    "codec run");
            wave = std::max(wave, waveByCodecRun.at(dependencyCodecRun));
        }
        waveByCodecRun.at(codecRunIndex) =
                materializesNothing(plan, codecRun) ? wave : wave + 1;
    }
    return waveByCodecRun;
}

/**
 * Builds dependency-ordered wave buckets from codec run wave assignments. The
 * runs of each wave keep their decode-plan order.
 */
WorkSchedule buildWaveBuckets(
        const DecodePlan& plan,
        const std::vector<size_t>& waveByCodecRun)
{
    WorkSchedule schedule;
    if (waveByCodecRun.empty()) {
        return schedule;
    }

    schedule.waves.resize(
            *std::max_element(waveByCodecRun.begin(), waveByCodecRun.end()));

    for (size_t codecRunIndex = 0; codecRunIndex < plan.codecRuns.size();
         ++codecRunIndex) {
        const PlannedCodecRun& codecRun = plan.codecRuns.at(codecRunIndex);
        if (materializesNothing(plan, codecRun)) {
            schedule.aliasingCodecRunIndices.push_back(codecRunIndex);
            continue;
        }

        schedule.waves.at(waveByCodecRun.at(codecRunIndex) - 1)
                .codecRunIndices.push_back(codecRunIndex);
    }
    return schedule;
}

} // namespace

Result<WorkSchedule> buildDecodeWorkSchedule(const DecodePlan& plan)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    try {
        ZL_ERR_IF_ERR(checkScheduledRanges(plan));
        const Result<std::vector<size_t>> waveByCodecRun =
                assignWavesToCodecRuns(plan);
        ZL_ERR_IF_ERR(waveByCodecRun.report());
        return buildWaveBuckets(plan, waveByCodecRun.value());
    } catch (const std::bad_alloc&) {
        ZL_ERR(allocation);
    }
}

} // namespace openzl::gpu
