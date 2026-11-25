// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/training/ace/automated_compressor_explorer.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <vector>

#include "openzl/common/a1cbor_helpers.h"
#include "openzl/common/allocation.h"
#include "openzl/shared/a1cbor.h"

namespace openzl {
namespace training {

std::vector<ACECompressor> AutomatedCompressorExplorer::initialPopulation()
{
    // Use all prebuilt compressors
    auto prebuilt = getPrebuiltCompressors(inputType());
    std::vector<ACECompressor> population;
    population.insert(population.end(), prebuilt.begin(), prebuilt.end());

    // Use populationSize() random compressors
    for (size_t i = 0; i < populationSize(); ++i) {
        population.push_back(
                buildRandomCompressor(rng(), inputType(), formatVersion_));
    }
    return population;
}

namespace {
/// Put small evolutionary pressure on the results to be simpler.
/// Adjusts cSize on the scale of 0.1%, and times on the scale of 1%, given a
/// typical number of components of ~10.
void adjustResults(const ACECompressor& gene, std::vector<float>& results)
{
    std::array<float, 3> scale = { 0.0001, 0.001, 0.001 };
    const size_t numComponents = gene.numComponents();
    for (size_t i = 0; i < results.size(); ++i) {
        const auto delta = results[i] * scale[i] * numComponents;
        results[i] += std::max<float>(numComponents, delta);
    }
}

std::vector<float> fitnessFromResult(
        const ACECompressor& gene,
        const poly::optional<ACECompressionResult>& result)
{
    if (!result.has_value()) {
        return std::vector<float>(3, std::numeric_limits<float>::infinity());
    }
    auto fitness = result->asFloatVector();
    adjustResults(gene, fitness);
    return fitness;
}

/// The bytes that `benchmark()` compresses for @p input. For a String input,
/// that includes the string lengths.
size_t benchmarkBytes(const Input& input)
{
    size_t bytes = input.contentSize();
    if (input.type() == Type::String) {
        bytes += input.numElts() * sizeof(uint32_t);
    }
    return bytes;
}

/// Benchmarks @p batch with contexts that the calling thread keeps across
/// calls, so that only a thread's first benchmark runs on cold contexts.
poly::optional<ACECompressionResult> benchmarkOnThreadContexts(
        const Compressor& compressor,
        poly::span<const Input> batch)
{
    thread_local CCtx cctx;
    thread_local DCtx dctx;
    return benchmark(cctx, dctx, compressor, batch);
}
} // namespace

/* static */ std::vector<poly::span<const Input>>
AutomatedCompressorExplorer::splitInputs(
        poly::span<const Input> inputs,
        size_t numBatches)
{
    size_t totalBytes = 0;
    for (const auto& input : inputs) {
        totalBytes += benchmarkBytes(input);
    }
    numBatches = std::clamp<size_t>(
            totalBytes / kMinBatchBytes, 1, std::max<size_t>(numBatches, 1));

    // Batch k ends before the first input that starts at or after k /
    // numBatches of the total bytes.
    std::vector<poly::span<const Input>> batches;
    size_t begin = 0;
    size_t bytes = 0;
    size_t batch = 1;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i > begin && batch < numBatches
            && bytes * numBatches >= batch * totalBytes) {
            batches.push_back(inputs.subspan(begin, i - begin));
            begin = i;
            while (batch < numBatches
                   && bytes * numBatches >= batch * totalBytes) {
                ++batch;
            }
        }
        bytes += benchmarkBytes(inputs[i]);
    }
    if (begin < inputs.size()) {
        batches.push_back(inputs.subspan(begin, inputs.size() - begin));
    }
    return batches;
}

std::vector<float> AutomatedCompressorExplorer::computeFitness(
        const ACECompressor& gene)
{
    if (gene.maxStreamExpansion() > kMaxStreamExpansion) {
        return fitnessFromResult(gene, poly::nullopt);
    }
    return fitnessFromResult(
            gene, gene.benchmark(inputs_, formatVersion_, compressionLevel_));
}

std::vector<std::vector<float>> AutomatedCompressorExplorer::computeFitness(
        poly::span<const ACECompressor> genes)
{
    struct GeneEval {
        std::reference_wrapper<const ACECompressor> gene;
        poly::optional<std::vector<float>> cachedFitness;
        std::vector<std::future<poly::optional<ACECompressionResult>>>
                batchResults;
    };
    // A gene that reuses its cached fitness gets no tasks. An uncached gene,
    // or a cached gene picked to run again, gets one task per input batch.
    std::vector<GeneEval> evals;
    evals.reserve(genes.size());
    for (const auto& gene : genes) {
        auto& eval = evals.emplace_back(
                GeneEval{ std::cref(gene), poly::nullopt, {} });
        auto cache = cachedFitness_.find(gene.hash());
        if (cache != cachedFitness_.end()) {
            std::uniform_int_distribution<size_t> dist(0, cache->second.first);
            if (dist(rng()) != 0) {
                eval.cachedFitness = cache->second.second;
                continue;
            }
        }
        // A candidate that multiplies its input is not run: see
        // kMaxStreamExpansion.
        if (gene.maxStreamExpansion() > kMaxStreamExpansion) {
            eval.cachedFitness = fitnessFromResult(gene, poly::nullopt);
            continue;
        }
        // The gene's batch tasks share one compressor. Each task holds a
        // reference, so the compressor outlives this call if it throws.
        const std::shared_ptr<const Compressor> compressor =
                std::make_shared<Compressor>(gene.buildBenchmarkCompressor(
                        formatVersion_, compressionLevel_));
        eval.batchResults.reserve(inputBatches_.size());
        for (const auto& batch : inputBatches_) {
            eval.batchResults.emplace_back(threadPool_.run([compressor, batch] {
                return benchmarkOnThreadContexts(*compressor, batch);
            }));
        }
    }

    std::vector<std::vector<float>> results;
    results.reserve(genes.size());
    for (auto& eval : evals) {
        std::vector<float> result;
        if (eval.cachedFitness.has_value()) {
            result = std::move(*eval.cachedFitness);
        } else {
            poly::optional<ACECompressionResult> total = ACECompressionResult{};
            for (auto& future : eval.batchResults) {
                auto batchResult = future.get();
                if (!batchResult.has_value()) {
                    total = poly::nullopt;
                } else if (total.has_value()) {
                    *total += *batchResult;
                }
            }
            result = fitnessFromResult(eval.gene.get(), total);
        }
        auto [cache, inserted] = cachedFitness_.emplace(
                eval.gene.get().hash(), std::make_pair(1, result));
        if (!inserted) {
            for (size_t j = 0; j < result.size(); ++j) {
                cache->second.second[j] =
                        std::min(cache->second.second[j], result[j]);
            }
            ++cache->second.first;
        }
        results.push_back(std::move(result));
    }
    return results;
}

namespace {
struct ArenaDeleter {
    void operator()(Arena* arena) const
    {
        ALLOC_Arena_freeArena(arena);
    }
};
} // namespace

std::string AutomatedCompressorExplorer::savePopulation() const
{
    std::unique_ptr<Arena, ArenaDeleter> arena(ALLOC_HeapArena_create());
    auto a         = A1C_Arena_wrap(arena.get());
    A1C_Item* item = A1C_Item_root(&a);
    if (item == nullptr) {
        throw std::bad_alloc();
    }
    A1C_Item* array = A1C_Item_array(item, population().size(), &a);
    if (array == nullptr) {
        throw std::bad_alloc();
    }
    for (const auto& gene : population()) {
        const auto s = gene.serialize();
        if (!A1C_Item_string_copy(array++, s.data(), s.size(), &a)) {
            throw std::bad_alloc();
        }
    }
    const size_t size = A1C_Item_encodedSize(item);
    std::string serialized(size, '\0');
    const size_t actual = A1C_Item_encode(
            item, (uint8_t*)serialized.data(), serialized.size(), nullptr);
    if (size != actual) {
        throw Exception("Serialization failed");
    }
    return serialized;
}

void AutomatedCompressorExplorer::loadPopulation(poly::string_view snapshot)
{
    std::unique_ptr<Arena, ArenaDeleter> arena(ALLOC_HeapArena_create());
    A1C_Decoder decoder;
    A1C_Decoder_init(&decoder, A1C_Arena_wrap(arena.get()), {});
    const A1C_Item* item = A1C_Decoder_decode(
            &decoder, (const uint8_t*)snapshot.data(), snapshot.size());
    if (item == nullptr) {
        throw Exception("Failed to deserialize");
    }
    if (item->type != A1C_ItemType_array) {
        throw Exception("Must be an array");
    }
    std::vector<ACECompressor> population;
    population.reserve(item->array.size);
    for (size_t i = 0; i < item->array.size; ++i) {
        const A1C_Item& p = item->array.items[i];
        if (p.type != A1C_ItemType_string) {
            throw Exception("Array elements must be strings");
        }
        population.emplace_back(
                poly::string_view{ p.string.data, p.string.size });
    }
    extendPopulation(std::move(population));
}

} // namespace training
} // namespace openzl
