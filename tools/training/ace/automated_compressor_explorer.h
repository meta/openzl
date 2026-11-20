// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <thread>
#include <vector>

#include "openzl/openzl.hpp"

#include "tools/training/ace/ace_compressor.h"
#include "tools/training/ace/ace_compressors.h"
#include "tools/training/ace/ace_crossover.h"
#include "tools/training/ace/ace_mutate.h"
#include "tools/training/utils/genetic_algorithm.h"
#include "tools/training/utils/thread_pool.h"

namespace openzl {
namespace training {

/**
 * @brief A genetic algorithm for finding a good OpenZL compressor for a set of
 * inputs.
 *
 * The AutomatedCompressorExplorer, or ACE for short, searches for a
 * Pareto-optimal set of compressors for a given set of inputs. After
 * construction, simply call `AutomatedCompressorExplorer::run()` and then
 * `AutomatedCompressorExplorer::solution()`. The solutions are sorted by
 * increasing compressed size. After selecting a solution, call
 * `ACECompressor::build()` to build the compression graph in a
 * `openzl::Compressor`.
 *
 * The state of the algorithm can be saved & reloaded with
 * `AutomatedCompressorExplorer::savePopulation()` and
 * `AutomatedCompressorExplorer::loadPopulation()`. If training on a similar set
 * of inputs, re-loading the population from a previous run can speed up
 * training. It also never hurts to load a previous population (except for the
 * one time cost to benchmark performance of the population), even if it is
 * completely unrelated to the current inputs.
 *
 * @warning ACE built compressors are not guaranteed to succeed on every
 * possible input. They are guaranteed to succeed on every input they are
 * trained on, but thats it. If the training inputs are representitive, it is
 * very likely the compressor will succeed on most inputs. It is recommended to
 * use them with permissive mode enabled so compression always succeeds.
 *
 * @note The compressors are currently only built from static components, so
 * they cannot react at runtime to differences in data. It is future work to
 * integrate ML selectors into ACE so it can compress different inputs
 * separately.
 */
class AutomatedCompressorExplorer : public GeneticAlgorithm<ACECompressor> {
   public:
    using Base = GeneticAlgorithm<ACECompressor>;

    struct Parameters : public Base::Parameters {
        size_t numThreads{ std::thread::hardware_concurrency() / 2 };
        /// Target format version. Must be set explicitly; a value of 0 is
        /// rejected at construction.
        uint32_t formatVersion{ 0 };
        /// Global compression level used while benchmarking candidates.
        int compressionLevel{ 0 };
    };

    /**
     * @param inputs The inputs to build a compressor for. These inputs
     * must outlive the `AutomatedCompressorExplorer`. Each input must be
     * the same type.
     */
    explicit AutomatedCompressorExplorer(poly::span<const Input> inputs)
            : AutomatedCompressorExplorer(inputs, Parameters{})
    {
    }

    /**
     * @param inputs The inputs to build a compressor for. These inputs
     * must outlive the `AutomatedCompressorExplorer`. Each input must be
     * the same type.
     * @param params Parameters that control the genetic algorithm. Good
     * defaults are chosen, but performance may be improved by tuning them.
     */
    AutomatedCompressorExplorer(
            poly::span<const Input> inputs,
            const Parameters& params)
            : Base(params),
              inputs_(std::move(inputs)),
              formatVersion_(params.formatVersion),
              compressionLevel_(params.compressionLevel),
              threadPool_(params.numThreads),
              crossover_(rng(), inputType(), formatVersion_),
              mutate_(rng(), inputType(), formatVersion_)
    {
        if (formatVersion_ == 0) {
            throw Exception("Format version must be set in the parameters");
        }
        for (auto const& input : inputs_) {
            if (input.type() != inputType()) {
                throw Exception("All inputs must have the same type");
            }
        }
        inputBatches_ = splitInputs(inputs_, params.numThreads);
    }

    AutomatedCompressorExplorer(
            poly::span<const Input> inputs,
            poly::string_view snapshot)
            : AutomatedCompressorExplorer(inputs, snapshot, Parameters{})
    {
    }

    AutomatedCompressorExplorer(
            poly::span<const Input> inputs,
            poly::string_view snapshot,
            const Parameters& params)
            : AutomatedCompressorExplorer(inputs, params)
    {
        loadPopulation(snapshot);
    }

    Type inputType() const
    {
        if (inputs_.empty()) {
            throw Exception("No inputs provided");
        }
        return inputs_[0].type();
    }

    poly::span<const Input> inputs() const
    {
        return inputs_;
    }

    uint32_t formatVersion() const
    {
        return formatVersion_;
    }

    int compressionLevel() const
    {
        return compressionLevel_;
    }

    /**
     * Saves the current population to a string.
     */
    std::string savePopulation() const;

    /**
     * Extends the current population with the population saved to @p snapshot.
     * This can be run at any point during the exploration to merge the current
     * population with a previously saved population. However, only
     * `populationSize()` compressors will be carried over to the next
     * generation.
     *
     * Extending the population from a snapshot can speed up convergence if the
     * snapshot compressors are relevant. If they are irrelevant, it doesn't
     * hurt, except for the cost to benchmark the snapshot's population.
     */
    void loadPopulation(poly::string_view snapshot);

    std::vector<ACECompressor> initialPopulation() override;

    ACECompressor crossover(
            const ACECompressor& parent1,
            const ACECompressor& parent2) override
    {
        return crossover_(parent1, parent2);
    }

    ACECompressor mutate(const ACECompressor& parent) override
    {
        return mutate_(parent);
    }

    std::vector<float> computeFitness(const ACECompressor& gene) override;

    std::vector<std::vector<float>> computeFitness(
            poly::span<const ACECompressor> genes) override;

    static constexpr size_t kAceStateParamId = 592;

    /// Candidates whose `ACECompressor::maxStreamExpansion()` is above this
    /// are given the fitness of a failed compression without being
    /// benchmarked. The memory such a candidate needs is set by its graph, not
    /// by the data, and grows by up to 8x for each chained bitunpack node. A
    /// single bitunpack of 1 bit (8x) stays allowed.
    static constexpr double kMaxStreamExpansion = 16.0;

   private:
    /// Small input sets get fewer batches, so that no batch holds much less
    /// than this.
    static constexpr size_t kMinBatchBytes = size_t(1) << 20;

    /**
     * Splits @p inputs into at most @p numBatches batches of contiguous whole
     * inputs, with about the same number of the bytes that `benchmark()`
     * compresses in each batch. Uses fewer batches when a batch would hold
     * less than about kMinBatchBytes.
     */
    static std::vector<poly::span<const Input>> splitInputs(
            poly::span<const Input> inputs,
            size_t numBatches);

    poly::span<const Input> inputs_;
    /// Each gene benchmarks every batch as its own thread pool task, about one
    /// batch per thread, so one slow gene does not leave the other threads
    /// idle for a whole generation. Batches hold whole inputs, so they do not
    /// change the compressed sizes.
    std::vector<poly::span<const Input>> inputBatches_;
    uint32_t formatVersion_;
    int compressionLevel_;
    ThreadPool threadPool_;
    ACECrossover crossover_;
    ACEMutate mutate_;
    std::unordered_map<uint64_t, std::pair<size_t, std::vector<float>>>
            cachedFitness_;
};

} // namespace training
} // namespace openzl
