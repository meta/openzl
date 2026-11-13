// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "openzl/cpp/poly/Optional.hpp"
#include "openzl/cpp/poly/Span.hpp"

namespace openzl {
namespace training {
namespace detail {
template <typename T, typename KeyFn>
void sortByKey(std::vector<T>& data, KeyFn&& keyFn, bool reverse = false)
{
    std::sort(
            data.begin(),
            data.end(),
            [&keyFn, &reverse](const T& lhs, const T& rhs) {
                return reverse ? keyFn(rhs) < keyFn(lhs)
                               : keyFn(lhs) < keyFn(rhs);
            });
}
} // namespace detail

/**
 * Computes the crowding distance for each point in @p subset,
 * which is a measure of how unique the solution is.
 *
 * @param fitness The fitness of each point
 * @param subset The subset of points to consider as indices into @p fitness
 * @returns The crowding distance
 */
std::vector<float> crowdingDistance(
        poly::span<const std::vector<float>> fitness,
        poly::span<const size_t> subset);

/// @returns true iff @p lhs dominates @p rhs, where smaller values are better
bool dominates(poly::span<const float> lhs, poly::span<const float> rhs);

/**
 * Computes the Pareto-optimal fronts for the given @p fitness values.
 * See the NSGA-II paper for details:
 * https://ieeexplore.ieee.org/document/996017
 *
 * @param fitness The fitness of each point
 * @returns The fronts and the rank. The fronts are each Pareto-optimal front.
 * fronts[0] are Pareto-optimal points, fronts[1] are only dominated by
 * Pareto-optimal points, and so on. rank is a reverse mapping from point to
 * front.
 */
std::pair<std::vector<std::vector<size_t>>, std::vector<size_t>>
fastNonDominatedSort(poly::span<const std::vector<float>> fitness);

class Selector {
   public:
    /**
     * Selects a parent for reproduction.
     *
     * @param rank The rank of each point.
     * @param crowdingDistance The crowding distance of each point.
     * @returns The index of the parent to select for reproduction.
     */
    virtual size_t select(
            poly::span<const size_t> rank,
            poly::span<const float> crowdingDistance) = 0;

    virtual ~Selector() = default;
};

/**
 * Selector that uses Tournament Selection:
 * https://en.wikipedia.org/wiki/Tournament_selection
 */
class TournamentSelector : public Selector {
   public:
    struct Parameters {
        size_t tournamentSize{ 3 };
        float torunamentSelectionProbability{ 0.9 };
        uint64_t seed{ 0 };
    };

    TournamentSelector() : TournamentSelector(Parameters{}) {}

    explicit TournamentSelector(Parameters const& params)
            : params_(params), gen_(params.seed)
    {
        if (params.tournamentSize < 1) {
            throw std::logic_error("Tournament size must be at least 1");
        }
    }

    const Parameters& parameters() const
    {
        return params_;
    }

    size_t select(
            poly::span<const size_t> rank,
            poly::span<const float> crowdingDistance) override;

   private:
    std::vector<size_t> getCandidates(size_t populationSize);

    Parameters params_;
    std::mt19937_64 gen_;
};

/**
 * Genetic algorithm base class.
 *
 * GeneT must be:
 * - copyable
 * - hashable
 */
template <typename GeneT>
class GeneticAlgorithm {
   public:
    static_assert(std::is_copy_constructible<GeneT>::value);
    static_assert(std::is_copy_assignable<GeneT>::value);

    struct Parameters {
        /// Size of the population to carry forward each generation
        size_t populationSize{ 100 };
        /// Maximum number of generations to run
        size_t maxGenerations{ 250 };
        /// Maximum time to run the algorithm in seconds
        poly::optional<std::chrono::seconds> maxTime{};
        /// Probability of mutating a child after crossover
        float mutationProbability{ 0.2 };
        /// Random seed
        uint32_t seed{ 0 };
        TournamentSelector::Parameters selectorParameters{};
    };

    GeneticAlgorithm(
            Parameters const& params                           = Parameters(),
            poly::optional<std::unique_ptr<Selector>> selector = poly::nullopt)
            : params_(params), gen_(params.seed), selector_(nullptr)
    {
        if (selector.has_value()) {
            selector_ = std::move(selector.value());
        } else {
            // Ensure the selector's seed isn't exactly the same as the GA's
            params_.selectorParameters.seed ^= gen_();
            selector_ = std::make_unique<TournamentSelector>(
                    TournamentSelector::Parameters{ .seed = gen_() });
        }
        if (params_.maxTime.has_value()) {
            deadline_ =
                    std::chrono::steady_clock::now() + params_.maxTime.value();
        }
    }

    virtual ~GeneticAlgorithm() = default;

    /**
     * Called before running the first generation to initialize the population.
     * This function should produce a diverse set of solutions. Additionally,
     * known possibly good solutions should be included in this set. E.g. hand
     * crafted solutions, or solutions from previous runs of the algorithm.
     */
    virtual std::vector<GeneT> initialPopulation() = 0;

    /**
     * Cross over @p parent1 and @p parent2 to produce a child gene.
     * @note This does not include mutation.
     * @returns A child that share properies from @p parent1 and @p parent2
     */
    virtual GeneT crossover(const GeneT& parent1, const GeneT& parent2) = 0;

    /**
     * Mutate the @p parent gene to produce a child.
     */
    virtual GeneT mutate(const GeneT& parent) = 0;

    /**
     * Compute the fitness of @p gene.
     *
     * @note Smaller values are considered better.
     */
    virtual std::vector<float> computeFitness(const GeneT& gene) = 0;

    /**
     * Computes the fitness for a list of @p genes.
     * This method may be overridden to allow optimizations like parallel
     * fitness computation.
     */
    virtual std::vector<std::vector<float>> computeFitness(
            poly::span<const GeneT> genes)
    {
        std::vector<std::vector<float>> f;
        f.reserve(genes.size());
        for (const auto& gene : genes) {
            f.push_back(computeFitness(gene));
        }
        return f;
    }

    /**
     * Selects two parents, crosses them over to produce a child, and maybe
     * mutates it.
     */
    virtual GeneT reproduce()
    {
        const auto& parent1 = selectParent();
        const auto& parent2 = selectParent();

        auto child = crossover(parent1, parent2);
        if (shouldMutate_(gen_) < mutationProbability()) {
            child = mutate(child);
        }

        return child;
    }

    /**
     * Produces @p numChildren using reproduce(), while ensuring the children
     * are unique and not in the parent generation.
     */
    virtual std::unordered_set<GeneT> reproduce(size_t numChildren)
    {
        std::unordered_set<GeneT> children;
        size_t maxIterations = 2 * numChildren;
        while (children.size() < numChildren && maxIterations-- > 0) {
            auto child = reproduce();
            if (populationSet_.count(child) == 0) {
                children.insert(std::move(child));
            }
        }
        return children;
    }

    const Parameters& parameters() const
    {
        return params_;
    }

    size_t populationSize() const
    {
        return params_.populationSize;
    }

    size_t maxGenerations() const
    {
        return params_.maxGenerations;
    }

    const poly::optional<std::chrono::steady_clock::time_point>& deadline()
            const
    {
        return deadline_;
    }

    float mutationProbability() const
    {
        return params_.mutationProbability;
    }

    Selector& selector()
    {
        return *selector_;
    }

    poly::span<const GeneT> population() const
    {
        return population_;
    }

    const std::unordered_set<GeneT> populationSet() const
    {
        return populationSet_;
    }

    poly::span<const size_t> rank() const
    {
        return rank_;
    }

    poly::span<const std::vector<float>> fitness() const
    {
        return fitness_;
    }

    poly::span<const float> crowdingDistance() const
    {
        return crowdingDistance_;
    }

    std::mt19937_64& rng()
    {
        return gen_;
    }

    size_t generation() const
    {
        return generation_;
    }

    /**
     * Selects a parent for reproduction.
     */
    const GeneT& selectParent()
    {
        auto idx = selector().select(rank(), crowdingDistance());
        return population()[idx];
    }

    /**
     * Adds @p genes to the population and updates the state accordingly.
     */
    std::vector<std::vector<size_t>> extendPopulation(
            std::vector<GeneT>&& genes)
    {
        return extendPopulation(genes.begin(), genes.end());
    }

    /**
     * Clears the population and updates the state accordingly.
     */
    void clearPopulation()
    {
        populationSet_.clear();
        population_.clear();
        fitness_.clear();
        rank_.clear();
        crowdingDistance_.clear();
    }

    /**
     * Reduces the population by preserving only the genes listed in @p subset
     * and updates the state accordingly.
     */
    void subsetPopulation(poly::span<const size_t> subset)
    {
        auto population = population_;
        auto fitness    = fitness_;

        clearPopulation();
        for (auto idx : subset) {
            populationSet_.insert(population[idx]);
            population_.push_back(population[idx]);
            fitness_.push_back(fitness[idx]);
        }
        updateRankAndCrowdingDistance();
    }

    /**
     * Runs one generation of the genetic algorithm.
     */
    void step()
    {
        if (generation_ == 0) {
            extendPopulation(initialPopulation());
        }

        std::vector<std::vector<size_t>> fronts;
        auto children = reproduce(populationSize());
        fronts        = extendPopulation(children.begin(), children.end());

        std::vector<size_t> subset;
        subset.reserve(populationSize());
        size_t rank;
        for (rank = 0; rank < fronts.size(); ++rank) {
            if (subset.size() + fronts[rank].size() > populationSize()) {
                break;
            }
            subset.insert(
                    subset.end(), fronts[rank].begin(), fronts[rank].end());
        }

        const size_t needed = populationSize() - subset.size();
        if (needed > 0) {
            detail::sortByKey(fronts[rank], [&](size_t idx) {
                return -crowdingDistance_[idx];
            });
            subset.insert(
                    subset.end(),
                    fronts[rank].begin(),
                    fronts[rank].begin() + needed);
        }
        subsetPopulation(subset);
        ++generation_;
    }

    /// @returns The progress in [0, 1] of the algorithm taking both generations
    /// and the deadline into account.
    double progress() const
    {
        auto generationProgress = (double)generation() / maxGenerations();

        if (!deadline().has_value()) {
            return generationProgress;
        }

        std::chrono::nanoseconds remaining =
                deadline().value() - std::chrono::steady_clock::now();
        std::chrono::nanoseconds maxTime = params_.maxTime.value();
        auto timeProgress = 1.0 - (double)remaining.count() / maxTime.count();

        return std::min(1.0, std::max(generationProgress, timeProgress));
    }

    bool finished() const
    {
        if (generation_ >= maxGenerations()) {
            return true;
        }
        if (generation_ == 0) {
            // Always run at least one generation
            return false;
        }
        if (deadline().has_value()
            && std::chrono::steady_clock::now() > deadline().value()) {
            return true;
        }
        return false;
    }

    /**
     * Runs the genetic algorithm.
     */
    void run()
    {
        while (!finished()) {
            step();
        }
    }

    /**
     * @returns The current Pareto-optimal solutions found by the algorithm
     * as (gene, fitness) pairs sorted by the fitness vector.
     */
    std::vector<std::pair<GeneT, std::vector<float>>> solution() const
    {
        std::vector<std::pair<GeneT, std::vector<float>>> result;
        for (size_t i = 0; i < population_.size(); ++i) {
            if (rank_[i] == 0) {
                result.emplace_back(population_[i], fitness_[i]);
            }
        }
        detail::sortByKey(result, [](auto r) { return r.second; });
        return result;
    }

   private:
    template <typename It>
    std::vector<std::vector<size_t>> extendPopulation(It begin, It end)
    {
        const size_t oldSize = population_.size();
        for (; begin != end; ++begin) {
            auto inserted = populationSet_.insert(*begin).second;
            if (inserted) {
                population_.push_back(std::move(*begin));
            }
        }
        auto fitness = computeFitness(
                poly::span<const GeneT>(
                        population_.data() + oldSize,
                        population_.data() + population_.size()));
        fitness_.insert(fitness_.end(), fitness.begin(), fitness.end());
        return updateRankAndCrowdingDistance();
    }

    void computeCrowdingDistance(const std::vector<std::vector<size_t>>& fronts)
    {
        crowdingDistance_.resize(population_.size(), 0.0);
        for (const auto& front : fronts) {
            auto dist = training::crowdingDistance(fitness_, front);
            for (size_t i = 0; i < front.size(); ++i) {
                crowdingDistance_[front[i]] = dist[i];
            }
        }
    }

    std::vector<std::vector<size_t>> updateRankAndCrowdingDistance()
    {
        auto [fronts, rank] = fastNonDominatedSort(fitness_);
        rank_               = std::move(rank);
        computeCrowdingDistance(fronts);
        return fronts;
    }

    Parameters params_;
    std::mt19937_64 gen_;
    std::unique_ptr<Selector> selector_;
    std::uniform_real_distribution<float> shouldMutate_{ 0, 1 };
    size_t generation_{ 0 };
    std::unordered_set<GeneT> populationSet_;
    std::vector<GeneT> population_;
    std::vector<std::vector<float>> fitness_;
    std::vector<size_t> rank_;
    std::vector<float> crowdingDistance_;
    poly::optional<std::chrono::steady_clock::time_point> deadline_;
};

} // namespace training
} // namespace openzl
