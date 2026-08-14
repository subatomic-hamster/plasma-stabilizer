#pragma once

// Proximal Policy Optimization, written out in full.
//
// PPO rather than an off-policy method because this environment is cheap to
// sample (thousands of episodes an hour) and the reward is a weighted sum of
// competing objectives whose balance shifts as the policy improves -- on-policy
// updates track that, a replay buffer full of stale trade-offs does not.
//
// The policy is a diagonal Gaussian with a state-independent standard
// deviation. State-dependent variance is standard in some implementations and
// is a liability here: it lets the policy become nearly deterministic in the
// states it has already mastered, which is exactly where a rate-limited
// actuator most needs exploration to discover that acting earlier is better.

#include <plasma/rl/Mlp.h>

#include <random>
#include <string>

namespace plasma {

/// Welford running mean and variance, used to standardize observations.
/// Diagnostic channels differ by orders of magnitude -- a log-compressed coil
/// envelope against a normalized temperature -- and an unnormalized network
/// spends its first thousand updates just learning the scales.
class RunningNormalizer
{
public:
    void configure(std::size_t size);
    void observe(std::span<const Real> sample);
    void apply(std::span<const Real> in, std::span<Real> out) const;

    [[nodiscard]] std::size_t size() const noexcept { return m_mean.size(); }
    [[nodiscard]] std::uint64_t count() const noexcept { return m_count; }

    void writeState(std::vector<Real>& out) const;
    bool readState(std::span<const Real> in);

private:
    std::vector<Real> m_mean;
    std::vector<Real> m_variance;
    std::uint64_t     m_count{ 0 };
};

struct PpoConfig
{
    std::vector<int> hidden{ 64, 64 };

    Real learningRate{ 3.0e-4 };
    Real clipRatio{ 0.2 };
    Real valueCoefficient{ 0.5 };
    /// Small. The usual reason to want a large entropy bonus is to escape local
    /// optima, but here it fights the thing that matters most: the deposition
    /// radius has to be held within about 0.15 of the rational surface, and an
    /// exploration width larger than that never lets the policy see the reward
    /// for aiming correctly.
    Real entropyCoefficient{ 0.0015 };
    Real discount{ 0.995 };
    Real gaeLambda{ 0.95 };
    Real maxGradientNorm{ 0.5 };

    int epochs{ 6 };
    int minibatchSize{ 256 };
    /// Environment steps gathered before each update.
    int stepsPerUpdate{ 4096 };

    /// Initial log standard deviation of the action distribution. exp(-1.6) is
    /// about 0.20, narrower than the window of deposition radii that actually
    /// suppress the island -- with the wider default the policy explores across
    /// the whole plasma and averages the aiming signal away.
    Real initialLogStd{ -1.0 };
    /// Floor, so exploration never collapses entirely.
    Real minimumLogStd{ -3.0 };

    bool normalizeObservations{ true };
    bool normalizeAdvantages{ true };
};

struct UpdateStats
{
    Real policyLoss{ 0 };
    Real valueLoss{ 0 };
    Real entropy{ 0 };
    Real approximateKl{ 0 };
    Real clipFraction{ 0 };
    Real explainedVariance{ 0 };
    int  minibatches{ 0 };
};

class PpoAgent
{
public:
    void configure(int observationSize, int actionSize, const PpoConfig& config, std::uint64_t seed);

    /// Samples an action and reports the log-probability and value estimate the
    /// update will need. `observation` is raw; normalization happens inside.
    void act(std::span<const Real> observation, std::span<Real> action,
             Real& logProbability, Real& value);

    /// Mean action with no exploration noise: what a deployed controller runs.
    void actGreedy(std::span<const Real> observation, std::span<Real> action) const;

    /// Value estimate alone, for bootstrapping a truncated episode.
    [[nodiscard]] Real value(std::span<const Real> observation) const;

    // --- Rollout collection -------------------------------------------------

    void beginRollout();
    void record(std::span<const Real> observation, std::span<const Real> action,
                Real logProbability, Real value, Real reward, bool terminated);
    /// Marks the end of a trajectory segment. `bootstrapValue` is the value of
    /// the state that follows, or zero if the episode genuinely ended.
    void endTrajectory(Real bootstrapValue);

    [[nodiscard]] int collectedSteps() const noexcept { return static_cast<int>(m_rewards.size()); }
    [[nodiscard]] bool rolloutFull() const noexcept
    {
        return collectedSteps() >= m_config.stepsPerUpdate;
    }

    /// Runs the PPO update over the collected rollout and clears it.
    UpdateStats update();

    // --- Persistence --------------------------------------------------------

    bool save(const std::string& path) const;
    bool load(const std::string& path);

    [[nodiscard]] const PpoConfig& config() const noexcept { return m_config; }
    [[nodiscard]] std::size_t parameterCount() const;
    [[nodiscard]] Real meanLogStd() const;

private:
    void computeAdvantages();
    void normalizedObservation(std::span<const Real> raw, std::span<Real> out) const;

    PpoConfig m_config;
    int m_observationSize{ 0 };
    int m_actionSize{ 0 };

    Mlp m_policy;
    Mlp m_critic;
    std::vector<Real> m_logStd;
    std::vector<Real> m_logStdGradient;
    std::vector<Real> m_logStdMoment1;
    std::vector<Real> m_logStdMoment2;

    RunningNormalizer m_normalizer;

    /// Running scale of the discounted returns. The critic is trained on
    /// standardized targets and its output is rescaled on the way out: episode
    /// returns here reach several hundred, and a freshly initialized network
    /// asked to predict that directly spends the whole run catching up, which
    /// showed as a large negative explained variance.
    Real m_returnMean{ 0 };
    Real m_returnScale{ 1 };
    bool m_returnStatsReady{ false };

    // Rollout storage, flat and reused between updates.
    std::vector<Real> m_observations; ///< normalized at collection time
    std::vector<Real> m_actions;
    std::vector<Real> m_logProbabilities;
    std::vector<Real> m_values;
    std::vector<Real> m_rewards;
    std::vector<std::uint8_t> m_terminated;
    /// Index one past the end of each trajectory segment, with its bootstrap.
    std::vector<int>  m_segmentEnd;
    std::vector<Real> m_segmentBootstrap;

    std::vector<Real> m_advantages;
    std::vector<Real> m_returns;

    mutable std::vector<Real> m_scratchObservation;
    std::vector<Real> m_scratchAction;
    std::vector<Real> m_scratchGradient;

    std::mt19937_64 m_rng;
    std::uint64_t   m_adamStep{ 0 };
};

} // namespace plasma
