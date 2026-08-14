#include <plasma/rl/Ppo.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace plasma {

namespace {

constexpr Real kLogTwoPi = 1.8378770664093454836;

/// Log-probability of `action` under a diagonal Gaussian.
Real gaussianLogProbability(std::span<const Real> action, std::span<const Real> mean,
                            std::span<const Real> logStd)
{
    Real total = 0;
    for (std::size_t i = 0; i < action.size(); ++i) {
        const Real sigma = std::exp(logStd[i]);
        const Real z     = (action[i] - mean[i]) / sigma;
        total += -0.5 * z * z - logStd[i] - 0.5 * kLogTwoPi;
    }
    return total;
}

} // namespace

// ---------------------------------------------------------------------------
// RunningNormalizer
// ---------------------------------------------------------------------------

void RunningNormalizer::configure(std::size_t size)
{
    m_mean.assign(size, 0.0);
    m_variance.assign(size, 1.0);
    m_count = 0;
}

void RunningNormalizer::observe(std::span<const Real> sample)
{
    ++m_count;
    const Real inverse = 1.0 / static_cast<Real>(m_count);
    for (std::size_t i = 0; i < m_mean.size() && i < sample.size(); ++i) {
        const Real delta = sample[i] - m_mean[i];
        m_mean[i] += delta * inverse;
        // Welford: the running second moment uses both the old and new means,
        // which is what makes it stable over millions of samples.
        m_variance[i] += (delta * (sample[i] - m_mean[i]) - m_variance[i]) * inverse;
    }
}

void RunningNormalizer::apply(std::span<const Real> in, std::span<Real> out) const
{
    for (std::size_t i = 0; i < out.size() && i < in.size(); ++i) {
        const Real deviation = std::sqrt(std::max(m_variance[i], 1e-8));
        out[i] = std::clamp((in[i] - m_mean[i]) / deviation, -8.0, 8.0);
    }
}

void RunningNormalizer::writeState(std::vector<Real>& out) const
{
    out.clear();
    out.push_back(static_cast<Real>(m_count));
    out.insert(out.end(), m_mean.begin(), m_mean.end());
    out.insert(out.end(), m_variance.begin(), m_variance.end());
}

bool RunningNormalizer::readState(std::span<const Real> in)
{
    if (in.size() != 1 + m_mean.size() * 2) return false;
    m_count = static_cast<std::uint64_t>(in[0]);
    std::copy_n(in.begin() + 1, m_mean.size(), m_mean.begin());
    std::copy_n(in.begin() + 1 + static_cast<std::ptrdiff_t>(m_mean.size()), m_variance.size(),
                m_variance.begin());
    return true;
}

// ---------------------------------------------------------------------------
// PpoAgent
// ---------------------------------------------------------------------------

void PpoAgent::configure(int observationSize, int actionSize, const PpoConfig& config,
                         std::uint64_t seed)
{
    m_config          = config;
    m_observationSize = observationSize;
    m_actionSize      = actionSize;

    MlpConfig policyConfig;
    policyConfig.inputs  = observationSize;
    policyConfig.hidden  = config.hidden;
    policyConfig.outputs = actionSize;
    policyConfig.outputScale = 0.01;
    m_policy.configure(policyConfig, seed);

    MlpConfig criticConfig = policyConfig;
    criticConfig.outputs     = 1;
    criticConfig.outputScale = 1.0;
    m_critic.configure(criticConfig, seed ^ 0x5DEECE66Dull);

    m_logStd.assign(static_cast<std::size_t>(actionSize), config.initialLogStd);
    m_logStdGradient.assign(static_cast<std::size_t>(actionSize), 0.0);
    m_logStdMoment1.assign(static_cast<std::size_t>(actionSize), 0.0);
    m_logStdMoment2.assign(static_cast<std::size_t>(actionSize), 0.0);

    m_normalizer.configure(static_cast<std::size_t>(observationSize));

    m_scratchObservation.assign(static_cast<std::size_t>(observationSize), 0.0);
    m_scratchAction.assign(static_cast<std::size_t>(actionSize), 0.0);
    m_scratchGradient.assign(static_cast<std::size_t>(actionSize), 0.0);

    m_rng.seed(seed == 0 ? 0x2545F4914F6CDD1Dull : seed ^ 0x9E3779B97F4A7C15ull);
    m_adamStep = 0;

    beginRollout();
}

void PpoAgent::normalizedObservation(std::span<const Real> raw, std::span<Real> out) const
{
    if (m_config.normalizeObservations && m_normalizer.count() > 1) {
        m_normalizer.apply(raw, out);
    } else {
        std::copy(raw.begin(), raw.begin() + static_cast<std::ptrdiff_t>(out.size()), out.begin());
    }
}

void PpoAgent::act(std::span<const Real> observation, std::span<Real> action,
                   Real& logProbability, Real& value)
{
    // The normalizer is updated with every observation the agent sees, so its
    // statistics track the distribution the current policy actually visits.
    if (m_config.normalizeObservations) m_normalizer.observe(observation);
    normalizedObservation(observation, m_scratchObservation);

    m_policy.forward(m_scratchObservation, m_scratchAction);

    std::normal_distribution<Real> gaussian(0.0, 1.0);
    for (int i = 0; i < m_actionSize; ++i) {
        const Real sigma = std::exp(m_logStd[static_cast<std::size_t>(i)]);
        action[static_cast<std::size_t>(i)] =
            m_scratchAction[static_cast<std::size_t>(i)] + sigma * gaussian(m_rng);
    }

    logProbability = gaussianLogProbability(action.subspan(0, static_cast<std::size_t>(m_actionSize)),
                                            m_scratchAction, m_logStd);

    Real critic = 0;
    m_critic.forward(m_scratchObservation, std::span<Real>(&critic, 1));
    value = critic * m_returnScale + m_returnMean;
}

void PpoAgent::actGreedy(std::span<const Real> observation, std::span<Real> action) const
{
    normalizedObservation(observation, m_scratchObservation);
    m_policy.predict(m_scratchObservation, action);
}

Real PpoAgent::value(std::span<const Real> observation) const
{
    normalizedObservation(observation, m_scratchObservation);
    Real result = 0;
    m_critic.predict(m_scratchObservation, std::span<Real>(&result, 1));
    return result * m_returnScale + m_returnMean;
}

void PpoAgent::beginRollout()
{
    m_observations.clear();
    m_actions.clear();
    m_logProbabilities.clear();
    m_values.clear();
    m_rewards.clear();
    m_terminated.clear();
    m_segmentEnd.clear();
    m_segmentBootstrap.clear();
}

void PpoAgent::record(std::span<const Real> observation, std::span<const Real> action,
                      Real logProbability, Real value, Real reward, bool terminated)
{
    // Observations are stored already normalized: the statistics move during a
    // rollout, and re-normalizing at update time with the final statistics would
    // evaluate the policy on inputs it never actually saw.
    normalizedObservation(observation, m_scratchObservation);
    m_observations.insert(m_observations.end(), m_scratchObservation.begin(),
                          m_scratchObservation.end());
    m_actions.insert(m_actions.end(), action.begin(),
                     action.begin() + static_cast<std::ptrdiff_t>(m_actionSize));
    m_logProbabilities.push_back(logProbability);
    m_values.push_back(value);
    m_rewards.push_back(reward);
    m_terminated.push_back(terminated ? 1 : 0);
}

void PpoAgent::endTrajectory(Real bootstrapValue)
{
    m_segmentEnd.push_back(static_cast<int>(m_rewards.size()));
    m_segmentBootstrap.push_back(bootstrapValue);
}

void PpoAgent::computeAdvantages()
{
    const int total = static_cast<int>(m_rewards.size());
    m_advantages.assign(static_cast<std::size_t>(total), 0.0);
    m_returns.assign(static_cast<std::size_t>(total), 0.0);

    int begin = 0;
    for (std::size_t segment = 0; segment < m_segmentEnd.size(); ++segment) {
        const int end = m_segmentEnd[segment];
        Real nextValue = m_segmentBootstrap[segment];
        Real advantage = 0;

        // Generalized advantage estimation, accumulated backwards through the
        // segment. A terminated step has no successor, so its bootstrap is zero
        // and the recursion restarts there.
        for (int t = end - 1; t >= begin; --t) {
            const auto index = static_cast<std::size_t>(t);
            const Real notDone = m_terminated[index] ? 0.0 : 1.0;
            const Real delta = m_rewards[index] + m_config.discount * nextValue * notDone -
                               m_values[index];
            advantage = delta + m_config.discount * m_config.gaeLambda * notDone * advantage;

            m_advantages[index] = advantage;
            m_returns[index]    = advantage + m_values[index];
            nextValue = m_values[index];
        }
        begin = end;
    }

    if (m_config.normalizeAdvantages && total > 1) {
        const Real mean = std::accumulate(m_advantages.begin(), m_advantages.end(), 0.0) /
                          static_cast<Real>(total);
        Real variance = 0;
        for (Real value : m_advantages) variance += (value - mean) * (value - mean);
        const Real deviation = std::sqrt(variance / static_cast<Real>(total)) + 1e-8;
        for (Real& value : m_advantages) value = (value - mean) / deviation;
    }
}

UpdateStats PpoAgent::update()
{
    UpdateStats stats;
    const int total = static_cast<int>(m_rewards.size());
    if (total < 2) {
        beginRollout();
        return stats;
    }

    // Any trajectory still open when the rollout filled is closed by the caller;
    // this guards the case where it was not.
    if (m_segmentEnd.empty() || m_segmentEnd.back() != total) {
        endTrajectory(0.0);
    }

    computeAdvantages();

    // Refresh the return statistics from this rollout before training the
    // critic on standardized targets.
    {
        const Real mean = std::accumulate(m_returns.begin(), m_returns.end(), 0.0) /
                          static_cast<Real>(total);
        Real variance = 0;
        for (Real value : m_returns) variance += (value - mean) * (value - mean);
        const Real deviation = std::sqrt(variance / static_cast<Real>(total)) + 1e-6;

        if (!m_returnStatsReady) {
            m_returnMean  = mean;
            m_returnScale = deviation;
            m_returnStatsReady = true;
        } else {
            // Blended slowly: a hard reset each update would move the critic's
            // output scale underneath it between rollouts.
            m_returnMean  = 0.9 * m_returnMean + 0.1 * mean;
            m_returnScale = 0.9 * m_returnScale + 0.1 * deviation;
        }
    }

    std::vector<int> order(static_cast<std::size_t>(total));
    std::iota(order.begin(), order.end(), 0);

    std::vector<Real> mean(static_cast<std::size_t>(m_actionSize));
    std::vector<Real> policyGradient(static_cast<std::size_t>(m_actionSize));

    Real klSum = 0;
    Real clipSum = 0;
    int  sampleCount = 0;

    for (int epoch = 0; epoch < m_config.epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), m_rng);

        for (int start = 0; start < total; start += m_config.minibatchSize) {
            const int stop = std::min(start + m_config.minibatchSize, total);
            const int size = stop - start;
            if (size < 2) continue;

            m_policy.zeroGradients();
            m_critic.zeroGradients();
            std::fill(m_logStdGradient.begin(), m_logStdGradient.end(), 0.0);

            Real batchPolicyLoss = 0;
            Real batchValueLoss  = 0;

            for (int k = start; k < stop; ++k) {
                const auto index = static_cast<std::size_t>(order[static_cast<std::size_t>(k)]);
                const std::span<const Real> observation(
                    m_observations.data() + index * static_cast<std::size_t>(m_observationSize),
                    static_cast<std::size_t>(m_observationSize));
                const std::span<const Real> action(
                    m_actions.data() + index * static_cast<std::size_t>(m_actionSize),
                    static_cast<std::size_t>(m_actionSize));

                m_policy.forward(observation, mean);

                const Real logProbability = gaussianLogProbability(action, mean, m_logStd);
                const Real ratio = std::exp(logProbability - m_logProbabilities[index]);
                const Real advantage = m_advantages[index];

                // Clipped surrogate. Where the clip is active the objective is
                // flat, so the gradient is exactly zero -- that is the whole
                // mechanism keeping the update inside the trust region.
                const Real clipped = std::clamp(ratio, 1.0 - m_config.clipRatio,
                                                1.0 + m_config.clipRatio);
                const bool clipActive = (ratio * advantage) > (clipped * advantage);
                if (clipActive) ++clipSum;

                const Real surrogate = std::min(ratio * advantage, clipped * advantage);
                batchPolicyLoss -= surrogate;

                const Real surrogateGradient = clipActive ? 0.0 : (ratio * advantage);

                // d(logProbability)/d(mean) and /d(logStd) for a diagonal Gaussian.
                for (int i = 0; i < m_actionSize; ++i) {
                    const auto slot  = static_cast<std::size_t>(i);
                    const Real sigma = std::exp(m_logStd[slot]);
                    const Real z     = (action[slot] - mean[slot]) / sigma;

                    policyGradient[slot] = -surrogateGradient * z / sigma;
                    m_logStdGradient[slot] += -surrogateGradient * (z * z - 1.0);
                    // Entropy bonus depends only on logStd for a Gaussian.
                    m_logStdGradient[slot] -= m_config.entropyCoefficient;
                }
                m_policy.backward(policyGradient);

                Real predicted = 0;
                m_critic.forward(observation, std::span<Real>(&predicted, 1));
                const Real target = (m_returns[index] - m_returnMean) / m_returnScale;
                const Real error  = predicted - target;
                batchValueLoss += 0.5 * error * error;

                const Real criticGradient = m_config.valueCoefficient * error;
                m_critic.backward(std::span<const Real>(&criticGradient, 1));

                klSum += m_logProbabilities[index] - logProbability;
                ++sampleCount;
            }

            const Real inverse = 1.0 / static_cast<Real>(size);
            m_policy.scaleGradients(inverse);
            m_critic.scaleGradients(inverse);
            for (Real& value : m_logStdGradient) value *= inverse;

            m_policy.clipGradients(m_config.maxGradientNorm);
            m_critic.clipGradients(m_config.maxGradientNorm);

            ++m_adamStep;
            m_policy.applyAdam(m_config.learningRate, m_adamStep);
            m_critic.applyAdam(m_config.learningRate, m_adamStep);

            // Adam on the log standard deviation, then a floor so exploration
            // cannot collapse to zero and freeze the policy.
            const Real correction1 = 1.0 - std::pow(0.9, static_cast<Real>(m_adamStep));
            const Real correction2 = 1.0 - std::pow(0.999, static_cast<Real>(m_adamStep));
            for (std::size_t i = 0; i < m_logStd.size(); ++i) {
                const Real g = m_logStdGradient[i];
                m_logStdMoment1[i] = 0.9 * m_logStdMoment1[i] + 0.1 * g;
                m_logStdMoment2[i] = 0.999 * m_logStdMoment2[i] + 0.001 * g * g;
                const Real corrected1 = m_logStdMoment1[i] / correction1;
                const Real corrected2 = m_logStdMoment2[i] / correction2;
                m_logStd[i] -= m_config.learningRate * corrected1 /
                               (std::sqrt(corrected2) + 1e-8);
                m_logStd[i] = std::max(m_logStd[i], m_config.minimumLogStd);
            }

            stats.policyLoss += batchPolicyLoss * inverse;
            stats.valueLoss  += batchValueLoss * inverse;
            ++stats.minibatches;
        }
    }

    if (stats.minibatches > 0) {
        stats.policyLoss /= static_cast<Real>(stats.minibatches);
        stats.valueLoss  /= static_cast<Real>(stats.minibatches);
    }
    if (sampleCount > 0) {
        stats.approximateKl = klSum / static_cast<Real>(sampleCount);
        stats.clipFraction  = clipSum / static_cast<Real>(sampleCount);
    }

    Real entropy = 0;
    for (Real value : m_logStd) entropy += value + 0.5 * (kLogTwoPi + 1.0);
    stats.entropy = entropy;

    // Explained variance of the value function: 1 means the critic accounts for
    // all the variation in returns, 0 means it is no better than the mean.
    const Real meanReturn = std::accumulate(m_returns.begin(), m_returns.end(), 0.0) /
                            static_cast<Real>(total);
    Real returnVariance = 0;
    Real residualVariance = 0;
    for (int i = 0; i < total; ++i) {
        const auto index = static_cast<std::size_t>(i);
        returnVariance += (m_returns[index] - meanReturn) * (m_returns[index] - meanReturn);
        const Real residual = m_returns[index] - m_values[index];
        residualVariance += residual * residual;
    }
    stats.explainedVariance = returnVariance > 0 ? 1.0 - residualVariance / returnVariance : 0.0;

    beginRollout();
    return stats;
}

std::size_t PpoAgent::parameterCount() const
{
    return m_policy.parameterCount() + m_critic.parameterCount() + m_logStd.size();
}

Real PpoAgent::meanLogStd() const
{
    if (m_logStd.empty()) return 0;
    return std::accumulate(m_logStd.begin(), m_logStd.end(), 0.0) /
           static_cast<Real>(m_logStd.size());
}

bool PpoAgent::save(const std::string& path) const
{
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) return false;

    std::vector<Real> policy, critic, normalizer;
    m_policy.writeParameters(policy);
    m_critic.writeParameters(critic);
    m_normalizer.writeState(normalizer);

    const std::int32_t header[4] = { m_observationSize, m_actionSize,
                                     static_cast<std::int32_t>(policy.size()),
                                     static_cast<std::int32_t>(critic.size()) };
    bool ok = std::fwrite(header, sizeof(header), 1, file) == 1;
    ok = ok && std::fwrite(policy.data(), sizeof(Real), policy.size(), file) == policy.size();
    ok = ok && std::fwrite(critic.data(), sizeof(Real), critic.size(), file) == critic.size();
    ok = ok && std::fwrite(m_logStd.data(), sizeof(Real), m_logStd.size(), file) == m_logStd.size();
    const std::int32_t normalizerSize = static_cast<std::int32_t>(normalizer.size());
    ok = ok && std::fwrite(&normalizerSize, sizeof(normalizerSize), 1, file) == 1;
    ok = ok && std::fwrite(normalizer.data(), sizeof(Real), normalizer.size(), file) == normalizer.size();

    std::fclose(file);
    return ok;
}

bool PpoAgent::load(const std::string& path)
{
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) return false;

    std::int32_t header[4] = {};
    bool ok = std::fread(header, sizeof(header), 1, file) == 1;
    if (!ok || header[0] != m_observationSize || header[1] != m_actionSize) {
        std::fclose(file);
        return false;
    }

    std::vector<Real> policy(static_cast<std::size_t>(header[2]));
    std::vector<Real> critic(static_cast<std::size_t>(header[3]));
    ok = ok && std::fread(policy.data(), sizeof(Real), policy.size(), file) == policy.size();
    ok = ok && std::fread(critic.data(), sizeof(Real), critic.size(), file) == critic.size();
    ok = ok && std::fread(m_logStd.data(), sizeof(Real), m_logStd.size(), file) == m_logStd.size();

    std::int32_t normalizerSize = 0;
    ok = ok && std::fread(&normalizerSize, sizeof(normalizerSize), 1, file) == 1;
    std::vector<Real> normalizer(static_cast<std::size_t>(std::max(normalizerSize, 0)));
    ok = ok && std::fread(normalizer.data(), sizeof(Real), normalizer.size(), file) == normalizer.size();

    std::fclose(file);

    ok = ok && m_policy.readParameters(policy);
    ok = ok && m_critic.readParameters(critic);
    ok = ok && m_normalizer.readState(normalizer);
    return ok;
}

} // namespace plasma
