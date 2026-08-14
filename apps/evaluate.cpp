// Evaluates a trained controller against the reference policies, and measures
// how long it takes to decide.
//
// The latency number is the one that decides whether any of this could run on a
// machine: a control decision has to be produced inside the control period, and
// the claim is only worth making if it is measured rather than assumed. What is
// timed is exactly what a deployed controller would run -- observation packing,
// normalization, and a forward pass -- with no exploration sampling and no
// gradient machinery.

#include <plasma/control/TokamakEnv.h>
#include <plasma/rl/Ppo.h>

#include <psim/core/Clock.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace plasma;

namespace {

struct Summary
{
    Real meanReturn{ 0 };
    Real returnDeviation{ 0 };
    Real disruptionRate{ 0 };
    Real meanSteps{ 0 };
    Real meanIsland{ 0 };
    Real peakIsland{ 0 };
    Real meanConfinement{ 0 };
    Real meanAlfvenic{ 0 };
    RewardTerms terms{};
};

enum class Mode
{
    Policy,
    Fixed,
    Passive,
};

Summary evaluate(TokamakEnv& env, const PpoAgent* agent, Mode mode, int episodes,
                 std::uint64_t seedBase, std::vector<double>& latencies)
{
    std::vector<Real> observation(env.observationSize());
    std::vector<Real> action(TokamakEnv::actionSize());
    std::vector<Real> returns;
    returns.reserve(static_cast<std::size_t>(episodes));

    // Current drive parked on the rational surface with the beam just below the
    // Alfvenic threshold; the best constant action found by sweeping.
    const std::vector<Real> tuned   = { 1.0, 0.05, -0.24, -1.0, 0.0, -0.385 };
    const std::vector<Real> passive = { -1.0, -1.0, 1.0, -1.0, 0.0, -1.0 };

    Summary summary;

    for (int episode = 0; episode < episodes; ++episode) {
        env.reset(seedBase + static_cast<std::uint64_t>(episode));

        Real episodeReturn = 0;
        int steps = 0;
        Real peak = 0;
        RewardTerms accumulated{};

        for (int step = 0; step < env.config().maxSteps; ++step) {
            switch (mode) {
            case Mode::Policy: {
                psim::Clock decision;
                env.observe(observation);
                agent->actGreedy(observation, action);
                latencies.push_back(decision.elapsed() * 1.0e6);
                break;
            }
            case Mode::Fixed:
                std::copy(tuned.begin(), tuned.end(), action.begin());
                break;
            case Mode::Passive:
                std::copy(passive.begin(), passive.end(), action.begin());
                break;
            }

            const StepResult result = env.step(action);
            episodeReturn += result.reward;
            ++steps;
            peak = std::max(peak, env.widestIsland());

            const RewardTerms& terms = env.rewardTerms();
            accumulated.confinement += terms.confinement;
            accumulated.tearing     += terms.tearing;
            accumulated.alfvenic    += terms.alfvenic;
            accumulated.smoothness  += terms.smoothness;
            accumulated.chatter     += terms.chatter;
            accumulated.saturation  += terms.saturation;

            summary.meanConfinement += env.equilibrium().confinementFraction();
            summary.meanAlfvenic += env.alfvenicAmplitude();

            if (result.terminated) {
                summary.disruptionRate += 1.0;
                break;
            }
            if (result.truncated) break;
        }

        const Real inverseSteps = 1.0 / static_cast<Real>(std::max(steps, 1));
        summary.terms.confinement += accumulated.confinement * inverseSteps;
        summary.terms.tearing     += accumulated.tearing * inverseSteps;
        summary.terms.alfvenic    += accumulated.alfvenic * inverseSteps;
        summary.terms.smoothness  += accumulated.smoothness * inverseSteps;
        summary.terms.chatter     += accumulated.chatter * inverseSteps;
        summary.terms.saturation  += accumulated.saturation * inverseSteps;

        summary.meanSteps += static_cast<Real>(steps);
        summary.meanIsland += env.widestIsland();
        summary.peakIsland = std::max(summary.peakIsland, peak);
        returns.push_back(episodeReturn);
        summary.meanReturn += episodeReturn;
    }

    // Confinement and Alfvenic amplitude were accumulated once per control
    // step, so they normalize by total steps, not by episodes.
    const Real totalSteps = summary.meanSteps;
    if (totalSteps > 0) {
        summary.meanConfinement /= totalSteps;
        summary.meanAlfvenic /= totalSteps;
    }

    const Real inverse = 1.0 / static_cast<Real>(std::max(episodes, 1));
    summary.meanReturn *= inverse;
    summary.disruptionRate *= inverse;
    summary.meanSteps *= inverse;
    summary.meanIsland *= inverse;
    summary.terms.confinement *= inverse;
    summary.terms.tearing *= inverse;
    summary.terms.alfvenic *= inverse;
    summary.terms.smoothness *= inverse;
    summary.terms.chatter *= inverse;
    summary.terms.saturation *= inverse;

    Real variance = 0;
    for (Real value : returns) variance += (value - summary.meanReturn) * (value - summary.meanReturn);
    summary.returnDeviation = std::sqrt(variance * inverse);

    return summary;
}

void report(const char* label, const Summary& summary)
{
    std::printf("%-22s %8.1f +/- %6.1f  %7.2f  %7.1f  %7.3f  %7.3f  %6.2f\n",
                label, summary.meanReturn, summary.returnDeviation, summary.disruptionRate,
                summary.meanSteps, summary.meanIsland, summary.peakIsland,
                summary.meanConfinement);
}

} // namespace

int main(int argc, char** argv)
{
    std::string policyPath = "policy.bin";
    int episodes = 40;
    std::uint64_t seedBase = 900000;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--policy") == 0 && i + 1 < argc) policyPath = argv[++i];
        else if (std::strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) episodes = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seedBase = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf("usage: %s [--policy FILE] [--episodes N] [--seed N]\n", argv[0]);
            return 0;
        }
    }

    TokamakEnvConfig config;
    config.equilibrium.edgeSafetyFactor  = 3.4;
    config.equilibrium.currentPeaking    = 2.4;
    config.equilibrium.bootstrapFraction = 0.35;
    config.equilibrium.lundquist         = 1.0e4;

    TokamakEnv env;
    env.configure(config);

    PpoAgent agent;
    PpoConfig ppo;
    agent.configure(static_cast<int>(env.observationSize()),
                    static_cast<int>(TokamakEnv::actionSize()), ppo, 1);
    const bool policyLoaded = agent.load(policyPath);

    std::printf("evaluating over %d episodes (seeds %llu..%llu)\n\n", episodes,
                static_cast<unsigned long long>(seedBase),
                static_cast<unsigned long long>(seedBase + static_cast<std::uint64_t>(episodes) - 1));
    std::printf("%-22s %8s     %6s  %7s  %7s  %7s  %7s  %6s\n",
                "controller", "return", "sd", "disrupt", "steps", "island", "peak", "conf");
    std::printf("-----------------------------------------------------------------------------------------\n");

    std::vector<double> latencies;
    std::vector<double> unused;

    if (policyLoaded) {
        report("learned policy", evaluate(env, &agent, Mode::Policy, episodes, seedBase, latencies));
    } else {
        std::printf("%-22s (no checkpoint at %s)\n", "learned policy", policyPath.c_str());
    }
    report("best constant action", evaluate(env, nullptr, Mode::Fixed, episodes, seedBase, unused));
    report("passive (full beam)", evaluate(env, nullptr, Mode::Passive, episodes, seedBase, unused));

    if (!latencies.empty()) {
        std::sort(latencies.begin(), latencies.end());
        const double median = latencies[latencies.size() / 2];
        const double p99    = latencies[static_cast<std::size_t>(
            static_cast<double>(latencies.size()) * 0.99)];
        const double worst  = latencies.back();
        const double budget = config.controlPeriod * config.alfvenTimeSeconds * 1.0e6;

        std::printf("\ncontrol decision latency over %zu decisions\n", latencies.size());
        std::printf("  median %.1f us, p99 %.1f us, worst %.1f us\n", median, p99, worst);
        std::printf("  control period %.0f us, so the worst decision uses %.2f%% of the budget\n",
                    budget, 100.0 * worst / budget);
        std::printf("  and %.4f%% of a 50 ms allowance\n", 100.0 * worst / 50000.0);
    }
    return 0;
}
