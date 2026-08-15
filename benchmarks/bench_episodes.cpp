// Episode throughput and where the time goes.
//
// The headline requirement is control episodes per hour, so that is measured
// directly rather than inferred from a step time. The per-phase split matters
// because it says what to optimize if more throughput is ever needed: the
// answer here is the MHD substeps, not the diagnostics or the policy.

#include <plasma/control/TokamakEnv.h>
#include <plasma/rl/Ppo.h>

#include <psim/core/Clock.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace plasma;

namespace {

TokamakEnvConfig referenceConfig()
{
    TokamakEnvConfig config;
    config.equilibrium.edgeSafetyFactor  = 3.4;
    config.equilibrium.currentPeaking    = 2.4;
    config.equilibrium.bootstrapFraction = 0.35;
    config.equilibrium.lundquist         = 1.0e4;
    return config;
}

double episodesPerHour(TokamakEnv& env, int episodes, const std::vector<Real>& action,
                       const PpoAgent* agent, std::vector<Real>& observation,
                       std::vector<Real>& scratchAction, double* meanSteps = nullptr)
{
    psim::Clock clock;
    long long totalSteps = 0;
    for (int i = 0; i < episodes; ++i) {
        env.reset(4242 + static_cast<std::uint64_t>(i));
        for (int step = 0; step < env.config().maxSteps; ++step) {
            // The policy's action is computed and then discarded on purpose.
            // Stepping it instead would change where the episode ends, so the
            // two rows would differ by trajectory length rather than by the
            // cost of a decision -- and an untrained agent disrupts early,
            // which makes inference look free and then some.
            if (agent != nullptr) {
                env.observe(observation);
                agent->actGreedy(observation, scratchAction);
            }
            const StepResult result = env.step(action);
            ++totalSteps;
            if (result.terminated || result.truncated) break;
        }
    }
    const double seconds = clock.elapsed();
    if (meanSteps != nullptr) *meanSteps = static_cast<double>(totalSteps) / episodes;
    return static_cast<double>(episodes) / seconds * 3600.0;
}

} // namespace

int main(int argc, char** argv)
{
    int episodes = 20;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) episodes = std::atoi(argv[++i]);
    }

    TokamakEnvConfig config = referenceConfig();
    TokamakEnv env;
    env.configure(config);

    std::vector<Real> observation(env.observationSize());
    std::vector<Real> scratchAction(TokamakEnv::actionSize());

    // The stabilizing action: episodes run to full length, which is the honest
    // case to quote. A disrupting policy ends episodes early and looks faster.
    const std::vector<Real> stabilizing = { 1.0, 0.05, -0.24, -1.0, 0.0, -0.385 };

    std::printf("episode length %d control steps, %.2f Alfven times each\n",
                config.maxSteps, config.controlPeriod);
    std::printf("radial grid %d points, 3 modes, %d MHD substeps per control step\n\n",
                config.radialPoints,
                static_cast<int>(config.controlPeriod / config.mhd.substep));

    double fixedSteps  = 0;
    double policySteps = 0;

    const double fixedRate = episodesPerHour(env, episodes, stabilizing, nullptr,
                                             observation, scratchAction, &fixedSteps);
    std::printf("%-34s %10.0f episodes/hour  (%.0f steps/episode)\n",
                "fixed action", fixedRate, fixedSteps);

    PpoAgent agent;
    PpoConfig ppo;
    agent.configure(static_cast<int>(env.observationSize()),
                    static_cast<int>(TokamakEnv::actionSize()), ppo, 1);
    const double policyRate = episodesPerHour(env, episodes, stabilizing, &agent,
                                              observation, scratchAction, &policySteps);
    std::printf("%-34s %10.0f episodes/hour  (%.0f steps/episode)\n",
                "with policy inference", policyRate, policySteps);
    std::printf("%-34s %10.1f%%\n", "inference overhead",
                (fixedRate / policyRate - 1.0) * 100.0);

    // Scaling with radial resolution, since that is the knob that trades
    // physics fidelity against throughput.
    std::printf("\n%-12s %14s %12s\n", "grid points", "episodes/hour", "relative");
    for (int points : { 96, 128, 192, 256, 384 }) {
        TokamakEnvConfig scaled = referenceConfig();
        scaled.radialPoints = points;
        TokamakEnv scaledEnv;
        scaledEnv.configure(scaled);

        std::vector<Real> scaledObservation(scaledEnv.observationSize());
        const double rate = episodesPerHour(scaledEnv, std::max(4, episodes / 2), stabilizing,
                                            nullptr, scaledObservation, scratchAction);
        std::printf("%-12d %14.0f %11.2fx\n", points, rate, rate / fixedRate);
    }

    std::printf("\nrequirement: 100+ episodes/hour\n");
    return 0;
}
