// Trains the stabilizing controller with the in-repo PPO implementation.
//
// Episodes are collected sequentially rather than from interleaved parallel
// environments, because the rollout buffer stores trajectories as contiguous
// segments and generalized advantage estimation walks backwards through them.
// Interleaving would scramble that ordering; the environment is fast enough
// that there is nothing to gain by it.

#include <plasma/control/TokamakEnv.h>
#include <plasma/rl/Ppo.h>

#include <psim/core/Clock.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

using namespace plasma;

namespace {

struct Options
{
    int updates{ 200 };
    int stepsPerUpdate{ 4096 };
    int evaluationEpisodes{ 16 };
    int evaluateEvery{ 10 };
    std::uint64_t seed{ 1 };
    std::string checkpoint{ "policy.bin" };
    std::string log{ "training.csv" };
    Real learningRate{ 3.0e-4 };
};

struct EpisodeSummary
{
    Real totalReward{ 0 };
    int  steps{ 0 };
    bool disrupted{ false };
    Real finalIsland{ 0 };
    Real meanConfinement{ 0 };
    Real meanAlfvenic{ 0 };
    RewardTerms meanTerms{};
};

void accumulate(RewardTerms& into, const RewardTerms& sample)
{
    into.confinement += sample.confinement;
    into.tearing     += sample.tearing;
    into.alfvenic    += sample.alfvenic;
    into.smoothness  += sample.smoothness;
    into.chatter     += sample.chatter;
    into.saturation  += sample.saturation;
    into.survival    += sample.survival;
    into.disruption  += sample.disruption;
    into.total       += sample.total;
}

void scale(RewardTerms& terms, Real factor)
{
    terms.confinement *= factor;
    terms.tearing     *= factor;
    terms.alfvenic    *= factor;
    terms.smoothness  *= factor;
    terms.chatter     *= factor;
    terms.saturation  *= factor;
    terms.survival    *= factor;
    terms.disruption  *= factor;
    terms.total       *= factor;
}

/// One episode under a fixed action, used for the baselines.
EpisodeSummary runFixed(TokamakEnv& env, std::uint64_t seed, std::span<const Real> action)
{
    env.reset(seed);
    EpisodeSummary summary;

    for (int step = 0; step < env.config().maxSteps; ++step) {
        const StepResult result = env.step(action);
        summary.totalReward += result.reward;
        summary.meanConfinement += env.equilibrium().confinementFraction();
        summary.meanAlfvenic += env.alfvenicAmplitude();
        accumulate(summary.meanTerms, env.rewardTerms());
        ++summary.steps;
        if (result.terminated) {
            summary.disrupted = true;
            break;
        }
        if (result.truncated) break;
    }

    const Real inverse = 1.0 / static_cast<Real>(std::max(summary.steps, 1));
    summary.meanConfinement *= inverse;
    summary.meanAlfvenic *= inverse;
    scale(summary.meanTerms, inverse);
    summary.finalIsland = env.widestIsland();
    return summary;
}

/// One episode under the greedy policy.
EpisodeSummary runPolicy(TokamakEnv& env, const PpoAgent& agent, std::uint64_t seed,
                         std::vector<Real>& observation, std::vector<Real>& action)
{
    env.reset(seed);
    EpisodeSummary summary;

    for (int step = 0; step < env.config().maxSteps; ++step) {
        env.observe(observation);
        agent.actGreedy(observation, action);

        const StepResult result = env.step(action);
        summary.totalReward += result.reward;
        summary.meanConfinement += env.equilibrium().confinementFraction();
        summary.meanAlfvenic += env.alfvenicAmplitude();
        accumulate(summary.meanTerms, env.rewardTerms());
        ++summary.steps;
        if (result.terminated) {
            summary.disrupted = true;
            break;
        }
        if (result.truncated) break;
    }

    const Real inverse = 1.0 / static_cast<Real>(std::max(summary.steps, 1));
    summary.meanConfinement *= inverse;
    summary.meanAlfvenic *= inverse;
    scale(summary.meanTerms, inverse);
    summary.finalIsland = env.widestIsland();
    return summary;
}

struct Evaluation
{
    Real meanReturn{ 0 };
    Real disruptionRate{ 0 };
    Real meanSteps{ 0 };
    Real meanIsland{ 0 };
    Real meanConfinement{ 0 };
    RewardTerms terms{};
};

Evaluation evaluatePolicy(TokamakEnv& env, const PpoAgent& agent, int episodes,
                          std::uint64_t seedBase, std::vector<Real>& observation,
                          std::vector<Real>& action)
{
    Evaluation evaluation;
    for (int i = 0; i < episodes; ++i) {
        // A fixed evaluation seed block, held constant across updates, so the
        // learning curve reflects the policy improving rather than the episodes
        // getting easier.
        const EpisodeSummary summary = runPolicy(env, agent, seedBase + static_cast<std::uint64_t>(i),
                                                 observation, action);
        evaluation.meanReturn += summary.totalReward;
        evaluation.disruptionRate += summary.disrupted ? 1.0 : 0.0;
        evaluation.meanSteps += static_cast<Real>(summary.steps);
        evaluation.meanIsland += summary.finalIsland;
        evaluation.meanConfinement += summary.meanConfinement;
        accumulate(evaluation.terms, summary.meanTerms);
    }
    const Real inverse = 1.0 / static_cast<Real>(std::max(episodes, 1));
    evaluation.meanReturn *= inverse;
    evaluation.disruptionRate *= inverse;
    evaluation.meanSteps *= inverse;
    evaluation.meanIsland *= inverse;
    evaluation.meanConfinement *= inverse;
    scale(evaluation.terms, inverse);
    return evaluation;
}

Options parseArguments(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](auto& target) {
            if (i + 1 < argc) {
                if constexpr (std::is_integral_v<std::decay_t<decltype(target)>>) {
                    target = static_cast<std::decay_t<decltype(target)>>(std::atoll(argv[++i]));
                } else {
                    target = static_cast<std::decay_t<decltype(target)>>(std::atof(argv[++i]));
                }
            }
        };
        if (std::strcmp(argv[i], "--updates") == 0) next(options.updates);
        else if (std::strcmp(argv[i], "--steps") == 0) next(options.stepsPerUpdate);
        else if (std::strcmp(argv[i], "--eval-episodes") == 0) next(options.evaluationEpisodes);
        else if (std::strcmp(argv[i], "--eval-every") == 0) next(options.evaluateEvery);
        else if (std::strcmp(argv[i], "--seed") == 0) next(options.seed);
        else if (std::strcmp(argv[i], "--lr") == 0) next(options.learningRate);
        else if (std::strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) options.checkpoint = argv[++i];
        else if (std::strcmp(argv[i], "--log") == 0 && i + 1 < argc) options.log = argv[++i];
        else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf("usage: %s [--updates N] [--steps N] [--eval-episodes N]\n"
                        "          [--eval-every N] [--seed N] [--lr X]\n"
                        "          [--checkpoint FILE] [--log FILE]\n", argv[0]);
            std::exit(0);
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv)
{
    const Options options = parseArguments(argc, argv);

    TokamakEnvConfig environmentConfig;
    environmentConfig.equilibrium.edgeSafetyFactor  = 3.4;
    environmentConfig.equilibrium.currentPeaking    = 2.4;
    environmentConfig.equilibrium.bootstrapFraction = 0.35;
    environmentConfig.equilibrium.lundquist         = 1.0e4;

    TokamakEnv env;
    env.configure(environmentConfig);

    TokamakEnv evaluationEnv;
    evaluationEnv.configure(environmentConfig);

    PpoConfig ppo;
    ppo.stepsPerUpdate = options.stepsPerUpdate;
    ppo.learningRate   = options.learningRate;

    PpoAgent agent;
    agent.configure(static_cast<int>(env.observationSize()),
                    static_cast<int>(TokamakEnv::actionSize()), ppo, options.seed);

    std::vector<Real> observation(env.observationSize());
    std::vector<Real> action(TokamakEnv::actionSize());

    std::printf("plasma-stabilizer training\n");
    std::printf("  observation %zu, action %zu, parameters %zu\n",
                env.observationSize(), TokamakEnv::actionSize(), agent.parameterCount());
    std::printf("  control period %.2f Alfven times (%.2f ms), %d steps per episode (%.0f ms pulse)\n",
                environmentConfig.controlPeriod,
                environmentConfig.controlPeriod * environmentConfig.alfvenTimeSeconds * 1e3,
                environmentConfig.maxSteps,
                environmentConfig.maxSteps * environmentConfig.controlPeriod *
                    environmentConfig.alfvenTimeSeconds * 1e3);

    // --- Baselines ----------------------------------------------------------
    // Two references: doing nothing, and the best constant action found by a
    // coarse sweep. A policy that cannot beat a well-chosen constant has not
    // learned to control anything, only to sit still in a good place.
    const std::vector<Real> passive = { -1.0, 0.0, 1.0, -1.0, 0.0, -1.0 };
    // Current drive at the rational surface with the beam just below the
    // Alfvenic threshold: the best fixed action found by sweeping the space.
    // It is a hard bar on purpose -- beating it requires actually reacting to
    // the plasma rather than settling on a good average.
    const std::vector<Real> tuned   = { 1.0, 0.05, -0.24, -1.0, 0.0, -0.385 };

    const std::uint64_t evaluationSeedBase = 900000;
    auto baseline = [&](const std::vector<Real>& fixed) {
        Evaluation result;
        for (int i = 0; i < options.evaluationEpisodes; ++i) {
            const EpisodeSummary summary =
                runFixed(evaluationEnv, evaluationSeedBase + static_cast<std::uint64_t>(i), fixed);
            result.meanReturn += summary.totalReward;
            result.disruptionRate += summary.disrupted ? 1.0 : 0.0;
            result.meanIsland += summary.finalIsland;
            result.meanConfinement += summary.meanConfinement;
        }
        const Real inverse = 1.0 / static_cast<Real>(options.evaluationEpisodes);
        result.meanReturn *= inverse;
        result.disruptionRate *= inverse;
        result.meanIsland *= inverse;
        result.meanConfinement *= inverse;
        return result;
    };

    const Evaluation passiveBaseline = baseline(passive);
    const Evaluation tunedBaseline   = baseline(tuned);

    std::printf("\nbaselines over %d fixed evaluation episodes\n", options.evaluationEpisodes);
    std::printf("  %-22s return %8.1f  disrupt %5.2f  island %6.3f  confinement %5.2f\n",
                "passive (full beam)", passiveBaseline.meanReturn, passiveBaseline.disruptionRate,
                passiveBaseline.meanIsland, passiveBaseline.meanConfinement);
    std::printf("  %-22s return %8.1f  disrupt %5.2f  island %6.3f  confinement %5.2f\n",
                "best constant action", tunedBaseline.meanReturn, tunedBaseline.disruptionRate,
                tunedBaseline.meanIsland, tunedBaseline.meanConfinement);

    std::FILE* logFile = std::fopen(options.log.c_str(), "w");
    if (logFile != nullptr) {
        std::fprintf(logFile,
                     "update,steps,episodes,train_return,eval_return,disruption_rate,eval_steps,"
                     "island,confinement,r_confinement,r_tearing,r_alfvenic,r_smoothness,"
                     "r_chatter,r_saturation,policy_loss,value_loss,entropy,kl,clip_fraction,"
                     "explained_variance,log_std,seconds\n");
    }

    std::printf("\n%6s %9s %10s %10s %8s %8s %8s %8s %7s\n",
                "update", "steps", "train ret", "eval ret", "disrupt", "island", "conf", "expl var", "sec");

    psim::Clock wallClock;
    std::uint64_t episodeSeed = options.seed * 7919 + 1;
    std::int64_t totalSteps = 0;
    int totalEpisodes = 0;
    Real bestEvaluation = -1e30;

    for (int update = 1; update <= options.updates; ++update) {
        agent.beginRollout();

        Real rolloutReturn = 0;
        int rolloutEpisodes = 0;

        while (!agent.rolloutFull()) {
            env.reset(episodeSeed++);
            Real episodeReturn = 0;

            for (int step = 0; step < env.config().maxSteps; ++step) {
                env.observe(observation);

                Real logProbability = 0;
                Real value = 0;
                agent.act(observation, action, logProbability, value);

                const StepResult result = env.step(action);
                agent.record(observation, action, logProbability, value, result.reward,
                             result.terminated);

                episodeReturn += result.reward;
                ++totalSteps;

                if (result.terminated || result.truncated) {
                    // A truncated episode still has a future, so its value has to
                    // be bootstrapped; a disrupted one does not.
                    Real bootstrap = 0;
                    if (!result.terminated) {
                        env.observe(observation);
                        bootstrap = agent.value(observation);
                    }
                    agent.endTrajectory(bootstrap);
                    break;
                }
            }

            rolloutReturn += episodeReturn;
            ++rolloutEpisodes;
            ++totalEpisodes;
        }

        const UpdateStats stats = agent.update();
        const Real trainReturn = rolloutReturn / static_cast<Real>(std::max(rolloutEpisodes, 1));

        if (update % options.evaluateEvery == 0 || update == options.updates || update == 1) {
            const Evaluation evaluation =
                evaluatePolicy(evaluationEnv, agent, options.evaluationEpisodes,
                               evaluationSeedBase, observation, action);

            std::printf("%6d %9lld %10.1f %10.1f %8.2f %8.3f %8.2f %8.2f %7.0f\n",
                        update, static_cast<long long>(totalSteps), trainReturn,
                        evaluation.meanReturn, evaluation.disruptionRate, evaluation.meanIsland,
                        evaluation.meanConfinement, stats.explainedVariance, wallClock.elapsed());
            std::fflush(stdout);

            if (logFile != nullptr) {
                std::fprintf(logFile,
                             "%d,%lld,%d,%.4f,%.4f,%.4f,%.2f,%.5f,%.4f,"
                             "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                             "%.5f,%.5f,%.4f,%.6f,%.4f,%.4f,%.4f,%.1f\n",
                             update, static_cast<long long>(totalSteps), totalEpisodes,
                             trainReturn, evaluation.meanReturn, evaluation.disruptionRate,
                             evaluation.meanSteps, evaluation.meanIsland, evaluation.meanConfinement,
                             evaluation.terms.confinement, evaluation.terms.tearing,
                             evaluation.terms.alfvenic, evaluation.terms.smoothness,
                             evaluation.terms.chatter, evaluation.terms.saturation,
                             stats.policyLoss, stats.valueLoss, stats.entropy,
                             stats.approximateKl, stats.clipFraction, stats.explainedVariance,
                             agent.meanLogStd(), wallClock.elapsed());
                std::fflush(logFile);
            }

            if (evaluation.meanReturn > bestEvaluation) {
                bestEvaluation = evaluation.meanReturn;
                agent.save(options.checkpoint);
            }
        }
    }

    if (logFile != nullptr) std::fclose(logFile);

    const double seconds = wallClock.elapsed();
    std::printf("\ntrained %d episodes (%lld steps) in %.0f s -> %.0f episodes/hour\n",
                totalEpisodes, static_cast<long long>(totalSteps), seconds,
                static_cast<double>(totalEpisodes) / seconds * 3600.0);
    std::printf("best evaluation return %.1f (passive %.1f, best constant %.1f)\n",
                bestEvaluation, passiveBaseline.meanReturn, tunedBaseline.meanReturn);
    std::printf("checkpoint written to %s\n", options.checkpoint.c_str());
    return 0;
}
