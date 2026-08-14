// Trains the same controller with RLTools' PPO instead of the in-repo one.
//
// Same environment, same reward, same domain randomization -- a different
// learner. Agreement between the two is the check that the control problem is
// well posed rather than an artifact of one implementation; disagreement points
// at whichever of them is wrong.

#include <rl_tools/operations/cpu_mux.h>

#include <plasma/rl/RlToolsEnvironment.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
// RLTools parameterizes on a numeric *policy* rather than a scalar type: it
// carries separate types for storage and accumulation so a build can drop to
// half precision without touching call sites.
using TYPE_POLICY = rlt::numeric_types::Policy<float>;
using T           = typename TYPE_POLICY::DEFAULT;
using TI          = typename DEVICE::index_t;
using RNG         = typename DEVICE::SPEC::RANDOM::template ENGINE<>;

using ENVIRONMENT_SPEC = rlt::rl::environments::plasma_tokamak::Specification<T, TI>;
using ENVIRONMENT      = rlt::rl::environments::PlasmaTokamak<ENVIRONMENT_SPEC>;

struct LoopCoreParameters
    : rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>
{
    static constexpr TI BATCH_SIZE = 256;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;

    // Four environments stepped in parallel, 1024 control steps each, gives a
    // 4096-step rollout -- the same batch the in-repo learner uses, so the two
    // curves are comparable per environment step.
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 1024;
    static constexpr TI N_ENVIRONMENTS = 4;
    static constexpr TI TOTAL_STEP_LIMIT = 800000;
    static constexpr TI STEP_LIMIT =
        TOTAL_STEP_LIMIT / (ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
    static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;

    using ACTOR_OPTIMIZER_PARAMETERS = rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>;
    using CRITIC_OPTIMIZER_PARAMETERS = ACTOR_OPTIMIZER_PARAMETERS;

    struct PPO_PARAMETERS : rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>
    {
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0015;
        static constexpr TI N_EPOCHS = 6;
        static constexpr T GAMMA = 0.995;
        // Matches the in-repo learner: exploration wider than the band of
        // deposition radii that suppress the island never finds the band.
        static constexpr T INITIAL_ACTION_STD = 0.37;
        static constexpr bool NORMALIZE_OBSERVATIONS = true;
    };
};

using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<
    TYPE_POLICY, TI, RNG, ENVIRONMENT, LoopCoreParameters,
    rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, true>;

template <typename NEXT>
struct LoopEvaluationParameters : rlt::rl::loop::steps::evaluation::Parameters<TYPE_POLICY, TI, NEXT>
{
    static constexpr TI EVALUATION_INTERVAL = 2;
    static constexpr TI NUM_EVALUATION_EPISODES = 10;
    static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};

using LOOP_EVAL_CONFIG =
    rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG,
                                             LoopEvaluationParameters<LOOP_CORE_CONFIG>>;
using LOOP_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_EVAL_CONFIG>;
using LOOP_STATE  = typename LOOP_CONFIG::template State<LOOP_CONFIG>;

int main(int argc, char** argv)
{
    TI seed = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = static_cast<TI>(std::atoll(argv[++i]));
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf("usage: %s [--seed N]\n", argv[0]);
            return 0;
        }
    }

    std::printf("plasma-stabilizer: RLTools PPO backend\n");
    std::printf("  observation %d, action %d, episode limit %d\n",
                static_cast<int>(ENVIRONMENT::OBSERVATION_DIM),
                static_cast<int>(ENVIRONMENT::ACTION_DIM),
                static_cast<int>(ENVIRONMENT::EPISODE_STEP_LIMIT));
    std::printf("  %d environments x %d steps per rollout, %d total steps\n\n",
                static_cast<int>(LoopCoreParameters::N_ENVIRONMENTS),
                static_cast<int>(LoopCoreParameters::ON_POLICY_RUNNER_STEPS_PER_ENV),
                static_cast<int>(LoopCoreParameters::TOTAL_STEP_LIMIT));

    DEVICE device;
    LOOP_STATE state;
    rlt::malloc(device, state);
    rlt::init(device, state, seed);

    while (!rlt::step(device, state)) {
        // The evaluation step in the loop prints its own returns.
    }

    rlt::free(device, state);
    return 0;
}
