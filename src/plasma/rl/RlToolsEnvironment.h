#pragma once

// Adapter exposing TokamakEnv to RLTools.
//
// The point of a second learner is not more training throughput; it is that an
// environment and a learning algorithm written by the same person can agree
// with each other and both be wrong. RLTools' PPO was written by somebody else,
// against a different interface, with its own advantage estimation, its own
// normalization and its own optimizer. If the plasma is controllable, it should
// be controllable by that too.
//
// RLTools is statically dispatched: an environment is a type with nested State,
// Parameters and Observation types plus free functions found by argument-
// dependent lookup. It also copies State values freely, so the heavyweight
// simulation cannot live there. Instead State is a lightweight cursor and the
// simulation lives in the environment object, one per parallel worker, which is
// consistent because the on-policy runner only ever steps an environment
// forward from the state it last produced.

#include <plasma/control/TokamakEnv.h>

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/environments.h>

#include <memory>

namespace rl_tools::rl::environments::plasma_tokamak {

template <typename T_T, typename T_TI>
struct StateSpecification
{
    using T  = T_T;
    using TI = T_TI;
};

/// Cursor into the simulation owned by the environment. `episodeStep` is what
/// makes the state meaningful on its own: RLTools compares and copies states,
/// and a state that carried no information at all would make its episode
/// bookkeeping ambiguous.
template <typename T_SPEC>
struct State
{
    using SPEC = T_SPEC;
    using T    = typename SPEC::T;
    using TI   = typename SPEC::TI;
    static constexpr TI DIM = 1;

    TI episodeStep = 0;
};

template <typename T_TI, T_TI T_DIM>
struct Observation
{
    static constexpr T_TI DIM = T_DIM;
};

template <typename T_T, typename T_TI>
struct Specification
{
    using T  = T_T;
    using TI = T_TI;
};

struct Parameters
{
};

} // namespace rl_tools::rl::environments::plasma_tokamak

namespace rl_tools::rl::environments {

/// The environment type handed to RLTools' PPO.
template <typename T_SPEC>
struct PlasmaTokamak : Environment<typename T_SPEC::T, typename T_SPEC::TI>
{
    using SPEC = T_SPEC;
    using T    = typename SPEC::T;
    using TI   = typename SPEC::TI;

    using State      = plasma_tokamak::State<plasma_tokamak::StateSpecification<T, TI>>;
    using Parameters = plasma_tokamak::Parameters;

    /// Fixed at the default diagnostic configuration; the environment asserts
    /// the runtime size matches when it is constructed.
    static constexpr TI OBSERVATION_DIM = ::plasma::TokamakEnv::kDefaultObservationSize;
    using Observation           = plasma_tokamak::Observation<TI, OBSERVATION_DIM>;
    using ObservationPrivileged = Observation;

    static constexpr TI N_AGENTS   = 1;
    static constexpr TI ACTION_DIM = ::plasma::kActuatorCount;
    static constexpr TI EPISODE_STEP_LIMIT = 400;

    /// Owned by malloc()/free() below. Mutable because RLTools passes the
    /// environment by const reference even to step(), which is reasonable for a
    /// stateless environment and simply not the shape of this one.
    mutable std::shared_ptr<::plasma::TokamakEnv> simulation;
    mutable ::plasma::Real lastReward = 0;
    mutable bool lastTerminated = false;
    mutable std::uint64_t episodeSeed = 1;
};

} // namespace rl_tools::rl::environments

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {

template <typename DEVICE, typename SPEC>
void malloc(DEVICE&, rl::environments::PlasmaTokamak<SPEC>& env)
{
    env.simulation = std::make_shared<::plasma::TokamakEnv>();

    ::plasma::TokamakEnvConfig config;
    config.equilibrium.edgeSafetyFactor  = 3.4;
    config.equilibrium.currentPeaking    = 2.4;
    config.equilibrium.bootstrapFraction = 0.35;
    config.equilibrium.lundquist         = 1.0e4;
    config.maxSteps = static_cast<int>(rl::environments::PlasmaTokamak<SPEC>::EPISODE_STEP_LIMIT);
    env.simulation->configure(config);
}

template <typename DEVICE, typename SPEC>
void free(DEVICE&, rl::environments::PlasmaTokamak<SPEC>& env)
{
    env.simulation.reset();
}

template <typename DEVICE, typename SPEC>
void init(DEVICE&, rl::environments::PlasmaTokamak<SPEC>&)
{
}

template <typename DEVICE, typename SPEC>
void initial_parameters(DEVICE&, const rl::environments::PlasmaTokamak<SPEC>&,
                        typename rl::environments::PlasmaTokamak<SPEC>::Parameters&)
{
}

template <typename DEVICE, typename SPEC, typename RNG>
void sample_initial_parameters(DEVICE&, const rl::environments::PlasmaTokamak<SPEC>&,
                               typename rl::environments::PlasmaTokamak<SPEC>::Parameters&, RNG&)
{
}

template <typename DEVICE, typename SPEC, typename STATE_SPEC>
void initial_state(DEVICE&, const rl::environments::PlasmaTokamak<SPEC>& env,
                   typename rl::environments::PlasmaTokamak<SPEC>::Parameters&,
                   rl::environments::plasma_tokamak::State<STATE_SPEC>& state)
{
    env.simulation->reset(env.episodeSeed);
    env.lastReward = 0;
    env.lastTerminated = false;
    state.episodeStep = 0;
}

template <typename DEVICE, typename SPEC, typename STATE_SPEC, typename RNG>
void sample_initial_state(DEVICE& device, const rl::environments::PlasmaTokamak<SPEC>& env,
                          typename rl::environments::PlasmaTokamak<SPEC>::Parameters& parameters,
                          rl::environments::plasma_tokamak::State<STATE_SPEC>& state, RNG& rng)
{
    // A fresh machine every episode, drawn from the same domain randomization
    // the in-repo learner sees, so the two are solving the same problem.
    env.episodeSeed = static_cast<std::uint64_t>(random::uniform_int_distribution(
        typename DEVICE::SPEC::RANDOM{}, 1, 1000000000, rng));
    initial_state(device, env, parameters, state);
}

template <typename DEVICE, typename SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
typename SPEC::T step(DEVICE&, const rl::environments::PlasmaTokamak<SPEC>& env,
                      typename rl::environments::PlasmaTokamak<SPEC>::Parameters&,
                      const rl::environments::plasma_tokamak::State<STATE_SPEC>& state,
                      const Matrix<ACTION_SPEC>& action,
                      rl::environments::plasma_tokamak::State<STATE_SPEC>& next_state, RNG&)
{
    ::plasma::Real buffer[::plasma::kActuatorCount];
    for (std::size_t i = 0; i < ::plasma::kActuatorCount; ++i) {
        buffer[i] = static_cast<::plasma::Real>(get(action, 0, static_cast<typename SPEC::TI>(i)));
    }

    const ::plasma::StepResult result =
        env.simulation->step(std::span<const ::plasma::Real>(buffer, ::plasma::kActuatorCount));

    env.lastReward     = result.reward;
    env.lastTerminated = result.terminated;
    next_state.episodeStep = state.episodeStep + 1;

    return static_cast<typename SPEC::T>(env.simulation->config().controlPeriod *
                                         env.simulation->config().alfvenTimeSeconds);
}

template <typename DEVICE, typename SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
typename SPEC::T reward(DEVICE&, const rl::environments::PlasmaTokamak<SPEC>& env,
                        typename rl::environments::PlasmaTokamak<SPEC>::Parameters&,
                        const rl::environments::plasma_tokamak::State<STATE_SPEC>&,
                        const Matrix<ACTION_SPEC>&,
                        const rl::environments::plasma_tokamak::State<STATE_SPEC>&, RNG&)
{
    // The environment computes the reward inside step(); RLTools asks for it
    // immediately afterwards for the same transition.
    return static_cast<typename SPEC::T>(env.lastReward);
}

template <typename DEVICE, typename SPEC, typename STATE_SPEC, typename OBS_TYPE_SPEC,
          typename OBS_SPEC, typename RNG>
void observe(DEVICE&, const rl::environments::PlasmaTokamak<SPEC>& env,
             const typename rl::environments::PlasmaTokamak<SPEC>::Parameters&,
             const rl::environments::plasma_tokamak::State<STATE_SPEC>&,
             const OBS_TYPE_SPEC&, Matrix<OBS_SPEC>& observation, RNG&)
{
    ::plasma::Real buffer[rl::environments::PlasmaTokamak<SPEC>::OBSERVATION_DIM];
    env.simulation->observe(std::span<::plasma::Real>(
        buffer, rl::environments::PlasmaTokamak<SPEC>::OBSERVATION_DIM));

    for (typename SPEC::TI i = 0; i < rl::environments::PlasmaTokamak<SPEC>::OBSERVATION_DIM; ++i) {
        set(observation, 0, i, static_cast<typename SPEC::T>(buffer[i]));
    }
}

template <typename DEVICE, typename SPEC, typename STATE_SPEC, typename RNG>
bool terminated(DEVICE&, const rl::environments::PlasmaTokamak<SPEC>& env,
                typename rl::environments::PlasmaTokamak<SPEC>::Parameters&,
                const rl::environments::plasma_tokamak::State<STATE_SPEC>, RNG&)
{
    return env.lastTerminated;
}

} // namespace rl_tools
RL_TOOLS_NAMESPACE_WRAPPER_END
