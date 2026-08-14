#pragma once

// The reinforcement-learning environment.
//
// One episode is one simulated discharge. Every control step the policy reads
// processed diagnostics, sets six actuator targets, and the plasma is advanced
// by one control period of reduced MHD. The episode ends when the discharge
// completes or when a mode grows large enough to disrupt it.
//
// The control problem is genuinely multi-objective, and the objectives really do
// fight each other:
//
//   Confinement wants beam power high and density low.
//   Alfvenic stability wants beam power low -- the beam's fast ions drive it.
//   Tearing stability wants current drive aimed at the island, but current drive
//     reshapes q, which moves the island, and the beam's pressure feeds the
//     bootstrap current that makes the island self-sustaining.
//   Actuator smoothness wants none of these to move quickly.
//   Control stability wants the policy not to sit against its limits, where it
//     has no authority left when something changes.
//
// No single actuator setting is good for all five. That is the point.

#include <plasma/control/Actuators.h>
#include <plasma/diagnostics/Diagnostics.h>
#include <plasma/mhd/ReducedMhd.h>

#include <random>

namespace plasma {

/// Relative importance of each objective. Exposed so the trade-off can be
/// re-weighted without touching the environment.
struct RewardWeights
{
    Real confinement{ 1.00 };
    Real tearing{ 2.60 };
    Real alfvenic{ 1.40 };
    Real smoothness{ 0.50 };
    Real chatter{ 0.35 };
    Real saturation{ 0.15 };
    Real survival{ 0.20 };

    /// Disruption penalty, charged *per control step the discharge failed to
    /// survive to complete*.
    ///
    /// A flat one-off penalty is wrong here, in a way that is easy to miss.
    /// Once an episode's per-step reward goes negative -- which it does as soon
    /// as an island is large -- ending the episode early stops the losses, so a
    /// flat penalty smaller than the remaining negative reward makes disrupting
    /// the *optimal* move. Training found exactly that: the policy pushed the
    /// beam up for confinement, let the island run away, and disrupted at a
    /// third of the pulse length on purpose, because that beat surviving.
    ///
    /// Scaling by the steps remaining makes a disruption cost what it destroys,
    /// so surviving badly always beats not surviving. The value is set above the
    /// worst sustainable per-step loss for that reason.
    Real disruption{ 6.0 };
};

/// Per-step breakdown, kept so training curves can show which objective a
/// policy is actually improving rather than only the scalar it optimizes.
struct RewardTerms
{
    Real confinement{ 0 };
    Real tearing{ 0 };
    Real alfvenic{ 0 };
    Real smoothness{ 0 };
    Real chatter{ 0 };
    Real saturation{ 0 };
    Real survival{ 0 };
    Real disruption{ 0 };
    Real total{ 0 };
};

struct TokamakEnvConfig
{
    int radialPoints{ 192 };

    EquilibriumConfig equilibrium{};
    MhdParameters     mhd{};
    DiagnosticConfig  diagnostics{};
    ActuatorLimits    actuators{};
    RewardWeights     reward{};

    /// Control period in Alfven times. With the mapping below this is a 1 kHz
    /// control loop, which is what a real neoclassical-tearing-mode controller
    /// runs at.
    Real controlPeriod{ 2.5 };
    int  maxSteps{ 400 };

    /// Physical duration of one Alfven time in this model. Chosen so the
    /// simulated tearing growth time lands in the tens of milliseconds where
    /// real tearing modes grow, which is what makes the control cadence and the
    /// latency budget meaningful numbers rather than arbitrary ones.
    Real alfvenTimeSeconds{ 3.8e-4 };

    /// Island width, as a fraction of the minor radius, that ends the discharge.
    ///
    /// Set above the width an uncontrolled discharge reaches (~0.48 here) so
    /// that losing the plasma is a genuine failure rather than the default
    /// outcome. With the threshold below that, every episode terminated and the
    /// reward became a cliff: identical returns either side of a discontinuity,
    /// no gradient for a policy-gradient method to follow, and learning stalled
    /// completely. The tearing penalty carries the signal smoothly instead.
    Real disruptionIslandWidth{ 0.46 };
    /// Island width used to normalize the tearing penalty. Kept at the old
    /// disruption threshold so the penalty still saturates where a real machine
    /// would already be in trouble.
    Real tearingPenaltyScale{ 0.35 };

    // --- Domain randomization ----------------------------------------------
    // Each episode draws a slightly different machine. A policy that only works
    // for one equilibrium has memorized rather than learned.

    /// Multiplicative spread on the Lundquist number, log-uniform.
    Real lundquistJitter{ 0.30 };
    /// Additive spread on the current peaking, which moves the rational surface.
    Real peakingJitter{ 0.15 };
    /// Spread on the rotation, which sets the mode frequency the filters see.
    Real rotationJitter{ 0.25 };
    /// Probability that the episode starts with a seed island already present,
    /// as a sawtooth crash or an edge-localized mode would leave.
    Real seedIslandProbability{ 0.45 };
    Real seedIslandAmplitude{ 6.0e-5 };
};

struct StepResult
{
    Real reward{ 0 };
    /// The discharge ended badly.
    bool terminated{ false };
    /// The discharge ran to its planned length.
    bool truncated{ false };
};

class TokamakEnv
{
public:
    void configure(const TokamakEnvConfig& config);

    /// Starts a new episode. `seed` drives both the mode seeding and the
    /// per-episode machine randomization.
    void reset(std::uint64_t seed);

    /// Applies one control action and advances the plasma one control period.
    StepResult step(std::span<const Real> action);

    [[nodiscard]] std::size_t observationSize() const;
    [[nodiscard]] static constexpr std::size_t actionSize() { return kActuatorCount; }

    /// Observation size for the default diagnostic configuration, available at
    /// compile time. RLTools dispatches statically and needs the dimension as a
    /// constant; configure() asserts that the runtime size agrees.
    ///   8 Mirnov coils x 2 frequency bands + 8 ECE channels + 6 scalars
    ///   + 6 actuator positions + 6 previous actions + 1 episode phase
    static constexpr std::size_t kDefaultObservationSize = 8 * 2 + 8 + 6 + kActuatorCount * 2 + 1;

    /// Writes exactly observationSize() values.
    void observe(std::span<Real> out) const;

    // --- Introspection ------------------------------------------------------
    // Used by the viewer and by training telemetry, never by the policy.

    [[nodiscard]] const RewardTerms& rewardTerms() const noexcept { return m_terms; }
    [[nodiscard]] const ReducedMhd& mhd() const noexcept { return m_mhd; }
    [[nodiscard]] const Equilibrium& equilibrium() const noexcept { return m_equilibrium; }
    [[nodiscard]] const DiagnosticSuite& diagnostics() const noexcept { return m_diagnostics; }
    [[nodiscard]] const ActuatorBank& actuators() const noexcept { return m_actuators; }
    [[nodiscard]] const RadialGrid& grid() const noexcept { return m_grid; }
    [[nodiscard]] const TokamakEnvConfig& config() const noexcept { return m_config; }

    [[nodiscard]] int stepIndex() const noexcept { return m_step; }
    [[nodiscard]] bool disrupted() const noexcept { return m_disrupted; }
    [[nodiscard]] Real episodeReturn() const noexcept { return m_return; }

    /// Ground truth, for logging and for the reward. Never in the observation:
    /// a controller that needs it could not be deployed.
    [[nodiscard]] Real widestIsland() const;
    [[nodiscard]] Real alfvenicAmplitude() const;

    /// Simulated wall-clock position of the episode, in seconds.
    [[nodiscard]] Real elapsedSeconds() const noexcept
    {
        return static_cast<Real>(m_step) * m_config.controlPeriod * m_config.alfvenTimeSeconds;
    }

    /// Index of each mode in the solver, for viewers and diagnostics.
    [[nodiscard]] int alfvenicModeIndex() const noexcept { return m_alfvenicMode; }
    [[nodiscard]] int primaryTearingModeIndex() const noexcept { return m_primaryTearingMode; }

private:
    void applyActuatorsToPlasma();
    RewardTerms computeReward(bool disrupted) const;

    TokamakEnvConfig m_config;

    RadialGrid      m_grid;
    Equilibrium     m_equilibrium;
    ReducedMhd      m_mhd;
    DiagnosticSuite m_diagnostics;
    ActuatorBank    m_actuators;

    std::vector<ModeSpec> m_modes;
    int m_alfvenicMode{ 0 };
    int m_primaryTearingMode{ 1 };

    /// Running variance per action channel, for the control-stability term.
    std::array<RunningVariance, kActuatorCount> m_actionVariance;
    std::array<Real, kActuatorCount> m_previousAction{};

    RewardTerms m_terms;
    std::mt19937_64 m_rng;

    int  m_step{ 0 };
    bool m_disrupted{ false };
    Real m_return{ 0 };
};

} // namespace plasma
