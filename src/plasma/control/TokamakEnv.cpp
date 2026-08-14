#include <plasma/control/TokamakEnv.h>

#include <algorithm>
#include <cassert>
#include <cmath>

namespace plasma {

namespace {

/// Alfvenic amplitude used to normalize its penalty. Matched to the fast-ion
/// saturation level, so a fully saturated mode scores about -1 before weights.
constexpr Real kAlfvenicReference = 2.0e-3;

} // namespace

void TokamakEnv::configure(const TokamakEnvConfig& config)
{
    m_config = config;
    m_grid.configure(config.radialPoints, config.equilibrium.minorRadius);

    // One Alfvenic mode with no rational surface in the plasma, and two tearing
    // modes that do have one. The (1,1) has q > 1 everywhere here, so it cannot
    // reconnect and simply rings on the Alfven continuum -- exactly the branch
    // the beam destabilizes.
    m_modes = {
        ModeSpec{ 1, 1, ModeClass::Alfvenic },
        ModeSpec{ 2, 1, ModeClass::Tearing },
        ModeSpec{ 3, 1, ModeClass::Tearing },
    };
    m_alfvenicMode       = 0;
    m_primaryTearingMode = 1;

    m_equilibrium.configure(m_grid, config.equilibrium);
    m_mhd.configure(m_grid, m_modes, config.mhd);
    m_diagnostics.configure(config.diagnostics, m_grid, config.controlPeriod);
    m_actuators.configure(config.actuators);

    for (RunningVariance& variance : m_actionVariance) {
        variance.configure(20.0 * config.controlPeriod);
    }

    reset(1);

    // The RLTools backend hard-codes the observation dimension because its
    // dispatch is static; if the diagnostic configuration changes, that constant
    // has to change with it.
    if (config.diagnostics.mirnovCoils == 8 && config.diagnostics.eceChannels == 8) {
        assert(observationSize() == kDefaultObservationSize &&
               "kDefaultObservationSize is out of date with the diagnostic layout");
    }
}

void TokamakEnv::reset(std::uint64_t seed)
{
    m_rng.seed(seed == 0 ? 0xA24BAED4963EE407ull : seed);
    m_step      = 0;
    m_disrupted = false;
    m_return    = 0;
    m_terms     = RewardTerms{};

    // --- Per-episode machine ------------------------------------------------
    std::uniform_real_distribution<Real> unit(-1.0, 1.0);
    std::uniform_real_distribution<Real> zeroToOne(0.0, 1.0);

    EquilibriumConfig equilibrium = m_config.equilibrium;
    equilibrium.lundquist *= std::exp(m_config.lundquistJitter * unit(m_rng));
    equilibrium.currentPeaking += m_config.peakingJitter * unit(m_rng);

    MhdParameters mhd = m_config.mhd;
    mhd.coreRotation *= 1.0 + m_config.rotationJitter * unit(m_rng);

    m_equilibrium.configure(m_grid, equilibrium);
    m_mhd.configure(m_grid, m_modes, mhd);

    m_actuators.reset();
    for (RunningVariance& variance : m_actionVariance) variance.reset();
    m_previousAction.fill(0.0);

    m_mhd.reset(m_rng());

    // A sawtooth crash or an edge-localized mode can drop a seed island into an
    // otherwise quiet plasma. Starting some episodes that way is what teaches
    // the policy to suppress an island that already exists, not only to prevent
    // one from forming.
    if (zeroToOne(m_rng) < m_config.seedIslandProbability) {
        const Real amplitude = m_config.seedIslandAmplitude * (0.5 + zeroToOne(m_rng));
        m_mhd.setResonantPerturbation(m_primaryTearingMode,
                                      std::polar(amplitude, constants::kTwoPi * zeroToOne(m_rng)));
    }

    applyActuatorsToPlasma();
    m_diagnostics.reset(m_rng());
    m_diagnostics.sample(m_mhd, m_equilibrium);
}

void TokamakEnv::applyActuatorsToPlasma()
{
    const ActuatorLimits& limits = m_config.actuators;

    DrivenCurrent driven;
    driven.fraction         = m_actuators.value(ActuatorChannel::CurrentDrivePower);
    driven.normalizedRadius = m_actuators.value(ActuatorChannel::CurrentDriveRadius);
    driven.width            = limits.depositionWidth;
    m_equilibrium.setDrivenCurrent(driven);

    const Real beam = m_actuators.value(ActuatorChannel::BeamPower);
    m_equilibrium.setHeatingScale(beam);
    m_equilibrium.setDensityScale(m_actuators.value(ActuatorChannel::GasPuff));

    // Quasilinear feedback: the islands the modes have grown flatten the
    // profiles, which changes the drive for the next step.
    const std::vector<IslandRegion> islands = m_mhd.islands();
    m_equilibrium.setIslands(islands);
    m_equilibrium.rebuild();

    // Beam power maps onto the fast-ion drive linearly between its limits.
    const Real beamFraction = (beam - limits.minimumBeamPower) /
                              std::max(limits.maximumBeamPower - limits.minimumBeamPower, 1e-9);
    m_mhd.setEnergeticDrive(limits.driveAtFullBeam * std::clamp(beamFraction, 0.0, 1.0));

    // The resonant coils apply a static vacuum field to the primary tearing
    // mode. Amplitude and phase are separate channels because the phase decides
    // whether the field opposes the island or feeds it.
    const Real resonant = m_actuators.value(ActuatorChannel::ResonantAmplitude);
    const Real phase    = m_actuators.value(ActuatorChannel::ResonantPhase) * constants::kPi;
    m_mhd.setResonantPerturbation(m_primaryTearingMode, std::polar(resonant, phase));

    m_mhd.onEquilibriumChanged(m_equilibrium);
}

Real TokamakEnv::widestIsland() const
{
    Real widest = 0;
    for (int i = 0; i < m_mhd.modeCount(); ++i) {
        widest = std::max(widest, m_mhd.islandWidth(i));
    }
    return widest / m_grid.minorRadius();
}

Real TokamakEnv::alfvenicAmplitude() const
{
    return m_mhd.amplitude(m_alfvenicMode);
}

RewardTerms TokamakEnv::computeReward(bool disrupted) const
{
    RewardTerms terms;
    const RewardWeights& weights = m_config.reward;

    // Confinement: stored energy relative to the reference discharge. Islands
    // flatten the pressure and cost energy; beam power buys it back.
    terms.confinement = weights.confinement * m_equilibrium.confinementFraction();

    // Instability, from ground truth rather than from the noisy diagnostics: the
    // reward is allowed privileged information, the observation is not.
    const Real island = widestIsland();
    terms.tearing = -weights.tearing * std::min(island / m_config.tearingPenaltyScale, 1.5);

    const Real alfvenic = alfvenicAmplitude() / kAlfvenicReference;
    terms.alfvenic = -weights.alfvenic * std::min(alfvenic, 2.0);

    // Actuator wear: how far every channel moved this step.
    terms.smoothness = -weights.smoothness * m_actuators.meanSquaredChange() * 1.0e3;

    // Control stability: a policy whose commands rattle around their own mean is
    // not a controller, it is a noise source with authority.
    Real variance = 0;
    for (const RunningVariance& channel : m_actionVariance) variance += channel.value();
    terms.chatter = -weights.chatter * variance / static_cast<Real>(kActuatorCount);

    // Headroom: sitting on a limit means no authority left in that direction.
    terms.saturation = -weights.saturation *
                       static_cast<Real>(m_actuators.saturatedCount()) /
                       static_cast<Real>(kActuatorCount);

    terms.survival = weights.survival;

    // Charged for the rest of the discharge that will now never happen, so that
    // ending the episode early can never be worth more than surviving it badly.
    const Real remaining = static_cast<Real>(std::max(0, m_config.maxSteps - m_step));
    terms.disruption = disrupted ? -weights.disruption * remaining : 0.0;

    terms.total = terms.confinement + terms.tearing + terms.alfvenic + terms.smoothness +
                  terms.chatter + terms.saturation + terms.survival + terms.disruption;
    return terms;
}

StepResult TokamakEnv::step(std::span<const Real> action)
{
    StepResult result;
    if (m_disrupted) {
        result.terminated = true;
        return result;
    }

    const Real dt = m_config.controlPeriod;

    m_actuators.drive(action, dt);
    for (std::size_t i = 0; i < kActuatorCount; ++i) {
        m_previousAction[i] = (i < action.size()) ? std::clamp(action[i], -1.0, 1.0) : 0.0;
        // Chatter is measured on where the actuator actually went, not on what
        // was commanded. During training the commanded action carries the
        // policy's exploration noise, and charging the policy for exploring is
        // a way of teaching it not to -- the slew-limited hardware position is
        // the honest measure of whether the machine is being rattled.
        const Actuator& channel = m_actuators.channel(static_cast<ActuatorChannel>(i));
        m_actionVariance[i].update(channel.normalized() * 2.0 - 1.0, dt);
    }

    applyActuatorsToPlasma();

    const bool finite = m_mhd.advance(dt);
    m_diagnostics.sample(m_mhd, m_equilibrium);
    ++m_step;

    // A discharge ends when an island grows past the width a tokamak can carry,
    // when the solver loses the solution, or when the planned pulse completes.
    const bool tooWide  = widestIsland() > m_config.disruptionIslandWidth;
    m_disrupted = !finite || tooWide || m_mhd.disrupted();

    m_terms = computeReward(m_disrupted);
    result.reward = m_terms.total;
    m_return += result.reward;

    result.terminated = m_disrupted;
    result.truncated  = !m_disrupted && m_step >= m_config.maxSteps;
    return result;
}

std::size_t TokamakEnv::observationSize() const
{
    // processed diagnostics + actuator positions + last action + episode phase
    return m_diagnostics.observationSize() + kActuatorCount * 2 + 1;
}

void TokamakEnv::observe(std::span<Real> out) const
{
    std::size_t written = m_diagnostics.writeObservation(out);

    if (written < out.size()) {
        written += m_actuators.writeState(out.subspan(written));
    }
    for (std::size_t i = 0; i < kActuatorCount && written < out.size(); ++i) {
        out[written++] = m_previousAction[i];
    }
    if (written < out.size()) {
        // How far through the pulse we are. Without it the problem is partially
        // observable in a way that has nothing to do with the physics: the value
        // function cannot know how much reward is left to collect.
        out[written++] = 2.0 * static_cast<Real>(m_step) /
                             static_cast<Real>(std::max(m_config.maxSteps, 1)) - 1.0;
    }
}

} // namespace plasma
