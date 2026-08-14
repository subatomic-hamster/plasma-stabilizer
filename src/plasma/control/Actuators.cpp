#include <plasma/control/Actuators.h>

#include <cmath>

namespace plasma {

void ActuatorBank::configure(const ActuatorLimits& limits)
{
    m_limits = limits;

    auto& channels = m_channels;
    channels[static_cast<std::size_t>(ActuatorChannel::CurrentDrivePower)]
        .configure(0.0, limits.maximumDrivenFraction, limits.powerSlewRate * limits.maximumDrivenFraction, 0.0);

    // Steering starts mid-plasma: a gyrotron mirror has to be somewhere, and
    // starting it on the rational surface would hand the policy the answer.
    channels[static_cast<std::size_t>(ActuatorChannel::CurrentDriveRadius)]
        .configure(0.05, 0.95, limits.steeringSlewRate, 0.5);

    channels[static_cast<std::size_t>(ActuatorChannel::BeamPower)]
        .configure(limits.minimumBeamPower, limits.maximumBeamPower,
                   limits.powerSlewRate * (limits.maximumBeamPower - limits.minimumBeamPower), 1.0);

    channels[static_cast<std::size_t>(ActuatorChannel::ResonantAmplitude)]
        .configure(0.0, limits.maximumResonantField,
                   limits.powerSlewRate * limits.maximumResonantField, 0.0);

    // Phase is in turns and wraps, but the actuator is a linear slew-limited
    // channel: coil supplies cannot teleport the field around the torus either.
    channels[static_cast<std::size_t>(ActuatorChannel::ResonantPhase)]
        .configure(-1.0, 1.0, limits.steeringSlewRate * 2.0, 0.0);

    channels[static_cast<std::size_t>(ActuatorChannel::GasPuff)]
        .configure(limits.minimumDensity, limits.maximumDensity,
                   limits.densitySlewRate * (limits.maximumDensity - limits.minimumDensity), 1.0);

    reset();
}

void ActuatorBank::reset()
{
    for (Actuator& actuator : m_channels) actuator.reset();
}

void ActuatorBank::drive(std::span<const Real> targets, Real dt)
{
    for (std::size_t i = 0; i < kActuatorCount; ++i) {
        // Policy outputs live in [-1, 1]; map onto the channel's physical range.
        const Real request = (i < targets.size()) ? std::clamp(targets[i], -1.0, 1.0) : 0.0;
        const Real unit    = 0.5 * (request + 1.0);

        Actuator& actuator = m_channels[i];
        actuator.drive(actuator.fromUnit(unit), dt);
    }
}

Real ActuatorBank::meanSquaredChange() const
{
    Real total = 0;
    for (const Actuator& actuator : m_channels) {
        const Real normalizedChange = actuator.lastChange() / actuator.range();
        total += normalizedChange * normalizedChange;
    }
    return total / static_cast<Real>(kActuatorCount);
}

int ActuatorBank::saturatedCount() const
{
    int count = 0;
    for (const Actuator& actuator : m_channels) {
        if (actuator.saturated()) ++count;
    }
    return count;
}

std::size_t ActuatorBank::writeState(std::span<Real> out) const
{
    std::size_t written = 0;
    for (const Actuator& actuator : m_channels) {
        if (written < out.size()) out[written] = actuator.normalized() * 2.0 - 1.0;
        ++written;
    }
    return written;
}

} // namespace plasma
