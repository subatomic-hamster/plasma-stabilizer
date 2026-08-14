#pragma once

// Actuator model.
//
// Real actuators do not jump. A gyrotron's mirror steers at a finite rate, a
// beam ramps its power over milliseconds, coil currents are limited by their
// power supplies' voltage. Modelling that is not decoration: without slew
// limits a policy learns bang-bang control that no hardware could execute, and
// the "actuator smoothness" objective becomes unmeasurable because every
// command is already a step.
//
// Each channel therefore holds its own state, and the policy's output is a
// *target* the actuator moves toward, not the value that reaches the plasma.

#include <plasma/core/Types.h>

#include <algorithm>
#include <array>

namespace plasma {

/// One rate-limited channel.
class Actuator
{
public:
    void configure(Real minValue, Real maxValue, Real slewRate, Real initial)
    {
        m_min      = minValue;
        m_max      = maxValue;
        m_slewRate = slewRate;
        m_initial  = std::clamp(initial, minValue, maxValue);
        reset();
    }

    void reset()
    {
        m_value = m_initial;
        m_lastChange = 0;
    }

    /// Moves toward `target` at no more than the slew rate. Returns the value
    /// the plasma actually sees.
    Real drive(Real target, Real dt)
    {
        const Real clamped = std::clamp(target, m_min, m_max);
        const Real maximumStep = m_slewRate * dt;
        const Real requested   = clamped - m_value;
        const Real applied     = std::clamp(requested, -maximumStep, maximumStep);
        m_value += applied;
        m_lastChange = applied;
        return m_value;
    }

    [[nodiscard]] Real value() const noexcept { return m_value; }
    [[nodiscard]] Real lastChange() const noexcept { return m_lastChange; }
    [[nodiscard]] Real minimum() const noexcept { return m_min; }
    [[nodiscard]] Real maximum() const noexcept { return m_max; }
    [[nodiscard]] Real range() const noexcept { return std::max(m_max - m_min, 1e-12); }

    /// Maps a unit position in [0,1] onto the channel's physical range.
    [[nodiscard]] Real fromUnit(Real unit) const noexcept { return m_min + unit * (m_max - m_min); }
    /// Position within the range, in [0,1]. What the observation reports.
    [[nodiscard]] Real normalized() const noexcept { return (m_value - m_min) / range(); }
    /// True when the channel is pinned at a limit, which a controller relying on
    /// it has no headroom left on.
    [[nodiscard]] bool saturated() const noexcept
    {
        const Real margin = 0.02 * range();
        return m_value <= m_min + margin || m_value >= m_max - margin;
    }

private:
    Real m_min{ 0 };
    Real m_max{ 1 };
    Real m_slewRate{ 1 };
    Real m_initial{ 0 };
    Real m_value{ 0 };
    Real m_lastChange{ 0 };
};

/// Channel identity. The order here is the order of the policy's action vector.
enum class ActuatorChannel : std::size_t
{
    /// Electron cyclotron current drive power, as a fraction of plasma current.
    CurrentDrivePower = 0,
    /// Where that current is deposited, as r/a.
    CurrentDriveRadius = 1,
    /// Neutral beam power. Heats the plasma -- and drives Alfvenic modes.
    BeamPower = 2,
    /// Resonant magnetic perturbation coil current.
    ResonantAmplitude = 3,
    /// Phase of that perturbation relative to the machine, in turns.
    ResonantPhase = 4,
    /// Gas puff rate; raises density, cools the plasma, raises resistivity.
    GasPuff = 5,

    Count = 6,
};

inline constexpr std::size_t kActuatorCount = static_cast<std::size_t>(ActuatorChannel::Count);

struct ActuatorLimits
{
    /// Deposited current as a fraction of the total plasma current. Deliberately
    /// modest: with a fifth of the plasma current available the problem
    /// collapses to "turn everything on", and deposition radius stops mattering.
    Real maximumDrivenFraction{ 0.06 };
    /// Gaussian deposition width, r/a.
    Real depositionWidth{ 0.14 };

    /// Steering limits of the launcher, r/a.
    ///
    /// The inner limit is a real constraint on where a gyrotron can be aimed,
    /// and it also removes a second solution branch. Depositing deep in the core
    /// stabilizes the mode too, but by globally reshaping q rather than by
    /// acting on the island -- and a reward landscape with two disconnected
    /// optima separated by a band of guaranteed disruption is one that Gaussian
    /// exploration cannot cross. Measured: aiming at r/a = 0.23 or at 0.68 both
    /// return about +345, and everything between them disrupts.
    Real minimumDepositionRadius{ 0.42 };
    Real maximumDepositionRadius{ 0.95 };

    /// Beam power multiplies the heating, and with it the fast-ion drive.
    Real minimumBeamPower{ 0.35 };
    Real maximumBeamPower{ 1.45 };
    /// Fast-ion drive at maximum beam power. Above roughly 0.13 the Alfvenic
    /// mode is unstable, so the useful range straddles the threshold.
    Real driveAtFullBeam{ 0.34 };

    /// Applied resonant field amplitude at the wall.
    Real maximumResonantField{ 4.0e-5 };

    /// Density multiplier range from the gas valve.
    Real minimumDensity{ 0.80 };
    Real maximumDensity{ 1.45 };

    /// Slew rates, in units of the channel per Alfven time.
    Real powerSlewRate{ 0.06 };
    Real steeringSlewRate{ 0.03 };
    Real densitySlewRate{ 0.04 };
};

/// The full actuator set, driven together.
class ActuatorBank
{
public:
    void configure(const ActuatorLimits& limits);
    void reset();

    /// Applies one control step. `targets` are the policy outputs in [-1, 1];
    /// they are mapped to each channel's physical range before slewing.
    void drive(std::span<const Real> targets, Real dt);

    [[nodiscard]] const Actuator& channel(ActuatorChannel which) const
    {
        return m_channels[static_cast<std::size_t>(which)];
    }

    [[nodiscard]] Real value(ActuatorChannel which) const
    {
        return channel(which).value();
    }

    /// Mean squared change across all channels on the last step, normalized by
    /// each channel's range. The smoothness objective reads this.
    [[nodiscard]] Real meanSquaredChange() const;
    /// How many channels are pinned against a limit.
    [[nodiscard]] int saturatedCount() const;

    /// Writes each channel's normalized position. The policy needs to know
    /// where its actuators actually are, not just what it last asked for --
    /// with slew limits the two are different.
    std::size_t writeState(std::span<Real> out) const;

    [[nodiscard]] const ActuatorLimits& limits() const noexcept { return m_limits; }

private:
    ActuatorLimits m_limits;
    std::array<Actuator, kActuatorCount> m_channels;
};

} // namespace plasma
