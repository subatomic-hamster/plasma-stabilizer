#pragma once

// Streaming signal processing for the diagnostic chain.
//
// Every filter here is fixed-state and O(1) per sample: no buffers to grow, no
// history to scan, no allocation. That is not an optimization, it is the
// requirement -- a plasma control system processes megasample-per-second
// diagnostic streams inside a control cycle measured in milliseconds, and any
// operator whose cost depends on how long the discharge has been running is
// disqualified before it starts.
//
// The same property is what makes an RL environment fast: these run once per
// control step per channel, thousands of times per episode.

#include <plasma/core/Types.h>

#include <algorithm>
#include <cmath>

namespace plasma {

/// First-order exponential smoother. `timeConstant` is in the same units as the
/// sample interval passed to update().
class ExponentialSmoother
{
public:
    ExponentialSmoother() = default;
    explicit ExponentialSmoother(Real timeConstant) : m_timeConstant(timeConstant) {}

    void configure(Real timeConstant) { m_timeConstant = std::max(timeConstant, 1e-9); }
    void reset(Real value = 0)
    {
        m_value = value;
        m_primed = false;
    }

    Real update(Real sample, Real dt)
    {
        if (!m_primed) {
            // Seeding with the first sample instead of zero avoids a spurious
            // ramp at the start of every episode, which the growth-rate
            // estimator downstream would read as a real transient.
            m_value  = sample;
            m_primed = true;
            return m_value;
        }
        const Real alpha = 1.0 - std::exp(-dt / m_timeConstant);
        m_value += alpha * (sample - m_value);
        return m_value;
    }

    [[nodiscard]] Real value() const noexcept { return m_value; }

private:
    Real m_timeConstant{ 1 };
    Real m_value{ 0 };
    bool m_primed{ false };
};

/// Direct-form-II biquad. Used as the band splitter that separates the slow
/// tearing activity from the fast Alfvenic activity -- the two instabilities
/// live in different frequency bands, and that is exactly how a real tokamak
/// tells them apart on the same coil.
class Biquad
{
public:
    /// Second-order band-pass with unit peak gain at `centerFrequency`
    /// (radians per unit time), sampled every `dt`.
    void configureBandPass(Real centerFrequency, Real quality, Real dt);
    /// Second-order low-pass, for the near-stationary tearing band.
    void configureLowPass(Real cutoffFrequency, Real dt);
    /// Second-order high-pass.
    void configureHighPass(Real cutoffFrequency, Real dt);

    void reset()
    {
        m_state1 = 0;
        m_state2 = 0;
    }

    Real update(Real sample)
    {
        const Real out = m_b0 * sample + m_state1;
        m_state1 = m_b1 * sample - m_a1 * out + m_state2;
        m_state2 = m_b2 * sample - m_a2 * out;
        return out;
    }

private:
    void normalize(Real b0, Real b1, Real b2, Real a0, Real a1, Real a2);

    Real m_b0{ 1 }, m_b1{ 0 }, m_b2{ 0 };
    Real m_a1{ 0 }, m_a2{ 0 };
    Real m_state1{ 0 }, m_state2{ 0 };
};

/// Streaming root-mean-square through an exponential window.
class RunningRms
{
public:
    RunningRms() = default;
    explicit RunningRms(Real timeConstant) : m_mean(timeConstant) {}

    void configure(Real timeConstant) { m_mean.configure(timeConstant); }
    void reset() { m_mean.reset(0); }

    Real update(Real sample, Real dt)
    {
        return std::sqrt(std::max(m_mean.update(sample * sample, dt), 0.0));
    }

    [[nodiscard]] Real value() const noexcept { return std::sqrt(std::max(m_mean.value(), 0.0)); }

private:
    ExponentialSmoother m_mean;
};

/// Estimates the exponential growth rate d(ln A)/dt of a positive envelope.
///
/// Differentiating the logarithm rather than fitting an exponential keeps the
/// state to two numbers and makes the estimate scale-free, which matters when
/// the amplitude sweeps six orders of magnitude during an episode. The
/// logarithm is floored so a quiet channel reports a rate near zero instead of
/// a huge negative one.
class GrowthRateEstimator
{
public:
    GrowthRateEstimator() = default;
    GrowthRateEstimator(Real timeConstant, Real floorValue)
        : m_smoother(timeConstant), m_floor(floorValue)
    {
    }

    void configure(Real timeConstant, Real floorValue)
    {
        m_smoother.configure(timeConstant);
        m_floor = floorValue;
    }

    void reset()
    {
        m_smoother.reset(0);
        m_haveLast = false;
        m_lastLog  = 0;
    }

    Real update(Real amplitude, Real dt)
    {
        const Real logAmplitude = std::log(std::max(amplitude, m_floor));
        if (!m_haveLast) {
            m_lastLog  = logAmplitude;
            m_haveLast = true;
            return 0;
        }
        const Real rate = (logAmplitude - m_lastLog) / std::max(dt, 1e-12);
        m_lastLog = logAmplitude;
        return m_smoother.update(rate, dt);
    }

    [[nodiscard]] Real value() const noexcept { return m_smoother.value(); }

private:
    ExponentialSmoother m_smoother;
    Real m_floor{ 1e-12 };
    Real m_lastLog{ 0 };
    bool m_haveLast{ false };
};

/// Tracks how much a signal moves around its own mean, through an exponential
/// window. The control-stability term of the reward reads this on the action
/// channels: a policy that chatters is one that would wear out a real actuator.
class RunningVariance
{
public:
    RunningVariance() = default;
    explicit RunningVariance(Real timeConstant) : m_mean(timeConstant), m_meanSquare(timeConstant) {}

    void configure(Real timeConstant)
    {
        m_mean.configure(timeConstant);
        m_meanSquare.configure(timeConstant);
    }

    void reset()
    {
        m_mean.reset(0);
        m_meanSquare.reset(0);
    }

    Real update(Real sample, Real dt)
    {
        const Real mean = m_mean.update(sample, dt);
        const Real second = m_meanSquare.update(sample * sample, dt);
        return std::max(second - mean * mean, 0.0);
    }

    [[nodiscard]] Real value() const noexcept
    {
        return std::max(m_meanSquare.value() - m_mean.value() * m_mean.value(), 0.0);
    }

private:
    ExponentialSmoother m_mean;
    ExponentialSmoother m_meanSquare;
};

} // namespace plasma
