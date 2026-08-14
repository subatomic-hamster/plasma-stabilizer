#include <plasma/diagnostics/SignalProcessing.h>

namespace plasma {

void Biquad::normalize(Real b0, Real b1, Real b2, Real a0, Real a1, Real a2)
{
    const Real inverse = 1.0 / a0;
    m_b0 = b0 * inverse;
    m_b1 = b1 * inverse;
    m_b2 = b2 * inverse;
    m_a1 = a1 * inverse;
    m_a2 = a2 * inverse;
    reset();
}

void Biquad::configureBandPass(Real centerFrequency, Real quality, Real dt)
{
    // Bilinear-transform design (RBJ cookbook). The prewarped angle is clamped
    // below Nyquist: a control loop sampling every few Alfven times cannot
    // resolve an Alfvenic mode's true frequency, and letting the design run past
    // Nyquist produces an unstable filter rather than a useless one.
    const Real omega = std::min(centerFrequency * dt, 0.95 * constants::kPi);
    const Real sinOmega = std::sin(omega);
    const Real cosOmega = std::cos(omega);
    const Real alpha = sinOmega / (2.0 * std::max(quality, 0.05));

    normalize(alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * cosOmega, 1.0 - alpha);
}

void Biquad::configureLowPass(Real cutoffFrequency, Real dt)
{
    const Real omega = std::min(cutoffFrequency * dt, 0.95 * constants::kPi);
    const Real sinOmega = std::sin(omega);
    const Real cosOmega = std::cos(omega);
    const Real alpha = sinOmega / std::sqrt(2.0); // Butterworth

    const Real b1 = 1.0 - cosOmega;
    normalize(b1 * 0.5, b1, b1 * 0.5, 1.0 + alpha, -2.0 * cosOmega, 1.0 - alpha);
}

void Biquad::configureHighPass(Real cutoffFrequency, Real dt)
{
    const Real omega = std::min(cutoffFrequency * dt, 0.95 * constants::kPi);
    const Real sinOmega = std::sin(omega);
    const Real cosOmega = std::cos(omega);
    const Real alpha = sinOmega / std::sqrt(2.0);

    const Real b1 = -(1.0 + cosOmega);
    normalize((1.0 + cosOmega) * 0.5, b1, (1.0 + cosOmega) * 0.5,
              1.0 + alpha, -2.0 * cosOmega, 1.0 - alpha);
}

} // namespace plasma
