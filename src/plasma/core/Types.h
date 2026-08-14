#pragma once

// Numeric types for the plasma model.
//
// Deliberately double, independently of particle-sim's `psim::Real` (float).
// The tearing layer is a thin region where the solution is set by a near
// cancellation between the field-line-bending and current-drive terms, and the
// mode amplitude spans six orders of magnitude over an episode. Float32 loses
// the layer. Conversions to psim's float types happen only at the render
// boundary, where precision stops mattering.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace plasma {

using Real  = double;
using Index = std::uint32_t;

namespace constants {
inline constexpr Real kPi      = 3.14159265358979323846;
inline constexpr Real kTwoPi   = 2.0 * kPi;
inline constexpr Real kEpsilon = 1e-30;
} // namespace constants

/// Non-owning mutable view over a profile.
using Profile      = std::span<Real>;
using ConstProfile = std::span<const Real>;

/// Linear interpolation of a profile sampled on a uniform grid, clamped at the
/// ends. Used wherever a diagnostic samples a field between grid points.
[[nodiscard]] inline Real sampleUniform(ConstProfile values, Real normalizedPosition)
{
    if (values.empty()) return 0;
    const Real clamped = normalizedPosition < 0 ? 0 : (normalizedPosition > 1 ? 1 : normalizedPosition);
    const Real x       = clamped * static_cast<Real>(values.size() - 1);
    const std::size_t i = static_cast<std::size_t>(x);
    if (i + 1 >= values.size()) return values.back();
    const Real t = x - static_cast<Real>(i);
    return values[i] * (1 - t) + values[i + 1] * t;
}

/// True when every entry is finite. Cheap guard for "did the solver diverge".
[[nodiscard]] inline bool allFinite(ConstProfile values)
{
    for (Real v : values) {
        if (!(v > -std::numeric_limits<Real>::infinity() && v < std::numeric_limits<Real>::infinity())) {
            return false;
        }
    }
    return true;
}

} // namespace plasma
