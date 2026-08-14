#include <plasma/mhd/Equilibrium.h>

#include <algorithm>
#include <cmath>

namespace plasma {

namespace {

/// Smooth 0->1 ramp used to blend the flattened island interior back into the
/// unperturbed profile. A hard cut would put a discontinuity in dp/dr, and the
/// mode drive is proportional to a gradient.
Real smoothStep(Real edge0, Real edge1, Real x)
{
    if (edge1 <= edge0) return x < edge0 ? 0.0 : 1.0;
    const Real t = std::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

/// Island interiors are flat: field lines close on themselves there, so
/// transport short-circuits across the island. Pressure and current both
/// flatten, and the two effects pull in opposite directions -- which is the
/// whole character of a neoclassical tearing mode:
///
///   flattening the current removes the gradient that drives the mode  -> saturating
///   flattening the pressure removes the bootstrap current it carried  -> destabilizing
///
/// They need different rules, though, and using one rule for both is wrong in a
/// way that is easy to miss. See the two functions below.

/// Locates the island on the grid. Returns false when it lies outside.
bool islandBounds(const RadialGrid& grid, const IslandRegion& island,
                  Real& inner, Real& outer, int& innerIndex, int& outerIndex)
{
    if (island.width <= 0) return false;
    const int n = grid.size();

    const Real half = 0.5 * island.width;
    inner = island.centerRadius - half;
    outer = island.centerRadius + half;
    if (outer <= grid.radius(0) || inner >= grid.radius(n - 1)) return false;

    innerIndex = grid.nearestIndex(inner);
    outerIndex = grid.nearestIndex(outer);
    return outerIndex > innerIndex;
}

/// Flattens the current to the value that conserves the current enclosed by the
/// island. Using the mean of the two edge values instead would change the total
/// plasma current, which the transformer holds fixed.
void flattenCurrentAcrossIsland(const RadialGrid& grid, const IslandRegion& island, Profile profile)
{
    Real inner = 0, outer = 0;
    int innerIndex = 0, outerIndex = 0;
    if (!islandBounds(grid, island, inner, outer, innerIndex, outerIndex)) return;

    Real weighted = 0;
    Real weight   = 0;
    for (int i = innerIndex; i <= outerIndex; ++i) {
        const Real r = grid.radius(i);
        weighted += profile[static_cast<std::size_t>(i)] * r;
        weight   += r;
    }
    if (weight <= constants::kEpsilon) return;
    const Real plateau = weighted / weight;

    for (int i = innerIndex; i <= outerIndex; ++i) {
        // Blend back to the original profile at the separatrix so the gradient
        // stays continuous; the drive term reads a gradient.
        const Real distance = std::abs(grid.radius(i) - island.centerRadius) /
                              (0.5 * island.width);
        const Real blend    = 1.0 - smoothStep(0.6, 1.0, distance);
        const auto index    = static_cast<std::size_t>(i);
        profile[index]      = profile[index] * (1.0 - blend) + plateau * blend;
    }
}

/// Flattens the pressure to the value at the island's *outer* edge, and drops
/// the whole core by the same amount.
///
/// This is the standard belt model of island-degraded confinement, and it is
/// the reason an island costs stored energy at all. Flattening to the average
/// of the two edge values conserves nothing useful and, on a convex profile,
/// actually *raises* the stored energy -- the chord midpoint of a convex curve
/// sits above the curve. That was the first version here, and it made islands
/// look beneficial.
void flattenPressureAcrossIsland(const RadialGrid& grid, const IslandRegion& island, Profile profile)
{
    Real inner = 0, outer = 0;
    int innerIndex = 0, outerIndex = 0;
    if (!islandBounds(grid, island, inner, outer, innerIndex, outerIndex)) return;

    const Real innerValue = profile[static_cast<std::size_t>(innerIndex)];
    const Real outerValue = profile[static_cast<std::size_t>(outerIndex)];
    const Real drop       = innerValue - outerValue;
    if (drop <= 0) return; // profile already flat or rising here

    for (int i = 0; i < grid.size(); ++i) {
        const auto index = static_cast<std::size_t>(i);
        if (i < innerIndex) {
            // Core: the gradient the island removed no longer holds it up.
            profile[index] = std::max(profile[index] - drop, 0.0);
        } else if (i <= outerIndex) {
            profile[index] = outerValue;
        }
    }
}

} // namespace

void Equilibrium::configure(const RadialGrid& grid, const EquilibriumConfig& config)
{
    m_grid   = &grid;
    m_config = config;

    const auto n = static_cast<std::size_t>(grid.size());
    m_current.assign(n, 0);
    m_currentGradient.assign(n, 0);
    m_ohmicCurrent.assign(n, 0);
    m_bootstrapCurrent.assign(n, 0);
    m_drivenProfile.assign(n, 0);
    m_fluxDerivative.assign(n, 0);
    m_safetyFactor.assign(n, 1);
    m_pressure.assign(n, 0);
    m_temperature.assign(n, 0);
    m_density.assign(n, 1);
    m_resistivity.assign(n, 0);

    // The first rebuild happens against a placeholder reference; capture the
    // real one, then rebuild again so every derived profile is consistent with it.
    m_referenceAxisTemperature = 1;
    reset();
    m_referenceAxisTemperature = std::max(m_temperature.front(), 1e-9);
    m_referenceStoredEnergy    = 1;
    rebuild();
    m_referenceStoredEnergy = std::max(m_storedEnergy, constants::kEpsilon);
}

void Equilibrium::reset()
{
    m_driven = DrivenCurrent{};
    m_islands.clear();
    m_heatingScale = 1;
    m_densityScale = 1;
    rebuild();
}

void Equilibrium::setIslands(std::span<const IslandRegion> islands)
{
    m_islands.assign(islands.begin(), islands.end());
}

void Equilibrium::buildPressureAndDensity()
{
    const RadialGrid& grid = *m_grid;
    const int n            = grid.size();
    const Real a           = m_config.minorRadius;

    // Reference pressure scale from poloidal beta. The absolute normalization
    // does not matter to the mode dynamics -- only gradients do -- but keeping
    // it tied to beta makes the bootstrap fraction meaningful.
    const Real pressureScale = m_config.betaPoloidal * m_heatingScale;

    for (int i = 0; i < n; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const Real x     = grid.radius(i) / a;
        const Real shape = std::pow(std::max(0.0, 1.0 - x * x), 2.0);

        m_pressure[index] = pressureScale * shape;
        m_density[index]  = m_densityScale *
                            (1.0 - (1.0 - m_config.edgeDensityFraction) * x * x);
    }

    // Islands flatten the pressure across their width, which removes the
    // bootstrap current that gradient was carrying.
    for (const IslandRegion& island : m_islands) {
        flattenPressureAcrossIsland(grid, island, m_pressure);
    }

    for (int i = 0; i < n; ++i) {
        const auto index = static_cast<std::size_t>(i);
        // T = p/n, in whatever units; only its shape enters the resistivity.
        m_temperature[index] = m_pressure[index] / std::max(m_density[index], 1e-3);
    }
}

void Equilibrium::buildCurrent()
{
    const RadialGrid& grid = *m_grid;
    const int n            = grid.size();
    const Real a           = m_config.minorRadius;

    // Fixing q(a) fixes the total current: q(a) = a^2 / I_total.
    const Real totalCurrent = a * a / std::max(m_config.edgeSafetyFactor, 0.1);
    m_plasmaCurrent = totalCurrent;

    // --- Driven current (ECCD) ---------------------------------------------
    std::fill(m_drivenProfile.begin(), m_drivenProfile.end(), 0.0);
    if (std::abs(m_driven.fraction) > 1e-9) {
        const Real center = std::clamp(m_driven.normalizedRadius, 0.0, 1.0) * a;
        const Real width  = std::max(m_driven.width, 0.5 * grid.spacing()) * a;

        Real normalization = 0;
        for (int i = 0; i < n; ++i) {
            const Real r     = grid.radius(i);
            const Real shape = std::exp(-((r - center) * (r - center)) / (width * width));
            m_drivenProfile[static_cast<std::size_t>(i)] = shape;
            normalization += shape * r;
        }
        normalization *= grid.spacing();

        // Scale so the deposited current is the requested fraction of I_total.
        const Real scale = (normalization > constants::kEpsilon)
                               ? m_driven.fraction * totalCurrent / normalization
                               : 0.0;
        for (Real& value : m_drivenProfile) value *= scale;
    }

    // --- Bootstrap current --------------------------------------------------
    // j_bs ~ -sqrt(r/R) * (dp/dr) / B_theta. B_theta depends on the current we
    // are building, so this is solved by Picard iteration -- two passes is
    // plenty because the bootstrap is a minority of the total.
    std::vector<Real> pressureGradient(static_cast<std::size_t>(n), 0.0);
    grid.derivative(m_pressure, pressureGradient);

    std::fill(m_bootstrapCurrent.begin(), m_bootstrapCurrent.end(), 0.0);

    for (int iteration = 0; iteration < 3; ++iteration) {
        // Ohmic current takes up whatever the other sources do not provide, so
        // that the total is always the transformer-imposed I_total.
        Real drivenTotal = 0;
        for (int i = 0; i < n; ++i) {
            drivenTotal += (m_drivenProfile[static_cast<std::size_t>(i)] +
                            m_bootstrapCurrent[static_cast<std::size_t>(i)]) * grid.radius(i);
        }
        drivenTotal *= grid.spacing();

        const Real peaking     = std::max(m_config.currentPeaking, 0.1);
        Real ohmicNormalization = 0;
        for (int i = 0; i < n; ++i) {
            const Real x     = grid.radius(i) / a;
            const Real shape = std::pow(std::max(0.0, 1.0 - x * x), peaking);
            m_ohmicCurrent[static_cast<std::size_t>(i)] = shape;
            ohmicNormalization += shape * grid.radius(i);
        }
        ohmicNormalization *= grid.spacing();

        const Real ohmicScale = (ohmicNormalization > constants::kEpsilon)
                                    ? (totalCurrent - drivenTotal) / ohmicNormalization
                                    : 0.0;
        for (Real& value : m_ohmicCurrent) value *= ohmicScale;

        for (int i = 0; i < n; ++i) {
            const auto index = static_cast<std::size_t>(i);
            m_current[index] = m_ohmicCurrent[index] + m_bootstrapCurrent[index] +
                               m_drivenProfile[index];
        }

        buildFluxAndSafetyFactor();

        if (iteration == 2) break;

        // Refresh the bootstrap estimate with the new poloidal field.
        const Real bootstrapScale = m_config.bootstrapFraction * m_heatingScale;
        for (int i = 0; i < n; ++i) {
            const auto index   = static_cast<std::size_t>(i);
            const Real r       = grid.radius(i);
            const Real inverseAspect = std::sqrt(r / m_config.majorRadius);
            const Real bTheta  = std::max(m_fluxDerivative[index], 1e-6);
            m_bootstrapCurrent[index] =
                -bootstrapScale * inverseAspect * pressureGradient[index] / bTheta;
        }

        // Renormalize the bootstrap so its integral is the configured fraction
        // of the total, independent of the pressure normalization.
        Real bootstrapTotal = 0;
        for (int i = 0; i < n; ++i) {
            bootstrapTotal += m_bootstrapCurrent[static_cast<std::size_t>(i)] * grid.radius(i);
        }
        bootstrapTotal *= grid.spacing();
        if (std::abs(bootstrapTotal) > constants::kEpsilon) {
            // Reference (island-free) bootstrap sets the scale; with islands the
            // integral drops, and that drop is the physical drive.
            const Real target = m_config.bootstrapFraction * totalCurrent * m_heatingScale;
            const Real factor = target / bootstrapTotal;
            for (Real& value : m_bootstrapCurrent) value *= factor;
        }
    }

    // The same flattening applies to the total current. Without it the mode has
    // no saturation mechanism at all: the drive is proportional to J', and a
    // linear model keeps growing until it overflows.
    for (const IslandRegion& island : m_islands) {
        flattenCurrentAcrossIsland(grid, island, m_current);
    }

    buildFluxAndSafetyFactor();
    grid.derivative(m_current, m_currentGradient);
}

void Equilibrium::buildFluxAndSafetyFactor()
{
    const RadialGrid& grid = *m_grid;
    const int n            = grid.size();
    const Real dr          = grid.spacing();

    // psi0'(r) = (1/r) * integral_0^r J r' dr', and q = r / psi0'.
    //
    // The integral has to be evaluated at the cell *centre*, which is where
    // psi0' is stored, not at the outer face where a running midpoint sum
    // naturally lands. Getting that wrong is a half-cell error everywhere and a
    // factor of four in the first cell, which lands directly on q(0) -- the
    // number that decides which rational surfaces exist at all.
    Real faceIntegral = 0;
    for (int i = 0; i < n; ++i) {
        const auto index   = static_cast<std::size_t>(i);
        const Real r       = grid.radius(i);
        const Real faceIn  = r - 0.5 * dr;
        const Real faceOut = r + 0.5 * dr;

        // integral(J r dr) is exact for piecewise-constant J.
        const Real toCentre = m_current[index] * (r * r - faceIn * faceIn) * 0.5;
        const Real centreIntegral = faceIntegral + toCentre;

        m_fluxDerivative[index] = centreIntegral / r;
        m_safetyFactor[index]   = r / std::max(m_fluxDerivative[index], 1e-9);

        faceIntegral += m_current[index] * (faceOut * faceOut - faceIn * faceIn) * 0.5;
    }
}

void Equilibrium::rebuild()
{
    if (m_grid == nullptr) return;

    buildPressureAndDensity();
    buildCurrent();

    const RadialGrid& grid = *m_grid;
    const int n            = grid.size();

    // Spitzer resistivity, eta ~ T^{-3/2}. The reference is the *initial* axis
    // temperature, not the current one: normalizing against the current value
    // would cancel any global change, and a gas puff that cools the plasma has
    // to be able to raise the resistivity -- that coupling is one of the
    // controller's levers on tearing growth.
    const Real referenceResistivity = 1.0 / std::max(m_config.lundquist, 1.0);
    for (int i = 0; i < n; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const Real ratio = std::max(m_temperature[index], 1e-6) / m_referenceAxisTemperature;
        m_resistivity[index] = referenceResistivity * std::pow(ratio, -1.5);
    }
    m_viscosity = referenceResistivity * m_config.magneticPrandtl;

    // Stored energy, the confinement figure of merit.
    m_storedEnergy = grid.integrate(m_pressure) * constants::kTwoPi;
}

Real Equilibrium::confinementFraction() const noexcept
{
    return m_storedEnergy / std::max(m_referenceStoredEnergy, constants::kEpsilon);
}

Real Equilibrium::rationalSurface(int m, int n) const
{
    if (m_grid == nullptr || n == 0) return -1;
    const Real target = static_cast<Real>(m) / static_cast<Real>(n);

    const int points = m_grid->size();
    for (int i = 0; i + 1 < points; ++i) {
        const Real q0 = m_safetyFactor[static_cast<std::size_t>(i)];
        const Real q1 = m_safetyFactor[static_cast<std::size_t>(i + 1)];
        if ((q0 - target) * (q1 - target) <= 0.0 && q1 != q0) {
            // Linear interpolation is exact enough: q varies smoothly and the
            // grid resolves the layer far more finely than this matters.
            const Real t = (target - q0) / (q1 - q0);
            return m_grid->radius(i) + t * m_grid->spacing();
        }
    }
    return -1;
}

Real Equilibrium::safetyFactorGradient(Real radius) const
{
    if (m_grid == nullptr) return 0;
    const int i = m_grid->nearestIndex(radius);
    const int n = m_grid->size();
    if (i <= 0) {
        return (m_safetyFactor[1] - m_safetyFactor[0]) / m_grid->spacing();
    }
    if (i >= n - 1) {
        return (m_safetyFactor[static_cast<std::size_t>(n - 1)] -
                m_safetyFactor[static_cast<std::size_t>(n - 2)]) / m_grid->spacing();
    }
    return (m_safetyFactor[static_cast<std::size_t>(i + 1)] -
            m_safetyFactor[static_cast<std::size_t>(i - 1)]) / (2.0 * m_grid->spacing());
}

} // namespace plasma
