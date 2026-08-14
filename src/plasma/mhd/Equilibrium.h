#pragma once

// Cylindrical (large-aspect-ratio) tokamak equilibrium.
//
// The current profile is the primary quantity and everything else is derived
// from it. That is the physically causal direction and it is what makes the
// actuators mean something: electron cyclotron current drive deposits current
// at a chosen radius, which reshapes q(r), which moves the rational surfaces
// and changes the free energy available to a tearing mode. Prescribing q(r)
// directly and bolting a current source on the side would break that chain.
//
//   J(r)      = ohmic + bootstrap + driven
//   psi0'(r)  = (1/r) * integral_0^r J r' dr'      (poloidal field)
//   q(r)      = r / psi0'(r)                        (safety factor)
//
// Total plasma current is held fixed, as an ohmic transformer would: driving
// current locally with ECCD redistributes it rather than adding to it.

#include <plasma/core/Types.h>
#include <plasma/mhd/RadialGrid.h>

namespace plasma {

struct EquilibriumConfig
{
    Real minorRadius{ 1.0 };   ///< a, the length unit
    Real majorRadius{ 3.0 };   ///< R0; the inverse aspect ratio a/R0 sets bootstrap strength

    /// Edge safety factor. Fixes the total plasma current, since q(a) = a^2 / I.
    Real edgeSafetyFactor{ 4.0 };
    /// Ohmic current peaking exponent in (1 - (r/a)^2)^peaking. With the edge q
    /// above, q(0) = q(a) / (peaking + 1), so this chooses which rational
    /// surfaces exist inside the plasma.
    Real currentPeaking{ 2.0 };

    /// Poloidal beta at reference heating. Sets the pressure scale, and through
    /// it the bootstrap current and the NTM drive.
    Real betaPoloidal{ 0.55 };
    /// Fraction of the plasma current carried by the bootstrap current at
    /// reference conditions.
    Real bootstrapFraction{ 0.35 };

    /// Lundquist number; resistivity on axis is 1/S in Alfven units.
    Real lundquist{ 1.0e4 };
    /// Ratio of viscosity to resistivity.
    Real magneticPrandtl{ 1.0 };

    /// Edge density as a fraction of the axis value.
    Real edgeDensityFraction{ 0.25 };
};

/// A magnetic island, as seen by the equilibrium: inside it the pressure and
/// the bootstrap current it carries are flattened.
struct IslandRegion
{
    Real centerRadius{ 0 };
    Real width{ 0 };
};

/// Externally driven current, i.e. the ECCD actuator.
struct DrivenCurrent
{
    /// Deposition location as r/a.
    Real normalizedRadius{ 0.5 };
    /// Gaussian deposition half-width as a fraction of the minor radius.
    Real width{ 0.05 };
    /// Driven current as a fraction of the total plasma current.
    Real fraction{ 0 };
};

class Equilibrium
{
public:
    void configure(const RadialGrid& grid, const EquilibriumConfig& config);

    /// Returns the model to its reference state: no driven current, no islands,
    /// nominal heating and density.
    void reset();

    // --- Actuator and feedback inputs --------------------------------------
    // Each of these only records the request; rebuild() applies them.

    void setDrivenCurrent(const DrivenCurrent& driven) { m_driven = driven; }
    /// Heating power as a multiple of the reference value; scales the pressure.
    void setHeatingScale(Real scale) { m_heatingScale = scale; }
    /// Density as a multiple of the reference value; slows Alfven waves and
    /// raises resistivity through the temperature.
    void setDensityScale(Real scale) { m_densityScale = scale; }
    /// Islands whose interior is flattened. Replaces the previous set.
    void setIslands(std::span<const IslandRegion> islands);

    /// Recomputes every derived profile. Cheap enough to call once per control
    /// step; not per MHD substep.
    void rebuild();

    // --- Derived profiles ---------------------------------------------------

    [[nodiscard]] ConstProfile current() const noexcept { return m_current; }
    [[nodiscard]] ConstProfile currentGradient() const noexcept { return m_currentGradient; }
    [[nodiscard]] ConstProfile poloidalFlux() const noexcept { return m_fluxDerivative; }
    [[nodiscard]] ConstProfile safetyFactor() const noexcept { return m_safetyFactor; }
    [[nodiscard]] ConstProfile pressure() const noexcept { return m_pressure; }
    [[nodiscard]] ConstProfile temperature() const noexcept { return m_temperature; }
    [[nodiscard]] ConstProfile density() const noexcept { return m_density; }
    [[nodiscard]] ConstProfile resistivity() const noexcept { return m_resistivity; }
    [[nodiscard]] ConstProfile bootstrapCurrent() const noexcept { return m_bootstrapCurrent; }
    [[nodiscard]] ConstProfile drivenCurrentProfile() const noexcept { return m_drivenProfile; }

    // --- Scalars ------------------------------------------------------------

    [[nodiscard]] Real plasmaCurrent() const noexcept { return m_plasmaCurrent; }
    [[nodiscard]] Real storedEnergy() const noexcept { return m_storedEnergy; }
    [[nodiscard]] Real axisSafetyFactor() const noexcept { return m_safetyFactor.empty() ? 0 : m_safetyFactor.front(); }
    [[nodiscard]] Real viscosity() const noexcept { return m_viscosity; }
    [[nodiscard]] Real heatingScale() const noexcept { return m_heatingScale; }
    [[nodiscard]] Real densityScale() const noexcept { return m_densityScale; }
    /// Fraction of the reference stored energy currently confined. The
    /// confinement term of the reward reads this.
    [[nodiscard]] Real confinementFraction() const noexcept;

    /// Radius where q = m/n, or a negative value when no such surface exists
    /// inside the plasma. q is monotonic here, so a bisection is exact.
    [[nodiscard]] Real rationalSurface(int m, int n) const;

    /// dq/dr at a radius, needed for the island width and for the shear term.
    [[nodiscard]] Real safetyFactorGradient(Real radius) const;

    [[nodiscard]] const RadialGrid& grid() const noexcept { return *m_grid; }
    [[nodiscard]] const EquilibriumConfig& config() const noexcept { return m_config; }

private:
    void buildPressureAndDensity();
    void buildCurrent();
    void buildFluxAndSafetyFactor();

    const RadialGrid* m_grid{ nullptr };
    EquilibriumConfig m_config;

    DrivenCurrent             m_driven;
    std::vector<IslandRegion> m_islands;
    Real m_heatingScale{ 1 };
    Real m_densityScale{ 1 };

    std::vector<Real> m_current;
    std::vector<Real> m_currentGradient;
    std::vector<Real> m_ohmicCurrent;
    std::vector<Real> m_bootstrapCurrent;
    std::vector<Real> m_drivenProfile;
    std::vector<Real> m_fluxDerivative;
    std::vector<Real> m_safetyFactor;
    std::vector<Real> m_pressure;
    std::vector<Real> m_temperature;
    std::vector<Real> m_density;
    std::vector<Real> m_resistivity;

    Real m_plasmaCurrent{ 0 };
    Real m_storedEnergy{ 0 };
    Real m_referenceStoredEnergy{ 1 };
    /// Axis temperature of the reference state. Resistivity is normalized
    /// against this, not against the current axis value, so that cooling the
    /// plasma actually raises eta instead of cancelling out.
    Real m_referenceAxisTemperature{ 1 };
    Real m_viscosity{ 0 };
};

} // namespace plasma
