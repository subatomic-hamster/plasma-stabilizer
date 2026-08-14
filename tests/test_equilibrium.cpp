// The equilibrium is where the actuators act. If current drive does not move
// q(r), or if an island does not flatten the profiles, the control problem is
// not a control problem -- the policy would be steering something disconnected
// from the physics.

#include "TestUtil.h"

#include <plasma/mhd/Equilibrium.h>

#include <algorithm>

using namespace plasma;

namespace {

EquilibriumConfig referenceConfig()
{
    EquilibriumConfig config;
    config.edgeSafetyFactor = 3.4;
    config.currentPeaking   = 2.6;
    config.bootstrapFraction = 0.35;
    return config;
}

void testSafetyFactorProfile()
{
    RadialGrid grid(256, 1.0);
    EquilibriumConfig config = referenceConfig();
    config.bootstrapFraction = 0.0; // isolate the ohmic profile

    Equilibrium equilibrium;
    equilibrium.configure(grid, config);

    const ConstProfile q = equilibrium.safetyFactor();

    // q(a) is what fixes the total current, so it must come back out.
    CHECK_NEAR(q.back(), config.edgeSafetyFactor, 0.05 * config.edgeSafetyFactor);
    CHECK_NEAR(equilibrium.plasmaCurrent(), 1.0 / config.edgeSafetyFactor, 1e-12);

    // For j ~ (1 - x^2)^nu with the current fixed, q(0) = q(a) / (nu + 1).
    const Real expectedAxis = config.edgeSafetyFactor / (config.currentPeaking + 1.0);
    CHECK_NEAR(q.front(), expectedAxis, 0.03 * expectedAxis);

    // Monotonic: a peaked current profile has monotonically rising q, which is
    // what makes the rational-surface search a simple bracket.
    for (std::size_t i = 1; i < q.size(); ++i) {
        CHECK(q[i] > q[i - 1] - 1e-12, "q is not monotonic at index %zu", i);
    }
    PASS("q profile matches the analytic axis and edge values and is monotonic");
}

void testRationalSurfaces()
{
    RadialGrid grid(256, 1.0);
    Equilibrium equilibrium;
    equilibrium.configure(grid, referenceConfig());

    const Real surface = equilibrium.rationalSurface(2, 1);
    CHECK(surface > 0, "no q = 2 surface found in the reference equilibrium");

    // q at the reported radius really is 2.
    const Real q = sampleUniform(equilibrium.safetyFactor(),
                                 (surface - grid.radius(0)) /
                                     (grid.radius(grid.size() - 1) - grid.radius(0)));
    CHECK_NEAR(q, 2.0, 0.02);

    // q never reaches 5 inside this plasma, so there is no such surface.
    CHECK(equilibrium.rationalSurface(5, 1) < 0, "found a q = 5 surface that should not exist");
    PASS("rational surfaces are located where q actually equals m/n");
}

void testDrivenCurrentReshapesQ()
{
    RadialGrid grid(256, 1.0);
    Equilibrium equilibrium;
    equilibrium.configure(grid, referenceConfig());

    const std::vector<Real> before(equilibrium.safetyFactor().begin(),
                                   equilibrium.safetyFactor().end());
    const Real currentBefore = equilibrium.plasmaCurrent();

    DrivenCurrent driven;
    driven.normalizedRadius = 0.6;
    driven.width            = 0.06;
    driven.fraction         = 0.15;
    equilibrium.setDrivenCurrent(driven);
    equilibrium.rebuild();

    // The transformer holds the total current: driving current locally
    // redistributes it rather than adding to it.
    CHECK_NEAR(equilibrium.plasmaCurrent(), currentBefore, 1e-12);

    // Deposition really is local.
    const int depositionIndex = grid.nearestIndex(0.6);
    const ConstProfile profile = equilibrium.drivenCurrentProfile();
    Real peak = 0;
    int peakIndex = 0;
    for (int i = 0; i < grid.size(); ++i) {
        if (profile[static_cast<std::size_t>(i)] > peak) {
            peak = profile[static_cast<std::size_t>(i)];
            peakIndex = i;
        }
    }
    CHECK(std::abs(peakIndex - depositionIndex) <= 2,
          "driven current peaked at index %d, expected near %d", peakIndex, depositionIndex);

    // And q actually moves.
    Real worstChange = 0;
    for (int i = 0; i < grid.size(); ++i) {
        worstChange = std::max(worstChange,
                               std::abs(equilibrium.safetyFactor()[static_cast<std::size_t>(i)] -
                                        before[static_cast<std::size_t>(i)]));
    }
    CHECK(worstChange > 1e-3, "current drive barely moved q: worst change %g", worstChange);
    PASS("driven current is local, conserves total current, and reshapes q");
}

void testIslandFlattening()
{
    RadialGrid grid(256, 1.0);
    Equilibrium equilibrium;
    equilibrium.configure(grid, referenceConfig());

    const Real surface = equilibrium.rationalSurface(2, 1);
    CHECK(surface > 0, "reference equilibrium has no q = 2 surface");

    const Real energyBefore = equilibrium.storedEnergy();
    const int index = grid.nearestIndex(surface);
    const Real gradientBefore = std::abs(equilibrium.currentGradient()[static_cast<std::size_t>(index)]);

    const IslandRegion island{ surface, 0.18 };
    equilibrium.setIslands(std::span<const IslandRegion>(&island, 1));
    equilibrium.rebuild();

    // Pressure flattening across the island costs stored energy: this is the
    // confinement degradation the reward has to trade against.
    CHECK(equilibrium.storedEnergy() < energyBefore,
          "island did not reduce stored energy: %g -> %g",
          energyBefore, equilibrium.storedEnergy());

    // Current flattening removes the gradient that drives the mode: this is the
    // saturation mechanism.
    const Real gradientAfter = std::abs(equilibrium.currentGradient()[static_cast<std::size_t>(index)]);
    CHECK(gradientAfter < gradientBefore,
          "island did not flatten the current gradient: %g -> %g", gradientBefore, gradientAfter);
    PASS("islands flatten pressure and current, costing confinement and drive");
}

void testHeatingAndDensityScales()
{
    RadialGrid grid(256, 1.0);
    Equilibrium equilibrium;
    equilibrium.configure(grid, referenceConfig());

    const Real baseline = equilibrium.confinementFraction();
    CHECK_NEAR(baseline, 1.0, 1e-9);

    equilibrium.setHeatingScale(1.4);
    equilibrium.rebuild();
    CHECK(equilibrium.confinementFraction() > baseline * 1.2,
          "extra heating did not raise stored energy");

    equilibrium.setHeatingScale(1.0);
    equilibrium.setDensityScale(1.5);
    equilibrium.rebuild();

    // Higher density at fixed pressure means lower temperature and so higher
    // resistivity -- the coupling that lets a gas puff change tearing growth.
    Equilibrium reference;
    reference.configure(grid, referenceConfig());
    CHECK(equilibrium.resistivity()[128] > reference.resistivity()[128],
          "raising density did not raise resistivity");
    PASS("heating raises confinement and density raises resistivity");
}

void testResetRestoresReference()
{
    RadialGrid grid(256, 1.0);
    Equilibrium equilibrium;
    equilibrium.configure(grid, referenceConfig());

    const std::vector<Real> reference(equilibrium.safetyFactor().begin(),
                                      equilibrium.safetyFactor().end());

    DrivenCurrent driven{ 0.3, 0.05, 0.2 };
    equilibrium.setDrivenCurrent(driven);
    equilibrium.setHeatingScale(1.6);
    const IslandRegion island{ 0.5, 0.2 };
    equilibrium.setIslands(std::span<const IslandRegion>(&island, 1));
    equilibrium.rebuild();

    equilibrium.reset();

    for (std::size_t i = 0; i < reference.size(); ++i) {
        CHECK_NEAR(equilibrium.safetyFactor()[i], reference[i], 1e-12);
    }
    PASS("reset returns the equilibrium to its reference state exactly");
}

} // namespace

int main()
{
    testSafetyFactorProfile();
    testRationalSurfaces();
    testDrivenCurrentReshapesQ();
    testIslandFlattening();
    testHeatingAndDensityScales();
    testResetRestoresReference();
    std::printf("test_equilibrium: all checks passed\n");
    return 0;
}
