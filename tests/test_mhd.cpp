// Physics validation for the reduced-MHD solver.
//
// These checks are against known results rather than against stored numbers:
// the resistive tearing growth rate has a textbook scaling, a shear Alfven wave
// has a known frequency, and the actuators must demonstrably change the growth
// rate or there is nothing for a controller to learn.

#include "TestUtil.h"

#include <plasma/mhd/ReducedMhd.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <tuple>

using namespace plasma;

namespace {

struct Setup
{
    RadialGrid  grid;
    Equilibrium equilibrium;
    ReducedMhd  mhd;
};

/// A configuration in which the (2,1) tearing mode is classically unstable.
EquilibriumConfig unstableConfig(Real lundquist = 1.0e4)
{
    EquilibriumConfig config;
    config.edgeSafetyFactor  = 3.0;
    config.currentPeaking    = 1.5;
    config.bootstrapFraction = 0.0;
    config.lundquist         = lundquist;
    return config;
}

Real measureGrowthRate(ReducedMhd& mhd, int mode, Real settle, Real window)
{
    mhd.advance(settle);
    const Real before = mhd.amplitude(mode);
    const Real start  = mhd.time();
    mhd.advance(window);
    return std::log(mhd.amplitude(mode) / before) / (mhd.time() - start);
}

Real tearingGrowth(Real lundquist, int points, Real settle, Real window,
                   const DrivenCurrent* driven = nullptr)
{
    static std::vector<std::unique_ptr<Setup>> keepAlive;
    auto setup = std::make_unique<Setup>();

    setup->grid.configure(points, 1.0);
    setup->equilibrium.configure(setup->grid, unstableConfig(lundquist));
    if (driven != nullptr) {
        setup->equilibrium.setDrivenCurrent(*driven);
        setup->equilibrium.rebuild();
    }

    const ModeSpec modes[] = { { 2, 1, ModeClass::Tearing } };
    MhdParameters parameters;
    parameters.substep      = 0.1;
    parameters.coreRotation = 0.0;

    setup->mhd.configure(setup->grid, modes, parameters);
    setup->mhd.reset(11);
    setup->mhd.onEquilibriumChanged(setup->equilibrium);

    const Real rate = measureGrowthRate(setup->mhd, 0, settle, window);
    keepAlive.push_back(std::move(setup));
    return rate;
}

void testTearingModeGrows()
{
    const Real rate = tearingGrowth(1.0e4, 256, 300, 300);
    CHECK(rate > 0.005, "tearing mode did not grow: gamma = %g", rate);
    CHECK(rate < 0.2, "growth rate %g is implausibly fast for a resistive mode", rate);
    PASS("the (2,1) tearing mode is unstable in a tearing-unstable equilibrium");
}

void testGrowthRateIsGridConverged()
{
    const Real coarse = tearingGrowth(1.0e4, 128, 300, 300);
    const Real medium = tearingGrowth(1.0e4, 256, 300, 300);
    const Real fine   = tearingGrowth(1.0e4, 512, 300, 300);

    CHECK(std::abs(medium - fine) < 0.01 * fine,
          "growth rate not converged: 256 gave %g, 512 gave %g", medium, fine);
    CHECK(std::abs(coarse - fine) < 0.03 * fine,
          "growth rate varies too much with resolution: %g vs %g", coarse, fine);
    PASS("growth rate is independent of radial resolution");
}

/// Resistive tearing theory gives gamma ~ eta^(3/5) asymptotically. The exponent
/// is only approached at large Lundquist number, so the assertion is that the
/// measured log-log slope moves toward -0.6 as S rises -- which is a far
/// stronger statement about the solver than any single number would be.
void testResistiveScaling()
{
    struct Sample { Real lundquist; Real rate; };
    std::vector<Sample> samples;

    for (Real S : { 3.0e3, 1.0e4, 3.0e4, 1.0e5 }) {
        // Window scaled to the expected growth time, so every sample gains
        // about the same number of e-foldings and none overflows.
        const Real expected = 0.027 * std::pow(S / 3.0e3, -0.6);
        const Real window   = 2.5 / expected;
        samples.push_back({ S, tearingGrowth(S, 512, window, window) });
    }

    for (const Sample& sample : samples) {
        CHECK(sample.rate > 0, "mode became stable at S = %g", sample.lundquist);
    }

    std::vector<Real> slopes;
    for (std::size_t i = 1; i < samples.size(); ++i) {
        slopes.push_back(std::log(samples[i].rate / samples[i - 1].rate) /
                         std::log(samples[i].lundquist / samples[i - 1].lundquist));
    }

    // Monotonically steepening toward the asymptotic value.
    for (std::size_t i = 1; i < slopes.size(); ++i) {
        CHECK(slopes[i] < slopes[i - 1] + 0.02,
              "slope stopped approaching -0.6: %g then %g", slopes[i - 1], slopes[i]);
    }
    CHECK(slopes.back() < -0.40 && slopes.back() > -0.75,
          "final slope %g is not near the -0.6 tearing exponent", slopes.back());
    PASS("growth rate follows the resistive tearing scaling toward eta^(3/5)");
}

/// The Alfvenic branch and its damping mechanism.
///
/// The naive expectation -- every radius rings at its own local |F| -- is only
/// true transiently. Phase mixing across the sheared continuum damps those
/// local oscillations, and what survives is a single global structure that
/// accumulates at a continuum extremum, where dF/dr vanishes and there is no
/// phase mixing left to damp it. So the checks are: the mode oscillates, at one
/// frequency everywhere, inside the continuum band, near the extremum, and it
/// is radially localized there.
void testAlfvenContinuum()
{
    RadialGrid grid(256, 1.0);
    EquilibriumConfig config;
    config.edgeSafetyFactor  = 3.4;
    config.currentPeaking    = 2.6;
    config.bootstrapFraction = 0.35;

    Equilibrium equilibrium;
    equilibrium.configure(grid, config);

    const ModeSpec modes[] = { { 1, 1, ModeClass::Alfvenic } };
    MhdParameters parameters;
    parameters.substep        = 0.05;
    parameters.coreRotation   = 0.0;
    parameters.energeticDrive = 0.0;

    ReducedMhd mhd;
    mhd.configure(grid, modes, parameters);
    mhd.reset(3);
    mhd.onEquilibriumChanged(equilibrium);

    const std::vector<int> probes = { 32, 96, 160, 224 };
    std::vector<Real> lastSign(probes.size(), 0);
    std::vector<Real> firstCrossing(probes.size(), -1);
    std::vector<Real> lastCrossing(probes.size(), 0);
    std::vector<int>  crossings(probes.size(), 0);
    std::vector<Real> peak(probes.size(), 0);

    mhd.advance(200.0); // let the transient continuum response phase-mix away
    for (int step = 0; step < 4000; ++step) {
        mhd.advance(0.25);
        for (std::size_t j = 0; j < probes.size(); ++j) {
            const Real value = mhd.flux(0)[probes[j]].real();
            peak[j] = std::max(peak[j], std::abs(value));
            const Real sign = value > 0 ? 1.0 : -1.0;
            if (lastSign[j] != 0 && sign != lastSign[j]) {
                ++crossings[j];
                if (firstCrossing[j] < 0) firstCrossing[j] = mhd.time();
                lastCrossing[j] = mhd.time();
            }
            lastSign[j] = sign;
        }
    }

    Real continuumMin = 1e30;
    Real continuumMax = -1e30;
    for (int i = 0; i < grid.size(); ++i) {
        const Real value = std::abs(mhd.parallelWavenumber(0)[static_cast<std::size_t>(i)]);
        continuumMin = std::min(continuumMin, value);
        continuumMax = std::max(continuumMax, value);
    }

    std::vector<Real> frequencies;
    for (std::size_t j = 0; j < probes.size(); ++j) {
        CHECK(crossings[j] > 10, "probe %zu did not oscillate: %d crossings", j, crossings[j]);
        const Real period = 2.0 * (lastCrossing[j] - firstCrossing[j]) /
                            static_cast<Real>(crossings[j] - 1);
        frequencies.push_back(constants::kTwoPi / period);
    }

    // One global frequency, not a spread of local ones.
    for (std::size_t j = 1; j < frequencies.size(); ++j) {
        CHECK_NEAR(frequencies[j], frequencies[0], 0.05 * frequencies[0]);
    }

    const Real frequency = frequencies[0];
    CHECK(frequency > continuumMin * 0.9 && frequency < continuumMax,
          "frequency %g lies outside the continuum band [%g, %g]",
          frequency, continuumMin, continuumMax);
    CHECK_NEAR(frequency, continuumMin, 0.15 * continuumMin);

    // Localized where it accumulated, not spread across the plasma.
    CHECK(peak.front() > 10.0 * peak.back(),
          "mode is not radially localized: inner peak %g, outer peak %g",
          peak.front(), peak.back());
    PASS("Alfvenic mode phase-mixes to a global frequency at the continuum minimum");
}

/// The beam drives the Alfvenic mode. Below threshold it is damped, above it
/// grows, and the fast-ion closure keeps it bounded instead of overflowing.
void testEnergeticDriveThreshold()
{
    auto run = [](Real drive, Real duration) {
        RadialGrid grid(256, 1.0);
        EquilibriumConfig config;
        config.edgeSafetyFactor  = 3.4;
        config.currentPeaking    = 2.6;
        config.bootstrapFraction = 0.35;

        Equilibrium equilibrium;
        equilibrium.configure(grid, config);

        const ModeSpec modes[] = { { 1, 1, ModeClass::Alfvenic } };
        MhdParameters parameters;
        parameters.substep        = 0.05;
        parameters.coreRotation   = 0.0;
        parameters.energeticDrive = drive;

        ReducedMhd mhd;
        mhd.configure(grid, modes, parameters);
        mhd.reset(3);
        mhd.onEquilibriumChanged(equilibrium);

        mhd.advance(100.0);
        const Real before = mhd.amplitude(0);
        const Real start  = mhd.time();
        const bool ok     = mhd.advance(duration);
        return std::tuple<bool, Real, Real>{
            ok, std::log(mhd.amplitude(0) / before) / (mhd.time() - start), mhd.amplitude(0)
        };
    };

    const auto [okQuiet, quiet, quietAmplitude] = run(0.0, 400.0);
    CHECK(okQuiet, "undriven Alfvenic mode diverged");
    CHECK(quiet < 0, "undriven Alfvenic mode should be damped, got gamma = %g", quiet);

    const auto [okDriven, driven, drivenAmplitude] = run(0.30, 400.0);
    CHECK(okDriven, "driven Alfvenic mode diverged");
    CHECK(driven > 0, "beam drive did not destabilize the mode: gamma = %g", driven);
    CHECK(drivenAmplitude > quietAmplitude, "driven mode is not larger than the damped one");

    // Well above threshold, the quasilinear closure has to hold it finite.
    const auto [okStrong, strong, strongAmplitude] = run(0.8, 3000.0);
    CHECK(okStrong, "strongly driven mode diverged instead of saturating");
    CHECK(std::isfinite(strongAmplitude) && strongAmplitude < 1.0,
          "strong drive did not saturate: amplitude %g", strongAmplitude);
    (void)strong;
    PASS("beam drive has a threshold and saturates through the fast-ion closure");
}

/// The point of the whole exercise: current drive near the rational surface must
/// change the tearing growth rate. If it does not, there is no control problem.
void testCurrentDriveHasAuthority()
{
    const Real baseline = tearingGrowth(1.0e4, 256, 300, 300);

    RadialGrid probeGrid(256, 1.0);
    Equilibrium probe;
    probe.configure(probeGrid, unstableConfig());
    const Real surface = probe.rationalSurface(2, 1);
    CHECK(surface > 0, "no rational surface to aim at");

    DrivenCurrent onSurface;
    onSurface.normalizedRadius = surface;
    onSurface.width            = 0.05;
    onSurface.fraction         = 0.20;
    const Real driven = tearingGrowth(1.0e4, 256, 300, 300, &onSurface);

    DrivenCurrent misplaced = onSurface;
    misplaced.normalizedRadius = 0.15; // deep in the core, far from the surface
    const Real elsewhere = tearingGrowth(1.0e4, 256, 300, 300, &misplaced);

    CHECK(std::abs(driven - baseline) > 0.15 * std::abs(baseline),
          "current drive on the rational surface barely changed growth: %g vs %g",
          baseline, driven);
    CHECK(driven < baseline, "co-current drive on the surface should be stabilizing: %g vs %g",
          baseline, driven);

    // Placement changes the answer materially -- which is what makes the
    // deposition radius a meaningful action dimension rather than a knob the
    // policy can ignore.
    //
    // Note which way round it comes out: at this driven fraction, core
    // deposition is *more* stabilizing than aiming at the island, because
    // redistributing a fifth of the plasma current reshapes q globally and
    // moves the rational surface. Localized deposition wins on efficiency at
    // small driven fractions, not on raw authority at large ones.
    CHECK(std::abs(driven - elsewhere) > 0.3 * std::abs(baseline),
          "deposition radius barely mattered: on-surface %g, core %g, baseline %g",
          driven, elsewhere, baseline);
    PASS("current drive changes the growth rate, and deposition radius matters");
}

/// An applied resonant field must be able to both drive and oppose the mode,
/// depending on its phase. That phase is an actuator channel.
void testResonantPerturbationPhaseMatters()
{
    auto finalAmplitude = [](Complex applied) {
        RadialGrid grid(256, 1.0);
        Equilibrium equilibrium;
        equilibrium.configure(grid, unstableConfig());

        const ModeSpec modes[] = { { 2, 1, ModeClass::Tearing } };
        MhdParameters parameters;
        parameters.substep      = 0.1;
        parameters.coreRotation = 0.0;

        ReducedMhd mhd;
        mhd.configure(grid, modes, parameters);
        mhd.reset(11);
        mhd.onEquilibriumChanged(equilibrium);

        mhd.advance(300.0); // let the eigenmode establish its phase
        const Complex reference = mhd.flux(0)[grid.nearestIndex(0.7)];
        const Real phase = std::arg(reference);

        mhd.setResonantPerturbation(0, applied * std::polar(1.0, phase));
        mhd.advance(200.0);
        return mhd.amplitude(0);
    };

    const Real aligned = finalAmplitude(Complex{ 2.0e-5, 0.0 });
    const Real opposed = finalAmplitude(Complex{ -2.0e-5, 0.0 });

    CHECK(aligned > opposed,
          "applied field phase had no effect: aligned %g, opposed %g", aligned, opposed);
    PASS("applied resonant field drives or opposes the mode depending on phase");
}

void testRotationProducesAFrequency()
{
    RadialGrid grid(256, 1.0);
    Equilibrium equilibrium;
    equilibrium.configure(grid, unstableConfig());

    const ModeSpec modes[] = { { 2, 1, ModeClass::Tearing } };
    MhdParameters parameters;
    parameters.substep      = 0.1;
    parameters.coreRotation = 0.05;

    ReducedMhd mhd;
    mhd.configure(grid, modes, parameters);
    mhd.reset(11);
    mhd.onEquilibriumChanged(equilibrium);
    mhd.advance(200.0);

    // A rotating island sweeps past the coils, so the wall signal has to change
    // phase. A static island would give a DC signal no inductive coil could see.
    const Real firstPhase = std::arg(mhd.wallField(0));
    mhd.advance(30.0);
    const Real secondPhase = std::arg(mhd.wallField(0));

    CHECK(std::abs(firstPhase - secondPhase) > 1e-3,
          "rotating plasma produced a stationary wall signal");
    PASS("plasma rotation makes the wall signal rotate");
}

} // namespace

int main()
{
    testTearingModeGrows();
    testGrowthRateIsGridConverged();
    testResistiveScaling();
    testAlfvenContinuum();
    testEnergeticDriveThreshold();
    testCurrentDriveHasAuthority();
    testResonantPerturbationPhaseMatters();
    testRotationProducesAFrequency();
    std::printf("test_mhd: all checks passed\n");
    return 0;
}
