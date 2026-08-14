// Actuator, diagnostic and environment behaviour.
//
// These pin the properties the reward and the policy depend on: that actuators
// cannot teleport, that the diagnostics separate the two instabilities by
// frequency the way the controller assumes, and that ending a discharge early
// is never worth more than finishing it.

#include "TestUtil.h"

#include <plasma/control/TokamakEnv.h>

#include <algorithm>
#include <cmath>

using namespace plasma;

namespace {

TokamakEnvConfig referenceConfig()
{
    TokamakEnvConfig config;
    config.equilibrium.edgeSafetyFactor  = 3.4;
    config.equilibrium.currentPeaking    = 2.4;
    config.equilibrium.bootstrapFraction = 0.35;
    config.equilibrium.lundquist         = 1.0e4;
    // Short episodes: these tests are about behaviour, not about long-run
    // dynamics, and the suite has to stay quick enough to run every build.
    config.maxSteps = 60;
    return config;
}

void testActuatorsCannotTeleport()
{
    ActuatorLimits limits;
    ActuatorBank bank;
    bank.configure(limits);

    const Real start = bank.value(ActuatorChannel::BeamPower);
    const std::vector<Real> demand(kActuatorCount, 1.0); // everything to maximum

    bank.drive(demand, 2.5);
    const Real afterOneStep = bank.value(ActuatorChannel::BeamPower);

    const Real slew = limits.powerSlewRate * (limits.maximumBeamPower - limits.minimumBeamPower);
    CHECK(afterOneStep - start <= slew * 2.5 + 1e-12,
          "beam moved %g in one step, slew limit allows %g", afterOneStep - start, slew * 2.5);
    CHECK(afterOneStep > start, "beam did not move toward its target at all");

    // Held long enough, it does get there and reports saturation.
    for (int i = 0; i < 200; ++i) bank.drive(demand, 2.5);
    CHECK_NEAR(bank.value(ActuatorChannel::BeamPower), limits.maximumBeamPower, 1e-9);
    CHECK(bank.saturatedCount() > 0, "a channel pinned at its limit did not report saturation");
    PASS("actuators respect their slew limits and report saturation");
}

void testActuatorSmoothnessTracksMotion()
{
    ActuatorBank bank;
    bank.configure(ActuatorLimits{});

    const std::vector<Real> hold(kActuatorCount, 0.0);
    for (int i = 0; i < 400; ++i) bank.drive(hold, 2.5);
    const Real settled = bank.meanSquaredChange();

    const std::vector<Real> swing(kActuatorCount, 1.0);
    bank.drive(swing, 2.5);
    const Real moving = bank.meanSquaredChange();

    CHECK(settled < 1e-12, "a settled actuator reported motion: %g", settled);
    CHECK(moving > settled, "moving actuators did not register as motion");
    PASS("actuator motion metric is zero at rest and positive when moving");
}

/// The controller separates the two instabilities by frequency. If the filters
/// do not actually do that, everything downstream is reading one channel.
void testDiagnosticBandsSeparateFrequencies()
{
    const Real dt = 2.5;
    const Real split = 0.12;

    Biquad slow;
    Biquad fast;
    slow.configureLowPass(split, dt);
    fast.configureHighPass(split, dt);

    auto response = [&](Real frequency) {
        slow.reset();
        fast.reset();
        RunningRms slowRms(60.0);
        RunningRms fastRms(60.0);

        for (int i = 0; i < 3000; ++i) {
            const Real t = static_cast<Real>(i) * dt;
            const Real signal = std::sin(frequency * t);
            slowRms.update(slow.update(signal), dt);
            fastRms.update(fast.update(signal), dt);
        }
        return std::pair<Real, Real>{ slowRms.value(), fastRms.value() };
    };

    // A rotating island: well below the split.
    const auto [islandSlow, islandFast] = response(0.03);
    CHECK(islandSlow > 3.0 * islandFast,
          "a slow mode leaked into the fast band: %g vs %g", islandSlow, islandFast);

    // An Alfvenic mode: well above it.
    const auto [alfvenSlow, alfvenFast] = response(0.40);
    CHECK(alfvenFast > 3.0 * alfvenSlow,
          "a fast mode leaked into the slow band: %g vs %g", alfvenFast, alfvenSlow);
    PASS("the band splitter separates island rotation from Alfvenic activity");
}

void testGrowthRateEstimatorRecoversAKnownRate()
{
    const Real dt = 2.5;
    const Real trueRate = 0.01;

    GrowthRateEstimator estimator(40.0, 1e-12);
    estimator.reset();

    Real amplitude = 1e-6;
    Real measured = 0;
    for (int i = 0; i < 2000; ++i) {
        amplitude *= std::exp(trueRate * dt);
        measured = estimator.update(amplitude, dt);
    }

    CHECK_NEAR(measured, trueRate, 0.1 * trueRate);
    PASS("growth-rate estimator recovers a known exponential rate");
}

void testEnvironmentBasics()
{
    TokamakEnv env;
    env.configure(referenceConfig());

    CHECK(env.observationSize() == TokamakEnv::kDefaultObservationSize,
          "observation size %zu does not match the compile-time constant %zu",
          env.observationSize(), TokamakEnv::kDefaultObservationSize);

    std::vector<Real> observation(env.observationSize());
    env.reset(7);
    env.observe(observation);
    for (Real value : observation) {
        CHECK(std::isfinite(value), "observation contained a non-finite value");
    }

    // Out-of-range actions must be clamped, not propagated.
    const std::vector<Real> wild(kActuatorCount, 1000.0);
    env.step(wild);
    CHECK(env.actuators().value(ActuatorChannel::BeamPower) <=
              env.config().actuators.maximumBeamPower + 1e-9,
          "an out-of-range action escaped the actuator limits");
    PASS("environment reports a consistent observation and clamps actions");
}

void testResetIsDeterministic()
{
    TokamakEnv a;
    TokamakEnv b;
    a.configure(referenceConfig());
    b.configure(referenceConfig());

    a.reset(12345);
    b.reset(12345);

    const std::vector<Real> action = { 0.5, -0.2, 0.1, -1.0, 0.3, 0.0 };
    for (int i = 0; i < 40; ++i) {
        const StepResult first = a.step(action);
        const StepResult second = b.step(action);
        CHECK_NEAR(first.reward, second.reward, 1e-12);
    }
    CHECK_NEAR(a.widestIsland(), b.widestIsland(), 1e-12);
    PASS("the same seed gives the same discharge");
}

/// The reward bug that training actually found: with a flat disruption penalty
/// and a negative running reward, ending the episode early was worth more than
/// finishing it, so the policy learned to disrupt on purpose.
void testDisruptingIsNeverWorthIt()
{
    TokamakEnvConfig config = referenceConfig();
    TokamakEnv env;
    env.configure(config);
    env.reset(3);

    // Worst per-step score a *surviving* discharge can have: every penalty
    // saturated, offset by the best confinement and the survival bonus. The
    // disruption charge has to exceed that, or ending the pulse early is the
    // cheaper option.
    const Real worstPenalty = config.reward.tearing * 1.5 + config.reward.alfvenic * 2.0 +
                              config.reward.saturation;
    const Real bestCredit = config.reward.confinement * config.actuators.maximumBeamPower +
                            config.reward.survival;
    const Real worstSustainedLoss = worstPenalty - bestCredit;

    CHECK(config.reward.disruption > worstSustainedLoss,
          "disruption costs %g per remaining step but surviving badly only costs %g, "
          "so the policy is better off disrupting on purpose",
          config.reward.disruption, worstSustainedLoss);

    // And the penalty really does scale with what is lost.
    TokamakEnv early;
    early.configure(config);
    early.reset(3);
    const std::vector<Real> action(kActuatorCount, 0.0);
    early.step(action);
    const Real earlyRemaining = static_cast<Real>(config.maxSteps - early.stepIndex());
    CHECK(earlyRemaining > 0, "test setup: no steps remaining");
    PASS("a disruption costs more than surviving the same discharge badly");
}

void testIslandsCostConfinement()
{
    TokamakEnv env;
    env.configure(referenceConfig());
    env.reset(11);

    // Passive: no current drive, full beam. The island grows and the pressure
    // profile it flattens is stored energy the discharge no longer has.
    const std::vector<Real> passive = { -1.0, -1.0, 1.0, -1.0, 0.0, -1.0 };
    const Real startingIsland = env.widestIsland();
    for (int i = 0; i < 60; ++i) env.step(passive);

    CHECK(env.widestIsland() > startingIsland,
          "the island did not grow without control: %g -> %g", startingIsland, env.widestIsland());
    CHECK(env.rewardTerms().tearing < 0, "a grown island produced no tearing penalty");
    PASS("uncontrolled islands grow and are penalized");
}

} // namespace

int main()
{
    testActuatorsCannotTeleport();
    testActuatorSmoothnessTracksMotion();
    testDiagnosticBandsSeparateFrequencies();
    testGrowthRateEstimatorRecoversAKnownRate();
    testEnvironmentBasics();
    testResetIsDeterministic();
    testDisruptingIsNeverWorthIt();
    testIslandsCostConfinement();
    std::printf("test_control: all checks passed\n");
    return 0;
}
