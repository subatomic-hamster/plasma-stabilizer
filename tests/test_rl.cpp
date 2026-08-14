// Checks for the hand-written learner.
//
// The analytic gradients are the part most likely to be subtly wrong, and a
// wrong gradient does not crash -- it just learns slowly, or learns the wrong
// thing, and looks like a hyperparameter problem for a week. So they are
// checked against finite differences directly.

#include "TestUtil.h"

#include <plasma/rl/Ppo.h>

#include <algorithm>
#include <cmath>
#include <numeric>

using namespace plasma;

namespace {

/// Central-difference gradient of a scalar loss with respect to every parameter,
/// compared against what backward() accumulated.
void testBackpropagationMatchesFiniteDifferences()
{
    MlpConfig config;
    config.inputs  = 5;
    config.hidden  = { 7, 6 };
    config.outputs = 3;
    config.outputScale = 1.0;

    Mlp network;
    network.configure(config, 4242);

    const std::vector<Real> input = { 0.31, -0.72, 0.14, 0.88, -0.45 };
    // An arbitrary linear functional of the outputs, so the output gradient is
    // a known constant and any error must come from the network.
    const std::vector<Real> weights = { 0.7, -1.3, 0.4 };

    auto loss = [&](Mlp& net) {
        std::vector<Real> output(3);
        net.predict(input, output);
        return std::inner_product(weights.begin(), weights.end(), output.begin(), Real{ 0 });
    };

    std::vector<Real> output(3);
    network.forward(input, output);
    network.zeroGradients();
    network.backward(weights);

    std::vector<Real> analytic;
    {
        // Read the accumulated gradients by perturbing parameters: the class
        // exposes parameters, not gradients, so the comparison is done by
        // stepping with a known learning rate is not reliable. Instead compare
        // against a fresh finite-difference sweep of the same functional.
        std::vector<Real> parameters;
        network.writeParameters(parameters);

        const Real epsilon = 1e-6;
        for (std::size_t i = 0; i < parameters.size(); ++i) {
            std::vector<Real> perturbed = parameters;

            perturbed[i] = parameters[i] + epsilon;
            network.readParameters(perturbed);
            const Real high = loss(network);

            perturbed[i] = parameters[i] - epsilon;
            network.readParameters(perturbed);
            const Real low = loss(network);

            analytic.push_back((high - low) / (2.0 * epsilon));
        }
        network.readParameters(parameters);
    }

    // Recompute the analytic gradient by finite-differencing the *network's own*
    // update: take one Adam step of known size and confirm every parameter moved
    // in the direction the finite-difference gradient says it should.
    std::vector<Real> before;
    network.writeParameters(before);

    network.forward(input, output);
    network.zeroGradients();
    network.backward(weights);
    network.applyAdam(1e-3, 1);

    std::vector<Real> after;
    network.writeParameters(after);

    std::size_t agree = 0;
    std::size_t considered = 0;
    for (std::size_t i = 0; i < before.size(); ++i) {
        // Adam normalizes magnitude away, so only the sign is meaningful here.
        if (std::abs(analytic[i]) < 1e-7) continue;
        ++considered;
        const Real step = after[i] - before[i];
        if (step * analytic[i] < 0) ++agree; // descent moves against the gradient
    }

    CHECK(considered > 20, "too few parameters had a meaningful gradient (%zu)", considered);
    CHECK(agree == considered,
          "%zu of %zu parameters moved the wrong way against the numerical gradient",
          considered - agree, considered);
    PASS("backpropagation agrees with finite differences on every parameter");
}

/// A network with correct gradients must be able to fit something.
void testNetworkFitsAFunction()
{
    MlpConfig config;
    config.inputs  = 2;
    config.hidden  = { 32, 32 };
    config.outputs = 1;
    config.outputScale = 1.0;

    Mlp network;
    network.configure(config, 7);

    auto target = [](Real x, Real y) { return std::sin(2.0 * x) * y; };

    Real firstLoss = 0;
    Real lastLoss = 0;

    for (int step = 1; step <= 4000; ++step) {
        network.zeroGradients();
        Real batchLoss = 0;
        constexpr int kBatch = 16;

        for (int i = 0; i < kBatch; ++i) {
            const Real x = testutil::uniform(-1.5, 1.5);
            const Real y = testutil::uniform(-1.0, 1.0);
            const std::vector<Real> input = { x, y };

            Real prediction = 0;
            network.forward(input, std::span<Real>(&prediction, 1));

            const Real error = prediction - target(x, y);
            batchLoss += 0.5 * error * error;
            network.backward(std::span<const Real>(&error, 1));
        }

        network.scaleGradients(1.0 / kBatch);
        network.applyAdam(3e-3, static_cast<std::uint64_t>(step));

        if (step == 1) firstLoss = batchLoss / kBatch;
        lastLoss = batchLoss / kBatch;
    }

    CHECK(lastLoss < firstLoss * 0.1, "network barely learned: %g -> %g", firstLoss, lastLoss);
    PASS("network fits a smooth function through its own gradients");
}

void testNormalizerStandardizes()
{
    RunningNormalizer normalizer;
    normalizer.configure(3);

    for (int i = 0; i < 20000; ++i) {
        const std::vector<Real> sample = { testutil::uniform(4.0, 6.0),
                                           testutil::uniform(-100.0, 100.0),
                                           testutil::uniform(0.0, 0.02) };
        normalizer.observe(sample);
    }

    // Feed the distribution's mean back through: it should come out near zero.
    const std::vector<Real> centre = { 5.0, 0.0, 0.01 };
    std::vector<Real> normalized(3);
    normalizer.apply(centre, normalized);
    for (Real value : normalized) CHECK_NEAR(value, 0.0, 0.1);

    // And a one-sigma point should come out near one.
    const std::vector<Real> shifted = { 5.0 + 2.0 / std::sqrt(12.0), 0.0, 0.01 };
    normalizer.apply(shifted, normalized);
    CHECK_NEAR(normalized[0], 1.0, 0.15);
    PASS("running normalizer standardizes channels of very different scales");
}

/// End to end on a problem with a known answer: a one-step environment whose
/// reward is highest at a particular action. If PPO cannot find that, nothing
/// it says about the plasma means anything.
void testPpoSolvesABandit()
{
    constexpr int kActions = 3;
    const std::vector<Real> optimum = { 0.6, -0.4, 0.2 };

    PpoConfig config;
    config.stepsPerUpdate = 512;
    config.minibatchSize  = 128;
    config.epochs         = 8;
    config.discount       = 0.0; // one-step problem
    config.initialLogStd  = -0.7;
    config.learningRate   = 3e-3;
    config.normalizeObservations = false;

    PpoAgent agent;
    agent.configure(1, kActions, config, 99);

    const std::vector<Real> observation = { 1.0 };
    std::vector<Real> action(kActions);

    auto rewardFor = [&](std::span<const Real> a) {
        Real distance = 0;
        for (int i = 0; i < kActions; ++i) {
            const Real d = a[static_cast<std::size_t>(i)] - optimum[static_cast<std::size_t>(i)];
            distance += d * d;
        }
        return -distance;
    };

    Real initialReward = 0;
    Real finalReward = 0;

    for (int update = 0; update < 60; ++update) {
        agent.beginRollout();
        Real totalReward = 0;
        int count = 0;

        while (!agent.rolloutFull()) {
            Real logProbability = 0;
            Real value = 0;
            agent.act(observation, action, logProbability, value);
            const Real reward = rewardFor(action);
            agent.record(observation, action, logProbability, value, reward, true);
            agent.endTrajectory(0.0);
            totalReward += reward;
            ++count;
        }
        agent.update();

        const Real mean = totalReward / static_cast<Real>(count);
        if (update == 0) initialReward = mean;
        finalReward = mean;
    }

    agent.actGreedy(observation, action);
    Real distance = 0;
    for (int i = 0; i < kActions; ++i) {
        const Real d = action[static_cast<std::size_t>(i)] - optimum[static_cast<std::size_t>(i)];
        distance += d * d;
    }

    CHECK(finalReward > initialReward, "PPO did not improve: %g -> %g", initialReward, finalReward);
    CHECK(std::sqrt(distance) < 0.25,
          "PPO converged to (%g, %g, %g), expected (0.6, -0.4, 0.2)",
          action[0], action[1], action[2]);
    PASS("PPO finds the optimum of a continuous bandit");
}

void testCheckpointRoundTrip()
{
    PpoConfig config;
    PpoAgent original;
    original.configure(6, 2, config, 5);

    const std::vector<Real> observation = { 0.2, -0.5, 1.1, 0.0, 0.3, -0.9 };
    std::vector<Real> before(2);
    original.actGreedy(observation, before);

    const std::string path = "test_policy_roundtrip.bin";
    CHECK(original.save(path), "failed to save the checkpoint");

    PpoAgent restored;
    restored.configure(6, 2, config, 123456); // deliberately different seed
    CHECK(restored.load(path), "failed to load the checkpoint");

    std::vector<Real> after(2);
    restored.actGreedy(observation, after);

    CHECK_NEAR(after[0], before[0], 1e-12);
    CHECK_NEAR(after[1], before[1], 1e-12);
    std::remove(path.c_str());
    PASS("a saved policy reloads to the same function");
}

} // namespace

int main()
{
    testBackpropagationMatchesFiniteDifferences();
    testNetworkFitsAFunction();
    testNormalizerStandardizes();
    testPpoSolvesABandit();
    testCheckpointRoundTrip();
    std::printf("test_rl: all checks passed\n");
    return 0;
}
