#pragma once

// A small dense network with hand-written gradients and Adam.
//
// Written out rather than pulled in because the whole model is three layers of
// at most a few thousand weights, and a dependency that brings a tensor
// library, a graph executor and a build system with it would outweigh the code
// it replaces. The second training backend (RLTools) exists precisely so this
// implementation is checked against one written by somebody else.
//
// Layout is row-major: weight[o * inputs + i] connects input i to output o.
// Activations from the last forward pass are kept because backward() needs
// them, so a Mlp instance is single-threaded by construction; parallel rollouts
// use one copy per worker.

#include <plasma/core/Types.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace plasma {

enum class Activation
{
    Tanh,
    ReLU,
    Identity,
};

struct MlpConfig
{
    int inputs{ 1 };
    std::vector<int> hidden{ 64, 64 };
    int outputs{ 1 };
    Activation hiddenActivation{ Activation::Tanh };
    /// Scale applied to the final layer's initial weights. Small values keep a
    /// fresh policy near the middle of its action range instead of saturated
    /// against a limit, which matters when the actuators are rate limited.
    Real outputScale{ 0.01 };
};

class Mlp
{
public:
    void configure(const MlpConfig& config, std::uint64_t seed);

    /// Runs the network, caching activations for a subsequent backward().
    void forward(std::span<const Real> input, std::span<Real> output);

    /// Inference only: no activation caching, so it is safe to call on a shared
    /// const network and is what the deployed controller would use.
    void predict(std::span<const Real> input, std::span<Real> output) const;

    /// Backpropagates `outputGradient`, accumulating parameter gradients.
    /// Optionally writes the gradient with respect to the input.
    void backward(std::span<const Real> outputGradient, std::span<Real> inputGradient = {});

    void zeroGradients();
    /// One Adam step. `step` is the global update counter, used for bias
    /// correction, and must start at 1.
    void applyAdam(Real learningRate, std::uint64_t step,
                   Real beta1 = 0.9, Real beta2 = 0.999, Real epsilon = 1e-8);

    /// Global L2 norm of the accumulated gradients.
    [[nodiscard]] Real gradientNorm() const;
    /// Scales all gradients so the global norm is at most `maximum`.
    void clipGradients(Real maximum);
    /// Divides all gradients by `count`, turning a sum into a mean.
    void scaleGradients(Real factor);

    [[nodiscard]] int inputSize() const noexcept { return m_config.inputs; }
    [[nodiscard]] int outputSize() const noexcept { return m_config.outputs; }
    [[nodiscard]] std::size_t parameterCount() const;

    /// Flat parameter access, for checkpointing.
    void writeParameters(std::vector<Real>& out) const;
    bool readParameters(std::span<const Real> in);

private:
    struct Layer
    {
        int inputs{ 0 };
        int outputs{ 0 };
        Activation activation{ Activation::Tanh };

        std::vector<Real> weight;
        std::vector<Real> bias;
        std::vector<Real> weightGradient;
        std::vector<Real> biasGradient;

        // Adam moments.
        std::vector<Real> weightMoment1, weightMoment2;
        std::vector<Real> biasMoment1, biasMoment2;

        // Cached from forward().
        std::vector<Real> input;
        std::vector<Real> preActivation;
        std::vector<Real> output;
    };

    MlpConfig m_config;
    std::vector<Layer> m_layers;
    std::vector<Real>  m_scratch;
    std::vector<Real>  m_gradientScratch;
};

} // namespace plasma
