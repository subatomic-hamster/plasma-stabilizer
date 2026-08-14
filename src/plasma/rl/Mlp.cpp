#include <plasma/rl/Mlp.h>

#include <algorithm>
#include <cmath>
#include <random>

namespace plasma {

namespace {

Real activate(Activation kind, Real x)
{
    switch (kind) {
    case Activation::Tanh:     return std::tanh(x);
    case Activation::ReLU:     return x > 0 ? x : 0;
    case Activation::Identity: return x;
    }
    return x;
}

/// Derivative expressed in terms of the *output*, which is already cached.
Real activationGradient(Activation kind, Real output)
{
    switch (kind) {
    case Activation::Tanh:     return 1.0 - output * output;
    case Activation::ReLU:     return output > 0 ? 1.0 : 0.0;
    case Activation::Identity: return 1.0;
    }
    return 1.0;
}

} // namespace

void Mlp::configure(const MlpConfig& config, std::uint64_t seed)
{
    m_config = config;
    m_layers.clear();

    std::vector<int> widths;
    widths.push_back(config.inputs);
    for (int hidden : config.hidden) widths.push_back(hidden);
    widths.push_back(config.outputs);

    std::mt19937_64 rng(seed == 0 ? 0xC3A5C85C97CB3127ull : seed);

    for (std::size_t i = 0; i + 1 < widths.size(); ++i) {
        Layer layer;
        layer.inputs  = widths[i];
        layer.outputs = widths[i + 1];
        const bool isOutput = (i + 2 == widths.size());
        layer.activation = isOutput ? Activation::Identity : config.hiddenActivation;

        const auto weights = static_cast<std::size_t>(layer.inputs) *
                             static_cast<std::size_t>(layer.outputs);
        layer.weight.resize(weights);
        layer.bias.assign(static_cast<std::size_t>(layer.outputs), 0.0);
        layer.weightGradient.assign(weights, 0.0);
        layer.biasGradient.assign(static_cast<std::size_t>(layer.outputs), 0.0);
        layer.weightMoment1.assign(weights, 0.0);
        layer.weightMoment2.assign(weights, 0.0);
        layer.biasMoment1.assign(static_cast<std::size_t>(layer.outputs), 0.0);
        layer.biasMoment2.assign(static_cast<std::size_t>(layer.outputs), 0.0);
        layer.input.assign(static_cast<std::size_t>(layer.inputs), 0.0);
        layer.preActivation.assign(static_cast<std::size_t>(layer.outputs), 0.0);
        layer.output.assign(static_cast<std::size_t>(layer.outputs), 0.0);

        // Xavier for tanh; the output layer is deliberately much smaller so a
        // fresh policy starts near the centre of its action range.
        const Real spread = std::sqrt(6.0 / static_cast<Real>(layer.inputs + layer.outputs)) *
                            (isOutput ? config.outputScale : 1.0);
        std::uniform_real_distribution<Real> distribution(-spread, spread);
        for (Real& value : layer.weight) value = distribution(rng);

        m_layers.push_back(std::move(layer));
    }

    m_scratch.assign(static_cast<std::size_t>(*std::max_element(widths.begin(), widths.end())), 0.0);
    m_gradientScratch.assign(m_scratch.size(), 0.0);
}

void Mlp::forward(std::span<const Real> input, std::span<Real> output)
{
    std::span<const Real> current = input;

    for (Layer& layer : m_layers) {
        std::copy(current.begin(), current.begin() + layer.inputs, layer.input.begin());

        for (int o = 0; o < layer.outputs; ++o) {
            const Real* row = layer.weight.data() + static_cast<std::size_t>(o) *
                                                        static_cast<std::size_t>(layer.inputs);
            Real sum = layer.bias[static_cast<std::size_t>(o)];
            for (int i = 0; i < layer.inputs; ++i) sum += row[i] * layer.input[static_cast<std::size_t>(i)];

            layer.preActivation[static_cast<std::size_t>(o)] = sum;
            layer.output[static_cast<std::size_t>(o)] = activate(layer.activation, sum);
        }
        current = layer.output;
    }

    const std::size_t count = std::min(output.size(), current.size());
    std::copy(current.begin(), current.begin() + static_cast<std::ptrdiff_t>(count), output.begin());
}

void Mlp::predict(std::span<const Real> input, std::span<Real> output) const
{
    std::vector<Real> buffer(input.begin(), input.end());
    std::vector<Real> next;

    for (const Layer& layer : m_layers) {
        next.assign(static_cast<std::size_t>(layer.outputs), 0.0);
        for (int o = 0; o < layer.outputs; ++o) {
            const Real* row = layer.weight.data() + static_cast<std::size_t>(o) *
                                                        static_cast<std::size_t>(layer.inputs);
            Real sum = layer.bias[static_cast<std::size_t>(o)];
            for (int i = 0; i < layer.inputs; ++i) sum += row[i] * buffer[static_cast<std::size_t>(i)];
            next[static_cast<std::size_t>(o)] = activate(layer.activation, sum);
        }
        buffer.swap(next);
    }

    const std::size_t count = std::min(output.size(), buffer.size());
    std::copy(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(count), output.begin());
}

void Mlp::backward(std::span<const Real> outputGradient, std::span<Real> inputGradient)
{
    std::vector<Real> delta(outputGradient.begin(), outputGradient.end());
    std::vector<Real> previous;

    for (std::size_t index = m_layers.size(); index-- > 0;) {
        Layer& layer = m_layers[index];

        // Through the activation.
        for (int o = 0; o < layer.outputs; ++o) {
            delta[static_cast<std::size_t>(o)] *=
                activationGradient(layer.activation, layer.output[static_cast<std::size_t>(o)]);
        }

        // Parameter gradients, accumulated across the minibatch.
        for (int o = 0; o < layer.outputs; ++o) {
            const Real d = delta[static_cast<std::size_t>(o)];
            layer.biasGradient[static_cast<std::size_t>(o)] += d;

            Real* row = layer.weightGradient.data() + static_cast<std::size_t>(o) *
                                                          static_cast<std::size_t>(layer.inputs);
            for (int i = 0; i < layer.inputs; ++i) {
                row[i] += d * layer.input[static_cast<std::size_t>(i)];
            }
        }

        // Gradient with respect to this layer's input, which is the next delta.
        const bool needed = (index > 0) || !inputGradient.empty();
        if (needed) {
            previous.assign(static_cast<std::size_t>(layer.inputs), 0.0);
            for (int o = 0; o < layer.outputs; ++o) {
                const Real d = delta[static_cast<std::size_t>(o)];
                if (d == 0) continue;
                const Real* row = layer.weight.data() + static_cast<std::size_t>(o) *
                                                            static_cast<std::size_t>(layer.inputs);
                for (int i = 0; i < layer.inputs; ++i) {
                    previous[static_cast<std::size_t>(i)] += d * row[i];
                }
            }
            delta.swap(previous);
        }
    }

    if (!inputGradient.empty()) {
        const std::size_t count = std::min(inputGradient.size(), delta.size());
        std::copy(delta.begin(), delta.begin() + static_cast<std::ptrdiff_t>(count),
                  inputGradient.begin());
    }
}

void Mlp::zeroGradients()
{
    for (Layer& layer : m_layers) {
        std::fill(layer.weightGradient.begin(), layer.weightGradient.end(), 0.0);
        std::fill(layer.biasGradient.begin(), layer.biasGradient.end(), 0.0);
    }
}

void Mlp::scaleGradients(Real factor)
{
    for (Layer& layer : m_layers) {
        for (Real& value : layer.weightGradient) value *= factor;
        for (Real& value : layer.biasGradient) value *= factor;
    }
}

Real Mlp::gradientNorm() const
{
    Real total = 0;
    for (const Layer& layer : m_layers) {
        for (Real value : layer.weightGradient) total += value * value;
        for (Real value : layer.biasGradient) total += value * value;
    }
    return std::sqrt(total);
}

void Mlp::clipGradients(Real maximum)
{
    const Real norm = gradientNorm();
    if (norm <= maximum || norm <= 0) return;
    scaleGradients(maximum / norm);
}

void Mlp::applyAdam(Real learningRate, std::uint64_t step, Real beta1, Real beta2, Real epsilon)
{
    const Real correction1 = 1.0 - std::pow(beta1, static_cast<Real>(step));
    const Real correction2 = 1.0 - std::pow(beta2, static_cast<Real>(step));

    auto update = [&](std::vector<Real>& values, std::vector<Real>& gradients,
                      std::vector<Real>& moment1, std::vector<Real>& moment2) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            const Real g = gradients[i];
            moment1[i] = beta1 * moment1[i] + (1.0 - beta1) * g;
            moment2[i] = beta2 * moment2[i] + (1.0 - beta2) * g * g;

            const Real corrected1 = moment1[i] / correction1;
            const Real corrected2 = moment2[i] / correction2;
            values[i] -= learningRate * corrected1 / (std::sqrt(corrected2) + epsilon);
        }
    };

    for (Layer& layer : m_layers) {
        update(layer.weight, layer.weightGradient, layer.weightMoment1, layer.weightMoment2);
        update(layer.bias, layer.biasGradient, layer.biasMoment1, layer.biasMoment2);
    }
}

std::size_t Mlp::parameterCount() const
{
    std::size_t total = 0;
    for (const Layer& layer : m_layers) total += layer.weight.size() + layer.bias.size();
    return total;
}

void Mlp::writeParameters(std::vector<Real>& out) const
{
    out.clear();
    out.reserve(parameterCount());
    for (const Layer& layer : m_layers) {
        out.insert(out.end(), layer.weight.begin(), layer.weight.end());
        out.insert(out.end(), layer.bias.begin(), layer.bias.end());
    }
}

bool Mlp::readParameters(std::span<const Real> in)
{
    if (in.size() != parameterCount()) return false;
    std::size_t offset = 0;
    for (Layer& layer : m_layers) {
        std::copy_n(in.begin() + static_cast<std::ptrdiff_t>(offset), layer.weight.size(),
                    layer.weight.begin());
        offset += layer.weight.size();
        std::copy_n(in.begin() + static_cast<std::ptrdiff_t>(offset), layer.bias.size(),
                    layer.bias.begin());
        offset += layer.bias.size();
    }
    return true;
}

} // namespace plasma
