#include <plasma/diagnostics/Diagnostics.h>

#include <algorithm>
#include <cmath>

namespace plasma {

namespace {

/// Coil-signal amplitude at which the inferred island width is calibrated to
/// match the simulation's own estimate. A real machine calibrates the same way,
/// against a modelled or previously measured discharge.
constexpr Real kIslandCalibration = 0.22;

} // namespace

void DiagnosticSuite::configure(const DiagnosticConfig& config, const RadialGrid& grid,
                                Real controlPeriod)
{
    m_config        = config;
    m_grid          = &grid;
    m_controlPeriod = std::max(controlPeriod, 1e-9);

    const auto coils    = static_cast<std::size_t>(std::max(1, config.mirnovCoils));
    const auto channels = static_cast<std::size_t>(std::max(1, config.eceChannels));

    m_outputs.mirnovRaw.assign(coils, 0);
    m_outputs.tearingBand.assign(coils, 0);
    m_outputs.alfvenicBand.assign(coils, 0);
    m_outputs.temperature.assign(channels, 0);

    m_tearingFilter.assign(coils, Biquad{});
    m_alfvenicFilter.assign(coils, Biquad{});
    m_tearingRms.assign(coils, RunningRms{});
    m_alfvenicRms.assign(coils, RunningRms{});

    for (std::size_t i = 0; i < coils; ++i) {
        m_tearingFilter[i].configureLowPass(config.bandSplitFrequency, m_controlPeriod);
        m_alfvenicFilter[i].configureHighPass(config.bandSplitFrequency, m_controlPeriod);
        m_tearingRms[i].configure(config.envelopeTimeConstant);
        m_alfvenicRms[i].configure(config.envelopeTimeConstant);
    }

    // The growth-rate floor is set an order of magnitude below the coil noise:
    // below that the channel carries no information and the estimator should
    // report "quiet", not a huge negative rate.
    const Real floorValue = 0.1 * std::max(config.mirnovNoise, 1e-9);
    m_tearingGrowth.configure(config.growthTimeConstant, floorValue);
    m_alfvenicGrowth.configure(config.growthTimeConstant, floorValue);
}

void DiagnosticSuite::reset(std::uint64_t seed)
{
    m_rng.seed(seed == 0 ? 0xD1B54A32D192ED03ull : seed);

    for (std::size_t i = 0; i < m_tearingFilter.size(); ++i) {
        m_tearingFilter[i].reset();
        m_alfvenicFilter[i].reset();
        m_tearingRms[i].reset();
        m_alfvenicRms[i].reset();
    }
    m_tearingGrowth.reset();
    m_alfvenicGrowth.reset();

    std::fill(m_outputs.mirnovRaw.begin(), m_outputs.mirnovRaw.end(), 0.0);
    std::fill(m_outputs.tearingBand.begin(), m_outputs.tearingBand.end(), 0.0);
    std::fill(m_outputs.alfvenicBand.begin(), m_outputs.alfvenicBand.end(), 0.0);
    std::fill(m_outputs.temperature.begin(), m_outputs.temperature.end(), 0.0);

    m_outputs.tearingAmplitude   = 0;
    m_outputs.tearingGrowthRate  = 0;
    m_outputs.alfvenicAmplitude  = 0;
    m_outputs.alfvenicGrowthRate = 0;
    m_outputs.inferredIslandWidth = 0;
}

void DiagnosticSuite::sample(const ReducedMhd& mhd, const Equilibrium& equilibrium)
{
    const auto coils = m_outputs.mirnovRaw.size();
    const Real dt    = m_controlPeriod;
    const Real invReference = 1.0 / std::max(m_config.referenceField, 1e-12);

    // --- Magnetic pickup coils ---------------------------------------------
    // Each coil sits at a poloidal angle and integrates the perturbed poloidal
    // field there. Every mode contributes, weighted by exp(i m theta): the array
    // sees a superposition, and separating it is the controller's problem.
    Real tearingSum  = 0;
    Real alfvenicSum = 0;

    for (std::size_t k = 0; k < coils; ++k) {
        const Real angle = constants::kTwoPi * static_cast<Real>(k) / static_cast<Real>(coils);

        Real field = 0;
        for (int mode = 0; mode < mhd.modeCount(); ++mode) {
            const Complex wall = mhd.wallField(mode);
            const Real phase   = static_cast<Real>(mhd.mode(mode).poloidal) * angle;
            field += (wall * std::polar(1.0, phase)).real();
        }

        field = field * invReference + m_config.mirnovNoise * m_gaussian(m_rng);
        m_outputs.mirnovRaw[k] = field;

        const Real slow = m_tearingFilter[k].update(field);
        const Real fast = m_alfvenicFilter[k].update(field);

        m_outputs.tearingBand[k]  = m_tearingRms[k].update(slow, dt);
        m_outputs.alfvenicBand[k] = m_alfvenicRms[k].update(fast, dt);

        tearingSum  += m_outputs.tearingBand[k];
        alfvenicSum += m_outputs.alfvenicBand[k];
    }

    const Real inverseCoils = 1.0 / static_cast<Real>(coils);
    m_outputs.tearingAmplitude  = tearingSum * inverseCoils;
    m_outputs.alfvenicAmplitude = alfvenicSum * inverseCoils;

    m_outputs.tearingGrowthRate  = m_tearingGrowth.update(m_outputs.tearingAmplitude, dt);
    m_outputs.alfvenicGrowthRate = m_alfvenicGrowth.update(m_outputs.alfvenicAmplitude, dt);

    // Island width inferred from the coil envelope. A magnetic island's wall
    // signal scales as the square of its width, so the inverse is a square root.
    m_outputs.inferredIslandWidth =
        kIslandCalibration * std::sqrt(std::max(m_outputs.tearingAmplitude, 0.0));

    // --- Electron cyclotron emission ---------------------------------------
    // A radial temperature profile. An island shows up here directly, as a flat
    // spot where transport has short-circuited across it.
    const ConstProfile temperature = equilibrium.temperature();
    const Real axisTemperature = std::max(temperature.front(), 1e-9);
    const auto channels = m_outputs.temperature.size();

    for (std::size_t j = 0; j < channels; ++j) {
        const Real position = static_cast<Real>(j) / static_cast<Real>(channels - 1);
        const Real value    = sampleUniform(temperature, position) / axisTemperature;
        m_outputs.temperature[j] = value + m_config.eceNoise * m_gaussian(m_rng);
    }

    // --- Global scalars -----------------------------------------------------
    m_outputs.lineDensity = m_grid->integrate(equilibrium.density()) * 2.0;
    m_outputs.confinement = equilibrium.confinementFraction();
}

std::size_t DiagnosticSuite::observationSize() const noexcept
{
    // tearing band + alfvenic band per coil, ECE channels, and six scalars.
    return m_outputs.tearingBand.size() * 2 + m_outputs.temperature.size() + 6;
}

std::size_t DiagnosticSuite::writeObservation(std::span<Real> out) const
{
    std::size_t written = 0;
    auto push = [&](Real value) {
        if (written < out.size()) out[written] = value;
        ++written;
    };

    // Band envelopes are log-compressed: they sweep several decades during an
    // episode, and a network fed the raw values would see nothing at all until
    // the mode was already large.
    auto compress = [](Real value) { return std::log10(std::max(value, 1e-8)) * 0.25 + 1.0; };

    for (Real value : m_outputs.tearingBand)  push(compress(value));
    for (Real value : m_outputs.alfvenicBand) push(compress(value));
    for (Real value : m_outputs.temperature)  push(value);

    push(compress(m_outputs.tearingAmplitude));
    push(compress(m_outputs.alfvenicAmplitude));
    // Growth rates are already small numbers; scale them into the same range.
    push(std::clamp(m_outputs.tearingGrowthRate * 20.0, -3.0, 3.0));
    push(std::clamp(m_outputs.alfvenicGrowthRate * 20.0, -3.0, 3.0));
    push(m_outputs.inferredIslandWidth * 5.0);
    push(m_outputs.confinement);

    return written;
}

} // namespace plasma
