#pragma once

// Synthetic diagnostics and the processing chain behind them.
//
// The controller never sees the mode amplitudes. It sees what a tokamak's
// control system sees: voltages from a ring of magnetic pickup coils, a radial
// temperature profile from electron cyclotron emission, a line-integrated
// density, all with noise on them. Everything it knows about the instabilities
// it has to infer from those.
//
// That distinction is the point. A policy trained on the true mode amplitude
// learns a controller that cannot be deployed; a policy trained on coil signals
// learns one that could be, and has to cope with the fact that a Mirnov array
// measures the sum of every mode at once, weighted by how close each one sits
// to the wall.
//
// The split between the two instabilities is made in frequency, exactly as it
// is on a real machine: the island rotates with the plasma at a low frequency,
// the Alfvenic mode rings near the Alfven continuum an order of magnitude
// faster, and a pair of filters separates them on the same coil.

#include <plasma/diagnostics/SignalProcessing.h>
#include <plasma/mhd/ReducedMhd.h>

#include <random>

namespace plasma {

struct DiagnosticConfig
{
    /// Magnetic pickup coils, equally spaced poloidally around the wall.
    int mirnovCoils{ 8 };
    /// Electron cyclotron emission channels, equally spaced in minor radius.
    int eceChannels{ 8 };

    /// Coil noise, as a fraction of a reference field level.
    Real mirnovNoise{ 1.5e-3 };
    /// ECE noise, as a fraction of the axis temperature.
    Real eceNoise{ 0.012 };

    /// Frequency separating the tearing band from the Alfvenic band, in
    /// radians per Alfven time.
    Real bandSplitFrequency{ 0.12 };

    /// Envelope and growth-rate averaging times, in Alfven times.
    Real envelopeTimeConstant{ 25.0 };
    Real growthTimeConstant{ 60.0 };

    /// Reference field used to normalize the coil signals into O(1) numbers.
    Real referenceField{ 1.0e-3 };
};

/// Everything the controller is allowed to know at a control step.
struct DiagnosticOutputs
{
    /// Instantaneous coil signals, normalized by the reference field.
    std::vector<Real> mirnovRaw;
    /// Per-coil envelope in the slow (tearing) band.
    std::vector<Real> tearingBand;
    /// Per-coil envelope in the fast (Alfvenic) band.
    std::vector<Real> alfvenicBand;
    /// Normalized ECE temperature profile.
    std::vector<Real> temperature;

    /// Array-averaged band amplitudes and their estimated growth rates.
    Real tearingAmplitude{ 0 };
    Real tearingGrowthRate{ 0 };
    Real alfvenicAmplitude{ 0 };
    Real alfvenicGrowthRate{ 0 };

    /// Island width inferred from the coil signal through a fixed calibration,
    /// the way a real machine estimates it. Not the simulation's true value.
    Real inferredIslandWidth{ 0 };

    /// Line-integrated density, normalized to the reference discharge.
    Real lineDensity{ 1 };
    /// Stored energy as a fraction of the reference discharge.
    Real confinement{ 1 };
};

class DiagnosticSuite
{
public:
    void configure(const DiagnosticConfig& config, const RadialGrid& grid, Real controlPeriod);

    /// Clears every filter and reseeds the noise generator.
    void reset(std::uint64_t seed);

    /// Reads the plasma and advances every filter by one control period.
    void sample(const ReducedMhd& mhd, const Equilibrium& equilibrium);

    [[nodiscard]] const DiagnosticOutputs& outputs() const noexcept { return m_outputs; }
    [[nodiscard]] const DiagnosticConfig& config() const noexcept { return m_config; }

    /// Number of values `writeObservation` emits.
    [[nodiscard]] std::size_t observationSize() const noexcept;
    /// Packs the processed diagnostics into a policy observation vector.
    /// Returns the number of values written.
    std::size_t writeObservation(std::span<Real> out) const;

private:
    DiagnosticConfig m_config;
    const RadialGrid* m_grid{ nullptr };
    Real m_controlPeriod{ 1 };

    DiagnosticOutputs m_outputs;

    /// Per-coil band splitters. Two biquads per coil, one per band.
    std::vector<Biquad> m_tearingFilter;
    std::vector<Biquad> m_alfvenicFilter;
    std::vector<RunningRms> m_tearingRms;
    std::vector<RunningRms> m_alfvenicRms;

    GrowthRateEstimator m_tearingGrowth;
    GrowthRateEstimator m_alfvenicGrowth;

    std::mt19937_64 m_rng;
    std::normal_distribution<Real> m_gaussian{ 0.0, 1.0 };
};

} // namespace plasma
