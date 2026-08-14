#include <plasma/mhd/ReducedMhd.h>

#include <algorithm>
#include <cmath>
#include <random>

namespace plasma {

namespace {
constexpr Complex kI{ 0.0, 1.0 };
} // namespace

void ReducedMhd::configure(const RadialGrid& grid, std::span<const ModeSpec> modes,
                           const MhdParameters& parameters)
{
    m_grid       = &grid;
    m_parameters = parameters;
    m_modes.assign(modes.begin(), modes.end());

    const auto n = static_cast<std::size_t>(grid.size());

    m_states.clear();
    m_states.resize(m_modes.size());
    m_rationalSurface.assign(m_modes.size(), -1);

    for (std::size_t i = 0; i < m_modes.size(); ++i) {
        ModeState& state = m_states[i];
        const ModeSpec& spec = m_modes[i];

        state.flux.assign(n, Complex{});
        state.vorticity.assign(n, Complex{});
        state.streamFunction.assign(n, Complex{});
        state.totalFlux.assign(n, Complex{});
        state.parallelWavenumber.assign(n, 0);
        state.vacuumShape.assign(n, 0);

        state.fluxStage.assign(n, Complex{});
        state.vorticityStage.assign(n, Complex{});
        state.fluxAccumulator.assign(n, Complex{});
        state.vorticityAccumulator.assign(n, Complex{});
        state.fluxDerivative.assign(n, Complex{});
        state.vorticityDerivative.assign(n, Complex{});
        state.laplacianFlux.assign(n, Complex{});
        state.laplacianVorticity.assign(n, Complex{});
        state.workFlux.assign(n, Complex{});
        state.workVorticity.assign(n, Complex{});

        state.laplacian.build(grid, spec.poloidal);

        // Vacuum solution of Delta* f = 0 that is regular on axis and equals 1
        // at the wall. An applied resonant field is exactly this shape.
        for (std::size_t j = 0; j < n; ++j) {
            state.vacuumShape[j] = std::pow(grid.normalized(static_cast<int>(j)),
                                            static_cast<Real>(spec.poloidal));
        }
    }

    m_density.assign(n, 1);
    m_currentGradient.assign(n, 0);
    m_rotation.assign(n, 0);
    m_energeticDriveShape.assign(n, 0);
    m_resistivity.assign(n, 0);
    m_viscosityProfile.assign(n, 0);

    m_time      = 0;
    m_substeps  = 0;
    m_disrupted = false;
    m_diffusionValid = false;
}

void ReducedMhd::reset(std::uint64_t seed)
{
    m_time      = 0;
    m_substeps  = 0;
    m_disrupted = false;

    std::mt19937_64 rng(seed == 0 ? 0x9E3779B97F4A7C15ull : seed);
    std::uniform_real_distribution<Real> phase(0.0, constants::kTwoPi);
    std::uniform_real_distribution<Real> scale(0.5, 1.5);

    const int n = m_grid->size();

    for (std::size_t i = 0; i < m_states.size(); ++i) {
        ModeState& state = m_states[i];
        const ModeSpec& spec = m_modes[i];

        // Seed with a smooth radial shape rather than white noise: a random
        // profile is dominated by grid-scale structure that the resistivity
        // erases in a few substeps, leaving nothing to grow.
        const Complex amplitude = std::polar(m_parameters.seedAmplitude * scale(rng), phase(rng));
        for (int j = 0; j < n; ++j) {
            const Real x = m_grid->normalized(j);
            const Real shape = std::pow(x, static_cast<Real>(spec.poloidal)) * (1.0 - x);
            state.flux[static_cast<std::size_t>(j)] = amplitude * shape;
            state.vorticity[static_cast<std::size_t>(j)] = Complex{};
        }
        state.appliedPerturbation = Complex{};
        state.totalFluxValid = false;
    }
}

void ReducedMhd::onEquilibriumChanged(const Equilibrium& equilibrium)
{
    const RadialGrid& grid = *m_grid;
    const int n            = grid.size();

    const ConstProfile q       = equilibrium.safetyFactor();
    const ConstProfile density = equilibrium.density();
    const ConstProfile current = equilibrium.currentGradient();
    const ConstProfile eta     = equilibrium.resistivity();

    for (int i = 0; i < n; ++i) {
        const auto index    = static_cast<std::size_t>(i);
        m_density[index]    = std::max(density[index], 0.05);
        m_currentGradient[index] = current[index];
        m_resistivity[index]     = eta[index];
        m_viscosityProfile[index] = equilibrium.viscosity();

        const Real x = grid.normalized(i);
        m_rotation[index] = m_parameters.coreRotation *
                            (1.0 - m_parameters.rotationShear * x * x);

        const Real offset = (x - m_parameters.energeticDriveRadius) / m_parameters.energeticDriveWidth;
        m_energeticDriveShape[index] = std::exp(-offset * offset);
    }

    const Real dt = m_parameters.substep;

    for (std::size_t i = 0; i < m_states.size(); ++i) {
        ModeState& state     = m_states[i];
        const ModeSpec& spec = m_modes[i];

        for (int j = 0; j < n; ++j) {
            const auto index = static_cast<std::size_t>(j);
            state.parallelWavenumber[index] =
                static_cast<Real>(spec.poloidal) / std::max(q[index], 1e-6) -
                static_cast<Real>(spec.toroidal);
        }

        state.laplacian.buildImplicitDiffusion(m_resistivity, dt, state.fluxDiffusion);
        state.laplacian.buildImplicitDiffusion(m_viscosityProfile, dt, state.vorticityDiffusion);

        const Real surface = equilibrium.rationalSurface(spec.poloidal, spec.toroidal);
        state.rationalSurfaceRadius = surface;
        m_rationalSurface[i]        = surface;

        if (surface > 0) {
            const int index = grid.nearestIndex(surface);
            const int lo    = std::max(index - 1, 0);
            const int hi    = std::min(index + 1, n - 1);
            state.wavenumberGradientAtSurface =
                (state.parallelWavenumber[static_cast<std::size_t>(hi)] -
                 state.parallelWavenumber[static_cast<std::size_t>(lo)]) /
                (static_cast<Real>(hi - lo) * grid.spacing());
        } else {
            state.wavenumberGradientAtSurface = 0;
        }

        state.totalFluxValid = false;
    }

    m_diffusionValid = true;
}

void ReducedMhd::computeDerivatives(ModeState& state, const ModeSpec& spec,
                                    std::span<const Complex> flux, std::span<const Complex> vorticity,
                                    std::span<Complex> fluxOut, std::span<Complex> vorticityOut) const
{
    const RadialGrid& grid = *m_grid;
    const int n            = grid.size();

    // Stream function from vorticity: Delta* phi = U, with phi(a) = 0.
    state.laplacian.solve(vorticity, std::span<Complex>(state.workFlux));
    const std::span<const Complex> phi{ state.workFlux };

    state.laplacian.apply(flux, std::span<Complex>(state.laplacianFlux));
    state.laplacian.apply(vorticity, std::span<Complex>(state.laplacianVorticity));

    const Real m       = static_cast<Real>(spec.poloidal);
    const Real toroidal = static_cast<Real>(spec.toroidal);
    const Real drive = state.saturatedDrive;

    for (int i = 0; i < n; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const Real r     = grid.radius(i);

        const Real F        = state.parallelWavenumber[index];
        const Real rho      = m_density[index];
        const Real omega    = toroidal * m_rotation[index];

        // Total perturbed flux drives the current-gradient term: the applied
        // vacuum field is just as real to the plasma as its own response.
        const Complex totalFlux = flux[index] + state.appliedPerturbation * state.vacuumShape[index];

        fluxOut[index] = kI * F * phi[index] - kI * omega * flux[index];

        const Complex bending = kI * F * state.laplacianFlux[index];
        const Complex current = -kI * (m / r) * m_currentGradient[index] * totalFlux;

        vorticityOut[index] = (bending + current) / rho
                            - kI * omega * vorticity[index]
                            + drive * m_energeticDriveShape[index] * vorticity[index];
    }
}

void ReducedMhd::applyImplicitDiffusion(ModeState& state)
{
    // Backward Euler on the resistive and viscous terms, split from the ideal
    // RK4 stage above. The split is first order, but the diffusive change over
    // one substep is tiny compared with the ideal one, so the error is well
    // below the RK4 truncation error it rides on.
    state.workFlux.assign(state.flux.begin(), state.flux.end());
    state.fluxDiffusion.solve(std::span<const Complex>(state.workFlux), std::span<Complex>(state.flux));

    state.workVorticity.assign(state.vorticity.begin(), state.vorticity.end());
    state.vorticityDiffusion.solve(std::span<const Complex>(state.workVorticity),
                                   std::span<Complex>(state.vorticity));
}

Real ReducedMhd::effectiveDrive(int index) const
{
    const ModeSpec& spec = m_modes[static_cast<std::size_t>(index)];
    if (spec.classification != ModeClass::Alfvenic) return 0.0;

    const Real saturation = std::max(m_parameters.energeticSaturation, 1e-12);
    const Real ratio      = amplitude(index) / saturation;
    return m_parameters.energeticDrive / (1.0 + ratio * ratio);
}

bool ReducedMhd::advance(Real duration)
{
    if (m_grid == nullptr || m_states.empty() || !m_diffusionValid) return false;
    if (m_disrupted) return true;

    const Real dt   = m_parameters.substep;
    const int steps = std::max(1, static_cast<int>(std::lround(duration / dt)));
    const int n     = m_grid->size();

    // Divergence and saturation are re-evaluated on this cadence. Every substep
    // would cost a full sweep for no benefit; never would let an unbounded mode
    // overflow to infinity long before advance() returns.
    constexpr int kCheckInterval = 32;

    for (int step = 0; step < steps; ++step) {
        if (step % kCheckInterval == 0) {
            for (int i = 0; i < modeCount(); ++i) {
                m_states[static_cast<std::size_t>(i)].saturatedDrive = effectiveDrive(i);

                const Real peak = amplitude(i);
                if (!std::isfinite(peak)) return false;
                if (peak > m_parameters.disruptionAmplitude) {
                    m_disrupted = true;
                    return true;
                }
            }
        }

        for (std::size_t modeIndex = 0; modeIndex < m_states.size(); ++modeIndex) {
            ModeState& state     = m_states[modeIndex];
            const ModeSpec& spec = m_modes[modeIndex];

            // Classical RK4 on the ideal terms. Lower-order explicit schemes do
            // not work here: the ideal spectrum is purely imaginary (Alfven
            // oscillation) and RK2's stability region excludes the imaginary
            // axis entirely, so it grows every wave without bound.
            auto accumulate = [&](Real weight) {
                for (int i = 0; i < n; ++i) {
                    const auto index = static_cast<std::size_t>(i);
                    state.fluxAccumulator[index] += weight * state.fluxDerivative[index];
                    state.vorticityAccumulator[index] += weight * state.vorticityDerivative[index];
                }
            };
            auto stageFrom = [&](Real weight) {
                for (int i = 0; i < n; ++i) {
                    const auto index = static_cast<std::size_t>(i);
                    state.fluxStage[index] = state.flux[index] + weight * state.fluxDerivative[index];
                    state.vorticityStage[index] =
                        state.vorticity[index] + weight * state.vorticityDerivative[index];
                }
            };

            std::fill(state.fluxAccumulator.begin(), state.fluxAccumulator.end(), Complex{});
            std::fill(state.vorticityAccumulator.begin(), state.vorticityAccumulator.end(), Complex{});

            computeDerivatives(state, spec, state.flux, state.vorticity,
                               state.fluxDerivative, state.vorticityDerivative);
            accumulate(dt / 6.0);
            stageFrom(dt * 0.5);

            computeDerivatives(state, spec, state.fluxStage, state.vorticityStage,
                               state.fluxDerivative, state.vorticityDerivative);
            accumulate(dt / 3.0);
            stageFrom(dt * 0.5);

            computeDerivatives(state, spec, state.fluxStage, state.vorticityStage,
                               state.fluxDerivative, state.vorticityDerivative);
            accumulate(dt / 3.0);
            stageFrom(dt);

            computeDerivatives(state, spec, state.fluxStage, state.vorticityStage,
                               state.fluxDerivative, state.vorticityDerivative);
            accumulate(dt / 6.0);

            for (int i = 0; i < n; ++i) {
                const auto index = static_cast<std::size_t>(i);
                state.flux[index] += state.fluxAccumulator[index];
                state.vorticity[index] += state.vorticityAccumulator[index];
            }

            applyImplicitDiffusion(state);
            state.totalFluxValid = false;
        }

        m_time += dt;
        ++m_substeps;
    }

    // One divergence check per advance, not per substep: it costs a full sweep
    // and the solver cannot recover once it has gone non-finite anyway.
    for (std::size_t i = 0; i < m_states.size(); ++i) {
        for (const Complex& value : m_states[i].flux) {
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) return false;
        }
        if (amplitude(static_cast<int>(i)) > m_parameters.disruptionAmplitude) {
            m_disrupted = true;
        }
    }
    return true;
}

void ReducedMhd::refreshTotalFlux(int index) const
{
    const ModeState& state = m_states[static_cast<std::size_t>(index)];
    if (state.totalFluxValid) return;

    const std::size_t n = state.flux.size();
    for (std::size_t i = 0; i < n; ++i) {
        state.totalFlux[i] = state.flux[i] + state.appliedPerturbation * state.vacuumShape[i];
    }
    state.totalFluxValid = true;
}

std::span<const Complex> ReducedMhd::flux(int index) const
{
    return m_states[static_cast<std::size_t>(index)].flux;
}

std::span<const Complex> ReducedMhd::totalFlux(int index) const
{
    refreshTotalFlux(index);
    return m_states[static_cast<std::size_t>(index)].totalFlux;
}

std::span<const Complex> ReducedMhd::vorticity(int index) const
{
    return m_states[static_cast<std::size_t>(index)].vorticity;
}

std::span<const Complex> ReducedMhd::streamFunction(int index) const
{
    ModeState& state = const_cast<ModeState&>(m_states[static_cast<std::size_t>(index)]);
    state.laplacian.solve(std::span<const Complex>(state.vorticity),
                          std::span<Complex>(state.streamFunction));
    return state.streamFunction;
}

ConstProfile ReducedMhd::parallelWavenumber(int index) const
{
    return m_states[static_cast<std::size_t>(index)].parallelWavenumber;
}

Real ReducedMhd::amplitude(int index) const
{
    refreshTotalFlux(index);
    Real peak = 0;
    for (const Complex& value : m_states[static_cast<std::size_t>(index)].totalFlux) {
        peak = std::max(peak, std::abs(value));
    }
    return peak;
}

Real ReducedMhd::islandWidth(int index) const
{
    const ModeState& state = m_states[static_cast<std::size_t>(index)];
    const ModeSpec& spec   = m_modes[static_cast<std::size_t>(index)];

    if (state.rationalSurfaceRadius <= 0) return 0;
    const Real gradient = std::abs(state.wavenumberGradientAtSurface);
    if (gradient < 1e-9) return 0;

    refreshTotalFlux(index);
    const int i = m_grid->nearestIndex(state.rationalSurfaceRadius);
    const Real fluxAtSurface = std::abs(state.totalFlux[static_cast<std::size_t>(i)]);

    // The helical flux near a rational surface is psi* ~ (r F'/2m)(r - rs)^2,
    // so an island of half width x satisfies (rs F'/2m) x^2 = 2|psi_s|.
    const Real width = 4.0 * std::sqrt(static_cast<Real>(spec.poloidal) * fluxAtSurface /
                                       (state.rationalSurfaceRadius * gradient));
    return std::min(width, m_grid->minorRadius());
}

Complex ReducedMhd::wallField(int index) const
{
    const ModeState& state = m_states[static_cast<std::size_t>(index)];
    const ModeSpec& spec   = m_modes[static_cast<std::size_t>(index)];
    const int n            = m_grid->size();
    const Real dr          = m_grid->spacing();
    const Real a           = m_grid->minorRadius();

    // A Mirnov coil measures the perturbed poloidal field, i.e. d(psi)/dr at the
    // wall. The evolved field vanishes there, so the derivative is one-sided
    // across the last half cell; the applied vacuum field contributes m*A/a.
    const Complex plasma = -2.0 * state.flux[static_cast<std::size_t>(n - 1)] / dr;
    const Complex applied = state.appliedPerturbation * static_cast<Real>(spec.poloidal) / a;
    return plasma + applied;
}

Real ReducedMhd::energy(int index) const
{
    const ModeState& state = m_states[static_cast<std::size_t>(index)];
    const int n            = m_grid->size();

    Real total = 0;
    for (int i = 0; i < n; ++i) {
        total += std::norm(state.flux[static_cast<std::size_t>(i)]) * m_grid->radius(i);
    }
    return total * m_grid->spacing();
}

std::vector<IslandRegion> ReducedMhd::islands() const
{
    std::vector<IslandRegion> result;
    result.reserve(m_states.size());
    for (int i = 0; i < modeCount(); ++i) {
        const Real surface = m_states[static_cast<std::size_t>(i)].rationalSurfaceRadius;
        if (surface <= 0) continue;
        const Real width = islandWidth(i);
        if (width <= 0) continue;
        result.push_back(IslandRegion{ surface, width });
    }
    return result;
}

void ReducedMhd::setResonantPerturbation(int index, Complex amplitude)
{
    ModeState& state = m_states[static_cast<std::size_t>(index)];
    state.appliedPerturbation = amplitude;
    state.totalFluxValid = false;
}

} // namespace plasma
