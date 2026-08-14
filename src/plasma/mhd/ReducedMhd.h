#pragma once

// Linear reduced-MHD evolution of a handful of helical modes.
//
// Reduced MHD (Strauss) in a periodic cylinder, linearized about the
// equilibrium and Fourier-decomposed as exp(i(m*theta - n*zeta)). For each mode
// two fields are evolved on the radial grid:
//
//   d(psi)/dt = i F phi + eta Delta* psi - i n Omega psi
//   d(U)/dt   = [ i F Delta* psi - i (m/r) J0' psi ] / rho + nu Delta* U
//               - i n Omega U + gamma_EP U
//   Delta* phi = U
//
// with F(r) = m/q(r) - n the parallel wavenumber, which vanishes on the
// rational surface q = m/n. Two instabilities live in this system:
//
//   Tearing modes. Where F = 0 inside the plasma, field lines can reconnect at
//   a rate set by the resistivity, and the equilibrium current gradient feeds
//   the perturbation. Growth is resistive, so it is slow, and the saturated
//   state is a magnetic island. Modes with a rational surface in the plasma are
//   tagged Tearing.
//
//   Alfvenic modes. Where F never vanishes, the same equations describe a shear
//   Alfven wave oscillating at omega ~ F, damped by continuum phase mixing.
//   Energetic particles from neutral beam injection can drive it unstable. That
//   drive enters as gamma_EP, a closure rather than a first-principles
//   wave-particle calculation -- the point here is a controllable instability
//   with the right frequency and the right response to beam power, not a
//   quantitative fast-ion code.
//
// Time integration splits the two timescales: RK4 on the ideal terms, whose
// eigenvalues are bounded by max|F| ~ 1, and backward Euler on the diffusive
// terms, whose explicit limit near the axis would otherwise be ~10^-3.

#include <plasma/core/Types.h>
#include <plasma/mhd/Equilibrium.h>
#include <plasma/mhd/RadialGrid.h>

namespace plasma {

enum class ModeClass
{
    Tearing,   ///< Has a rational surface in the plasma; grows resistively.
    Alfvenic,  ///< No rational surface; oscillates, driven by fast ions.
};

struct ModeSpec
{
    int        poloidal{ 2 };
    int        toroidal{ 1 };
    ModeClass  classification{ ModeClass::Tearing };

    [[nodiscard]] Real resonantSafetyFactor() const
    {
        return static_cast<Real>(poloidal) / static_cast<Real>(toroidal);
    }
};

struct MhdParameters
{
    /// Core toroidal rotation in rad per Alfven time. Rotation is what gives an
    /// island a measurable frequency at the Mirnov coils, and what lets a
    /// rotating plasma screen a static applied perturbation.
    Real coreRotation{ 0.04 };
    /// Rotation profile shape: Omega(r) = coreRotation * (1 - shear*(r/a)^2).
    /// The shear is the phase-mixing source that damps the Alfven continuum.
    Real rotationShear{ 0.85 };

    /// Fast-ion drive strength, set from the beam power actuator.
    Real energeticDrive{ 0.0 };
    Real energeticDriveRadius{ 0.45 };
    Real energeticDriveWidth{ 0.30 };
    /// Mode amplitude at which the fast-ion drive is halved. A growing Alfven
    /// mode flattens the fast-ion pressure profile that feeds it, so the drive
    /// weakens as the mode grows. Without this closure the Alfvenic branch has
    /// no saturation at all and simply overflows.
    Real energeticSaturation{ 2.0e-3 };

    /// Amplitude of the random seed perturbation at reset.
    Real seedAmplitude{ 1.0e-6 };
    /// Integration substep, in Alfven times.
    Real substep{ 0.1 };

    /// Hard ceiling on mode amplitude. Reaching it is a disruption, not a
    /// number to keep integrating.
    Real disruptionAmplitude{ 0.35 };
};

class ReducedMhd
{
public:
    void configure(const RadialGrid& grid, std::span<const ModeSpec> modes, const MhdParameters& parameters);

    /// Seeds every mode with a small random perturbation. Deterministic in `seed`.
    void reset(std::uint64_t seed);

    /// Recomputes the equilibrium-dependent coefficients: F(r), the diffusion
    /// operators, the density weighting. Call whenever the equilibrium is
    /// rebuilt -- once per control step, not per substep.
    void onEquilibriumChanged(const Equilibrium& equilibrium);

    /// Advances by `duration` Alfven times. Returns false if the solution went
    /// non-finite, which the environment treats as a failed episode.
    bool advance(Real duration);

    // --- State --------------------------------------------------------------

    [[nodiscard]] int modeCount() const noexcept { return static_cast<int>(m_modes.size()); }
    [[nodiscard]] const ModeSpec& mode(int index) const { return m_modes[static_cast<std::size_t>(index)]; }

    /// Evolved plasma response; zero at the wall by construction.
    [[nodiscard]] std::span<const Complex> flux(int index) const;
    /// Plasma response plus the vacuum field of the applied perturbation.
    [[nodiscard]] std::span<const Complex> totalFlux(int index) const;
    [[nodiscard]] std::span<const Complex> vorticity(int index) const;
    [[nodiscard]] std::span<const Complex> streamFunction(int index) const;
    /// F(r) = m/q(r) - n.
    [[nodiscard]] ConstProfile parallelWavenumber(int index) const;

    // --- Scalars per mode ---------------------------------------------------

    /// Peak |psi| over the radius, the headline amplitude of the mode.
    [[nodiscard]] Real amplitude(int index) const;
    /// Magnetic island full width, zero for modes with no rational surface.
    [[nodiscard]] Real islandWidth(int index) const;
    /// Perturbed poloidal field at the wall: what a Mirnov coil integrates.
    [[nodiscard]] Complex wallField(int index) const;
    /// Perturbation energy, the quantity whose log-derivative is the growth rate.
    [[nodiscard]] Real energy(int index) const;
    /// Radius of the rational surface, or a negative value if there is none.
    [[nodiscard]] Real rationalSurface(int index) const { return m_rationalSurface[static_cast<std::size_t>(index)]; }

    /// True once any mode exceeds the disruption amplitude.
    [[nodiscard]] bool disrupted() const noexcept { return m_disrupted; }

    /// Island regions for the quasilinear feedback into the equilibrium.
    [[nodiscard]] std::vector<IslandRegion> islands() const;

    // --- Actuator inputs ----------------------------------------------------

    /// Applied resonant magnetic perturbation for one mode, as the complex
    /// amplitude of the vacuum field at the wall. The phase matters: aligned
    /// with the island it drives, opposed it cancels.
    void setResonantPerturbation(int index, Complex amplitude);
    void setEnergeticDrive(Real value) { m_parameters.energeticDrive = value; }
    void setRotation(Real coreRotation) { m_parameters.coreRotation = coreRotation; }

    [[nodiscard]] const MhdParameters& parameters() const noexcept { return m_parameters; }
    [[nodiscard]] Real time() const noexcept { return m_time; }
    [[nodiscard]] std::uint64_t substepCount() const noexcept { return m_substeps; }

private:
    struct ModeState
    {
        std::vector<Complex> flux;
        std::vector<Complex> vorticity;
        std::vector<Complex> streamFunction;

        /// psi + vacuum RMP field, refreshed lazily for readers.
        mutable std::vector<Complex> totalFlux;
        mutable bool totalFluxValid{ false };

        std::vector<Real> parallelWavenumber;
        /// Vacuum profile (r/a)^m of the applied perturbation, unit amplitude.
        std::vector<Real> vacuumShape;
        Complex appliedPerturbation{ 0, 0 };

        LaplacianOperator laplacian;
        Tridiagonal       fluxDiffusion;
        Tridiagonal       vorticityDiffusion;

        Real rationalSurfaceRadius{ -1 };
        /// Drive after saturation, refreshed once per advance() rather than per
        /// substep: it depends on a full-profile reduction and changes slowly.
        Real saturatedDrive{ 0 };
        Real wavenumberGradientAtSurface{ 0 };

        // RK4 scratch.
        std::vector<Complex> fluxStage;
        std::vector<Complex> vorticityStage;
        std::vector<Complex> fluxAccumulator;
        std::vector<Complex> vorticityAccumulator;
        std::vector<Complex> fluxDerivative;
        std::vector<Complex> vorticityDerivative;
        std::vector<Complex> laplacianFlux;
        std::vector<Complex> laplacianVorticity;
        std::vector<Complex> workFlux;
        std::vector<Complex> workVorticity;
    };

    /// Fast-ion drive after quasilinear reduction by the mode's own amplitude.
    [[nodiscard]] Real effectiveDrive(int index) const;

    void computeDerivatives(ModeState& state, const ModeSpec& spec,
                            std::span<const Complex> flux, std::span<const Complex> vorticity,
                            std::span<Complex> fluxOut, std::span<Complex> vorticityOut) const;

    void applyImplicitDiffusion(ModeState& state);
    void refreshTotalFlux(int index) const;

    const RadialGrid* m_grid{ nullptr };
    MhdParameters     m_parameters;

    std::vector<ModeSpec>  m_modes;
    std::vector<ModeState> m_states;
    std::vector<Real>      m_rationalSurface;

    // Equilibrium-derived coefficients, shared across modes.
    std::vector<Real> m_density;
    std::vector<Real> m_currentGradient;
    std::vector<Real> m_rotation;
    std::vector<Real> m_energeticDriveShape;
    std::vector<Real> m_resistivity;
    std::vector<Real> m_viscosityProfile;

    Real          m_time{ 0 };
    std::uint64_t m_substeps{ 0 };
    bool          m_disrupted{ false };
    bool          m_diffusionValid{ false };
};

} // namespace plasma
