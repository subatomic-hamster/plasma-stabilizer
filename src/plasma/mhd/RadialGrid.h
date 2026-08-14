#pragma once

// Radial discretization, the perpendicular Laplacian, and the tridiagonal
// solves both of them need.
//
// Everything in the model is spectral in the poloidal and toroidal angles and
// finite-difference in minor radius, so a "field" is one radial profile per
// (m, n) mode. That reduces a 3D MHD problem to a handful of 1D problems, which
// is what makes thousands of control episodes affordable.
//
// The grid is cell-centred, r_i = (i + 1/2) dr, so no point sits on the
// coordinate singularity at r = 0 and the 1/r and m^2/r^2 terms never blow up.
//
// Field amplitudes are complex: a mode has a phase, and the phase is not
// cosmetic. It is what makes plasma rotation, mode locking, and the relative
// phase between an applied resonant magnetic perturbation and the island chain
// representable at all -- and that relative phase is one of the actuator
// channels the controller learns to use.

#include <plasma/core/Types.h>

#include <complex>

namespace plasma {

using Complex = std::complex<Real>;

class RadialGrid
{
public:
    RadialGrid() = default;
    RadialGrid(int points, Real minorRadius) { configure(points, minorRadius); }

    void configure(int points, Real minorRadius);

    [[nodiscard]] int size() const noexcept { return static_cast<int>(m_radius.size()); }
    [[nodiscard]] Real minorRadius() const noexcept { return m_minorRadius; }
    [[nodiscard]] Real spacing() const noexcept { return m_spacing; }

    [[nodiscard]] Real radius(int i) const noexcept { return m_radius[static_cast<std::size_t>(i)]; }
    [[nodiscard]] ConstProfile radii() const noexcept { return m_radius; }
    /// r/a, the usual normalized flux-surface label.
    [[nodiscard]] Real normalized(int i) const noexcept
    {
        return m_radius[static_cast<std::size_t>(i)] / m_minorRadius;
    }

    /// Index whose radius is nearest `r`, clamped into range.
    [[nodiscard]] int nearestIndex(Real r) const noexcept;

    /// integral(f r dr) over the plasma. Volume-weighted, so it is the right
    /// thing for stored energy, plasma current, and line averages.
    [[nodiscard]] Real integrate(ConstProfile f) const;

    /// Radial derivative by central differences, one-sided at the ends.
    void derivative(ConstProfile f, Profile out) const;

private:
    std::vector<Real> m_radius;
    Real              m_minorRadius{ 1 };
    Real              m_spacing{ 1 };
};

/// Tridiagonal matrix with real coefficients, applied to real or complex data.
///
/// The Thomas forward sweep depends only on the matrix, so it is factored once
/// and every subsequent solve is two linear passes. Both the elliptic solve for
/// the stream function and the implicit diffusion step run several times per
/// substep, so this is one of the hottest routines in the simulation.
class Tridiagonal
{
public:
    void resize(int n);

    [[nodiscard]] int size() const noexcept { return static_cast<int>(m_diagonal.size()); }

    Real& sub(int i) noexcept { return m_sub[static_cast<std::size_t>(i)]; }
    Real& diagonal(int i) noexcept { return m_diagonal[static_cast<std::size_t>(i)]; }
    Real& super(int i) noexcept { return m_super[static_cast<std::size_t>(i)]; }

    [[nodiscard]] Real sub(int i) const noexcept { return m_sub[static_cast<std::size_t>(i)]; }
    [[nodiscard]] Real diagonal(int i) const noexcept { return m_diagonal[static_cast<std::size_t>(i)]; }
    [[nodiscard]] Real super(int i) const noexcept { return m_super[static_cast<std::size_t>(i)]; }

    /// Must be called after the coefficients are set and before any solve().
    void factorize();

    /// out = M x, using the raw coefficients. Homogeneous outside the domain.
    template <typename T>
    void apply(std::span<const T> x, std::span<T> out) const
    {
        const int n = size();
        for (int i = 0; i < n; ++i) {
            const auto index = static_cast<std::size_t>(i);
            T value = x[index] * m_diagonal[index];
            if (i > 0)     value += x[index - 1] * m_sub[index];
            if (i + 1 < n) value += x[index + 1] * m_super[index];
            out[index] = value;
        }
    }

    /// Solves M out = rhs. Requires factorize().
    template <typename T>
    void solve(std::span<const T> rhs, std::span<T> out) const
    {
        const int n = size();
        if (n == 0) return;

        T previous = rhs[0] * m_pivot[0];
        out[0] = previous;
        for (int i = 1; i < n; ++i) {
            const auto index = static_cast<std::size_t>(i);
            previous = (rhs[index] - previous * m_sub[index]) * m_pivot[index];
            out[index] = previous;
        }

        for (int i = n - 2; i >= 0; --i) {
            const auto index = static_cast<std::size_t>(i);
            out[index] -= out[index + 1] * m_cPrime[index];
        }
    }

private:
    std::vector<Real> m_sub;
    std::vector<Real> m_diagonal;
    std::vector<Real> m_super;
    std::vector<Real> m_cPrime;
    std::vector<Real> m_pivot;
};

/// Delta*_m f = f'' + f'/r - m^2 f / r^2, with homogeneous Dirichlet conditions
/// at the wall and regularity at the axis.
///
/// Homogeneous is not a limitation: an externally applied resonant field is a
/// vacuum solution of Delta* f = 0, namely (r/a)^m, so it is carried separately
/// and added wherever the total perturbed flux is needed. Keeping the evolved
/// field homogeneous means every operator here is a plain tridiagonal solve
/// with no boundary bookkeeping.
class LaplacianOperator
{
public:
    void build(const RadialGrid& grid, int m);

    template <typename T>
    void apply(std::span<const T> f, std::span<T> out) const { m_matrix.apply(f, out); }

    /// Solves Delta*_m f = rhs with f(a) = 0.
    template <typename T>
    void solve(std::span<const T> rhs, std::span<T> out) const { m_matrix.solve(rhs, out); }

    /// Builds and factorizes (I - coefficient(r) * Delta*_m) into `out`, the
    /// operator a backward-Euler diffusion step inverts.
    void buildImplicitDiffusion(ConstProfile coefficient, Real dt, Tridiagonal& out) const;

    [[nodiscard]] int poloidalMode() const noexcept { return m_m; }
    [[nodiscard]] const Tridiagonal& matrix() const noexcept { return m_matrix; }

private:
    int         m_m{ 0 };
    Tridiagonal m_matrix;

    /// Unmodified stencil, kept because the implicit operator needs the
    /// coefficients before the boundary corrections were folded in.
    std::vector<Real> m_rawSub;
    std::vector<Real> m_rawDiagonal;
    std::vector<Real> m_rawSuper;
    /// Boundary correction already applied to the first and last diagonal.
    Real m_axisCorrection{ 0 };
    Real m_wallCorrection{ 0 };
};

} // namespace plasma
