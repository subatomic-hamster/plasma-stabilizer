#include <plasma/mhd/RadialGrid.h>

#include <algorithm>
#include <cmath>

namespace plasma {

void RadialGrid::configure(int points, Real minorRadius)
{
    const int n   = std::max(4, points);
    m_minorRadius = minorRadius > 0 ? minorRadius : 1;
    m_spacing     = m_minorRadius / static_cast<Real>(n);

    m_radius.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        m_radius[static_cast<std::size_t>(i)] = (static_cast<Real>(i) + 0.5) * m_spacing;
    }
}

int RadialGrid::nearestIndex(Real r) const noexcept
{
    if (m_radius.empty()) return -1;
    const int i = static_cast<int>(std::lround(r / m_spacing - 0.5));
    return std::clamp(i, 0, size() - 1);
}

Real RadialGrid::integrate(ConstProfile f) const
{
    // Midpoint rule; the cell-centred grid makes it second-order with no
    // endpoint special cases.
    Real total  = 0;
    const int n = std::min<int>(size(), static_cast<int>(f.size()));
    for (int i = 0; i < n; ++i) {
        total += f[static_cast<std::size_t>(i)] * m_radius[static_cast<std::size_t>(i)];
    }
    return total * m_spacing;
}

void RadialGrid::derivative(ConstProfile f, Profile out) const
{
    const int n = size();
    if (n < 2 || static_cast<int>(f.size()) < n || static_cast<int>(out.size()) < n) return;

    const Real inv2dr = 1.0 / (2.0 * m_spacing);
    for (int i = 1; i < n - 1; ++i) {
        out[static_cast<std::size_t>(i)] =
            (f[static_cast<std::size_t>(i + 1)] - f[static_cast<std::size_t>(i - 1)]) * inv2dr;
    }
    out[0] = (f[1] - f[0]) / m_spacing;
    out[static_cast<std::size_t>(n - 1)] =
        (f[static_cast<std::size_t>(n - 1)] - f[static_cast<std::size_t>(n - 2)]) / m_spacing;
}

// ---------------------------------------------------------------------------
// Tridiagonal
// ---------------------------------------------------------------------------

void Tridiagonal::resize(int n)
{
    const auto size = static_cast<std::size_t>(std::max(0, n));
    m_sub.assign(size, 0);
    m_diagonal.assign(size, 0);
    m_super.assign(size, 0);
    m_cPrime.assign(size, 0);
    m_pivot.assign(size, 0);
}

void Tridiagonal::factorize()
{
    const int n = size();
    if (n == 0) return;

    m_pivot[0]  = 1.0 / m_diagonal[0];
    m_cPrime[0] = m_super[0] * m_pivot[0];

    for (int i = 1; i < n; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const Real denominator = m_diagonal[index] - m_sub[index] * m_cPrime[index - 1];
        m_pivot[index]  = 1.0 / denominator;
        m_cPrime[index] = m_super[index] * m_pivot[index];
    }
}

// ---------------------------------------------------------------------------
// LaplacianOperator
// ---------------------------------------------------------------------------

void LaplacianOperator::build(const RadialGrid& grid, int m)
{
    m_m = m;

    const int n       = grid.size();
    const Real dr     = grid.spacing();
    const Real invDr2 = 1.0 / (dr * dr);

    m_rawSub.assign(static_cast<std::size_t>(n), 0);
    m_rawDiagonal.assign(static_cast<std::size_t>(n), 0);
    m_rawSuper.assign(static_cast<std::size_t>(n), 0);

    // Conservative (flux) form: (1/r) d/dr (r df/dr) discretized as the
    // difference of fluxes through the cell faces at r +/- dr/2.
    //
    // The naive form f'' + f'/r looks equivalent and is not: its truncation
    // error carries a 1/r, so near the axis it fails to annihilate the vacuum
    // solution (r/a)^m by ~dr, which injects a spurious current exactly where
    // the fields should be smallest. The flux form cancels that term
    // identically for m = 1 and m = 2, and it is self-adjoint, which the energy
    // bookkeeping depends on.
    for (int i = 0; i < n; ++i) {
        const auto index   = static_cast<std::size_t>(i);
        const Real r       = grid.radius(i);
        const Real invR    = 1.0 / r;
        const Real faceIn  = r - 0.5 * dr;   // exactly 0 at i = 0
        const Real faceOut = r + 0.5 * dr;

        m_rawSub[index]      = faceIn * invR * invDr2;
        m_rawSuper[index]    = faceOut * invR * invDr2;
        m_rawDiagonal[index] = -(faceIn + faceOut) * invR * invDr2 -
                               static_cast<Real>(m) * static_cast<Real>(m) * invR * invR;
    }

    // Axis: the inner face of the first cell has zero area, so the flux through
    // it is zero with no ghost point and no special case. That is the discrete
    // statement of regularity.
    m_axisCorrection = 0;
    // Wall: r = a is exactly the outer face of the last cell, and f(a) = 0, so
    // the ghost value is -f_{n-1}.
    m_wallCorrection = -m_rawSuper[static_cast<std::size_t>(n - 1)];

    m_matrix.resize(n);
    for (int i = 0; i < n; ++i) {
        m_matrix.sub(i)      = (i == 0) ? 0.0 : m_rawSub[static_cast<std::size_t>(i)];
        m_matrix.super(i)    = (i == n - 1) ? 0.0 : m_rawSuper[static_cast<std::size_t>(i)];
        m_matrix.diagonal(i) = m_rawDiagonal[static_cast<std::size_t>(i)];
    }
    m_matrix.diagonal(n - 1) += m_wallCorrection;
    m_matrix.factorize();
}

void LaplacianOperator::buildImplicitDiffusion(ConstProfile coefficient, Real dt, Tridiagonal& out) const
{
    const int n = static_cast<int>(m_rawDiagonal.size());
    out.resize(n);

    // I - dt * D(r) * Delta*, with the same boundary corrections. Backward
    // Euler on the diffusive terms is what removes the crippling explicit
    // timestep limit: near the axis m^2/r^2 reaches ~10^6, which would cap dt
    // at ~10^-3 Alfven times and make an episode unaffordable.
    for (int i = 0; i < n; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const Real scale = -dt * coefficient[index];

        out.sub(i)      = (i == 0) ? 0.0 : scale * m_rawSub[index];
        out.super(i)    = (i == n - 1) ? 0.0 : scale * m_rawSuper[index];
        out.diagonal(i) = 1.0 + scale * m_rawDiagonal[index];
    }
    out.diagonal(0)     += -dt * coefficient[0] * m_axisCorrection;
    out.diagonal(n - 1) += -dt * coefficient[static_cast<std::size_t>(n - 1)] * m_wallCorrection;
    out.factorize();
}

} // namespace plasma
