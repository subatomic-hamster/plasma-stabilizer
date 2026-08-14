// The Laplacian operator is the foundation everything else stands on: the
// stream function, the perturbed current, and the implicit diffusion step all
// go through it. If it is wrong the mode physics is wrong in ways that look
// plausible, so it is checked against analytic solutions rather than itself.

#include "TestUtil.h"

#include <plasma/mhd/RadialGrid.h>

#include <vector>

using namespace plasma;

namespace {

void testGridLayout()
{
    RadialGrid grid(64, 1.0);
    CHECK(grid.size() == 64, "unexpected grid size %d", grid.size());
    CHECK_NEAR(grid.spacing(), 1.0 / 64.0, 1e-15);

    // Cell centred: no point on the axis, none on the wall.
    CHECK(grid.radius(0) > 0, "first point sits on the coordinate singularity");
    CHECK(grid.radius(63) < grid.minorRadius(), "last point sits on the wall");
    CHECK_NEAR(grid.radius(0), 0.5 / 64.0, 1e-15);

    // integral(r dr) from 0 to 1 = 1/2.
    std::vector<Real> ones(64, 1.0);
    CHECK_NEAR(grid.integrate(ones), 0.5, 1e-4);
    PASS("cell-centred grid avoids the axis and integrates correctly");
}

void testDerivative()
{
    RadialGrid grid(256, 1.0);
    std::vector<Real> f(256), df(256);
    for (int i = 0; i < 256; ++i) f[static_cast<std::size_t>(i)] = grid.radius(i) * grid.radius(i);

    grid.derivative(f, df);
    // d(r^2)/dr = 2r; skip the one-sided ends.
    for (int i = 2; i < 254; ++i) {
        CHECK_NEAR(df[static_cast<std::size_t>(i)], 2.0 * grid.radius(i), 1e-10);
    }
    PASS("central-difference derivative is exact on a quadratic");
}

/// Delta*_m (r/a)^m = 0 analytically. This is the vacuum solution the applied
/// resonant perturbation rides on, so it had better be in the operator's null
/// space to the level the discretization allows.
///
/// For m = 1 and 2 the conservative stencil annihilates it exactly -- worth
/// asserting to machine precision, because it pins the flux form rather than
/// merely a consistent one. For higher m a residual survives in the innermost
/// cells, where r ~ dr makes the O(dr^2 / r) truncation term O(dr); no
/// second-order scheme avoids that, so the meaningful claim there is
/// convergence, not exactness.
Real worstNullSpaceResidual(int points, int m, int skipInnerCells)
{
    RadialGrid grid(points, 1.0);
    LaplacianOperator laplacian;
    laplacian.build(grid, m);

    std::vector<Real> f(static_cast<std::size_t>(points)), out(static_cast<std::size_t>(points));
    for (int i = 0; i < points; ++i) {
        f[static_cast<std::size_t>(i)] = std::pow(grid.normalized(i), static_cast<Real>(m));
    }
    laplacian.apply(std::span<const Real>(f), std::span<Real>(out));

    // The wall row is excluded: there f(a) = 1, and the operator is built for
    // homogeneous conditions, so it deliberately disagrees.
    Real worst = 0;
    for (int i = skipInnerCells; i < points - 2; ++i) {
        worst = std::max(worst, std::abs(out[static_cast<std::size_t>(i)]));
    }
    return worst;
}

void testVacuumSolutionIsAnnihilated()
{
    // Scaled by dr^2 because the operator's own entries are O(1/dr^2): an
    // exact cancellation between terms of that size still leaves a rounding
    // residual of eps/dr^2, and calling that a failure would be measuring
    // floating point, not the stencil.
    const int points   = 512;
    const Real spacing = 1.0 / static_cast<Real>(points);
    for (int m : { 1, 2 }) {
        const Real scaled = worstNullSpaceResidual(points, m, 0) * spacing * spacing;
        CHECK(scaled < 1e-14, "m=%d: flux form should be exact, scaled residual was %g", m, scaled);
    }
    PASS("(r/a)^m is annihilated to machine precision for m = 1, 2");

    // Second order away from the axis: quadrupling the resolution should cut
    // the residual by about sixteen.
    const Real coarse = worstNullSpaceResidual(128, 3, 16);
    const Real fine   = worstNullSpaceResidual(512, 3, 64);
    CHECK(fine < coarse / 8.0, "m=3 residual only fell from %g to %g under 4x refinement",
          coarse, fine);
    PASS("m = 3 residual converges at second order away from the axis");
}

/// apply and solve must be inverses. This catches boundary-condition mismatches
/// between the two, which is the failure mode that silently changes the mode
/// structure without ever producing a NaN.
void testApplyAndSolveAreInverse()
{
    RadialGrid grid(128, 1.0);

    for (int m : { 1, 2, 3, 4 }) {
        LaplacianOperator laplacian;
        laplacian.build(grid, m);

        // A field that respects the boundary conditions: zero at axis and wall.
        std::vector<Real> f(128), lap(128), recovered(128);
        for (int i = 0; i < 128; ++i) {
            const Real x = grid.normalized(i);
            f[static_cast<std::size_t>(i)] = std::pow(x, static_cast<Real>(m)) * (1.0 - x);
        }

        laplacian.apply(std::span<const Real>(f), std::span<Real>(lap));
        laplacian.solve(std::span<const Real>(lap), std::span<Real>(recovered));

        for (int i = 0; i < 128; ++i) {
            CHECK_NEAR(recovered[static_cast<std::size_t>(i)], f[static_cast<std::size_t>(i)], 1e-9);
        }
    }
    PASS("apply and solve round-trip exactly");
}

void testComplexAndRealAgree()
{
    RadialGrid grid(64, 1.0);
    LaplacianOperator laplacian;
    laplacian.build(grid, 2);

    std::vector<Real> realInput(64), realOutput(64);
    std::vector<Complex> complexInput(64), complexOutput(64);
    for (int i = 0; i < 64; ++i) {
        const Real value = testutil::uniform(-1.0, 1.0);
        realInput[static_cast<std::size_t>(i)]    = value;
        complexInput[static_cast<std::size_t>(i)] = Complex{ value, -2.0 * value };
    }

    laplacian.solve(std::span<const Real>(realInput), std::span<Real>(realOutput));
    laplacian.solve(std::span<const Complex>(complexInput), std::span<Complex>(complexOutput));

    for (int i = 0; i < 64; ++i) {
        CHECK_NEAR(complexOutput[static_cast<std::size_t>(i)].real(),
                   realOutput[static_cast<std::size_t>(i)], 1e-12);
        CHECK_NEAR(complexOutput[static_cast<std::size_t>(i)].imag(),
                   -2.0 * realOutput[static_cast<std::size_t>(i)], 1e-12);
    }
    PASS("complex solve matches the real one component-wise");
}

/// The implicit diffusion operator must reduce to the identity as dt -> 0 and
/// must actually damp a perturbation for finite dt.
void testImplicitDiffusion()
{
    RadialGrid grid(128, 1.0);
    LaplacianOperator laplacian;
    laplacian.build(grid, 2);

    std::vector<Real> coefficient(128, 1e-3);
    std::vector<Real> field(128), out(128);
    for (int i = 0; i < 128; ++i) {
        const Real x = grid.normalized(i);
        field[static_cast<std::size_t>(i)] = x * x * (1.0 - x);
    }

    Tridiagonal identityLike;
    laplacian.buildImplicitDiffusion(coefficient, 0.0, identityLike);
    identityLike.solve(std::span<const Real>(field), std::span<Real>(out));
    for (int i = 0; i < 128; ++i) {
        CHECK_NEAR(out[static_cast<std::size_t>(i)], field[static_cast<std::size_t>(i)], 1e-12);
    }

    Tridiagonal diffusing;
    laplacian.buildImplicitDiffusion(coefficient, 10.0, diffusing);

    Real before = 0;
    for (Real v : field) before += v * v;

    std::vector<Real> current = field;
    for (int step = 0; step < 20; ++step) {
        diffusing.solve(std::span<const Real>(current), std::span<Real>(out));
        current = out;
    }

    Real after = 0;
    for (Real v : current) after += v * v;

    CHECK(after < before * 0.5, "diffusion barely damped: %g -> %g", before, after);
    CHECK(after >= 0.0, "diffusion produced a non-finite result");
    PASS("implicit diffusion is the identity at dt=0 and damps for dt>0");
}

} // namespace

int main()
{
    testGridLayout();
    testDerivative();
    testVacuumSolutionIsAnnihilated();
    testApplyAndSolveAreInverse();
    testComplexAndRealAgree();
    testImplicitDiffusion();
    std::printf("test_radial_grid: all checks passed\n");
    return 0;
}
