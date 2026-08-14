#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

/// Aborts with file/line context when `cond` is false. Deliberately not
/// assert(): it must fire in release builds too, which is where the optimized
/// solver actually runs.
#define CHECK(cond, ...)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
            std::fprintf(stderr, "     " __VA_ARGS__);                           \
            std::fprintf(stderr, "\n");                                          \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                    \
    do {                                                                         \
        const double lhs = static_cast<double>(a);                               \
        const double rhs = static_cast<double>(b);                               \
        if (!(std::fabs(lhs - rhs) <= static_cast<double>(tol))) {               \
            std::fprintf(stderr, "FAIL %s:%d: |%g - %g| > %g\n",                 \
                         __FILE__, __LINE__, lhs, rhs, static_cast<double>(tol)); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

#define PASS(name) std::printf("ok   %s\n", name)

namespace testutil {

/// Deterministic RNG so a failure is always reproducible.
inline std::mt19937_64& rng()
{
    static std::mt19937_64 generator(20260814u);
    return generator;
}

inline double uniform(double lo, double hi)
{
    return std::uniform_real_distribution<double>(lo, hi)(rng());
}

} // namespace testutil
