#include <plasma/render/TokamakScene.h>

#include <psim/core/Clock.h>

#include <algorithm>
#include <cmath>
#include <random>

namespace plasma {

TokamakScene::TokamakScene() : m_env(std::make_unique<TokamakEnv>()) {}
TokamakScene::~TokamakScene() = default;

void TokamakScene::configure(const TokamakEnvConfig& environment, const SceneConfig& scene)
{
    m_scene = scene;
    m_env->configure(environment);

    m_action.assign(TokamakEnv::actionSize(), 0.0);
    m_observation.assign(m_env->observationSize(), 0.0);
    if (m_fixedAction.empty()) {
        // Current drive on the rational surface with the beam below the
        // Alfvenic threshold: the reference the policy is compared against.
        m_fixedAction = { 1.0, 0.335, -0.24, -1.0, 0.0, -0.385 };
    }

    rebuildSamples();
    reset(m_seed);
}

void TokamakScene::setFixedAction(std::span<const Real> action)
{
    m_fixedAction.assign(action.begin(), action.end());
    m_fixedAction.resize(TokamakEnv::actionSize(), 0.0);
}

void TokamakScene::rebuildSamples()
{
    const int radial = std::max(8, m_scene.radialSamples);
    const int poloidal = std::max(8, m_scene.poloidalSamples);
    const int tracers = std::max(0, m_scene.tracerCount);

    const std::size_t sampleCount = static_cast<std::size_t>(radial) *
                                    static_cast<std::size_t>(poloidal);
    m_tracerBegin = static_cast<int>(sampleCount);

    m_particles.clear();
    m_particles.reserve(sampleCount + static_cast<std::size_t>(tracers));

    const Real minor = m_env->grid().minorRadius();

    for (int i = 0; i < radial; ++i) {
        // Skip the axis itself: the perturbation vanishes there and the polar
        // sampling would pile every poloidal point on one spot.
        const Real r = minor * (static_cast<Real>(i) + 0.5) / static_cast<Real>(radial);
        for (int j = 0; j < poloidal; ++j) {
            const Real theta = constants::kTwoPi * static_cast<Real>(j) /
                               static_cast<Real>(poloidal);
            m_particles.add(psim::Vec3{ static_cast<psim::Real>(r * std::cos(theta)),
                                        static_cast<psim::Real>(r * std::sin(theta)),
                                        0.0f });
        }
    }

    std::mt19937_64 rng(20260814u);
    std::uniform_real_distribution<Real> radiusDistribution(0.05, 0.98);
    std::uniform_real_distribution<Real> angleDistribution(0.0, constants::kTwoPi);
    for (int i = 0; i < tracers; ++i) {
        const Real r = minor * std::sqrt(radiusDistribution(rng)); // uniform in area
        const Real theta = angleDistribution(rng);
        m_particles.add(psim::Vec3{ static_cast<psim::Real>(r * std::cos(theta)),
                                    static_cast<psim::Real>(r * std::sin(theta)),
                                    0.0f });
    }

    m_scalars.assign(m_particles.size(), 0.0f);
    m_equilibriumHelicalFlux.assign(static_cast<std::size_t>(m_env->grid().size()), 0.0);

    // Sized so the cross-section reads as a filled disc rather than dots.
    m_particleRadius = static_cast<psim::Real>(1.6 * minor / static_cast<Real>(radial));
}

void TokamakScene::reset(std::uint64_t seed)
{
    m_seed = seed == 0 ? 1 : seed;
    m_env->reset(m_seed);
    m_finished = false;
    m_lastReward = 0;
    std::fill(m_action.begin(), m_action.end(), 0.0);
    updateFlux();
}

void TokamakScene::updateFlux()
{
    const ReducedMhd& mhd = m_env->mhd();
    const RadialGrid& grid = m_env->grid();
    const int mode = m_env->primaryTearingModeIndex();
    const ModeSpec& spec = mhd.mode(mode);

    const int n = grid.size();
    const Real dr = grid.spacing();
    const Real poloidalNumber = static_cast<Real>(spec.poloidal);

    // Equilibrium helical flux: d(psi*)/dr = (r/m) * F(r), which vanishes at the
    // rational surface. That stationary point is where the island opens.
    const ConstProfile wavenumber = mhd.parallelWavenumber(mode);
    Real accumulated = 0;
    for (int i = 0; i < n; ++i) {
        const Real r = grid.radius(i);
        accumulated += (r / poloidalNumber) * wavenumber[static_cast<std::size_t>(i)] * dr;
        m_equilibriumHelicalFlux[static_cast<std::size_t>(i)] = accumulated;
    }

    const std::span<const Complex> perturbation = mhd.totalFlux(mode);

    // Scale the perturbation so the island is legible even when it is small.
    // Without it a 1% island is invisible next to the equilibrium flux, which
    // defeats the point of watching the controller work.
    Real peak = 0;
    for (const Complex& value : perturbation) peak = std::max(peak, std::abs(value));

    Real equilibriumSpan = 0;
    for (Real value : m_equilibriumHelicalFlux) {
        equilibriumSpan = std::max(equilibriumSpan, std::abs(value));
    }
    const Real visibility = (peak > 1e-14)
                                ? std::min(1.0, 0.35 * equilibriumSpan / peak)
                                : 1.0;

    const int radial = m_scene.radialSamples;
    const int poloidal = m_scene.poloidalSamples;
    const Real minor = grid.minorRadius();

    Real lowest = 1e30;
    Real highest = -1e30;

    for (int i = 0; i < radial; ++i) {
        const Real r = minor * (static_cast<Real>(i) + 0.5) / static_cast<Real>(radial);
        const int gridIndex = grid.nearestIndex(r);

        const Real background = m_equilibriumHelicalFlux[static_cast<std::size_t>(gridIndex)];
        const Complex mode1 = perturbation[static_cast<std::size_t>(gridIndex)] * visibility;

        for (int j = 0; j < poloidal; ++j) {
            const Real theta = constants::kTwoPi * static_cast<Real>(j) /
                               static_cast<Real>(poloidal);
            // Re[psi_m e^{i m theta}] is the physical perturbation on this plane.
            const Real value = background +
                               (mode1 * std::polar(1.0, poloidalNumber * theta)).real();

            const std::size_t index = static_cast<std::size_t>(i) *
                                          static_cast<std::size_t>(poloidal) +
                                      static_cast<std::size_t>(j);
            m_scalars[index] = static_cast<psim::Real>(value);
            lowest = std::min(lowest, value);
            highest = std::max(highest, value);
        }
    }

    // Tracers are coloured by their own local flux so they blend into the map.
    for (std::size_t i = static_cast<std::size_t>(m_tracerBegin); i < m_scalars.size(); ++i) {
        const psim::Vec3 position = m_particles.positions()[i];
        const Real px = static_cast<Real>(position.x);
        const Real py = static_cast<Real>(position.y);
        const Real r = std::sqrt(px * px + py * py);
        const Real theta = std::atan2(py, px);
        const int gridIndex = grid.nearestIndex(std::min(r, minor * 0.999));

        const Real value = m_equilibriumHelicalFlux[static_cast<std::size_t>(gridIndex)] +
                           (perturbation[static_cast<std::size_t>(gridIndex)] * visibility *
                            std::polar(1.0, poloidalNumber * theta)).real();
        m_scalars[i] = static_cast<psim::Real>(value);
    }

    if (highest > lowest) {
        m_scalarRange = psim::Vec2{ static_cast<psim::Real>(lowest),
                                    static_cast<psim::Real>(highest) };
    }
}

void TokamakScene::advanceTracers(Real dt)
{
    if (m_tracerBegin >= static_cast<int>(m_particles.size())) return;

    const ReducedMhd& mhd = m_env->mhd();
    const RadialGrid& grid = m_env->grid();
    const int mode = m_env->primaryTearingModeIndex();
    const Real poloidalNumber = static_cast<Real>(mhd.mode(mode).poloidal);
    const Real minor = grid.minorRadius();

    const std::span<const Complex> stream = mhd.streamFunction(mode);

    auto positions = m_particles.positions();
    const Real rotation = m_env->mhd().parameters().coreRotation;

    for (std::size_t i = static_cast<std::size_t>(m_tracerBegin); i < positions.size(); ++i) {
        const Real x = static_cast<Real>(positions[i].x);
        const Real y = static_cast<Real>(positions[i].y);
        Real r = std::sqrt(x * x + y * y);
        Real theta = std::atan2(y, x);
        if (r < 1e-4) continue;

        const int gridIndex = grid.nearestIndex(std::min(r, minor * 0.999));
        const Complex phi = stream[static_cast<std::size_t>(gridIndex)];

        // Perpendicular flow from the stream function: v_r = (1/r) d(phi)/d(theta),
        // v_theta = -d(phi)/dr. Only the radial part is worth the derivative --
        // the poloidal part is dominated by the equilibrium rotation.
        const Complex phase = std::polar(1.0, poloidalNumber * theta);
        Real radialVelocity = (poloidalNumber / r) * (phi * phase * Complex{ 0, 1 }).real();

        // Tracers are a visualization, not a measurement: cap their excursion so
        // a large stream function cannot fling them across the plasma in one
        // step and draw radial streaks over the island structure.
        const Real maximumStep = 0.02 * minor;
        radialVelocity = std::clamp(radialVelocity * dt, -maximumStep, maximumStep) / dt;

        r += radialVelocity * dt;
        theta += rotation * dt;

        // Keep tracers inside the plasma; a lost tracer is a dead pixel.
        r = std::clamp(r, 0.03 * minor, 0.985 * minor);

        positions[i] = psim::Vec3{ static_cast<psim::Real>(r * std::cos(theta)),
                                   static_cast<psim::Real>(r * std::sin(theta)), 0.0f };
    }
}

void TokamakScene::chooseAction()
{
    psim::Clock clock;

    switch (m_controller) {
    case SceneController::Policy:
        if (m_policy != nullptr) {
            m_env->observe(m_observation);
            m_policy->actGreedy(m_observation, m_action);
        } else {
            std::copy(m_fixedAction.begin(), m_fixedAction.end(), m_action.begin());
        }
        break;
    case SceneController::Fixed:
        std::copy(m_fixedAction.begin(), m_fixedAction.end(), m_action.begin());
        break;
    case SceneController::Passive:
        m_action.assign(TokamakEnv::actionSize(), 0.0);
        m_action[static_cast<std::size_t>(ActuatorChannel::CurrentDrivePower)] = -1.0;
        m_action[static_cast<std::size_t>(ActuatorChannel::ResonantAmplitude)] = -1.0;
        break;
    }

    m_decisionMicroseconds = clock.elapsed() * 1.0e6;
}

void TokamakScene::step(psim::Real /*dt*/)
{
    if (m_finished) return;

    for (int i = 0; i < std::max(1, m_scene.stepsPerFrame); ++i) {
        chooseAction();
        const StepResult result = m_env->step(m_action);
        m_lastReward = result.reward;

        advanceTracers(m_env->config().controlPeriod);

        if (result.terminated || result.truncated) {
            m_finished = true;
            break;
        }
    }

    updateFlux();
    advanceClock(static_cast<psim::Real>(m_env->config().controlPeriod));
}

psim::AABB TokamakScene::bounds() const
{
    const auto minor = static_cast<psim::Real>(m_env->grid().minorRadius());
    return psim::AABB(psim::Vec3{ -minor, -minor, -minor * 0.15f },
                      psim::Vec3{ minor, minor, minor * 0.15f });
}

psim::SimulationStats TokamakScene::stats() const
{
    psim::SimulationStats stats;
    stats.particleCount = m_particles.size();
    stats.solverResidual = static_cast<psim::Real>(m_env->widestIsland());
    stats.totalKineticEnergy = static_cast<psim::Real>(m_env->equilibrium().confinementFraction());
    stats.lastStepMs = m_decisionMicroseconds * 1.0e-3;
    return stats;
}

std::vector<psim::SimulationParameter> TokamakScene::parameters()
{
    // Nothing directly tunable: the actuators are the policy's business, and
    // exposing them here would let a slider fight the controller.
    return {};
}

} // namespace plasma
