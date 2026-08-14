#pragma once

// Visualization adapter: presents a running control episode as something the
// particle-sim renderer already knows how to draw.
//
// The engine's renderer asks a simulation for particle positions, an optional
// scalar field, and optional topology. It never learns what it is drawing. So
// the plasma is presented as a polar sampling of the poloidal cross-section --
// one point per sample -- coloured by the helical flux of the tearing mode.
//
// Helical flux is the right quantity to colour by because its contours *are*
// the field lines of the perturbed helical field. Where the mode is small the
// contours are nested circles; where an island forms they reconnect into the
// characteristic chain, and the island grows and shrinks on screen exactly as
// the controller acts on it. Plotting the perturbation amplitude instead would
// show a smooth blob and hide the topology change that matters.
//
// The scene also carries a set of drifting tracer particles advected by the
// perturbed flow, which makes the rotation and the flow pattern around the
// island visible in a way a static field plot cannot.

#include <plasma/control/TokamakEnv.h>
#include <plasma/rl/Ppo.h>

#include <psim/sim/Simulation.h>

#include <memory>

namespace plasma {

struct SceneConfig
{
    /// Radial and poloidal sampling of the cross-section.
    int radialSamples{ 120 };
    int poloidalSamples{ 256 };
    /// Tracer particles advected by the perturbed flow.
    int tracerCount{ 4000 };

    /// Control periods advanced per rendered frame. Above one the episode plays
    /// faster than real time, which is usually what you want when a pulse lasts
    /// 400 steps.
    int stepsPerFrame{ 1 };
};

/// How the episode is being driven.
enum class SceneController
{
    /// The trained policy.
    Policy,
    /// A fixed action; the reference the policy is measured against.
    Fixed,
    /// No current drive at all, so the instability runs unopposed.
    Passive,
};

class TokamakScene final : public psim::Simulation
{
public:
    TokamakScene();
    ~TokamakScene() override;

    void configure(const TokamakEnvConfig& environment, const SceneConfig& scene);

    /// Attaches a trained policy. Ownership stays with the caller.
    void setPolicy(const PpoAgent* policy) { m_policy = policy; }
    void setController(SceneController controller) { m_controller = controller; }
    [[nodiscard]] SceneController controller() const noexcept { return m_controller; }

    void setFixedAction(std::span<const Real> action);

    // --- psim::Simulation ---------------------------------------------------

    [[nodiscard]] const char* name() const override { return "tokamak"; }

    void reset(std::uint64_t seed = 0) override;
    void step(psim::Real dt) override;

    [[nodiscard]] const psim::ParticleSystem& particles() const override { return m_particles; }
    [[nodiscard]] psim::ParticleSystem& particles() override { return m_particles; }
    [[nodiscard]] psim::Real particleRadius() const override { return m_particleRadius; }
    [[nodiscard]] psim::AABB bounds() const override;

    [[nodiscard]] std::span<const psim::Real> scalarField() const override { return m_scalars; }
    [[nodiscard]] const char* scalarFieldName() const override { return "helical flux"; }
    [[nodiscard]] psim::Vec2 scalarFieldRange() const override { return m_scalarRange; }

    [[nodiscard]] psim::SimulationStats stats() const override;
    [[nodiscard]] std::vector<psim::SimulationParameter> parameters() override;

    // --- Introspection for the viewer's overlay -----------------------------

    [[nodiscard]] const TokamakEnv& environment() const noexcept { return *m_env; }
    [[nodiscard]] TokamakEnv& environment() noexcept { return *m_env; }
    [[nodiscard]] bool episodeFinished() const noexcept { return m_finished; }
    [[nodiscard]] Real lastReward() const noexcept { return m_lastReward; }
    /// Wall-clock microseconds the controller took to produce its last action.
    [[nodiscard]] double lastDecisionMicroseconds() const noexcept { return m_decisionMicroseconds; }

private:
    void rebuildSamples();
    void updateFlux();
    void advanceTracers(Real dt);
    void chooseAction();

    std::unique_ptr<TokamakEnv> m_env;
    SceneConfig m_scene;

    const PpoAgent*   m_policy{ nullptr };
    SceneController   m_controller{ SceneController::Policy };
    std::vector<Real> m_action;
    std::vector<Real> m_fixedAction;
    std::vector<Real> m_observation;

    psim::ParticleSystem   m_particles;
    std::vector<psim::Real> m_scalars;
    psim::Vec2 m_scalarRange{ -1, 1 };
    psim::Real m_particleRadius{ 0.01f };

    /// Equilibrium part of the helical flux, integrated once per control step.
    std::vector<Real> m_equilibriumHelicalFlux;

    /// Index of the first tracer particle; everything before it is the fixed
    /// polar sampling of the cross-section.
    int m_tracerBegin{ 0 };

    bool m_finished{ false };
    Real m_lastReward{ 0 };
    double m_decisionMicroseconds{ 0 };
    std::uint64_t m_seed{ 1 };
};

} // namespace plasma
