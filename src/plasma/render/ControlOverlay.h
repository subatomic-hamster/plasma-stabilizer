#pragma once

// Screen-space overlay showing what the controller is doing.
//
// The plasma view alone tells you the island grew; it does not tell you which
// actuator the policy reached for, or whether the diagnostics saw the mode
// before it acted. Those are the questions you actually have when inspecting a
// learned controller, so they get their own panel: a bar per actuator, a
// scrolling trace per diagnostic band, and the reward split into the objectives
// it is trading against.
//
// Drawn with the engine's shader and mesh wrappers under an orthographic
// projection, so it composites over the 3D scene without a second renderer.

#include <plasma/control/TokamakEnv.h>

#include <psim/render/GpuMesh.h>
#include <psim/render/Shader.h>

#include <array>
#include <deque>
#include <string>
#include <vector>

namespace plasma {

class ControlOverlay
{
public:
    bool initialize();

    /// Samples the environment. Call once per control step, not per frame, so
    /// the traces advance at the control rate.
    void record(const TokamakEnv& environment);

    void clear();

    /// Composites the panel over the current framebuffer.
    void draw(int framebufferWidth, int framebufferHeight) const;

    /// Number of control steps kept in the scrolling traces.
    [[nodiscard]] std::size_t historyLength() const noexcept { return m_capacity; }

private:
    struct Vertex
    {
        float x{ 0 };
        float y{ 0 };
    };

    void pushQuad(std::vector<Vertex>& out, float x0, float y0, float x1, float y1) const;
    void pushTrace(std::vector<Vertex>& out, const std::deque<float>& values,
                   float x0, float y0, float width, float height,
                   float minimum, float maximum) const;

    void submit(const std::vector<Vertex>& vertices, unsigned int primitive,
                float r, float g, float b, float a) const;

    psim::Shader m_shader;
    unsigned int m_vertexArray{ 0 };
    mutable unsigned int m_vertexBuffer{ 0 };
    mutable std::size_t m_bufferCapacity{ 0 };

    std::size_t m_capacity{ 420 };

    std::array<std::deque<float>, kActuatorCount> m_actuatorHistory;
    std::deque<float> m_tearingBand;
    std::deque<float> m_alfvenicBand;
    std::deque<float> m_islandWidth;
    std::deque<float> m_confinement;
    std::deque<float> m_reward;

    /// Latest values, for the bar column.
    std::array<float, kActuatorCount> m_actuatorNow{};
    RewardTerms m_terms{};
    float m_islandNow{ 0 };
    float m_disruptionLimit{ 0.46f };
};

} // namespace plasma
