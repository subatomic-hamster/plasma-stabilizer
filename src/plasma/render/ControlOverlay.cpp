#include <plasma/render/ControlOverlay.h>

#include <psim/render/GlCommon.h>

#include <algorithm>
#include <cmath>

namespace plasma {

namespace {

constexpr std::string_view kVertexSource = R"(#version 410 core
layout(location = 0) in vec2 aPosition;
uniform vec4 uViewport; // x0, y0, width, height in pixels
void main()
{
    // Pixel coordinates, origin top-left, to clip space.
    vec2 normalized = (aPosition - uViewport.xy) / uViewport.zw;
    gl_Position = vec4(normalized.x * 2.0 - 1.0, 1.0 - normalized.y * 2.0, 0.0, 1.0);
}
)";

constexpr std::string_view kFragmentSource = R"(#version 410 core
uniform vec4 uColor;
out vec4 FragColor;
void main() { FragColor = uColor; }
)";

/// Panel geometry, in pixels from the top-left of the window.
constexpr float kPanelX = 18.0f;
constexpr float kPanelY = 18.0f;
constexpr float kPanelWidth = 430.0f;
constexpr float kBarHeight = 15.0f;
constexpr float kBarGap = 5.0f;
constexpr float kTraceHeight = 52.0f;

} // namespace

bool ControlOverlay::initialize()
{
    if (!m_shader.build(kVertexSource, kFragmentSource, "control-overlay")) return false;

    glGenVertexArrays(1, &m_vertexArray);
    glGenBuffers(1, &m_vertexBuffer);

    glBindVertexArray(m_vertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);
    glBindVertexArray(0);

    clear();
    return true;
}

void ControlOverlay::clear()
{
    for (auto& history : m_actuatorHistory) history.clear();
    m_tearingBand.clear();
    m_alfvenicBand.clear();
    m_islandWidth.clear();
    m_confinement.clear();
    m_reward.clear();
    m_actuatorNow.fill(0.0f);
    m_terms = RewardTerms{};
    m_islandNow = 0;
}

void ControlOverlay::record(const TokamakEnv& environment)
{
    auto push = [this](std::deque<float>& series, float value) {
        series.push_back(value);
        while (series.size() > m_capacity) series.pop_front();
    };

    const ActuatorBank& actuators = environment.actuators();
    for (std::size_t i = 0; i < kActuatorCount; ++i) {
        const float value =
            static_cast<float>(actuators.channel(static_cast<ActuatorChannel>(i)).normalized());
        m_actuatorNow[i] = value;
        push(m_actuatorHistory[i], value);
    }

    const DiagnosticOutputs& diagnostics = environment.diagnostics().outputs();
    // Log-compressed, because the band envelopes sweep several decades over a
    // pulse and a linear trace would be a flat line then a vertical wall.
    push(m_tearingBand,
         static_cast<float>(std::log10(std::max(diagnostics.tearingAmplitude, 1e-8)) * 0.2 + 1.0));
    push(m_alfvenicBand,
         static_cast<float>(std::log10(std::max(diagnostics.alfvenicAmplitude, 1e-8)) * 0.2 + 1.0));

    m_islandNow = static_cast<float>(environment.widestIsland());
    m_disruptionLimit = static_cast<float>(environment.config().disruptionIslandWidth);
    push(m_islandWidth, m_islandNow);
    push(m_confinement, static_cast<float>(environment.equilibrium().confinementFraction()));
    push(m_reward, static_cast<float>(environment.rewardTerms().total));

    m_terms = environment.rewardTerms();
}

void ControlOverlay::pushQuad(std::vector<Vertex>& out, float x0, float y0, float x1, float y1) const
{
    out.push_back({ x0, y0 });
    out.push_back({ x1, y0 });
    out.push_back({ x1, y1 });
    out.push_back({ x0, y0 });
    out.push_back({ x1, y1 });
    out.push_back({ x0, y1 });
}

void ControlOverlay::pushTrace(std::vector<Vertex>& out, const std::deque<float>& values,
                               float x0, float y0, float width, float height,
                               float minimum, float maximum) const
{
    if (values.size() < 2) return;
    const float span = std::max(maximum - minimum, 1e-6f);
    const float step = width / static_cast<float>(m_capacity - 1);

    // A line strip would need its own draw call per trace; emitting explicit
    // segment pairs lets every trace of one colour go out together.
    for (std::size_t i = 1; i < values.size(); ++i) {
        const float x1 = x0 + step * static_cast<float>(i - 1);
        const float x2 = x0 + step * static_cast<float>(i);
        const float v1 = std::clamp((values[i - 1] - minimum) / span, 0.0f, 1.0f);
        const float v2 = std::clamp((values[i] - minimum) / span, 0.0f, 1.0f);
        out.push_back({ x1, y0 + height - v1 * height });
        out.push_back({ x2, y0 + height - v2 * height });
    }
}

void ControlOverlay::submit(const std::vector<Vertex>& vertices, unsigned int primitive,
                            float r, float g, float b, float a) const
{
    if (vertices.empty()) return;

    glBindVertexArray(m_vertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);

    const std::size_t bytes = vertices.size() * sizeof(Vertex);
    if (bytes > m_bufferCapacity) {
        m_bufferCapacity = bytes * 2;
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(m_bufferCapacity), nullptr,
                     GL_STREAM_DRAW);
    }
    glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(bytes), vertices.data());

    m_shader.setVec4("uColor", glm::vec4(r, g, b, a));
    glDrawArrays(primitive, 0, static_cast<GLsizei>(vertices.size()));
    glBindVertexArray(0);
}

void ControlOverlay::draw(int framebufferWidth, int framebufferHeight) const
{
    if (!m_shader.valid()) return;

    const float panelHeight = kBarHeight * kActuatorCount + kBarGap * (kActuatorCount + 1) +
                              kTraceHeight * 3.0f + 56.0f;

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    m_shader.bind();
    m_shader.setVec4("uViewport", glm::vec4(0.0f, 0.0f, static_cast<float>(framebufferWidth),
                                            static_cast<float>(framebufferHeight)));

    std::vector<Vertex> vertices;

    // Backing panel.
    pushQuad(vertices, kPanelX - 8.0f, kPanelY - 8.0f,
             kPanelX + kPanelWidth + 8.0f, kPanelY + panelHeight);
    submit(vertices, GL_TRIANGLES, 0.05f, 0.06f, 0.09f, 0.82f);

    // --- Actuator bars ------------------------------------------------------
    float cursorY = kPanelY;
    const float barWidth = kPanelWidth - 96.0f;

    vertices.clear();
    for (std::size_t i = 0; i < kActuatorCount; ++i) {
        const float y = cursorY + static_cast<float>(i) * (kBarHeight + kBarGap);
        pushQuad(vertices, kPanelX + 92.0f, y, kPanelX + 92.0f + barWidth, y + kBarHeight);
    }
    submit(vertices, GL_TRIANGLES, 0.16f, 0.18f, 0.23f, 1.0f);

    vertices.clear();
    for (std::size_t i = 0; i < kActuatorCount; ++i) {
        const float y = cursorY + static_cast<float>(i) * (kBarHeight + kBarGap);
        const float filled = barWidth * std::clamp(m_actuatorNow[i], 0.0f, 1.0f);
        pushQuad(vertices, kPanelX + 92.0f, y, kPanelX + 92.0f + filled, y + kBarHeight);
    }
    // Amber: these are the commands leaving the controller.
    submit(vertices, GL_TRIANGLES, 0.95f, 0.68f, 0.20f, 1.0f);

    cursorY += kActuatorCount * (kBarHeight + kBarGap) + 14.0f;

    // --- Island width against the disruption limit --------------------------
    vertices.clear();
    pushQuad(vertices, kPanelX, cursorY, kPanelX + kPanelWidth, cursorY + kBarHeight);
    submit(vertices, GL_TRIANGLES, 0.16f, 0.18f, 0.23f, 1.0f);

    vertices.clear();
    const float islandFraction = std::clamp(m_islandNow / std::max(m_disruptionLimit, 1e-6f), 0.0f, 1.0f);
    pushQuad(vertices, kPanelX, cursorY, kPanelX + kPanelWidth * islandFraction,
             cursorY + kBarHeight);
    // Red as the island approaches the width that ends the discharge.
    submit(vertices, GL_TRIANGLES, 0.20f + 0.75f * islandFraction, 0.75f - 0.55f * islandFraction,
           0.35f, 1.0f);

    cursorY += kBarHeight + 16.0f;

    // --- Diagnostic band traces ---------------------------------------------
    vertices.clear();
    pushQuad(vertices, kPanelX, cursorY, kPanelX + kPanelWidth, cursorY + kTraceHeight);
    pushQuad(vertices, kPanelX, cursorY + kTraceHeight + 8.0f, kPanelX + kPanelWidth,
             cursorY + kTraceHeight * 2.0f + 8.0f);
    pushQuad(vertices, kPanelX, cursorY + kTraceHeight * 2.0f + 16.0f, kPanelX + kPanelWidth,
             cursorY + kTraceHeight * 3.0f + 16.0f);
    submit(vertices, GL_TRIANGLES, 0.09f, 0.10f, 0.13f, 1.0f);

    vertices.clear();
    pushTrace(vertices, m_tearingBand, kPanelX, cursorY, kPanelWidth, kTraceHeight, 0.0f, 1.4f);
    submit(vertices, GL_LINES, 0.35f, 0.80f, 1.00f, 1.0f);

    vertices.clear();
    pushTrace(vertices, m_alfvenicBand, kPanelX, cursorY, kPanelWidth, kTraceHeight, 0.0f, 1.4f);
    submit(vertices, GL_LINES, 1.00f, 0.45f, 0.55f, 1.0f);

    vertices.clear();
    pushTrace(vertices, m_islandWidth, kPanelX, cursorY + kTraceHeight + 8.0f, kPanelWidth,
              kTraceHeight, 0.0f, std::max(m_disruptionLimit, 1e-6f));
    submit(vertices, GL_LINES, 0.95f, 0.60f, 0.30f, 1.0f);

    vertices.clear();
    pushTrace(vertices, m_confinement, kPanelX, cursorY + kTraceHeight + 8.0f, kPanelWidth,
              kTraceHeight, 0.0f, 1.6f);
    submit(vertices, GL_LINES, 0.45f, 0.90f, 0.55f, 1.0f);

    vertices.clear();
    pushTrace(vertices, m_reward, kPanelX, cursorY + kTraceHeight * 2.0f + 16.0f, kPanelWidth,
              kTraceHeight, -3.0f, 2.0f);
    submit(vertices, GL_LINES, 0.90f, 0.90f, 0.95f, 1.0f);

    // Zero line for the reward trace, so positive and negative are separable.
    vertices.clear();
    const float zeroY = cursorY + kTraceHeight * 2.0f + 16.0f +
                        kTraceHeight * (1.0f - (0.0f - (-3.0f)) / 5.0f);
    vertices.push_back({ kPanelX, zeroY });
    vertices.push_back({ kPanelX + kPanelWidth, zeroY });
    submit(vertices, GL_LINES, 0.45f, 0.45f, 0.50f, 0.9f);

    psim::Shader::unbind();
    glEnable(GL_DEPTH_TEST);
}

} // namespace plasma
