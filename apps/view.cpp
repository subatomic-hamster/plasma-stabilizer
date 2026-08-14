// Interactive viewer for a control episode.
//
// Reuses the particle-sim renderer wholesale: the plasma cross-section is
// handed over as a Simulation and drawn by code that knows nothing about
// tokamaks. Only the control overlay is specific to this project.

#include <plasma/render/ControlOverlay.h>
#include <plasma/render/TokamakScene.h>

#include <psim/core/Clock.h>
#include <psim/core/Log.h>
#include <psim/render/Camera.h>
#include <psim/render/SimulationRenderer.h>
#include <psim/render/Window.h>

#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>

using namespace plasma;

namespace {

struct Options
{
    std::string policyPath;
    std::string controller{ "policy" };
    std::uint64_t seed{ 1 };
    int width{ 1600 };
    int height{ 900 };
    int stepsPerFrame{ 1 };
    bool vsync{ true };
};

Options parseArguments(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--policy") == 0 && i + 1 < argc) options.policyPath = argv[++i];
        else if (std::strcmp(argv[i], "--controller") == 0 && i + 1 < argc) options.controller = argv[++i];
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) options.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc) options.width = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc) options.height = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--speed") == 0 && i + 1 < argc) options.stepsPerFrame = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--no-vsync") == 0) options.vsync = false;
        else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf("usage: %s [--policy FILE] [--controller policy|fixed|passive]\n"
                        "          [--seed N] [--speed N] [--width px] [--height px] [--no-vsync]\n",
                        argv[0]);
            std::exit(0);
        }
    }
    return options;
}

void printControls()
{
    std::printf(
        "\ncontrols\n"
        "  left drag    orbit         scroll   zoom\n"
        "  middle drag  pan           f        re-frame\n"
        "  space        pause         r        restart episode\n"
        "  1 2 3        controller: learned policy / best fixed / passive\n"
        "  [ ]          slower / faster\n"
        "  esc          quit\n\n");
}

const char* controllerName(SceneController controller)
{
    switch (controller) {
    case SceneController::Policy:  return "learned policy";
    case SceneController::Fixed:   return "best fixed action";
    case SceneController::Passive: return "passive (no control)";
    }
    return "?";
}

} // namespace

int main(int argc, char** argv)
{
    const Options options = parseArguments(argc, argv);

    TokamakEnvConfig environmentConfig;
    environmentConfig.equilibrium.edgeSafetyFactor  = 3.4;
    environmentConfig.equilibrium.currentPeaking    = 2.4;
    environmentConfig.equilibrium.bootstrapFraction = 0.35;
    environmentConfig.equilibrium.lundquist         = 1.0e4;

    SceneConfig sceneConfig;
    sceneConfig.stepsPerFrame = std::max(1, options.stepsPerFrame);

    TokamakScene scene;
    scene.configure(environmentConfig, sceneConfig);

    // The policy is optional: without one the viewer still runs the fixed and
    // passive references, which is how you look at the problem before there is
    // anything trained to look at.
    PpoAgent agent;
    bool policyLoaded = false;
    if (!options.policyPath.empty()) {
        PpoConfig ppo;
        agent.configure(static_cast<int>(scene.environment().observationSize()),
                        static_cast<int>(TokamakEnv::actionSize()), ppo, 1);
        policyLoaded = agent.load(options.policyPath);
        if (policyLoaded) {
            scene.setPolicy(&agent);
            psim::log::info("loaded policy from {}", options.policyPath);
        } else {
            psim::log::warn("could not load policy from {}; falling back to the fixed action",
                            options.policyPath);
        }
    }

    SceneController controller = SceneController::Policy;
    if (options.controller == "fixed" || !policyLoaded) controller = SceneController::Fixed;
    if (options.controller == "passive") controller = SceneController::Passive;
    scene.setController(controller);

    psim::Window window;
    psim::WindowConfig windowConfig;
    windowConfig.width  = options.width;
    windowConfig.height = options.height;
    windowConfig.vsync  = options.vsync;
    windowConfig.title  = "plasma-stabilizer";
    if (!window.create(windowConfig)) return 1;

    psim::SimulationRenderer renderer;
    if (!renderer.initialize()) {
        psim::log::error("renderer initialization failed");
        return 1;
    }

    ControlOverlay overlay;
    if (!overlay.initialize()) {
        psim::log::error("overlay initialization failed");
        return 1;
    }

    psim::Camera camera;
    camera.setViewport(window.width(), window.height());
    // Face-on: this is a poloidal cross-section, and the island chain only
    // reads correctly when looked at down the machine axis.
    camera.setOrientation(0.0, 0.0);
    camera.frame(scene.bounds());

    psim::RenderOptions renderOptions;
    renderOptions.mode = psim::DrawMode::Points;
    renderOptions.showGrid = false;
    renderOptions.colorByScalar = true;
    renderOptions.background = glm::vec4(0.04f, 0.05f, 0.07f, 1.0f);

    printControls();
    scene.reset(options.seed);

    psim::FrameStats frameStats;
    psim::Clock frameClock;
    bool paused = false;
    std::uint64_t episodeSeed = options.seed;
    double worstDecisionMicroseconds = 0;

    while (!window.shouldClose()) {
        window.pollEvents();
        frameStats.addFrame(frameClock.tick());
        camera.setViewport(window.width(), window.height());

        if (window.keyPressed(GLFW_KEY_ESCAPE)) window.requestClose();
        if (window.keyPressed(GLFW_KEY_SPACE)) paused = !paused;
        if (window.keyPressed(GLFW_KEY_R)) {
            scene.reset(++episodeSeed);
            overlay.clear();
            worstDecisionMicroseconds = 0;
        }
        if (window.keyPressed(GLFW_KEY_1) && policyLoaded) scene.setController(SceneController::Policy);
        if (window.keyPressed(GLFW_KEY_2)) scene.setController(SceneController::Fixed);
        if (window.keyPressed(GLFW_KEY_3)) scene.setController(SceneController::Passive);
        if (window.keyPressed(GLFW_KEY_F)) camera.frame(scene.bounds());

        const psim::Vec2 mouseDelta = window.mouseDelta();
        const bool shiftHeld = window.keyDown(GLFW_KEY_LEFT_SHIFT) || window.keyDown(GLFW_KEY_RIGHT_SHIFT);
        if (window.mouseDown(psim::MouseButton::Left) && !shiftHeld) camera.orbit(mouseDelta.x, mouseDelta.y);
        if (window.mouseDown(psim::MouseButton::Middle) ||
            (window.mouseDown(psim::MouseButton::Left) && shiftHeld)) {
            camera.pan(mouseDelta.x, mouseDelta.y);
        }
        if (window.scrollDelta() != 0) camera.zoom(window.scrollDelta());

        if (!paused && !scene.episodeFinished()) {
            scene.step(1.0f / 60.0f);
            overlay.record(scene.environment());
            worstDecisionMicroseconds = std::max(worstDecisionMicroseconds,
                                                 scene.lastDecisionMicroseconds());
        }

        renderer.draw(scene, camera, renderOptions);
        overlay.draw(window.width(), window.height());
        window.swapBuffers();

        if (frameStats.frameCount() % 15 == 0) {
            const TokamakEnv& environment = scene.environment();
            window.setTitle(std::format(
                "plasma-stabilizer | {} | step {}/{} ({:.0f} ms) | island {:.3f} | "
                "confinement {:.2f} | return {:.0f} | decision {:.0f} us | {:.0f} FPS{}",
                controllerName(scene.controller()), environment.stepIndex(),
                environment.config().maxSteps, environment.elapsedSeconds() * 1e3,
                environment.widestIsland(), environment.equilibrium().confinementFraction(),
                environment.episodeReturn(), worstDecisionMicroseconds, frameStats.fps(),
                scene.episodeFinished() ? (environment.disrupted() ? " | DISRUPTED" : " | COMPLETE")
                                        : (paused ? " | PAUSED" : "")));
        }
    }

    std::printf("worst controller decision latency: %.1f us (budget %.0f us per control step)\n",
                worstDecisionMicroseconds,
                environmentConfig.controlPeriod * environmentConfig.alfvenTimeSeconds * 1e6);
    return 0;
}
