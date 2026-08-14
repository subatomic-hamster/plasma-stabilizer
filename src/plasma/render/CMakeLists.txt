# plasma_render: visualization only. Depends on the engine's renderer; nothing
# in the physics or RL path depends on this.
add_library(plasma_render
    TokamakScene.cpp
    ControlOverlay.cpp)
add_library(plasma::render ALIAS plasma_render)
target_link_libraries(plasma_render PUBLIC plasma::rl psim::render PRIVATE plasma::flags)
