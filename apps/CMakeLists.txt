add_executable(plasma-train train.cpp)
target_link_libraries(plasma-train PRIVATE plasma::rl plasma::flags)

if(PLASMA_BUILD_VIEWER)
    add_executable(plasma-view view.cpp)
    target_link_libraries(plasma-view PRIVATE plasma::render plasma::flags glfw)
endif()

if(PLASMA_WITH_RLTOOLS)
    add_executable(plasma-train-rltools train_rltools.cpp)
    # No plasma::flags here: RLTools' headers are not written to this project's
    # warning settings, and the environment adapter is mostly their templates.
    target_link_libraries(plasma-train-rltools PRIVATE plasma::control rl_tools::rl_tools)
endif()
