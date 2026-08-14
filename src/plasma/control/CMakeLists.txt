# plasma_control: actuators, reward, and the RL environment. Headless: a
# training run links this and nothing else.
add_library(plasma_control
    Actuators.cpp
    TokamakEnv.cpp)
add_library(plasma::control ALIAS plasma_control)
target_link_libraries(plasma_control PUBLIC plasma::diagnostics PRIVATE plasma::flags)
