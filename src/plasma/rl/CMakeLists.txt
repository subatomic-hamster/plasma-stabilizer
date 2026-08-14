# plasma_rl: the in-repo learner. Deliberately dependency-free beyond the
# environment -- the second backend (RLTools) is the independent check.
add_library(plasma_rl
    Mlp.cpp
    Ppo.cpp)
add_library(plasma::rl ALIAS plasma_rl)
target_link_libraries(plasma_rl PUBLIC plasma::control PRIVATE plasma::flags)
