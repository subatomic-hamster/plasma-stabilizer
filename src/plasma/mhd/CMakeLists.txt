# plasma_mhd: the physics model. No rendering, no RL, no windowing -- a training
# run vectorized across processes links only this.
add_library(plasma_mhd
    RadialGrid.cpp
    Equilibrium.cpp
    ReducedMhd.cpp)

add_library(plasma::mhd ALIAS plasma_mhd)

target_include_directories(plasma_mhd PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(plasma_mhd PUBLIC psim::core PRIVATE plasma::flags)
