# Dependency resolution.
#
# particle-sim is embedded as a subproject rather than vendored or copied: it is
# actively developed alongside this repository, and a source-level dependency
# means a fix there is available here immediately.

include(FetchContent)

# --- particle-sim -----------------------------------------------------------
if(NOT EXISTS "${PLASMA_PARTICLE_SIM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "particle-sim not found at '${PLASMA_PARTICLE_SIM_DIR}'.\n"
        "Clone it next to this repository, or point -DPLASMA_PARTICLE_SIM_DIR=<path> at it.")
endif()

# The engine's own tests, benchmarks and viewer are not part of this build.
set(PSIM_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(PSIM_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(PSIM_BUILD_RENDERER   ${PLASMA_BUILD_VIEWER} CACHE BOOL "" FORCE)

add_subdirectory(${PLASMA_PARTICLE_SIM_DIR} ${CMAKE_BINARY_DIR}/particle-sim)

# --- RLTools ----------------------------------------------------------------
# Header-only C++ deep-RL library. Used as the second training backend, after
# the in-repo PPO implementation, to check the environment against an
# independently written learner.
if(PLASMA_WITH_RLTOOLS)
    FetchContent_Declare(rl_tools
        GIT_REPOSITORY https://github.com/rl-tools/rl-tools.git
        GIT_TAG        v2.0.1
        GIT_SHALLOW    TRUE)

    # Populate only: RLTools' own CMake pulls in tests, examples and optional
    # backends that this project has no use for.
    FetchContent_GetProperties(rl_tools)
    if(NOT rl_tools_POPULATED)
        FetchContent_Populate(rl_tools)
    endif()

    add_library(rl_tools INTERFACE)
    target_include_directories(rl_tools SYSTEM INTERFACE ${rl_tools_SOURCE_DIR}/include)
    target_compile_definitions(rl_tools INTERFACE
        RL_TOOLS_DISABLE_TENSORBOARD
        RL_TOOLS_DISABLE_HDF5
        RL_TOOLS_DISABLE_CLI11)
    add_library(rl_tools::rl_tools ALIAS rl_tools)
endif()
