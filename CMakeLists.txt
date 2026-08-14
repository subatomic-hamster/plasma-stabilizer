cmake_minimum_required(VERSION 3.20)
project(plasma-stabilizer
    VERSION 0.1.0
    DESCRIPTION "Reinforcement-learning control of MHD instabilities in a simulated tokamak"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(PLASMA_BUILD_TESTS      "Build physics and RL self-checks"        ON)
option(PLASMA_BUILD_BENCHMARKS "Build throughput benchmarks"             ON)
option(PLASMA_BUILD_VIEWER     "Build the OpenGL viewer"                 ON)
option(PLASMA_WITH_RLTOOLS     "Build the RLTools training backend"      ON)
option(PLASMA_NATIVE_ARCH      "Compile with -march=native"              ON)

# Path to the particle-sim engine this project is built on. It supplies the
# Simulation interface the environment implements, the math and parallel-for
# core, and the whole renderer.
set(PLASMA_PARTICLE_SIM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../particle-sim"
    CACHE PATH "Source directory of the particle-sim engine")

include(cmake/CompilerWarnings.cmake)
include(cmake/Dependencies.cmake)

add_subdirectory(src)

if(PLASMA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(PLASMA_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

add_subdirectory(apps)

message(STATUS "")
message(STATUS "plasma-stabilizer ${PROJECT_VERSION}")
message(STATUS "  build type ......... ${CMAKE_BUILD_TYPE}")
message(STATUS "  particle-sim ....... ${PLASMA_PARTICLE_SIM_DIR}")
message(STATUS "  viewer ............. ${PLASMA_BUILD_VIEWER}")
message(STATUS "  rltools backend .... ${PLASMA_WITH_RLTOOLS}")
message(STATUS "  tests .............. ${PLASMA_BUILD_TESTS}")
message(STATUS "")
