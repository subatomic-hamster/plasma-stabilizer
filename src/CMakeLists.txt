add_subdirectory(plasma/mhd)
add_subdirectory(plasma/diagnostics)
add_subdirectory(plasma/control)
add_subdirectory(plasma/rl)

if(PLASMA_BUILD_VIEWER)
    add_subdirectory(plasma/render)
endif()
