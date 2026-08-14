# Warning and optimization flags for first-party targets only. Third-party code
# (particle-sim included) keeps its own settings.

add_library(plasma_flags INTERFACE)
add_library(plasma::flags ALIAS plasma_flags)

if(MSVC)
    target_compile_options(plasma_flags INTERFACE /W4 /permissive-)
else()
    target_compile_options(plasma_flags INTERFACE
        -Wall -Wextra -Wpedantic
        -Wshadow -Wnon-virtual-dtor -Wcast-align
        -Wunused -Woverloaded-virtual -Wdouble-promotion)

    target_compile_options(plasma_flags INTERFACE
        $<$<CONFIG:Release>:-O3 -fno-math-errno -fno-trapping-math>)

    if(PLASMA_NATIVE_ARCH)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-march=native PLASMA_HAS_MARCH_NATIVE)
        if(PLASMA_HAS_MARCH_NATIVE)
            target_compile_options(plasma_flags INTERFACE $<$<CONFIG:Release>:-march=native>)
        endif()
    endif()
endif()
