# Assert-based self-checks, same discipline as the engine: each file is a main()
# that aborts on the first violated invariant, and the checks fire in release
# builds, which is where the solver actually runs.
function(plasma_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE plasma::mhd plasma::flags)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

plasma_add_test(test_radial_grid)
plasma_add_test(test_equilibrium)
plasma_add_test(test_mhd)
