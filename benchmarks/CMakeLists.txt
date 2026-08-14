function(plasma_add_benchmark name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE plasma::mhd plasma::flags)
endfunction()
