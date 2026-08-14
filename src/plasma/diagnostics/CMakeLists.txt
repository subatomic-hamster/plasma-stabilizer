add_library(plasma_diagnostics
    SignalProcessing.cpp
    Diagnostics.cpp)
add_library(plasma::diagnostics ALIAS plasma_diagnostics)
target_link_libraries(plasma_diagnostics PUBLIC plasma::mhd PRIVATE plasma::flags)
