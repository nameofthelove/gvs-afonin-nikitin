add_library(core STATIC
    src/vecadd.cu
)

target_include_directories(core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(core PUBLIC
    CUDA::cudart
)

set_target_properties(core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
