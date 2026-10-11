add_executable(benchmarks
    src/bench_matmul.cu
)

target_include_directories(benchmarks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(benchmarks PRIVATE EIGEN_NO_CUDA)

target_link_libraries(benchmarks PRIVATE
    core
    benchmark::benchmark
    Eigen3::Eigen
)
