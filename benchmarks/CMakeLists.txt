add_executable(benchmarks
    src/bench_vecadd.cu
)

target_link_libraries(benchmarks PRIVATE
    core
    benchmark::benchmark
    Eigen3::Eigen
)