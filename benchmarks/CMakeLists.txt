add_executable(benchmarks
    src/bench_vecred.cu
)

target_include_directories(benchmarks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


target_link_libraries(benchmarks PRIVATE
    core
    benchmark::benchmark
)
