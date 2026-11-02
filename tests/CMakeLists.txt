add_executable(unit_tests
    src/test_vecred.cu
)

target_include_directories(unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(unit_tests PRIVATE EIGEN_NO_CUDA)

target_link_libraries(unit_tests PRIVATE
    core
    GTest::gtest_main
    Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(unit_tests DISCOVERY_TIMEOUT 60)
