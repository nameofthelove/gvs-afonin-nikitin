add_executable(unit_tests
    src/test_vecadd.cu
)

target_link_libraries(unit_tests PRIVATE
    core
    GTest::gtest_main
    Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(unit_tests)
