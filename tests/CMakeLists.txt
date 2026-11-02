add_executable(unit_tests
    src/conv2d_instances.cu
    src/test_conv2d.cpp
)

target_include_directories(unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(unit_tests PRIVATE
    hsys::nn
    GTest::gtest_main
)

# Windows ищет DLL рядом с исполняемым файлом: копируем библиотеки LibTorch после сборки.
if(WIN32)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET unit_tests POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:unit_tests>
      VERBATIM
  )
endif()

# PRE_TEST: список тестов запрашивается при запуске ctest, когда DLL уже скопированы.
include(GoogleTest)
gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 120)
