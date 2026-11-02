cmake_minimum_required(VERSION 3.24)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# Архитектура GPU (Ampere: RTX 3050 Ti -> sm_86); переопределяется через -DCMAKE_CUDA_ARCHITECTURES.
# Задаётся до project(): иначе project() подставляет архитектуру по умолчанию nvcc (sm_75).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(work6 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Сборки LibTorch для Windows бывают только Release (/MD): Debug несовместим по CRT.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(TORCH_CUDA_ARCH_LIST "8.6" CACHE STRING "GPU architectures for the Torch CMake config")

find_package(CUDAToolkit REQUIRED)

# LibTorch: Torch_DIR или CMAKE_PREFIX_PATH, иначе — пакет torch из текущего Python
# (torch.utils.cmake_prefix_path указывает на site-packages/torch/share/cmake).
if(NOT DEFINED Torch_DIR AND NOT DEFINED ENV{Torch_DIR})
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE TORCH_PYTHON_RESULT
      ERROR_QUIET
    )
    if(TORCH_PYTHON_RESULT EQUAL 0)
      list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
    endif()
  endif()
endif()
find_package(Torch REQUIRED)

include(FetchContent)

# Google Test
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_subdirectory(core)
add_subdirectory(tests)
