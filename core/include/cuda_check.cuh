#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Преобразует код ошибки CUDA Runtime API в исключение std::runtime_error.
inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}
