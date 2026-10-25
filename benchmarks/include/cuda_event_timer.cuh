#pragma once

#include <cuda_runtime.h>

#include "cuda_check.cuh"

// RAII-обёртка над парой CUDA Events для измерения времени работы кода на устройстве.
class CudaEventTimer {
private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};

public:
    CudaEventTimer() {
        check_cuda(cudaEventCreate(&start_), "cudaEventCreate");
        const cudaError_t err = cudaEventCreate(&stop_);
        if (err != cudaSuccess) {
            cudaEventDestroy(start_);
            check_cuda(err, "cudaEventCreate");
        }
    }

    ~CudaEventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void start(cudaStream_t stream = nullptr) {
        check_cuda(cudaEventRecord(start_, stream), "cudaEventRecord(start)");
    }

    // Возвращает время между start() и stop() в миллисекундах.
    float stop(cudaStream_t stream = nullptr) {
        check_cuda(cudaEventRecord(stop_, stream), "cudaEventRecord(stop)");
        check_cuda(cudaEventSynchronize(stop_), "cudaEventSynchronize");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start_, stop_), "cudaEventElapsedTime");
        return elapsed_ms;
    }
};
