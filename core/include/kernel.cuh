#pragma once
#include <cuda_runtime.h>
#include "vector_view.cuh"

template <typename AtomT>
__global__ void kernel_vecadd(VectorView<AtomT> a, VectorView<AtomT> b, VectorView<AtomT> result) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < result.size()) {
        result[idx] = a[idx] + b[idx];
    }
}