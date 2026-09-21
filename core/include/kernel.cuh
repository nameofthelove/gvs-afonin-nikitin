#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#include "vector_view.cuh"

template <typename AtomT>
__global__ void kernel_vecadd(VectorView<AtomT> a,
                              VectorView<AtomT> b,
                              VectorView<AtomT> result) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (idx < result.size()) {
        result[idx] = a[idx] + b[idx];
    }
}

template <typename AtomT>
inline void launch_vecadd(VectorView<AtomT> a,
                          VectorView<AtomT> b,
                          VectorView<AtomT> result) {
    constexpr int threads_per_block = 256;
    const int blocks_per_grid =
        static_cast<int>((result.size() + threads_per_block - 1) / threads_per_block);
    kernel_vecadd<<<blocks_per_grid, threads_per_block>>>(a, b, result);
}
