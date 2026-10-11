#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "matrix_view.cuh"

// Сторона квадратного блока нитей (16 x 16 = 256 нитей в блоке).
inline constexpr unsigned int kMatmulBlockDim = 16;

// Наивное умножение C = A * B без разделяемой памяти.
// Двумерная сетка: ось x — столбцы C, ось y — строки C; одна нить вычисляет один элемент C.
template <typename AtomT>
__global__ void kernel_matmul_naive(MatrixView<AtomT> a, MatrixView<AtomT> b,
                                    MatrixView<AtomT> c) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row < c.nrows() && col < c.ncols()) {
        AtomT acc{};
        for (std::size_t p = 0; p < a.ncols(); ++p) {
            acc += a(row, p) * b(p, col);
        }
        c(row, col) = acc;
    }
}

// Асинхронный запуск kernel_matmul_naive; синхронизацию выполняет вызывающая сторона.
template <typename AtomT>
void launch_matmul_naive(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c,
                         cudaStream_t stream = nullptr) {
    if (c.size() == 0) {
        return;  // Пустая сетка — недопустимая конфигурация запуска.
    }

    const dim3 block(kMatmulBlockDim, kMatmulBlockDim);
    const dim3 grid(static_cast<unsigned int>((c.ncols() + block.x - 1) / block.x),
                    static_cast<unsigned int>((c.nrows() + block.y - 1) / block.y));

    kernel_matmul_naive<<<grid, block, 0, stream>>>(a, b, c);
    check_cuda(cudaGetLastError(), "kernel_matmul_naive launch");
}
