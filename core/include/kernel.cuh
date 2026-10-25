#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "matrix_view.cuh"

// Сторона квадратного блока нитей для наивного кёрнела (16 x 16 = 256 нитей в блоке).
inline constexpr unsigned int kMatmulBlockDim = 16;

// Сторона квадратного тайла для кёрнела с разделяемой памятью по умолчанию.
// 16 x 16 = 256 нитей: на SM архитектуры Ampere (sm_86, до 1536 нитей) помещается 6 блоков,
// тогда как при тайле 32 x 32 (1024 нити) — только один, что снижает загрузку SM до 67 %.
inline constexpr unsigned int kMatmulTileDim = 16;

// Наивное умножение C = A * B без разделяемой памяти (версия из практической работы №2).
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

// Блочное умножение C = A * B с разделяемой памятью.
// Блок нитей TileDim x TileDim вычисляет тайл C того же размера. Внутренняя размерность k
// проходится полосами ширины TileDim: на каждом шаге нити блока совместно загружают тайл A
// и тайл B в разделяемую память (каждая нить — по одному элементу каждого тайла), после чего
// каждая нить накапливает скалярное произведение строки тайла A на столбец тайла B.
// Элементы за границами матриц заменяются нулями, поэтому размеры могут быть не кратны TileDim.
// Кёрнел ожидает blockDim = (TileDim, TileDim).
template <typename AtomT, unsigned int TileDim>
__global__ void kernel_matmul_shmem(MatrixView<AtomT> a, MatrixView<AtomT> b,
                                    MatrixView<AtomT> c) {
    __shared__ AtomT tile_a[TileDim][TileDim];
    __shared__ AtomT tile_b[TileDim][TileDim];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * TileDim + ty;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * TileDim + tx;

    AtomT acc{};
    for (std::size_t tile_start = 0; tile_start < a.ncols(); tile_start += TileDim) {
        // Соседние по tx нити читают соседние элементы строки — доступ объединяется.
        const std::size_t a_col = tile_start + tx;
        const std::size_t b_row = tile_start + ty;
        tile_a[ty][tx] = (row < a.nrows() && a_col < a.ncols()) ? a(row, a_col) : AtomT{};
        tile_b[ty][tx] = (b_row < b.nrows() && col < b.ncols()) ? b(b_row, col) : AtomT{};

        // Тайлы должны быть полностью загружены до начала вычислений.
        __syncthreads();

#pragma unroll
        for (unsigned int p = 0; p < TileDim; ++p) {
            // tile_a[ty][p] одинаков для всего варпа (broadcast),
            // tile_b[p][tx] — последовательные слова, конфликтов банков нет.
            acc += tile_a[ty][p] * tile_b[p][tx];
        }

        // Тайлы нельзя перезаписывать, пока все нити блока не закончат их чтение.
        __syncthreads();
    }

    // Проверка выполняется после цикла: все нити блока, включая лежащие за границами C,
    // должны участвовать в загрузке тайлов и доходить до __syncthreads.
    if (row < c.nrows() && col < c.ncols()) {
        c(row, col) = acc;
    }
}

// Размер сетки, покрывающей матрицу c блоками block_dim x block_dim.
template <typename AtomT>
dim3 matmul_grid(const MatrixView<AtomT>& c, unsigned int block_dim) {
    return dim3(static_cast<unsigned int>((c.ncols() + block_dim - 1) / block_dim),
                static_cast<unsigned int>((c.nrows() + block_dim - 1) / block_dim));
}

// Асинхронный запуск kernel_matmul_naive; синхронизацию выполняет вызывающая сторона.
template <typename AtomT>
void launch_matmul_naive(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c,
                         cudaStream_t stream = nullptr) {
    if (c.size() == 0) {
        return;  // Пустая сетка — недопустимая конфигурация запуска.
    }

    const dim3 block(kMatmulBlockDim, kMatmulBlockDim);
    kernel_matmul_naive<<<matmul_grid(c, kMatmulBlockDim), block, 0, stream>>>(a, b, c);
    check_cuda(cudaGetLastError(), "kernel_matmul_naive launch");
}

// Асинхронный запуск kernel_matmul_shmem; синхронизацию выполняет вызывающая сторона.
template <typename AtomT, unsigned int TileDim = kMatmulTileDim>
void launch_matmul_shmem(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c,
                         cudaStream_t stream = nullptr) {
    static_assert(TileDim > 0 && TileDim * TileDim <= 1024,
                  "Block of TileDim x TileDim threads exceeds the CUDA limit of 1024");

    if (c.size() == 0) {
        return;  // Пустая сетка — недопустимая конфигурация запуска.
    }

    const dim3 block(TileDim, TileDim);
    kernel_matmul_shmem<AtomT, TileDim>
        <<<matmul_grid(c, TileDim), block, 0, stream>>>(a, b, c);
    check_cuda(cudaGetLastError(), "kernel_matmul_shmem launch");
}
