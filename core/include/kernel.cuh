#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "cuda_check.cuh"
#include "matrix_view.cuh"

// Сторона квадратного блока нитей для наивного кёрнела (16 x 16 = 256 нитей в блоке).
inline constexpr unsigned int kMatmulBlockDim = 16;

// Сторона квадратного тайла для кёрнела с разделяемой памятью по умолчанию.
// 16 x 16 = 256 нитей: на SM архитектуры Ampere (sm_86, до 1536 нитей) помещается 6 блоков,
// тогда как при тайле 32 x 32 (1024 нити) — только один, что снижает загрузку SM до 67 %.
inline constexpr unsigned int kMatmulTileDim = 16;

// Геометрия тензорного кёрнела: блок из 8 варпов образует решётку 2 x 4 (рис. 1 задания),
// каждый варп вычисляет в регистрах прямоугольник C из фрагментов 16 x 16.
inline constexpr unsigned int kWmmaFragDim = 16;
inline constexpr unsigned int kWmmaFragItems = kWmmaFragDim * kWmmaFragDim;
inline constexpr unsigned int kWarpSize = 32;
inline constexpr unsigned int kWmmaWarpRows = 2;
inline constexpr unsigned int kWmmaWarpCols = 4;
inline constexpr unsigned int kWmmaWarps = kWmmaWarpRows * kWmmaWarpCols;
inline constexpr unsigned int kWmmaBlockThreads = kWmmaWarps * kWarpSize;

// store_matrix_sync требует выравнивания каждой строки фрагмента на 16 байт.
inline constexpr unsigned int kWmmaStoreAlignment = 16;

// Размеры тайлов по умолчанию: BlockItemsY x BlockItemsX — тайл C, BlockItemsK — ширина
// полосы по внутренней размерности. Объём загрузки тайлов на одну операцию MMA обратно
// пропорционален размеру тайла C, поэтому тайл C выбран большим: каждый варп вычисляет 8
// фрагментов. Тайл 256 x 256 при n <= 512 не загружает все SM (не более 4 блоков).
inline constexpr unsigned int kWmmaBlockItemsY = 128;
inline constexpr unsigned int kWmmaBlockItemsX = 128;
inline constexpr unsigned int kWmmaBlockItemsK = 32;

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

// Нити блока совместно копируют прямоугольник Rows x Cols матрицы src с началом в (row, col)
// в тайл разделяемой памяти. Каждая нить копирует Rows * Cols / kWmmaBlockThreads
// последовательных элементов пакетами по kPacketItems: если пакет лежит в матрице и выровнен,
// он читается одной векторной инструкцией, иначе — поэлементно с заменой нулями элементов
// за границами src.
template <unsigned int Rows, unsigned int Cols>
__device__ __forceinline__ void wmma_load_tile(half* tile, const MatrixView<half>& src,
                                               std::size_t row, std::size_t col) {
    constexpr unsigned int kThreadItems = Rows * Cols / kWmmaBlockThreads;
    constexpr unsigned int kPacketItems = kThreadItems < 8 ? kThreadItems : 8;
    constexpr unsigned int kPacketBytes = kPacketItems * sizeof(half);

#pragma unroll
    for (unsigned int packet = 0; packet < kThreadItems / kPacketItems; ++packet) {
        const unsigned int first = threadIdx.x * kThreadItems + packet * kPacketItems;
        const std::size_t src_row = row + first / Cols;
        const std::size_t src_col = col + first % Cols;
        const half* packet_src = &src(src_row, src_col);

        if (src_row < src.nrows() && src_col + kPacketItems <= src.ncols() &&
            reinterpret_cast<std::uintptr_t>(packet_src) % kPacketBytes == 0) {
            __pipeline_memcpy_async(&tile[first], packet_src, kPacketBytes);
        } else {
#pragma unroll
            for (unsigned int i = 0; i < kPacketItems; ++i) {
                const bool in_bounds = src_row < src.nrows() && src_col + i < src.ncols();
                tile[first + i] = in_bounds ? src(src_row, src_col + i) : half{};
            }
        }
    }
}

// Умножение C = A * B на тензорных ядрах (CUDA WMMA, half).
// Блок из 8 варпов вычисляет тайл C размером BlockItemsY x BlockItemsX. Внутренняя размерность k
// проходится полосами ширины BlockItemsK: на каждом шаге нити блока совместно загружают в
// разделяемую память тайл A (BlockItemsY x BlockItemsK) и тайл B (BlockItemsK x BlockItemsX),
// после чего каждый варп накапливает в регистрах свою часть тайла C операциями mma_sync.
// Фрагмент, лежащий в матрице C целиком, записывается в неё напрямую, остальные — через буфер
// в разделяемой памяти с проверкой границ, поэтому размеры могут быть не кратны 16.
// Кёрнел ожидает blockDim = (kWmmaBlockThreads, 1).
template <unsigned int BlockItemsY, unsigned int BlockItemsX, unsigned int BlockItemsK>
__global__ void kernel_matmul_wmma(MatrixView<half> a, MatrixView<half> b, MatrixView<half> c) {
    namespace wmma = nvcuda::wmma;

    constexpr unsigned int kWarpItemsY = BlockItemsY / kWmmaWarpRows;
    constexpr unsigned int kWarpItemsX = BlockItemsX / kWmmaWarpCols;
    constexpr unsigned int kFragsY = kWarpItemsY / kWmmaFragDim;
    constexpr unsigned int kFragsX = kWarpItemsX / kWmmaFragDim;

    using FragmentA = wmma::fragment<wmma::matrix_a, kWmmaFragDim, kWmmaFragDim, kWmmaFragDim,
                                     half, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, kWmmaFragDim, kWmmaFragDim, kWmmaFragDim,
                                     half, wmma::row_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, kWmmaFragDim, kWmmaFragDim, kWmmaFragDim,
                                     half>;

    __shared__ alignas(16) half tile_a[BlockItemsY][BlockItemsK];
    __shared__ alignas(16) half tile_b[BlockItemsK][BlockItemsX];
    __shared__ alignas(16) half frag_c[kWmmaWarps][kWmmaFragDim][kWmmaFragDim];

    // Варпы занимают решётку по столбцам: Warp0 и Warp1 — первый столбец (рис. 1).
    const unsigned int warp = threadIdx.x / kWarpSize;
    const unsigned int warp_y = warp % kWmmaWarpRows * kWarpItemsY;
    const unsigned int warp_x = warp / kWmmaWarpRows * kWarpItemsX;
    const std::size_t block_row = static_cast<std::size_t>(blockIdx.y) * BlockItemsY;
    const std::size_t block_col = static_cast<std::size_t>(blockIdx.x) * BlockItemsX;

    FragmentC acc[kFragsY][kFragsX];
#pragma unroll
    for (unsigned int fy = 0; fy < kFragsY; ++fy) {
#pragma unroll
        for (unsigned int fx = 0; fx < kFragsX; ++fx) {
            wmma::fill_fragment(acc[fy][fx], half{});
        }
    }

    for (std::size_t strip = 0; strip < a.ncols(); strip += BlockItemsK) {
        wmma_load_tile<BlockItemsY, BlockItemsK>(&tile_a[0][0], a, block_row, strip);
        wmma_load_tile<BlockItemsK, BlockItemsX>(&tile_b[0][0], b, strip, block_col);
        __pipeline_commit();
        __pipeline_wait_prior(0);

        // Тайлы должны быть полностью загружены до начала вычислений.
        __syncthreads();

#pragma unroll
        for (unsigned int p = 0; p < BlockItemsK; p += kWmmaFragDim) {
            // Фрагмент A умножается на все фрагменты B варпа и наоборот, поэтому каждый
            // из них загружается в регистры один раз.
            FragmentA frag_a[kFragsY];
            FragmentB frag_b[kFragsX];
#pragma unroll
            for (unsigned int fy = 0; fy < kFragsY; ++fy) {
                wmma::load_matrix_sync(frag_a[fy], &tile_a[warp_y + fy * kWmmaFragDim][p],
                                       BlockItemsK);
            }
#pragma unroll
            for (unsigned int fx = 0; fx < kFragsX; ++fx) {
                wmma::load_matrix_sync(frag_b[fx], &tile_b[p][warp_x + fx * kWmmaFragDim],
                                       BlockItemsX);
            }
#pragma unroll
            for (unsigned int fy = 0; fy < kFragsY; ++fy) {
#pragma unroll
                for (unsigned int fx = 0; fx < kFragsX; ++fx) {
                    wmma::mma_sync(acc[fy][fx], frag_a[fy], frag_b[fx], acc[fy][fx]);
                }
            }
        }

        // Тайлы нельзя перезаписывать, пока все варпы блока не закончат их чтение.
        __syncthreads();
    }

    // Условие прямой записи одинаково для всех нитей варпа: зависит только от координат варпа.
    const bool stride_aligned = c.ncols() % (kWmmaStoreAlignment / sizeof(half)) == 0;
#pragma unroll
    for (unsigned int fy = 0; fy < kFragsY; ++fy) {
#pragma unroll
        for (unsigned int fx = 0; fx < kFragsX; ++fx) {
            const std::size_t row = block_row + warp_y + fy * kWmmaFragDim;
            const std::size_t col = block_col + warp_x + fx * kWmmaFragDim;
            half* frag_dst = &c(row, col);

            if (stride_aligned && row + kWmmaFragDim <= c.nrows() &&
                col + kWmmaFragDim <= c.ncols() &&
                reinterpret_cast<std::uintptr_t>(frag_dst) % kWmmaStoreAlignment == 0) {
                wmma::store_matrix_sync(frag_dst, acc[fy][fx],
                                        static_cast<unsigned int>(c.ncols()),
                                        wmma::mem_row_major);
                continue;
            }

            // Фрагмент выходит за границы C или строка C не выровнена: запись идёт через
            // разделяемую память. Варп читает только записанный им самим фрагмент, поэтому
            // барьер не требуется.
            half* buffer = &frag_c[warp][0][0];
            wmma::store_matrix_sync(buffer, acc[fy][fx], kWmmaFragDim, wmma::mem_row_major);
#pragma unroll
            for (unsigned int i = 0; i < kWmmaFragItems / kWarpSize; ++i) {
                const unsigned int item =
                    (threadIdx.x % kWarpSize) * (kWmmaFragItems / kWarpSize) + i;
                const std::size_t item_row = row + item / kWmmaFragDim;
                const std::size_t item_col = col + item % kWmmaFragDim;
                if (item_row < c.nrows() && item_col < c.ncols()) {
                    c(item_row, item_col) = buffer[item];
                }
            }
        }
    }
}

constexpr bool is_power_of_two(unsigned int value) {
    return value != 0 && (value & (value - 1)) == 0;
}

// Размер сетки, покрывающей матрицу c прямоугольниками block_y x block_x.
template <typename AtomT>
dim3 matmul_grid(const MatrixView<AtomT>& c, unsigned int block_y, unsigned int block_x) {
    return dim3(static_cast<unsigned int>((c.ncols() + block_x - 1) / block_x),
                static_cast<unsigned int>((c.nrows() + block_y - 1) / block_y));
}

// Асинхронный запуск kernel_matmul_naive; синхронизацию выполняет вызывающая сторона.
template <typename AtomT>
void launch_matmul_naive(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c,
                         cudaStream_t stream = nullptr) {
    if (c.size() == 0) {
        return;  // Пустая сетка — недопустимая конфигурация запуска.
    }

    const dim3 block(kMatmulBlockDim, kMatmulBlockDim);
    const dim3 grid = matmul_grid(c, kMatmulBlockDim, kMatmulBlockDim);
    kernel_matmul_naive<<<grid, block, 0, stream>>>(a, b, c);
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
    const dim3 grid = matmul_grid(c, TileDim, TileDim);
    kernel_matmul_shmem<AtomT, TileDim><<<grid, block, 0, stream>>>(a, b, c);
    check_cuda(cudaGetLastError(), "kernel_matmul_shmem launch");
}

// Асинхронный запуск kernel_matmul_wmma; синхронизацию выполняет вызывающая сторона.
template <unsigned int BlockItemsY = kWmmaBlockItemsY, unsigned int BlockItemsX = kWmmaBlockItemsX,
          unsigned int BlockItemsK = kWmmaBlockItemsK>
void launch_matmul_wmma(MatrixView<half> a, MatrixView<half> b, MatrixView<half> c,
                        cudaStream_t stream = nullptr) {
    static_assert(BlockItemsY % (kWmmaWarpRows * kWmmaFragDim) == 0,
                  "BlockItemsY must be split evenly between warp rows into 16 x 16 fragments");
    static_assert(BlockItemsX % (kWmmaWarpCols * kWmmaFragDim) == 0,
                  "BlockItemsX must be split evenly between warp columns into 16 x 16 fragments");
    static_assert(BlockItemsK > 0 && BlockItemsK % kWmmaFragDim == 0,
                  "BlockItemsK must be split into 16 x 16 fragments");
    static_assert((BlockItemsY * BlockItemsK) % kWmmaBlockThreads == 0 &&
                      (BlockItemsK * BlockItemsX) % kWmmaBlockThreads == 0,
                  "Tiles must be split evenly between the threads of a block");
    static_assert(is_power_of_two(BlockItemsY * BlockItemsK / kWmmaBlockThreads) &&
                      is_power_of_two(BlockItemsK * BlockItemsX / kWmmaBlockThreads),
                  "Threads must copy packets of 4, 8 or 16 bytes");
    static_assert(sizeof(half) * (BlockItemsK * (BlockItemsY + BlockItemsX) +
                                  kWmmaWarps * kWmmaFragItems) <= 48 * 1024,
                  "Tiles exceed the shared memory limit of 48 KB per block");

    if (c.size() == 0) {
        return;  // Пустая сетка — недопустимая конфигурация запуска.
    }

    const dim3 grid = matmul_grid(c, BlockItemsY, BlockItemsX);
    kernel_matmul_wmma<BlockItemsY, BlockItemsX, BlockItemsK>
        <<<grid, kWmmaBlockThreads, 0, stream>>>(a, b, c);
    check_cuda(cudaGetLastError(), "kernel_matmul_wmma launch");
}
