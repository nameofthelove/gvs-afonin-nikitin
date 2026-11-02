#pragma once

#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "vector_view.cuh"

inline constexpr unsigned int kWarpSize = 32;

// Маска __shfl_down_sync: в обмене участвуют все нити варпа.
inline constexpr unsigned int kWarpFullMask = 0xFFFFFFFFu;

// Геометрия по умолчанию: блок из 256 нитей, не более 120 блоков (все слоты 20 SM по 6 блоков).
// На больших векторах геометрия почти не влияет на время: его ограничивает полоса памяти.
inline constexpr unsigned int kVecredBlockDim = 256;
inline constexpr unsigned int kVecredMaxBlocks = 120;

constexpr bool is_power_of_two(unsigned int value) {
    return value != 0 && (value & (value - 1)) == 0;
}

// Чтение: блок обрабатывает непрерывный участок вектора длиной chunk, нить блока складывает
// каждый BlockDim-й элемент участка. Соседние нити варпа читают соседние элементы, поэтому
// их обращения объединяются. Нить, которой не досталось элементов, возвращает ноль.
template <typename AtomT, typename SumT, unsigned int BlockDim>
__device__ __forceinline__ SumT vecred_thread_sum(const VectorView<AtomT>& in) {
    const std::size_t chunk = (in.size() + gridDim.x - 1) / gridDim.x;
    const std::size_t begin = blockIdx.x * chunk;
    const std::size_t end = begin + chunk < in.size() ? begin + chunk : in.size();

    SumT acc{};
    for (std::size_t i = begin + threadIdx.x; i < end; i += BlockDim) {
        acc += static_cast<SumT>(in[i]);
    }
    return acc;
}

// Редукция блока через разделяемую память: на каждом шаге первая половина оставшихся нитей
// прибавляет значения второй половины, итого log2(BlockDim) шагов с барьером после каждого.
// partial указывает на массив из BlockDim элементов; сумма блока возвращается в нити 0.
template <typename SumT, unsigned int BlockDim>
__device__ __forceinline__ SumT block_reduce_shared(SumT* partial, SumT value) {
    const unsigned int tid = threadIdx.x;

    partial[tid] = value;
    __syncthreads();

#pragma unroll
    for (unsigned int stride = BlockDim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    return partial[0];
}

// Редукция варпа широковещанием регистров: сумма 32 значений оказывается в нити 0 варпа за
// log2(32) = 5 шагов. Нити с lane + offset >= 32 получают от __shfl_down_sync своё же значение,
// но в дерево суммирования нити 0 они не входят: на шаге offset участвуют только кратные 2*offset.
template <typename SumT>
__device__ __forceinline__ SumT warp_reduce(SumT value) {
#pragma unroll
    for (unsigned int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(kWarpFullMask, value, offset);
    }
    return value;
}

// Редукция блока на уровне варпов: каждый варп сворачивает значения нитей в регистрах и кладёт
// сумму в разделяемую память, после барьера первый варп сворачивает суммы варпов так же.
// warp_sums указывает на массив из BlockDim / 32 элементов; сумма блока возвращается в нити 0.
template <typename SumT, unsigned int BlockDim>
__device__ __forceinline__ SumT block_reduce_warp(SumT* warp_sums, SumT value) {
    constexpr unsigned int kWarps = BlockDim / kWarpSize;

    const unsigned int warp = threadIdx.x / kWarpSize;
    const unsigned int lane = threadIdx.x % kWarpSize;

    const SumT warp_sum = warp_reduce(value);
    if (lane == 0) {
        warp_sums[warp] = warp_sum;
    }
    __syncthreads();

    SumT block_sum{};
    if (warp == 0) {
        block_sum = warp_reduce(lane < kWarps ? warp_sums[lane] : SumT{});
    }
    return block_sum;
}

// Суммирование без широковещания: частичные суммы нитей сворачиваются в разделяемой памяти,
// нить 0 записывает сумму блока в out[blockIdx.x]. Вектор разбивается на gridDim.x участков.
// Кёрнел ожидает blockDim = (BlockDim, 1).
template <typename AtomT, typename SumT, unsigned int BlockDim>
__global__ void kernel_vecred_nobr(VectorView<AtomT> in, VectorView<SumT> out) {
    __shared__ SumT partial[BlockDim];

    const SumT value = vecred_thread_sum<AtomT, SumT, BlockDim>(in);
    const SumT block_sum = block_reduce_shared<SumT, BlockDim>(partial, value);

    if (threadIdx.x == 0) {
        out[blockIdx.x] = block_sum;
    }
}

// Суммирование с широковещанием регистров: варпы сворачивают значения функцией
// __shfl_down_sync, в разделяемой памяти хранятся только суммы варпов.
// Кёрнел ожидает blockDim = (BlockDim, 1).
template <typename AtomT, typename SumT, unsigned int BlockDim>
__global__ void kernel_vecred_br(VectorView<AtomT> in, VectorView<SumT> out) {
    __shared__ SumT warp_sums[BlockDim / kWarpSize];

    const SumT value = vecred_thread_sum<AtomT, SumT, BlockDim>(in);
    const SumT block_sum = block_reduce_warp<SumT, BlockDim>(warp_sums, value);

    if (threadIdx.x == 0) {
        out[blockIdx.x] = block_sum;
    }
}

// Число блоков первого прохода: блок читает не меньше BlockDim элементов, если их хватает.
inline unsigned int vecred_grid(std::size_t size, unsigned int block_dim, unsigned int max_blocks) {
    return static_cast<unsigned int>(std::min<std::size_t>((size + block_dim - 1) / block_dim,
                                                           max_blocks));
}

template <unsigned int BlockDim, unsigned int MaxBlocks>
constexpr void check_vecred_geometry() {
    static_assert(is_power_of_two(BlockDim) && BlockDim >= kWarpSize && BlockDim <= 1024,
                  "BlockDim must be a power of two between 32 and 1024");
    static_assert(MaxBlocks > 0 && MaxBlocks <= BlockDim,
                  "Partial sums of the second pass must fit into a single block");
}

// Асинхронная двухпроходная редукция без широковещания; синхронизацию выполняет вызывающая
// сторона. Первый проход записывает суммы блоков в partial, второй одним блоком сворачивает их
// на месте: чтение завершается раньше записи partial[0], поэтому наложение буферов безопасно.
// Результат — partial[0]; вместимость partial — не меньше MaxBlocks.
template <typename AtomT, typename SumT, unsigned int BlockDim = kVecredBlockDim,
          unsigned int MaxBlocks = kVecredMaxBlocks>
void launch_vecred_nobr(VectorView<AtomT> in, VectorView<SumT> partial,
                        cudaStream_t stream = nullptr) {
    check_vecred_geometry<BlockDim, MaxBlocks>();

    if (in.size() == 0) {
        return;  // Пустая сетка — недопустимая конфигурация запуска.
    }

    const unsigned int grid = vecred_grid(in.size(), BlockDim, MaxBlocks);
    kernel_vecred_nobr<AtomT, SumT, BlockDim><<<grid, BlockDim, 0, stream>>>(in, partial);
    check_cuda(cudaGetLastError(), "kernel_vecred_nobr launch");

    if (grid > 1) {
        const VectorView<SumT> block_sums(&partial[0], grid);
        kernel_vecred_nobr<SumT, SumT, BlockDim><<<1, BlockDim, 0, stream>>>(block_sums, partial);
        check_cuda(cudaGetLastError(), "kernel_vecred_nobr launch");
    }
}

// Асинхронная двухпроходная редукция с широковещанием; контракт как у launch_vecred_nobr.
template <typename AtomT, typename SumT, unsigned int BlockDim = kVecredBlockDim,
          unsigned int MaxBlocks = kVecredMaxBlocks>
void launch_vecred_br(VectorView<AtomT> in, VectorView<SumT> partial,
                      cudaStream_t stream = nullptr) {
    check_vecred_geometry<BlockDim, MaxBlocks>();

    if (in.size() == 0) {
        return;  // Пустая сетка — недопустимая конфигурация запуска.
    }

    const unsigned int grid = vecred_grid(in.size(), BlockDim, MaxBlocks);
    kernel_vecred_br<AtomT, SumT, BlockDim><<<grid, BlockDim, 0, stream>>>(in, partial);
    check_cuda(cudaGetLastError(), "kernel_vecred_br launch");

    if (grid > 1) {
        const VectorView<SumT> block_sums(&partial[0], grid);
        kernel_vecred_br<SumT, SumT, BlockDim><<<1, BlockDim, 0, stream>>>(block_sums, partial);
        check_cuda(cudaGetLastError(), "kernel_vecred_br launch");
    }
}
