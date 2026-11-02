#pragma once

#include <ATen/AccumulateType.h>
#include <torch/types.h>

namespace hsys::nn::functional::kernels {

template <class T>
using view_4d_t = torch::PackedTensorAccessor32<T, 4, torch::RestrictPtrTraits>;

// Число выходных каналов, которые блок вычисляет над одним загруженным тайлом входа:
// тайл читается из глобальной памяти один раз на группу каналов, а не на каждый канал.
inline constexpr long kOutChannelsPerBlock = 8;

// Размер выхода по одной оси, формулы (3)–(4) задания. При отрицательном числителе выход
// пуст: целочисленное деление в C++ округляет к нулю, а не вниз, и дало бы размер 1.
__host__ __device__ constexpr long output_size(long input_size, long filter_size, long stride,
                                               long padding, long dilation) {
    const long span = input_size + 2 * padding - dilation * (filter_size - 1) - 1;
    return span < 0 ? 0 : span / stride + 1;
}

// Сторона тайла входа для тайла выхода block_size x block_size: крайний выход тайла читает
// вход со смещением (block_size - 1) * stride + (filter_size - 1) * dilation от первого.
__host__ __device__ constexpr long input_tile_size(long filter_size, long stride, long dilation,
                                                   long block_size) {
    return (block_size - 1) * stride + (filter_size - 1) * dilation + 1;
}

// Пакетная многоканальная свёртка (кросс-корреляция), формулы (1)–(2) задания.
//
// Блок block_size x block_size нитей вычисляет тайл выхода того же размера для изображения n
// и группы из kOutChannelsPerBlock выходных каналов; нить отвечает за один пиксель тайла во
// всех каналах группы и держит суммы в регистрах. Для каждого входного канала блок загружает
// в разделяемую память тайл входа (с нулями вместо отступа) и веса группы, после чего каждый
// элемент тайла переиспользуется filter_size^2 соседними нитями и kOutChannelsPerBlock каналами.
// Параметры фильтра — константы времени компиляции, поэтому циклы по фильтру развёрнуты,
// а индексы в тайле вычисляются без умножений во время выполнения.
//
// Ожидается blockDim = (block_size, block_size), gridDim = (ceil(W_out / block_size),
// ceil(H_out / block_size), Z) при любом Z >= 1: пары (n, группа каналов) обходятся с шагом
// gridDim.z. Значения накапливаются в at::acc_type (float для half и bfloat16).
// __launch_bounds__ ограничивает число регистров так, чтобы блок из block_size^2 нитей
// помещался на SM: без него при block_size = 32 запуск завершается ошибкой
// cudaErrorLaunchOutOfResources.
template <class AtomT, long filter_size, long stride = 1, long padding = 0, long dilation = 1,
          long block_size = 16>
__global__ void __launch_bounds__(block_size * block_size) conv2d(view_4d_t<AtomT> output, const view_4d_t<AtomT> input,
                       const view_4d_t<AtomT> weights) {
    static_assert(filter_size > 0 && stride > 0 && padding >= 0 && dilation > 0,
                  "Invalid convolution parameters");
    static_assert(block_size > 0 && block_size * block_size <= 1024,
                  "Block must contain between 1 and 1024 threads");

    using AccT = at::acc_type<AtomT, /*is_cuda=*/true>;

    constexpr int kTile = input_tile_size(filter_size, stride, dilation, block_size);
    constexpr int kFilterArea = filter_size * filter_size;
    constexpr int kThreads = block_size * block_size;

    static_assert(sizeof(AccT) * (kTile * kTile + kOutChannelsPerBlock * kFilterArea) <=
                      48 * 1024,
                  "Tiles do not fit into 48 KiB of static shared memory");

    __shared__ AccT tile[kTile][kTile];
    __shared__ AccT filters[kOutChannelsPerBlock][filter_size][filter_size];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * block_size + tx;

    const int h_out = blockIdx.y * block_size + ty;
    const int w_out = blockIdx.x * block_size + tx;

    // Левый верхний угол тайла во входе; отрицателен в первом тайле при padding > 0.
    const int h_origin = static_cast<int>(blockIdx.y * block_size * stride) - padding;
    const int w_origin = static_cast<int>(blockIdx.x * block_size * stride) - padding;

    const int in_channels = input.size(1);
    const int out_channels = output.size(1);
    const int channel_groups = (out_channels + kOutChannelsPerBlock - 1) / kOutChannelsPerBlock;
    const int work_items = output.size(0) * channel_groups;

    // Условие цикла одинаково для всех нитей блока, поэтому барьеры внутри него безопасны.
    for (int item = blockIdx.z; item < work_items; item += gridDim.z) {
        const int n = item / channel_groups;
        const int c_first = item % channel_groups * kOutChannelsPerBlock;

        AccT acc[kOutChannelsPerBlock] = {};

        for (int c_in = 0; c_in < in_channels; ++c_in) {
            // Соседние нити загружают соседние элементы строки тайла: обращения объединяются.
            // Вне изображения записывается ноль — это и есть X^ из формулы (2).
            for (int i = tid; i < kTile * kTile; i += kThreads) {
                const int h = h_origin + i / kTile;
                const int w = w_origin + i % kTile;
                const bool inside = 0 <= h && h < input.size(2) && 0 <= w && w < input.size(3);
                tile[i / kTile][i % kTile] =
                    inside ? static_cast<AccT>(input[n][c_in][h][w]) : AccT{};
            }

            // Веса отсутствующих каналов последней группы заменяются нулями.
            for (int i = tid; i < kOutChannelsPerBlock * kFilterArea; i += kThreads) {
                const int c = i / kFilterArea;
                const int r = i % kFilterArea / filter_size;
                const int q = i % filter_size;
                filters[c][r][q] = c_first + c < out_channels
                                       ? static_cast<AccT>(weights[c_first + c][c_in][r][q])
                                       : AccT{};
            }
            __syncthreads();

            // Все нити варпа читают один и тот же вес — широковещательное чтение без конфликтов.
#pragma unroll
            for (int r = 0; r < filter_size; ++r) {
#pragma unroll
                for (int q = 0; q < filter_size; ++q) {
                    const AccT x = tile[ty * stride + r * dilation][tx * stride + q * dilation];
#pragma unroll
                    for (int c = 0; c < kOutChannelsPerBlock; ++c) {
                        acc[c] += x * filters[c][r][q];
                    }
                }
            }
            __syncthreads();
        }

        // Нити за пределами выхода участвовали только в загрузке тайла.
        if (h_out < output.size(2) && w_out < output.size(3)) {
#pragma unroll
            for (int c = 0; c < kOutChannelsPerBlock; ++c) {
                if (c_first + c < out_channels) {
                    output[n][c_first + c][h_out][w_out] = static_cast<AtomT>(acc[c]);
                }
            }
        }
    }
}

}  // namespace hsys::nn::functional::kernels
