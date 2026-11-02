#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <type_traits>

#include "kernel.cuh"
#include "matrix_view.cuh"

// Стратегия умножения матриц на устройстве (паттерн «Стратегия»).
// Реализация получает уже согласованные по размерам представления A (m x k), B (k x n)
// и C (m x n) и асинхронно ставит вычисление C = A * B в поток stream.
template <typename AtomT>
class MatmulStrategy {
public:
    using Atom = AtomT;

    virtual ~MatmulStrategy() = default;

    virtual void multiply(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c,
                          cudaStream_t stream = nullptr) const = 0;
};

// Наивный алгоритм: каждая нить читает строку A и столбец B из глобальной памяти.
template <typename AtomT>
class NaiveMatmul final : public MatmulStrategy<AtomT> {
public:
    void multiply(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c,
                  cudaStream_t stream = nullptr) const override {
        launch_matmul_naive(a, b, c, stream);
    }
};

// Блочный алгоритм с буферизацией тайлов TileDim x TileDim в разделяемой памяти.
template <typename AtomT, unsigned int TileDim = kMatmulTileDim>
class SharedMemoryMatmul final : public MatmulStrategy<AtomT> {
public:
    void multiply(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c,
                  cudaStream_t stream = nullptr) const override {
        launch_matmul_shmem<AtomT, TileDim>(a, b, c, stream);
    }
};

// Умножение на тензорных ядрах (WMMA): только для half, размеры тайлов задают геометрию блока.
template <unsigned int BlockItemsY = kWmmaBlockItemsY, unsigned int BlockItemsX = kWmmaBlockItemsX,
          unsigned int BlockItemsK = kWmmaBlockItemsK>
class WmmaMatmul final : public MatmulStrategy<half> {
public:
    void multiply(MatrixView<half> a, MatrixView<half> b, MatrixView<half> c,
                  cudaStream_t stream = nullptr) const override {
        launch_matmul_wmma<BlockItemsY, BlockItemsX, BlockItemsK>(a, b, c, stream);
    }
};

// Стратегия, используемая Matrix::operator* по умолчанию: тензорные ядра поддерживают только
// half, остальные типы умножаются блочным алгоритмом с разделяемой памятью.
template <typename AtomT>
std::shared_ptr<const MatmulStrategy<AtomT>> default_matmul_strategy() {
    if constexpr (std::is_same_v<AtomT, half>) {
        static const auto strategy = std::make_shared<const WmmaMatmul<>>();
        return strategy;
    } else {
        static const auto strategy = std::make_shared<const SharedMemoryMatmul<AtomT>>();
        return strategy;
    }
}
