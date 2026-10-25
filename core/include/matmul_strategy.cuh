#pragma once

#include <cuda_runtime.h>
#include <memory>

#include "kernel.cuh"
#include "matrix_view.cuh"

// Стратегия умножения матриц на устройстве (паттерн «Стратегия»).
// Реализация получает уже согласованные по размерам представления A (m x k), B (k x n)
// и C (m x n) и асинхронно ставит вычисление C = A * B в поток stream.
template <typename AtomT>
class MatmulStrategy {
public:
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

// Стратегия, используемая Matrix::operator* по умолчанию.
template <typename AtomT>
std::shared_ptr<const MatmulStrategy<AtomT>> default_matmul_strategy() {
    static const auto strategy = std::make_shared<const SharedMemoryMatmul<AtomT>>();
    return strategy;
}
