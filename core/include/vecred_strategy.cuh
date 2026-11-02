#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <memory>

#include "kernel.cuh"
#include "vector_view.cuh"

// Стратегия суммирования вектора на устройстве (паттерн «Стратегия»).
// Реализация асинхронно ставит вычисление в поток stream и оставляет результат в partial[0].
// AtomT — тип элемента, SumT — тип накопителя (шире AtomT для малоразрядных типов).
template <typename AtomT, typename SumT = AtomT>
class VecredStrategy {
public:
    using Atom = AtomT;
    using Sum = SumT;

    virtual ~VecredStrategy() = default;

    // Вместимость буфера partial, достаточная для вектора любого размера.
    virtual std::size_t partial_capacity() const = 0;

    virtual void reduce(VectorView<AtomT> in, VectorView<SumT> partial,
                        cudaStream_t stream = nullptr) const = 0;
};

// Частичные суммы нитей сворачиваются в разделяемой памяти (kernel_vecred_nobr).
template <typename AtomT, typename SumT = AtomT, unsigned int BlockDim = kVecredBlockDim,
          unsigned int MaxBlocks = kVecredMaxBlocks>
class NoBroadcastVecred final : public VecredStrategy<AtomT, SumT> {
public:
    std::size_t partial_capacity() const override { return MaxBlocks; }

    void reduce(VectorView<AtomT> in, VectorView<SumT> partial,
                cudaStream_t stream = nullptr) const override {
        launch_vecred_nobr<AtomT, SumT, BlockDim, MaxBlocks>(in, partial, stream);
    }
};

// Варпы сворачивают значения широковещанием регистров (kernel_vecred_br).
template <typename AtomT, typename SumT = AtomT, unsigned int BlockDim = kVecredBlockDim,
          unsigned int MaxBlocks = kVecredMaxBlocks>
class BroadcastVecred final : public VecredStrategy<AtomT, SumT> {
public:
    std::size_t partial_capacity() const override { return MaxBlocks; }

    void reduce(VectorView<AtomT> in, VectorView<SumT> partial,
                cudaStream_t stream = nullptr) const override {
        launch_vecred_br<AtomT, SumT, BlockDim, MaxBlocks>(in, partial, stream);
    }
};

// Стратегия, используемая Vector::sum по умолчанию.
template <typename AtomT, typename SumT = AtomT>
std::shared_ptr<const VecredStrategy<AtomT, SumT>> default_vecred_strategy() {
    static const auto strategy = std::make_shared<const BroadcastVecred<AtomT, SumT>>();
    return strategy;
}
