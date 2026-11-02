#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

#include "cuda_check.cuh"
#include "data.cuh"
#include "vecred_strategy.cuh"
#include "vector_view.cuh"

// Фасад над данными на устройстве.
// Агрегация: Data разделяется между копиями Vector через std::shared_ptr.
// Композиция: каждый экземпляр Vector владеет собственным VectorView.
// Алгоритм суммирования задаётся объектом-стратегией VecredStrategy; стратегия разделяется
// между векторами.
template <typename AtomT, typename SumT = AtomT>
class Vector {
public:
    using Strategy = VecredStrategy<AtomT, SumT>;

private:
    std::shared_ptr<Data<AtomT>> data_;
    VectorView<AtomT> view_;
    std::shared_ptr<const Strategy> vecred_strategy_;

public:
    explicit Vector(std::size_t size, std::shared_ptr<const Strategy> vecred_strategy =
                                          default_vecred_strategy<AtomT, SumT>())
        : data_(std::make_shared<Data<AtomT>>(size)),
          view_(data_->data(), size),
          vecred_strategy_(std::move(vecred_strategy)) {
        if (!vecred_strategy_) {
            throw std::invalid_argument("Vector: reduction strategy must not be null");
        }
    }

    std::size_t size() const { return view_.size(); }

    Data<AtomT>& data() { return *data_; }
    const Data<AtomT>& data() const { return *data_; }

    VectorView<AtomT>& view() { return view_; }
    const VectorView<AtomT>& view() const { return view_; }

    const Strategy& vecred_strategy() const { return *vecred_strategy_; }

    void set_vecred_strategy(std::shared_ptr<const Strategy> vecred_strategy) {
        if (!vecred_strategy) {
            throw std::invalid_argument("Vector: reduction strategy must not be null");
        }
        vecred_strategy_ = std::move(vecred_strategy);
    }

    // Сумма элементов; для пустого вектора — SumT{}.
    SumT sum() const {
        if (size() == 0) {
            return SumT{};
        }

        Data<SumT> partial(vecred_strategy_->partial_capacity());
        const VectorView<SumT> partial_view(partial.data(), partial.size());
        vecred_strategy_->reduce(view_, partial_view);

        SumT result{};
        partial.copy_to_host(&result, 1);
        return result;
    }
};
