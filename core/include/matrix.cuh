#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

#include "cuda_check.cuh"
#include "data.cuh"
#include "matmul_strategy.cuh"
#include "matrix_view.cuh"

// Фасад над данными на устройстве.
// Агрегация: Data разделяется между копиями Matrix через std::shared_ptr.
// Композиция: каждый экземпляр Matrix владеет собственным MatrixView.
// Алгоритм умножения задаётся объектом-стратегией MatmulStrategy (по умолчанию для half —
// тензорные ядра, для остальных типов — блочный алгоритм с разделяемой памятью);
// стратегия разделяется между матрицами.
template <typename AtomT>
class Matrix {
public:
    using Strategy = MatmulStrategy<AtomT>;

private:
    std::shared_ptr<Data<AtomT>> data_;
    MatrixView<AtomT> view_;
    std::shared_ptr<const Strategy> matmul_strategy_;

public:
    Matrix(std::size_t nrows, std::size_t ncols,
           std::shared_ptr<const Strategy> matmul_strategy = default_matmul_strategy<AtomT>())
        : data_(std::make_shared<Data<AtomT>>(nrows * ncols)),
          view_(data_->data(), nrows, ncols),
          matmul_strategy_(std::move(matmul_strategy)) {
        if (!matmul_strategy_) {
            throw std::invalid_argument("Matrix: matmul strategy must not be null");
        }
    }

    std::size_t size() const { return view_.size(); }
    std::size_t nrows() const { return view_.nrows(); }
    std::size_t ncols() const { return view_.ncols(); }

    Data<AtomT>& data() { return *data_; }
    const Data<AtomT>& data() const { return *data_; }

    MatrixView<AtomT>& view() { return view_; }
    const MatrixView<AtomT>& view() const { return view_; }

    const Strategy& matmul_strategy() const { return *matmul_strategy_; }

    void set_matmul_strategy(std::shared_ptr<const Strategy> matmul_strategy) {
        if (!matmul_strategy) {
            throw std::invalid_argument("Matrix: matmul strategy must not be null");
        }
        matmul_strategy_ = std::move(matmul_strategy);
    }

    // C = (*this) * other, где (*this) — m x k, other — k x n, C — m x n.
    // Умножение выполняет стратегия левого операнда; результат наследует её.
    Matrix operator*(const Matrix& other) const {
        if (ncols() != other.nrows()) {
            throw std::invalid_argument("Matrix dimensions mismatch: lhs.ncols != rhs.nrows");
        }

        Matrix result(nrows(), other.ncols(), matmul_strategy_);
        matmul_strategy_->multiply(view_, other.view_, result.view_);
        check_cuda(cudaDeviceSynchronize(), "matmul execution");
        return result;
    }
};
