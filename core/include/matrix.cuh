#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

#include "cuda_check.cuh"
#include "data.cuh"
#include "kernel.cuh"
#include "matrix_view.cuh"

// Фасад над данными на устройстве.
// Агрегация: Data разделяется между копиями Matrix через std::shared_ptr.
// Композиция: каждый экземпляр Matrix владеет собственным MatrixView.
template <typename AtomT>
class Matrix {
private:
    std::shared_ptr<Data<AtomT>> data_;
    MatrixView<AtomT> view_;

public:
    Matrix(std::size_t nrows, std::size_t ncols)
        : data_(std::make_shared<Data<AtomT>>(nrows * ncols)),
          view_(data_->data(), nrows, ncols) {}

    std::size_t size() const { return view_.size(); }
    std::size_t nrows() const { return view_.nrows(); }
    std::size_t ncols() const { return view_.ncols(); }

    Data<AtomT>& data() { return *data_; }
    const Data<AtomT>& data() const { return *data_; }

    MatrixView<AtomT>& view() { return view_; }
    const MatrixView<AtomT>& view() const { return view_; }

    // C = (*this) * other, где (*this) — m x k, other — k x n, C — m x n.
    Matrix operator*(const Matrix& other) const {
        if (ncols() != other.nrows()) {
            throw std::invalid_argument("Matrix dimensions mismatch: lhs.ncols != rhs.nrows");
        }

        Matrix result(nrows(), other.ncols());
        launch_matmul_naive(view_, other.view_, result.view_);
        check_cuda(cudaDeviceSynchronize(), "kernel_matmul_naive execution");
        return result;
    }
};
