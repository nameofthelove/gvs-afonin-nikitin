#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <type_traits>

// Невладеющее представление матрицы nrows x ncols в линейной памяти (row-major).
// Тривиально копируемо, поэтому передаётся в кёрнелы по значению.
template <typename AtomT>
class MatrixView {
private:
    AtomT* data_;
    std::size_t nrows_;
    std::size_t ncols_;

public:
    __host__ __device__ MatrixView(AtomT* data, std::size_t nrows, std::size_t ncols)
        : data_(data), nrows_(nrows), ncols_(ncols) {}

    __host__ __device__ std::size_t size() const { return nrows_ * ncols_; }
    __host__ __device__ std::size_t nrows() const { return nrows_; }
    __host__ __device__ std::size_t ncols() const { return ncols_; }

    // Доступ по линейному индексу.
    __host__ __device__ AtomT& operator[](std::size_t n) { return data_[n]; }
    __host__ __device__ const AtomT& operator[](std::size_t n) const { return data_[n]; }

    // Доступ по (строка, столбец): элемент (i, j) лежит по смещению i * ncols + j.
    __host__ __device__ AtomT& operator()(std::size_t i, std::size_t j) {
        return data_[i * ncols_ + j];
    }
    __host__ __device__ const AtomT& operator()(std::size_t i, std::size_t j) const {
        return data_[i * ncols_ + j];
    }
};

static_assert(std::is_trivially_copyable_v<MatrixView<float>>,
              "MatrixView must be trivially copyable to be passed to kernels by value");
