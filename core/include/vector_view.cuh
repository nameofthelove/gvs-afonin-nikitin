#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <type_traits>

// Невладеющее представление вектора в линейной памяти.
// Тривиально копируемо, поэтому передаётся в кёрнелы по значению.
template <typename AtomT>
class VectorView {
private:
    AtomT* data_;
    std::size_t size_;

public:
    __host__ __device__ VectorView(AtomT* data, std::size_t size) : data_(data), size_(size) {}

    __host__ __device__ std::size_t size() const { return size_; }

    __host__ __device__ AtomT& operator[](std::size_t n) { return data_[n]; }
    __host__ __device__ const AtomT& operator[](std::size_t n) const { return data_[n]; }
};

static_assert(std::is_trivially_copyable_v<VectorView<float>>,
              "VectorView must be trivially copyable to be passed to kernels by value");
