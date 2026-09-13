#pragma once
#include <cstddef>
#include <cuda_runtime.h>

template <typename AtomT>
class VectorView {
private:
	AtomT* data_;
	size_t size_;
public:
	__host__ __device__ VectorView(AtomT* data, size_t size) : data_(data), size_(size) {}
	__host__ __device__ AtomT& operator[](std::size_t n) { return data_[n]; }
	__host__ __device__ const AtomT& operator[](std::size_t n) const { return data_[n]; }
	__host__ __device__ AtomT& operator()(std::size_t i) { return data_[i]; }
	__host__ __device__ const AtomT& operator()(std::size_t i) const { return data_[i]; }
};