#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
using namespace std;

template<typename AtomT>
class Data {
private:
	size_t size_;
	AtomT* data_;
public:
	explicit Data(size_t size) : size_(size), data_(nullptr) {
		if (size_ > 0) {
			cudaError_t err = cudaMalloc(&data_, size_ * sizeof(AtomT));
			if (err != cudaSuccess) {
				throw runtime_error("cudaMalloc failed");
			}
		}
	}
	~Data() {
		if (data_) cudaFree(data_);
	}

	Data(const Data& other) : size_t(size), data_(nullptr) {
		if (size_ > 0) {
			cudaMalloc(&data_, size_ * sizeof(AtomT));
			cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice);
		}
	}

	Data(Data&& other) noexcept : size_(other.size_), data_(other.data_) {
		other.size_ = 0;
		other.data_ = nullptr;
	}

	Data& operator=(const Data& other) {
		if (this != &other) {
			Data temp(other);
			swap(size_, temp.size_);
			swap(data_, temp.data_);
		}
		return *this;
	}

	Data& operator=(Data&& other) noexcept {
		if (this != &other) {
			if (data_) cudaFree(data_);
			size_ = other.size_;
			data_ = other.data_;
			other.size_ = 0;
			other.data_ = nullptr;
		}
		return *this;
	}

	void copy_to_host(AtomT* host_ptr) const {
		cudaMemcpy(host_ptr, data_), size_ * sizeof(AtomT), cudaMemcpyDeviceToHost);
	}

	void copy_from_host(AtomT* host_ptr) const {
		cudaMemcpy(data_, host_ptr, size_ * sizeof(AtomT), cudaMemcpyHostToDevice);
	}

	AtomT* data() { return data_; }
	const AtomT* data() const { return data_; }
	size_t size() const { return size_; }
};