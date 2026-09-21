#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

template <typename AtomT>
class Data {
private:
    std::size_t size_;
    AtomT* data_;

public:
    explicit Data(std::size_t size) : size_(size), data_(nullptr) {
        if (size_ > 0) {
            const cudaError_t err = cudaMalloc(&data_, size_ * sizeof(AtomT));
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed");
            }
        }
    }

    ~Data() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    Data(const Data& other) : size_(other.size_), data_(nullptr) {
        if (size_ > 0) {
            const cudaError_t err = cudaMalloc(&data_, size_ * sizeof(AtomT));
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed in copy constructor");
            }
            const cudaError_t copy_err = cudaMemcpy(
                data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice);
            if (copy_err != cudaSuccess) {
                cudaFree(data_);
                data_ = nullptr;
                throw std::runtime_error("cudaMemcpy failed in copy constructor");
            }
        }
    }

    Data(Data&& other) noexcept : size_(other.size_), data_(other.data_) {
        other.size_ = 0;
        other.data_ = nullptr;
    }

    Data& operator=(const Data& other) {
        if (this != &other) {
            Data temp(other);
            std::swap(size_, temp.size_);
            std::swap(data_, temp.data_);
        }
        return *this;
    }

    Data& operator=(Data&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFree(data_);
            }
            size_ = other.size_;
            data_ = other.data_;
            other.size_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    void copy_to_host(AtomT* host_ptr) const {
        if (size_ == 0) {
            return;
        }
        const cudaError_t err = cudaMemcpy(
            host_ptr, data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy (D2H) failed");
        }
    }

    void copy_from_host(const AtomT* host_ptr) {
        if (size_ == 0) {
            return;
        }
        const cudaError_t err = cudaMemcpy(
            data_, host_ptr, size_ * sizeof(AtomT), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy (H2D) failed");
        }
    }

    AtomT* data() { return data_; }
    const AtomT* data() const { return data_; }
    std::size_t size() const { return size_; }
};
