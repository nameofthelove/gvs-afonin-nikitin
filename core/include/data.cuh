#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <utility>

#include "cuda_check.cuh"

// Владеющий RAII-буфер в глобальной памяти устройства.
template <typename AtomT>
class Data {
private:
    std::size_t size_;
    AtomT* data_;

public:
    explicit Data(std::size_t size) : size_(size), data_(nullptr) {
        if (size_ > 0) {
            check_cuda(cudaMalloc(&data_, size_ * sizeof(AtomT)), "cudaMalloc");
        }
    }

    ~Data() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    Data(const Data& other) : Data(other.size_) { //cop
        if (size_ > 0) {
            // При исключении память освободит деструктор делегирующего конструктора.
            check_cuda(cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT),
                                  cudaMemcpyDeviceToDevice),
                       "cudaMemcpy (D2D)");
        }
    }

    Data(Data&& other) noexcept
        : size_(std::exchange(other.size_, 0)), data_(std::exchange(other.data_, nullptr)) {}

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
            size_ = std::exchange(other.size_, 0);
            data_ = std::exchange(other.data_, nullptr);
        }
        return *this;
    }

    void copy_to_host(AtomT* host_ptr) const {
        if (size_ == 0) {
            return;
        }
        check_cuda(cudaMemcpy(host_ptr, data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToHost),
                   "cudaMemcpy (D2H)");
    }

    void copy_from_host(const AtomT* host_ptr) {
        if (size_ == 0) {
            return;
        }
        check_cuda(cudaMemcpy(data_, host_ptr, size_ * sizeof(AtomT), cudaMemcpyHostToDevice),
                   "cudaMemcpy (H2D)");
    }

    AtomT* data() { return data_; }
    const AtomT* data() const { return data_; }
    std::size_t size() const { return size_; }
};
