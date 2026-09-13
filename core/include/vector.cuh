#pragma once
#include "data.cuh"
#include "vector_view.cuh"
#include "kernel.cuh"
#include <memory>
#include <cstddef>
#include <stdexcept>

template <typename AtomT>
class Vector {
private:
    std::shared_ptr<Data<AtomT>> data_;
    VectorView<AtomT> view_;

public:
    explicit Vector(std::size_t size) 
        : data_(std::make_shared<Data<AtomT>>(size)), view_(data_->data(), size) {}

    size_t size() const { return view_.size(); }
    VectorView<AtomT>& view() { return view_; }
    const VectorView<AtomT>& view() const { return view_; }

    Vector operator+(const Vector& other) const {
        if (this->size() != other.size()) {
            throw std::invalid_argument("Vector dimensions must match for addition");
        }

        Vector result(this->size());

        int threadsPerBlock = 256;
        int blocksPerGrid = (this->size() + threadsPerBlock - 1) / threadsPerBlock;

        // Вызов кёрнела с передачей VectorView по значению
        kernel_vecadd<<<blocksPerGrid, threadsPerBlock>>>(this->view_, other.view_, result.view_);
        cudaDeviceSynchronize();

        return result;
    }
};