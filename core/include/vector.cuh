#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

#include "data.cuh"
#include "kernel.cuh"
#include "vector_view.cuh"

template <typename AtomT>
class Vector {
private:
    std::shared_ptr<Data<AtomT>> data_;
    VectorView<AtomT> view_;

public:
    explicit Vector(std::size_t size)
        : data_(std::make_shared<Data<AtomT>>(size)),
          view_(data_->data(), size) {}

    std::size_t size() const { return view_.size(); }

    Data<AtomT>& data() { return *data_; }
    const Data<AtomT>& data() const { return *data_; }

    VectorView<AtomT>& view() { return view_; }
    const VectorView<AtomT>& view() const { return view_; }

    Vector operator+(const Vector& other) const {
        if (size() != other.size()) {
            throw std::invalid_argument("Vector dimensions must match for addition");
        }

        Vector result(size());
        launch_vecadd(view_, other.view_, result.view_);

        const cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaDeviceSynchronize failed after vecadd");
        }
        return result;
    }
};
