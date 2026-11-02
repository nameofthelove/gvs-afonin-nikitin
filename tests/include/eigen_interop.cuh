#pragma once

#include <Eigen/Dense>

#include "vector.cuh"

template <typename AtomT>
using EigenVector = Eigen::Matrix<AtomT, Eigen::Dynamic, 1>;

// Копирует вектор Eigen на устройство.
template <typename AtomT, typename SumT = AtomT>
Vector<AtomT, SumT> to_device(const EigenVector<AtomT>& host) {
    Vector<AtomT, SumT> device(static_cast<std::size_t>(host.size()));
    device.data().copy_from_host(host.data());
    return device;
}
