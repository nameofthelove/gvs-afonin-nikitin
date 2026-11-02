#pragma once

#include <Eigen/Dense>

#include "matrix.cuh"

// Eigen::MatrixXf хранит элементы по столбцам (column-major), Matrix — по строкам (row-major),
// поэтому обмен данными идёт через промежуточную row-major матрицу Eigen.
template <typename AtomT>
using RowMajorMatrix = Eigen::Matrix<AtomT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Эталон для half: совпадает с Matrix<half> по порядку хранения и формату элемента (binary16).
using HalfMatrix = RowMajorMatrix<Eigen::half>;

static_assert(sizeof(Eigen::half) == sizeof(half), "half types must share the binary16 layout");

template <typename AtomT>
Matrix<AtomT> to_device(const Eigen::Matrix<AtomT, Eigen::Dynamic, Eigen::Dynamic>& host) {
    const RowMajorMatrix<AtomT> row_major = host;
    Matrix<AtomT> device(static_cast<std::size_t>(host.rows()),
                         static_cast<std::size_t>(host.cols()));
    device.data().copy_from_host(row_major.data());
    return device;
}

template <typename AtomT>
Eigen::Matrix<AtomT, Eigen::Dynamic, Eigen::Dynamic> to_host(const Matrix<AtomT>& device) {
    RowMajorMatrix<AtomT> row_major(static_cast<Eigen::Index>(device.nrows()),
                                    static_cast<Eigen::Index>(device.ncols()));
    device.data().copy_to_host(row_major.data());
    return row_major;
}

inline Matrix<half> to_device(const HalfMatrix& host) {
    Matrix<half> device(static_cast<std::size_t>(host.rows()),
                        static_cast<std::size_t>(host.cols()));
    device.data().copy_from_host(reinterpret_cast<const half*>(host.data()));
    return device;
}

inline HalfMatrix to_host(const Matrix<half>& device) {
    HalfMatrix host(static_cast<Eigen::Index>(device.nrows()),
                    static_cast<Eigen::Index>(device.ncols()));
    device.data().copy_to_host(reinterpret_cast<half*>(host.data()));
    return host;
}
