#include "matrix.cuh"

// Явные инстанцирования: проверяют компилируемость шаблонов для основных типов.
template class Data<float>;
template class Data<double>;
template class Data<int>;

template class MatrixView<float>;
template class MatrixView<double>;
template class MatrixView<int>;

template class NaiveMatmul<float>;
template class NaiveMatmul<double>;
template class NaiveMatmul<int>;

template class SharedMemoryMatmul<float>;
template class SharedMemoryMatmul<double>;
template class SharedMemoryMatmul<int>;

template class Matrix<float>;
template class Matrix<double>;
template class Matrix<int>;
