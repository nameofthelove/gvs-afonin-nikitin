#include "matrix.cuh"

// Явные инстанцирования: проверяют компилируемость шаблонов для основных типов.
template class Data<float>;
template class Data<double>;
template class Data<int>;

template class MatrixView<float>;
template class MatrixView<double>;
template class MatrixView<int>;

template class Matrix<float>;
template class Matrix<double>;
template class Matrix<int>;
