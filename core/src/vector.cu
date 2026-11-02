#include <cstdint>

#include "vector.cuh"

// Явные инстанцирования: проверяют компилируемость шаблонов для основных типов.
template class Data<float>;
template class Data<double>;
template class Data<int>;
template class Data<std::uint8_t>;
template class Data<std::uint64_t>;

template class VectorView<float>;
template class VectorView<double>;
template class VectorView<int>;

template class NoBroadcastVecred<float>;
template class NoBroadcastVecred<double>;
template class NoBroadcastVecred<int>;
template class NoBroadcastVecred<std::uint8_t, std::uint64_t>;

template class BroadcastVecred<float>;
template class BroadcastVecred<double>;
template class BroadcastVecred<int>;
template class BroadcastVecred<std::uint8_t, std::uint64_t>;

template class Vector<float>;
template class Vector<double>;
template class Vector<int>;
template class Vector<std::uint8_t, std::uint64_t>;
