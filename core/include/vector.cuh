#pragma once
#include "data.cuh"
#include "vector_view.cuh"
#include <memory>
#include <cstddef>

template <typename T>
class Vector {
private:
	shared_ptr<Data<AtomT>> data_;
	VectorView<AtomT> view_;
public:
	explicit Vector(std::size_t size) : data_(std::make_shared<Data<AtomT>>(size)), view_(data_->data(), size) {}
	size_t size() const { return *view_.size(); }
	Data<AtomT> data() { return &data_}
	const Data<AtomT> data() const { return &data_ }
	VectorView<AtomT>& view() { return view_ }
	const VectorView<AtomT>& view() const { return view_ }
};