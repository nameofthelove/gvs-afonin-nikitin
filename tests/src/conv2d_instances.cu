#include "conv2d_instances.h"

#include <array>
#include <stdexcept>
#include <utility>

#include "hsys/nn/functional/conv2d.cuh"

namespace hsys::tests {

namespace {

template <long filter_size, long stride, long padding, long dilation, long block_size = 16>
constexpr std::pair<FilterParams, Conv2dFn> instance() {
    return {{filter_size, stride, padding, dilation, block_size},
            &nn::functional::conv2d<filter_size, stride, padding, dilation, block_size>};
}

const std::array kInstances = {
    // Табл. 1 задания.
    instance<3, 2, 1, 1>(),
    instance<3, 1, 1, 1>(),
    instance<7, 2, 3, 1>(),
    instance<1, 2, 0, 1>(),
    instance<3, 1, 2, 2>(),
    instance<3, 2, 2, 2>(),
    // Другие размеры блока: тайл 8x8 и блок из 1024 нитей с наибольшим тайлом входа (69x69).
    instance<3, 1, 1, 1, 8>(),
    instance<3, 1, 1, 1, 32>(),
    instance<7, 2, 3, 1, 32>(),
    // Параметры вне таблицы: чётный фильтр, шаг больше фильтра, отступ больше половины
    // фильтра, фильтр без отступа (выход меньше входа).
    instance<2, 1, 0, 3>(),
    instance<1, 3, 0, 1>(),
    instance<5, 3, 0, 1>(),
    instance<3, 1, 3, 1>(),
};

}  // namespace

std::string to_string(const FilterParams& params) {
    std::string name = "k" + std::to_string(params.filter_size) + "_s" +
                       std::to_string(params.stride) + "_p" + std::to_string(params.padding) +
                       "_d" + std::to_string(params.dilation);
    if (params.block_size != 16) {
        name += "_b" + std::to_string(params.block_size);
    }
    return name;
}

Conv2dFn conv2d_instance(const FilterParams& params) {
    for (const auto& [instance_params, fn] : kInstances) {
        if (instance_params == params) {
            return fn;
        }
    }
    throw std::out_of_range("conv2d is not instantiated for " + to_string(params));
}

}  // namespace hsys::tests
