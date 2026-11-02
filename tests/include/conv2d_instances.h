#pragma once

#include <torch/types.h>

#include <string>

namespace hsys::tests {

// Параметры фильтра и блока, с которыми скомпилирован экземпляр hsys::nn::functional::conv2d.
struct FilterParams {
    long filter_size;
    long stride;
    long padding;
    long dilation;
    long block_size = 16;

    bool operator==(const FilterParams&) const = default;
};

// Имя для параметризованных тестов, например k3_s2_p1_d1 или k3_s1_p1_d1_b32.
std::string to_string(const FilterParams& params);

using Conv2dFn = torch::Tensor (*)(const torch::Tensor& input, const torch::Tensor& weights);

// Экземпляр conv2d для params; std::out_of_range, если он не скомпилирован. Параметры фильтра
// — аргументы шаблона, поэтому каждый проверяемый набор порождается в conv2d_instances.cu,
// а сами тесты компилируются обычным компилятором C++ без кода CUDA.
Conv2dFn conv2d_instance(const FilterParams& params);

}  // namespace hsys::tests
