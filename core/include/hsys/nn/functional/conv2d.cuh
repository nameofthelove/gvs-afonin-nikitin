#pragma once

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>

#include "hsys/nn/functional/kernels/conv2d.cuh"

namespace hsys::nn::functional {

// Предельное значение gridDim.y и gridDim.z.
inline constexpr std::int64_t kMaxGridYZ = 65535;

constexpr std::int64_t ceil_div(std::int64_t value, std::int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

// Свёртка input [N, C_in, H_in, W_in] с весами weights [C_out, C_in, k, k] без смещения —
// то же, что torch::nn::functional::conv2d с параметрами stride, padding, dilation.
// Тензоры должны находиться на одном устройстве CUDA и иметь один тип (float, double, half или
// bfloat16); непрерывность в памяти не требуется: аксессоры учитывают шаги тензора.
// Кёрнел ставится в текущий поток PyTorch (at::cuda::getCurrentCUDAStream) и выполняется
// асинхронно; результат — новый непрерывный тензор [N, C_out, H_out, W_out].
template <long filter_size, long stride = 1, long padding = 0, long dilation = 1,
          long block_size = 16>
torch::Tensor conv2d(const torch::Tensor& input, const torch::Tensor& weights) {
    TORCH_CHECK(input.is_cuda() && weights.is_cuda(),
                "hsys::conv2d: tensors must be on a CUDA device, got ", input.device(), " and ",
                weights.device());
    TORCH_CHECK(input.device() == weights.device(),
                "hsys::conv2d: tensors must be on the same device, got ", input.device(),
                " and ", weights.device());
    TORCH_CHECK(input.scalar_type() == weights.scalar_type(),
                "hsys::conv2d: input and weights must have the same dtype, got ",
                input.scalar_type(), " and ", weights.scalar_type());
    TORCH_CHECK(input.dim() == 4, "hsys::conv2d: expected input [N, C_in, H_in, W_in], got ",
                input.sizes());
    TORCH_CHECK(weights.dim() == 4, "hsys::conv2d: expected weights [C_out, C_in, k, k], got ",
                weights.sizes());
    TORCH_CHECK(weights.size(1) == input.size(1), "hsys::conv2d: weights ", weights.sizes(),
                " expect ", weights.size(1), " input channels, got input ", input.sizes());
    TORCH_CHECK(weights.size(2) == filter_size && weights.size(3) == filter_size,
                "hsys::conv2d: kernel is compiled for a ", filter_size, "x", filter_size,
                " filter, got weights ", weights.sizes());

    const std::int64_t h_out =
        kernels::output_size(input.size(2), filter_size, stride, padding, dilation);
    const std::int64_t w_out =
        kernels::output_size(input.size(3), filter_size, stride, padding, dilation);
    TORCH_CHECK(h_out > 0 && w_out > 0, "hsys::conv2d: input ", input.sizes(),
                " is too small for a ", filter_size, "x", filter_size, " filter with padding ",
                padding, " and dilation ", dilation);

    const c10::cuda::CUDAGuard device_guard(input.device());

    torch::Tensor output =
        torch::empty({input.size(0), weights.size(0), h_out, w_out}, input.options());
    if (output.numel() == 0) {
        return output;  // Пустой пакет или C_out = 0: запуск с пустой сеткой недопустим.
    }

    const std::int64_t tiles_y = ceil_div(h_out, block_size);
    TORCH_CHECK(tiles_y <= kMaxGridYZ, "hsys::conv2d: output height ", h_out,
                " exceeds the grid limit");

    // Пары (изображение, группа каналов), которые не поместились в gridDim.z, кёрнел
    // обходит в цикле.
    const std::int64_t work_items =
        input.size(0) * ceil_div(weights.size(0), kernels::kOutChannelsPerBlock);

    const dim3 block(block_size, block_size);
    const dim3 grid(static_cast<unsigned int>(ceil_div(w_out, block_size)),
                    static_cast<unsigned int>(tiles_y),
                    static_cast<unsigned int>(std::min(work_items, kMaxGridYZ)));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "hsys::conv2d", [&] {
            kernels::conv2d<scalar_t, filter_size, stride, padding, dilation, block_size>
                <<<grid, block, 0, stream>>>(
                    output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });

    return output;
}

}  // namespace hsys::nn::functional
