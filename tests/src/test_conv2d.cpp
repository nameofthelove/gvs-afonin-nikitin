#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "conv2d_instances.h"

using hsys::tests::conv2d_instance;
using hsys::tests::FilterParams;

namespace F = torch::nn::functional;

namespace {

// Размер пакета в тестах табл. 1 (в задании не задан).
constexpr std::int64_t kBatchSize = 2;

struct Shape {
    std::int64_t batch;
    std::int64_t in_channels;
    std::int64_t height;
    std::int64_t width;
    std::int64_t out_channels;
};

std::string to_string(const Shape& shape) {
    return "n" + std::to_string(shape.batch) + "_" + std::to_string(shape.in_channels) + "x" +
           std::to_string(shape.height) + "x" + std::to_string(shape.width) + "_to" +
           std::to_string(shape.out_channels);
}

std::string to_string(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kHalf:
            return "Half";
        case torch::kBFloat16:
            return "BFloat16";
        case torch::kDouble:
            return "Double";
        case torch::kFloat:
        default:
            return "Float";
    }
}

// Табл. 1 задания: параметры фильтра x размеры изображений.
const std::vector<FilterParams> kTableFilters = {
    {3, 2, 1, 1}, {3, 1, 1, 1}, {7, 2, 3, 1}, {1, 2, 0, 1}, {3, 1, 2, 2}, {3, 2, 2, 2},
};

const std::vector<Shape> kTableShapes = {
    {kBatchSize, 3, 244, 244, 64},   {kBatchSize, 64, 112, 112, 64},
    {kBatchSize, 64, 56, 56, 64},    {kBatchSize, 128, 28, 28, 128},
    {kBatchSize, 64, 56, 56, 128},   {kBatchSize, 128, 28, 28, 256},
    {kBatchSize, 256, 14, 14, 512},  {kBatchSize, 256, 14, 14, 256},
    {kBatchSize, 512, 7, 7, 512},
};

struct Tolerance {
    double rtol;
    double atol;
};

// Допуски задания: 1e-5 для float, 1e-2 для типов пониженной точности.
Tolerance tolerance(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kHalf:
        case torch::kBFloat16:
            return {1e-2, 1e-2};
        case torch::kDouble:
            return {1e-12, 1e-12};
        case torch::kFloat:
        default:
            return {1e-5, 1e-5};
    }
}

torch::TensorOptions cuda_options(torch::ScalarType dtype) {
    return torch::TensorOptions().device(torch::kCUDA).dtype(dtype);
}

// Равномерное распределение на [-bound, bound]; генерируется в float, затем приводится к dtype,
// поэтому тензоры разных типов содержат одни и те же значения с точностью до округления.
torch::Tensor uniform(torch::IntArrayRef sizes, double bound, torch::ScalarType dtype) {
    return ((torch::rand(sizes, cuda_options(torch::kFloat)) * 2 - 1) * bound).to(dtype);
}

// Вход из [-1, 1] и веса из [-1/sqrt(fan_in), 1/sqrt(fan_in)], как при инициализации
// torch::nn::Conv2d: выходы имеют порядок единицы при любом числе входных каналов.
std::pair<torch::Tensor, torch::Tensor> random_problem(const FilterParams& params,
                                                       const Shape& shape,
                                                       torch::ScalarType dtype) {
    const std::int64_t k = params.filter_size;
    const double bound = 1.0 / std::sqrt(static_cast<double>(shape.in_channels * k * k));
    return {uniform({shape.batch, shape.in_channels, shape.height, shape.width}, 1.0, dtype),
            uniform({shape.out_channels, shape.in_channels, k, k}, bound, dtype)};
}

F::Conv2dFuncOptions conv_options(const FilterParams& params) {
    return F::Conv2dFuncOptions()
        .stride(static_cast<std::int64_t>(params.stride))
        .padding(static_cast<std::int64_t>(params.padding))
        .dilation(static_cast<std::int64_t>(params.dilation));
}

// Эталон — torch::nn::functional::conv2d над теми же значениями в double, приведённый к типу
// входа: погрешность эталона не складывается с погрешностью проверяемого кёрнела, а TF32
// cuDNN (включён по умолчанию на Ampere) не участвует.
torch::Tensor reference(const torch::Tensor& input, const torch::Tensor& weights,
                        const FilterParams& params) {
    return F::conv2d(input.to(torch::kDouble), weights.to(torch::kDouble), conv_options(params))
        .to(input.scalar_type());
}

::testing::AssertionResult matches_reference(const torch::Tensor& input,
                                             const torch::Tensor& weights,
                                             const FilterParams& params) {
    const torch::Tensor actual = conv2d_instance(params)(input, weights);
    const torch::Tensor expected = reference(input, weights, params);

    if (actual.sizes() != expected.sizes()) {
        return ::testing::AssertionFailure()
               << "shape " << actual.sizes() << ", expected " << expected.sizes();
    }
    if (actual.scalar_type() != input.scalar_type()) {
        return ::testing::AssertionFailure() << "dtype " << actual.scalar_type();
    }

    const auto [rtol, atol] = tolerance(input.scalar_type());
    if (torch::allclose(actual, expected, rtol, atol)) {
        return ::testing::AssertionSuccess();
    }
    const double max_error =
        (actual.to(torch::kDouble) - expected.to(torch::kDouble)).abs().max().item<double>();
    return ::testing::AssertionFailure() << "max |actual - expected| = " << max_error;
}

// Тесты пропускаются, если устройство CUDA недоступно.
class CudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA device is not available";
        }
        torch::manual_seed(42);
    }
};

template <class Param>
class CudaParamTest : public CudaTest, public ::testing::WithParamInterface<Param> {};

}  // namespace

// Основной тест задания: декартово произведение табл. 1 для float, half и bfloat16.
using TableParam = std::tuple<torch::ScalarType, FilterParams, Shape>;

class Conv2dTableTest : public CudaParamTest<TableParam> {};

TEST_P(Conv2dTableTest, MatchesTorchConv2d) {
    const auto& [dtype, params, shape] = GetParam();
    const auto [input, weights] = random_problem(params, shape, dtype);
    EXPECT_TRUE(matches_reference(input, weights, params));
}

INSTANTIATE_TEST_SUITE_P(Table1, Conv2dTableTest,
                         ::testing::Combine(::testing::Values(torch::kFloat, torch::kHalf,
                                                              torch::kBFloat16),
                                            ::testing::ValuesIn(kTableFilters),
                                            ::testing::ValuesIn(kTableShapes)),
                         [](const auto& info) {
                             return to_string(std::get<0>(info.param)) + "_" +
                                    hsys::tests::to_string(std::get<1>(info.param)) + "_" +
                                    to_string(std::get<2>(info.param));
                         });

// Граничные случаи: размеры, не кратные тайлу и группе каналов, выход 1x1, фильтр шире
// изображения, неквадратные изображения, размеры блока 8 и 32, параметры вне табл. 1.
namespace {

struct EdgeCase {
    std::string name;
    FilterParams params;
    Shape shape;
};

const std::vector<EdgeCase> kEdgeCases = {
    {"SinglePixel", {3, 1, 1, 1}, {1, 1, 1, 1, 1}},
    {"SingleChannel", {3, 2, 1, 1}, {2, 1, 31, 29, 1}},
    {"NotMultipleOfTile", {3, 1, 1, 1}, {3, 5, 17, 33, 5}},
    {"WideImage", {3, 1, 1, 1}, {1, 4, 5, 300, 13}},
    {"TallImage", {7, 2, 3, 1}, {1, 4, 300, 5, 9}},
    {"FilterWiderThanImage", {7, 2, 3, 1}, {2, 3, 1, 1, 9}},
    {"DilatedFilterWiderThanImage", {3, 1, 2, 2}, {2, 3, 1, 2, 7}},
    {"PointwiseOnSinglePixel", {1, 2, 0, 1}, {3, 8, 1, 1, 8}},
    {"PointwiseOddSize", {1, 2, 0, 1}, {2, 3, 15, 17, 9}},
    {"OutChannelsBelowGroup", {3, 2, 2, 2}, {2, 6, 23, 19, 3}},
    {"OutChannelsAboveGroup", {3, 2, 2, 2}, {2, 6, 23, 19, 9}},
    {"LargeBatch", {3, 1, 1, 1}, {17, 3, 9, 9, 4}},
    {"Block8", {3, 1, 1, 1, 8}, {2, 16, 33, 35, 24}},
    {"Block32", {3, 1, 1, 1, 32}, {2, 16, 33, 35, 24}},
    {"Block32LargestTile", {7, 2, 3, 1, 32}, {2, 16, 70, 66, 9}},
    {"EvenDilatedFilter", {2, 1, 0, 3}, {2, 3, 9, 4, 3}},
    {"StrideAboveFilter", {1, 3, 0, 1}, {2, 6, 23, 31, 17}},
    {"NoPadding", {5, 3, 0, 1}, {2, 6, 23, 31, 17}},
    {"PaddingAboveHalfFilter", {3, 1, 3, 1}, {2, 4, 11, 13, 10}},
};

}  // namespace

using EdgeParam = std::tuple<torch::ScalarType, EdgeCase>;

class Conv2dEdgeTest : public CudaParamTest<EdgeParam> {};

TEST_P(Conv2dEdgeTest, MatchesTorchConv2d) {
    const auto& [dtype, edge] = GetParam();
    const auto [input, weights] = random_problem(edge.params, edge.shape, dtype);
    EXPECT_TRUE(matches_reference(input, weights, edge.params));
}

INSTANTIATE_TEST_SUITE_P(EdgeCases, Conv2dEdgeTest,
                         ::testing::Combine(::testing::Values(torch::kFloat, torch::kDouble,
                                                              torch::kHalf),
                                            ::testing::ValuesIn(kEdgeCases)),
                         [](const auto& info) {
                             return to_string(std::get<0>(info.param)) + "_" +
                                    std::get<1>(info.param).name;
                         });

// Точное совпадение: целые значения из [-2, 2] в double суммируются без округления, поэтому
// любой пропущенный, лишний или сдвинутый элемент отступа меняет результат.
class Conv2dExactTest : public CudaParamTest<FilterParams> {};

TEST_P(Conv2dExactTest, IntegerValuesMatchExactly) {
    const FilterParams& params = GetParam();
    const std::int64_t k = params.filter_size;
    const auto options = cuda_options(torch::kDouble);
    const torch::Tensor input = torch::randint(-2, 3, {2, 5, 19, 23}, options);
    const torch::Tensor weights = torch::randint(-2, 3, {11, 5, k, k}, options);

    EXPECT_TRUE(torch::equal(conv2d_instance(params)(input, weights),
                             reference(input, weights, params)));
}

TEST_P(Conv2dExactTest, ImpulseReproducesFilter) {
    // Единичный импульс в центре входа: каждый выход, накрывающий импульс, равен одному
    // весу фильтра, остальные — нулю. Проверяет индексацию весов и тайла.
    const FilterParams& params = GetParam();
    const std::int64_t k = params.filter_size;
    const auto options = cuda_options(torch::kDouble);
    torch::Tensor input = torch::zeros({1, 1, 21, 21}, options);
    input.index_put_({0, 0, 10, 10}, 1.0);
    const torch::Tensor weights = torch::arange(1, k * k + 1, options).reshape({1, 1, k, k});

    EXPECT_TRUE(torch::equal(conv2d_instance(params)(input, weights),
                             reference(input, weights, params)));
}

INSTANTIATE_TEST_SUITE_P(Table1, Conv2dExactTest, ::testing::ValuesIn(kTableFilters),
                         [](const auto& info) { return hsys::tests::to_string(info.param); });

// Аксессоры учитывают шаги тензора: непрерывность входа и весов не требуется.
class Conv2dLayoutTest : public CudaTest {};

TEST_F(Conv2dLayoutTest, TransposedInput) {
    const FilterParams params{3, 2, 1, 1};
    const torch::Tensor input = uniform({2, 8, 27, 19}, 1.0, torch::kFloat).transpose(2, 3);
    const torch::Tensor weights = uniform({10, 8, 3, 3}, 0.1, torch::kFloat);
    ASSERT_FALSE(input.is_contiguous());
    EXPECT_TRUE(matches_reference(input, weights, params));
}

TEST_F(Conv2dLayoutTest, ChannelsLast) {
    const FilterParams params{3, 1, 1, 1};
    const torch::Tensor input = uniform({2, 8, 20, 20}, 1.0, torch::kFloat)
                                    .contiguous(torch::MemoryFormat::ChannelsLast);
    const torch::Tensor weights = uniform({16, 8, 3, 3}, 0.1, torch::kFloat)
                                      .contiguous(torch::MemoryFormat::ChannelsLast);
    EXPECT_TRUE(matches_reference(input, weights, params));
}

TEST_F(Conv2dLayoutTest, SlicedInputAndWeights) {
    // Каждый второй канал и столбец: шаги не совпадают с размерами.
    const FilterParams params{7, 2, 3, 1};
    const torch::Tensor input = uniform({2, 12, 30, 60}, 1.0, torch::kFloat)
                                    .slice(1, 0, 12, 2)
                                    .slice(3, 0, 60, 2);
    const torch::Tensor weights = uniform({20, 6, 7, 7}, 0.1, torch::kFloat).slice(0, 0, 20, 2);
    EXPECT_TRUE(matches_reference(input, weights, params));
}

TEST_F(Conv2dLayoutTest, OutputIsContiguous) {
    const torch::Tensor input = uniform({1, 3, 8, 8}, 1.0, torch::kFloat).transpose(2, 3);
    const torch::Tensor weights = uniform({4, 3, 3, 3}, 0.1, torch::kFloat);
    EXPECT_TRUE(conv2d_instance({3, 1, 1, 1})(input, weights).is_contiguous());
}

// Поведение функции запуска.
class Conv2dLaunchTest : public CudaTest {};

TEST_F(Conv2dLaunchTest, GridStrideOverBatch) {
    // N * ceil(C_out / 8) = 70000 > 65535: часть пар обходится циклом по blockIdx.z.
    const FilterParams params{1, 2, 0, 1};
    const torch::Tensor input = uniform({70000, 1, 1, 1}, 1.0, torch::kFloat);
    const torch::Tensor weights = uniform({1, 1, 1, 1}, 1.0, torch::kFloat);
    EXPECT_TRUE(matches_reference(input, weights, params));
}

TEST_F(Conv2dLaunchTest, EmptyBatch) {
    const torch::Tensor input = torch::empty({0, 3, 8, 8}, cuda_options(torch::kFloat));
    const torch::Tensor weights = uniform({4, 3, 3, 3}, 0.1, torch::kFloat);

    const torch::Tensor output = conv2d_instance({3, 1, 1, 1})(input, weights);
    EXPECT_EQ(output.sizes().vec(), (std::vector<std::int64_t>{0, 4, 8, 8}));
}

TEST_F(Conv2dLaunchTest, UsesCurrentStream) {
    const FilterParams params{3, 1, 1, 1};
    const torch::Tensor input = uniform({2, 16, 64, 64}, 1.0, torch::kFloat);
    const torch::Tensor weights = uniform({32, 16, 3, 3}, 0.1, torch::kFloat);
    const torch::Tensor expected = reference(input, weights, params);

    const c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool();
    torch::Tensor actual;
    {
        const c10::cuda::CUDAStreamGuard guard(stream);
        actual = conv2d_instance(params)(input, weights);
    }
    stream.synchronize();
    EXPECT_TRUE(torch::allclose(actual, expected, 1e-5, 1e-5));
}

TEST_F(Conv2dLaunchTest, DeterministicAndInputsUnchanged) {
    const FilterParams params{3, 2, 2, 2};
    const torch::Tensor input = uniform({2, 32, 28, 28}, 1.0, torch::kFloat);
    const torch::Tensor weights = uniform({24, 32, 3, 3}, 0.1, torch::kFloat);
    const torch::Tensor input_copy = input.clone();
    const torch::Tensor weights_copy = weights.clone();

    const torch::Tensor first = conv2d_instance(params)(input, weights);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(torch::equal(conv2d_instance(params)(input, weights), first));
    }
    EXPECT_TRUE(torch::equal(input, input_copy));
    EXPECT_TRUE(torch::equal(weights, weights_copy));
}

TEST_F(Conv2dLaunchTest, OutputSizeMatchesFormula) {
    // Формулы (3)–(4) против размеров, которые возвращает PyTorch.
    for (const FilterParams& params : kTableFilters) {
        for (const std::int64_t size : {1, 2, 3, 7, 8, 14, 15, 28, 56, 112, 244}) {
            const auto options = cuda_options(torch::kFloat);
            const torch::Tensor input = torch::zeros({1, 1, size, size + 1}, options);
            const torch::Tensor weights =
                torch::zeros({1, 1, params.filter_size, params.filter_size}, options);
            EXPECT_EQ(conv2d_instance(params)(input, weights).sizes(),
                      F::conv2d(input, weights, conv_options(params)).sizes())
                << hsys::tests::to_string(params) << ", size " << size;
        }
    }
}

// Некорректные аргументы отклоняются TORCH_CHECK (c10::Error) до запуска кёрнела.
class Conv2dErrorTest : public CudaTest {
protected:
    void SetUp() override {
        CudaTest::SetUp();
        if (IsSkipped()) {
            return;
        }
        input_ = uniform({1, 4, 8, 8}, 1.0, torch::kFloat);
        weights_ = uniform({2, 4, 3, 3}, 0.1, torch::kFloat);
    }

    const hsys::tests::Conv2dFn conv3x3_ = conv2d_instance({3, 1, 1, 1});
    torch::Tensor input_;
    torch::Tensor weights_;
};

TEST_F(Conv2dErrorTest, CpuTensors) {
    EXPECT_THROW(conv3x3_(input_.cpu(), weights_.cpu()), c10::Error);
    EXPECT_THROW(conv3x3_(input_, weights_.cpu()), c10::Error);
}

TEST_F(Conv2dErrorTest, DtypeMismatch) {
    EXPECT_THROW(conv3x3_(input_, weights_.to(torch::kDouble)), c10::Error);
}

TEST_F(Conv2dErrorTest, UnsupportedDtype) {
    EXPECT_THROW(conv3x3_(input_.to(torch::kInt), weights_.to(torch::kInt)), c10::Error);
}

TEST_F(Conv2dErrorTest, WrongRank) {
    EXPECT_THROW(conv3x3_(input_[0], weights_), c10::Error);
    EXPECT_THROW(conv3x3_(input_, weights_[0]), c10::Error);
}

TEST_F(Conv2dErrorTest, InputChannelsMismatch) {
    EXPECT_THROW(conv3x3_(input_.slice(1, 0, 3), weights_), c10::Error);
}

TEST_F(Conv2dErrorTest, FilterSizeMismatch) {
    EXPECT_THROW(conv3x3_(input_, uniform({2, 4, 5, 5}, 0.1, torch::kFloat)), c10::Error);
    EXPECT_THROW(conv3x3_(input_, uniform({2, 4, 3, 1}, 0.1, torch::kFloat)), c10::Error);
}

TEST_F(Conv2dErrorTest, InputTooSmall) {
    // Фильтр 2x2 с расширением 3 охватывает 4 пикселя: вход 3x3 без отступа для него мал.
    // Числитель формулы (3) равен -1, усечение деления к нулю дало бы размер 1.
    const torch::Tensor input = uniform({1, 1, 3, 3}, 1.0, torch::kFloat);
    const torch::Tensor weights = uniform({1, 1, 2, 2}, 1.0, torch::kFloat);
    EXPECT_THROW(conv2d_instance({2, 1, 0, 3})(input, weights), c10::Error);
    EXPECT_THROW(F::conv2d(input, weights, conv_options({2, 1, 0, 3})), c10::Error);
}

TEST(Conv2dInstancesTest, MissingInstanceThrows) {
    EXPECT_THROW(conv2d_instance({11, 1, 0, 1}), std::out_of_range);
}
