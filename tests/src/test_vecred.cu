#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "eigen_interop.cuh"
#include "vector.cuh"

namespace {

constexpr float kPrecision = 1e-4f;
constexpr double kPrecisionDouble = 1e-9;

const std::vector<std::size_t> kSizes = {1, 2, 3, 127, 129, 512, 541, 1037};

// Размеры на границах варпа, блока и сетки (256 x 120 = 30720 для геометрии по умолчанию).
const std::vector<std::size_t> kBoundarySizes = {4,    31,   32,    33,    63,    64,    65,
                                                 255,  256,  257,   511,   513,   1023,  1024,
                                                 1025, 4095, 30719, 30720, 30721, 65536, 100003};

enum class Kernel { Nobr, Br };

// Геометрии блока: BlockDim, MaxBlocks.
enum class Geometry { Default, Warp32, Block64, Block512, Block1024, SingleBlock };

std::string to_string(Kernel kernel) { return kernel == Kernel::Nobr ? "Nobr" : "Br"; }

std::string to_string(Geometry geometry) {
    switch (geometry) {
        case Geometry::Warp32:
            return "Warp32";
        case Geometry::Block64:
            return "Block64";
        case Geometry::Block512:
            return "Block512";
        case Geometry::Block1024:
            return "Block1024";
        case Geometry::SingleBlock:
            return "SingleBlock";
        case Geometry::Default:
        default:
            return "Default";
    }
}

template <typename AtomT, typename SumT, unsigned int BlockDim, unsigned int MaxBlocks>
std::shared_ptr<const VecredStrategy<AtomT, SumT>> make_with_geometry(Kernel kernel) {
    if (kernel == Kernel::Nobr) {
        return std::make_shared<const NoBroadcastVecred<AtomT, SumT, BlockDim, MaxBlocks>>();
    }
    return std::make_shared<const BroadcastVecred<AtomT, SumT, BlockDim, MaxBlocks>>();
}

template <typename AtomT, typename SumT = AtomT>
std::shared_ptr<const VecredStrategy<AtomT, SumT>> make_strategy(
    Kernel kernel, Geometry geometry = Geometry::Default) {
    switch (geometry) {
        case Geometry::Warp32:
            return make_with_geometry<AtomT, SumT, 32, 32>(kernel);
        case Geometry::Block64:
            return make_with_geometry<AtomT, SumT, 64, 16>(kernel);
        case Geometry::Block512:
            return make_with_geometry<AtomT, SumT, 512, 60>(kernel);
        case Geometry::Block1024:
            return make_with_geometry<AtomT, SumT, 1024, 20>(kernel);
        case Geometry::SingleBlock:
            return make_with_geometry<AtomT, SumT, 256, 1>(kernel);
        case Geometry::Default:
        default:
            return make_with_geometry<AtomT, SumT, kVecredBlockDim, kVecredMaxBlocks>(kernel);
    }
}

// Случайный вектор с элементами из [-1, 1]; зерно зависит от размера.
template <typename AtomT>
EigenVector<AtomT> random_vector(std::size_t n) {
    std::srand(static_cast<unsigned int>(n * 7919 + 13));
    return EigenVector<AtomT>::Random(static_cast<Eigen::Index>(n));
}

template <typename AtomT, typename SumT>
SumT sum_with(Kernel kernel, const EigenVector<AtomT>& host, Geometry geometry) {
    Vector<AtomT, SumT> device = to_device<AtomT, SumT>(host);
    device.set_vecred_strategy(make_strategy<AtomT, SumT>(kernel, geometry));
    return device.sum();
}

// Вектор 0, 1, ..., n - 1 с заданным типом элемента (по модулю разрядности типа).
template <typename AtomT>
EigenVector<AtomT> iota_vector(std::size_t n) {
    EigenVector<AtomT> host(static_cast<Eigen::Index>(n));
    for (std::size_t i = 0; i < n; ++i) {
        host[static_cast<Eigen::Index>(i)] = static_cast<AtomT>(i);
    }
    return host;
}

}  // namespace

// Основной тест задания: размеры n и эталон Eigen::Matrix::sum.
class VecredParamTest : public ::testing::TestWithParam<std::tuple<Kernel, std::size_t>> {};

TEST_P(VecredParamTest, MatchesEigen) {
    const auto [kernel, n] = GetParam();

    const EigenVector<float> host = random_vector<float>(n);
    const float expected = host.sum();
    const float actual = sum_with<float, float>(kernel, host, Geometry::Default);

    EXPECT_NEAR(actual, expected, kPrecision);
}

INSTANTIATE_TEST_SUITE_P(AllSizes, VecredParamTest,
                         ::testing::Combine(::testing::Values(Kernel::Nobr, Kernel::Br),
                                            ::testing::ValuesIn(kSizes)),
                         [](const auto& info) {
                             return to_string(std::get<0>(info.param)) + "_n" +
                                    std::to_string(std::get<1>(info.param));
                         });

// Граничные размеры для всех геометрий блока. Тип double: погрешность округления не маскирует
// потерю или двойной учёт одного элемента.
class VecredBoundaryParamTest
    : public ::testing::TestWithParam<std::tuple<Kernel, Geometry, std::size_t>> {};

TEST_P(VecredBoundaryParamTest, MatchesEigen) {
    const auto [kernel, geometry, n] = GetParam();

    const EigenVector<double> host = random_vector<double>(n);
    EXPECT_NEAR((sum_with<double, double>(kernel, host, geometry)), host.sum(), kPrecisionDouble);
}

INSTANTIATE_TEST_SUITE_P(
    BoundarySizes, VecredBoundaryParamTest,
    ::testing::Combine(::testing::Values(Kernel::Nobr, Kernel::Br),
                       ::testing::Values(Geometry::Default, Geometry::Warp32, Geometry::Block64,
                                         Geometry::Block512, Geometry::Block1024,
                                         Geometry::SingleBlock),
                       ::testing::ValuesIn(kBoundarySizes)),
    [](const auto& info) {
        return to_string(std::get<0>(info.param)) + "_" + to_string(std::get<1>(info.param)) +
               "_n" + std::to_string(std::get<2>(info.param));
    });

// Точные проверки: целочисленная сумма не зависит от порядка сложения.
class VecredExactParamTest : public ::testing::TestWithParam<Kernel> {};

TEST_P(VecredExactParamTest, EveryElementIsCountedExactlyOnce) {
    constexpr std::size_t kSize = 1037;

    // Единица в одной позиции: потерянный или повторно учтённый элемент изменит сумму.
    Vector<int> device(kSize, make_strategy<int>(GetParam()));
    for (std::size_t position = 0; position < kSize; position += 7) {
        std::vector<int> host(kSize, 0);
        host[position] = 1;
        device.data().copy_from_host(host.data());
        ASSERT_EQ(device.sum(), 1) << "position " << position;
    }
}

TEST_P(VecredExactParamTest, ArithmeticProgression) {
    for (const std::size_t n : {std::size_t{1000}, std::size_t{30721}, std::size_t{65535}}) {
        const auto expected = static_cast<std::uint64_t>(n) * (n - 1) / 2;
        EXPECT_EQ((sum_with<std::uint64_t, std::uint64_t>(
                      GetParam(), iota_vector<std::uint64_t>(n), Geometry::Default)),
                  expected)
            << "n = " << n;
    }
}

TEST_P(VecredExactParamTest, NarrowAtomWideAccumulator) {
    // Сумма байтов не помещается в байт: накопитель шире элемента.
    constexpr std::size_t kSize = std::size_t{1} << 24;

    Vector<std::uint8_t, std::uint64_t> device(
        kSize, make_strategy<std::uint8_t, std::uint64_t>(GetParam()));
    const std::vector<std::uint8_t> host(kSize, 255);
    device.data().copy_from_host(host.data());

    EXPECT_EQ(device.sum(), std::uint64_t{255} * kSize);
}

TEST_P(VecredExactParamTest, AllElementsEqual) {
    const EigenVector<double> host = EigenVector<double>::Constant(4097, 0.5);
    EXPECT_EQ((sum_with<double, double>(GetParam(), host, Geometry::Default)), 2048.5);
}

TEST_P(VecredExactParamTest, AlternatingSignsCancel) {
    EigenVector<double> host(1024);
    for (Eigen::Index i = 0; i < host.size(); ++i) {
        host[i] = i % 2 == 0 ? 1.0 : -1.0;
    }
    EXPECT_EQ((sum_with<double, double>(GetParam(), host, Geometry::Default)), 0.0);
}

TEST_P(VecredExactParamTest, EmptyVector) {
    Vector<float> device(0, make_strategy<float>(GetParam()));
    EXPECT_EQ(device.sum(), 0.0f);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST_P(VecredExactParamTest, RepeatedSumIsStable) {
    const EigenVector<float> host = random_vector<float>(1037);
    Vector<float> device = to_device(host);
    device.set_vecred_strategy(make_strategy<float>(GetParam()));

    const float first = device.sum();
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(device.sum(), first);
    }
}

TEST_P(VecredExactParamTest, InputIsNotModified) {
    const EigenVector<float> host = random_vector<float>(1037);
    Vector<float> device = to_device(host);
    device.set_vecred_strategy(make_strategy<float>(GetParam()));

    device.sum();

    EigenVector<float> after(host.size());
    device.data().copy_to_host(after.data());
    EXPECT_EQ(after, host);
}

TEST_P(VecredExactParamTest, UnalignedView) {
    constexpr std::size_t kSize = 541;

    // Представление начинается с нечётного адреса: чтение не должно зависеть от выравнивания.
    std::vector<std::uint8_t> bytes(kSize + 1);
    std::iota(bytes.begin(), bytes.end(), std::uint8_t{1});
    Data<std::uint8_t> storage(bytes.size());
    storage.copy_from_host(bytes.data());

    Data<std::uint64_t> partial(kVecredMaxBlocks);
    const auto strategy = make_strategy<std::uint8_t, std::uint64_t>(GetParam());
    strategy->reduce(VectorView<std::uint8_t>(storage.data() + 1, kSize),
                     VectorView<std::uint64_t>(partial.data(), partial.size()));

    std::uint64_t actual = 0;
    partial.copy_to_host(&actual, 1);
    EXPECT_EQ(actual, std::accumulate(bytes.begin() + 1, bytes.end(), std::uint64_t{0}));
}

TEST_P(VecredExactParamTest, DoesNotWriteOutsidePartial) {
    constexpr double kGuard = -7.0;

    // Частичные суммы первого прохода не выходят за вместимость буфера.
    const auto strategy = make_strategy<double>(GetParam());
    const std::size_t capacity = strategy->partial_capacity();
    const EigenVector<double> host = random_vector<double>(1000000);
    const Vector<double> device = to_device(host);

    std::vector<double> init(capacity + 16, kGuard);
    Data<double> partial(init.size());
    partial.copy_from_host(init.data());
    strategy->reduce(device.view(), VectorView<double>(partial.data(), capacity));

    partial.copy_to_host(init.data());
    EXPECT_NEAR(init[0], host.sum(), kPrecisionDouble);
    for (std::size_t i = capacity; i < init.size(); ++i) {
        EXPECT_EQ(init[i], kGuard) << "index " << i;
    }
}

TEST_P(VecredExactParamTest, QueuedOnUserStream) {
    const EigenVector<double> host = random_vector<double>(100000);
    const Vector<double> device = to_device(host);
    const auto strategy = make_strategy<double>(GetParam());
    Data<double> partial(strategy->partial_capacity());

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    strategy->reduce(device.view(), VectorView<double>(partial.data(), partial.size()), stream);
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    double actual = 0.0;
    partial.copy_to_host(&actual, 1);
    EXPECT_NEAR(actual, host.sum(), kPrecisionDouble);
}

TEST_P(VecredExactParamTest, NoDeviceMemoryLeak) {
    std::size_t free_before = 0;
    std::size_t free_after = 0;
    std::size_t total = 0;

    {
        // Первый запуск загружает модуль кёрнела, его память не считается утечкой.
        Vector<float> warm_up(1037, make_strategy<float>(GetParam()));
        warm_up.sum();
    }
    check_cuda(cudaMemGetInfo(&free_before, &total), "cudaMemGetInfo");

    for (int i = 0; i < 50; ++i) {
        Vector<float> device(std::size_t{1} << 20, make_strategy<float>(GetParam()));
        Vector<float> copy = device;
        Data<float> deep_copy(device.data());
        copy.sum();
    }

    check_cuda(cudaMemGetInfo(&free_after, &total), "cudaMemGetInfo");
    EXPECT_EQ(free_before, free_after);
}

INSTANTIATE_TEST_SUITE_P(Kernels, VecredExactParamTest,
                         ::testing::Values(Kernel::Nobr, Kernel::Br),
                         [](const auto& info) { return to_string(info.param); });

// Контрольный запуск на наибольшем размере бенчмарков, требует 2 ГиБ памяти устройства:
// --gtest_also_run_disabled_tests --gtest_filter='*MaxBenchmarkSize*'
TEST(VecredLargeTest, DISABLED_MaxBenchmarkSize) {
    constexpr std::size_t kSize = std::size_t{1} << 31;

    Vector<std::uint8_t, std::uint64_t> device(kSize);
    const std::vector<std::uint8_t> host(kSize, 3);
    device.data().copy_from_host(host.data());

    for (const Kernel kernel : {Kernel::Nobr, Kernel::Br}) {
        device.set_vecred_strategy(make_strategy<std::uint8_t, std::uint64_t>(kernel));
        EXPECT_EQ(device.sum(), std::uint64_t{3} * kSize);
    }
}

TEST(VecredStrategyTest, DefaultIsBroadcast) {
    const Vector<float> vector(8);
    EXPECT_NE(dynamic_cast<const BroadcastVecred<float>*>(&vector.vecred_strategy()), nullptr);
}

TEST(VecredStrategyTest, NullStrategyThrows) {
    EXPECT_THROW(Vector<float>(8, nullptr), std::invalid_argument);

    Vector<float> vector(8);
    EXPECT_THROW(vector.set_vecred_strategy(nullptr), std::invalid_argument);
}

TEST(VecredStrategyTest, KernelsAgree) {
    for (const std::size_t n : kBoundarySizes) {
        const EigenVector<double> host = random_vector<double>(n);
        EXPECT_NEAR((sum_with<double, double>(Kernel::Nobr, host, Geometry::Default)),
                    (sum_with<double, double>(Kernel::Br, host, Geometry::Default)),
                    kPrecisionDouble);
    }
}

TEST(VecredGridTest, GridCoversVector) {
    EXPECT_EQ(vecred_grid(1, 256, 120), 1u);
    EXPECT_EQ(vecred_grid(256, 256, 120), 1u);
    EXPECT_EQ(vecred_grid(257, 256, 120), 2u);
    EXPECT_EQ(vecred_grid(30720, 256, 120), 120u);
    EXPECT_EQ(vecred_grid(std::size_t{1} << 31, 256, 120), 120u);
}

TEST(VectorViewTest, IsTriviallyCopyable) {
    EXPECT_TRUE(std::is_trivially_copyable_v<VectorView<float>>);
    EXPECT_TRUE(std::is_trivially_copyable_v<VectorView<std::uint64_t>>);
}

TEST(VectorViewTest, Indexing) {
    float values[3] = {1.0f, 2.0f, 3.0f};
    VectorView<float> view(values, 3);

    EXPECT_EQ(view.size(), 3u);
    view[1] = 5.0f;
    EXPECT_EQ(values[1], 5.0f);
}

TEST(VectorTest, CopiesShareData) {
    const EigenVector<float> host = EigenVector<float>::Ones(16);
    Vector<float> first = to_device(host);
    Vector<float> second = first;

    const EigenVector<float> zeros = EigenVector<float>::Zero(16);
    second.data().copy_from_host(zeros.data());

    EXPECT_EQ(first.sum(), 0.0f);
    EXPECT_EQ(first.view().size(), second.view().size());
}

TEST(DataTest, CopyIsDeepAndMoveTransfersOwnership) {
    const std::vector<int> values = {1, 2, 3, 4};
    Data<int> source(values.size());
    source.copy_from_host(values.data());

    Data<int> copy(source);
    const std::vector<int> zeros(values.size(), 0);
    source.copy_from_host(zeros.data());

    std::vector<int> actual(values.size());
    copy.copy_to_host(actual.data());
    EXPECT_EQ(actual, values);

    const int* const raw = copy.data();
    Data<int> moved(std::move(copy));
    EXPECT_EQ(moved.data(), raw);
    EXPECT_EQ(copy.data(), nullptr);
    EXPECT_EQ(copy.size(), 0u);
}

TEST(DataTest, PartialCopyToHost) {
    const std::vector<int> values = {1, 2, 3, 4};
    Data<int> data(values.size());
    data.copy_from_host(values.data());

    int first = 0;
    data.copy_to_host(&first, 1);
    EXPECT_EQ(first, 1);
    EXPECT_THROW(data.copy_to_host(&first, 5), std::out_of_range);
}
