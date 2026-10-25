#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "eigen_interop.cuh"
#include "matrix.cuh"

namespace {

constexpr double kPrecision = 1e-5;

const std::vector<std::size_t> kSizes = {1, 2, 3, 127, 128, 129, 512};

// Проверяемые стратегии operator*.
enum class StrategyKind { Default, Naive, Shmem16, Shmem32 };

std::shared_ptr<const MatmulStrategy<float>> make_strategy(StrategyKind kind) {
    switch (kind) {
        case StrategyKind::Naive:
            return std::make_shared<const NaiveMatmul<float>>();
        case StrategyKind::Shmem16:
            return std::make_shared<const SharedMemoryMatmul<float, 16>>();
        case StrategyKind::Shmem32:
            return std::make_shared<const SharedMemoryMatmul<float, 32>>();
        case StrategyKind::Default:
        default:
            return default_matmul_strategy<float>();
    }
}

std::string to_string(StrategyKind kind) {
    switch (kind) {
        case StrategyKind::Naive:
            return "Naive";
        case StrategyKind::Shmem16:
            return "Shmem16";
        case StrategyKind::Shmem32:
            return "Shmem32";
        case StrategyKind::Default:
        default:
            return "Default";
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Параметризованные тесты: A (m x k) * B (k x n) для всех m, n, k из kSizes.
// ---------------------------------------------------------------------------

using MatmulParams = std::tuple<std::size_t, std::size_t, std::size_t>;  // m, k, n

class MatmulParamTest : public ::testing::TestWithParam<MatmulParams> {};

// Основной набор: operator* со стратегией по умолчанию (kernel_matmul_shmem).
TEST_P(MatmulParamTest, MatchesEigen) {
    const auto [m, k, n] = GetParam();

    std::srand(static_cast<unsigned int>(m * 1'000'003 + k * 1'009 + n));
    const Eigen::MatrixXf a_host = Eigen::MatrixXf::Random(m, k);
    const Eigen::MatrixXf b_host = Eigen::MatrixXf::Random(k, n);
    const Eigen::MatrixXf expected = a_host * b_host;

    const Matrix<float> a = to_device(a_host);
    const Matrix<float> b = to_device(b_host);
    const Matrix<float> c = a * b;

    ASSERT_EQ(c.nrows(), m);
    ASSERT_EQ(c.ncols(), n);
    ASSERT_EQ(c.size(), m * n);

    const Eigen::MatrixXf actual = to_host(c);
    EXPECT_TRUE(actual.isApprox(expected, kPrecision))
        << "max |diff| = " << (actual - expected).cwiseAbs().maxCoeff();
}

std::string matmul_param_name(const ::testing::TestParamInfo<MatmulParams>& info) {
    const auto [m, k, n] = info.param;
    return "m" + std::to_string(m) + "_k" + std::to_string(k) + "_n" + std::to_string(n);
}

INSTANTIATE_TEST_SUITE_P(AllSizes, MatmulParamTest,
                         ::testing::Combine(::testing::ValuesIn(kSizes),
                                            ::testing::ValuesIn(kSizes),
                                            ::testing::ValuesIn(kSizes)),
                         matmul_param_name);

// ---------------------------------------------------------------------------
// Те же размеры для каждой стратегии в отдельности (в т. ч. предыдущей версии и
// блочного алгоритма с тайлами 16 x 16 и 32 x 32).
// ---------------------------------------------------------------------------

using StrategyParams = std::tuple<StrategyKind, std::size_t, std::size_t, std::size_t>;

class MatmulStrategyParamTest : public ::testing::TestWithParam<StrategyParams> {};

TEST_P(MatmulStrategyParamTest, MatchesEigen) {
    const auto [kind, m, k, n] = GetParam();

    std::srand(static_cast<unsigned int>(m * 1'000'003 + k * 1'009 + n));
    const Eigen::MatrixXf a_host = Eigen::MatrixXf::Random(m, k);
    const Eigen::MatrixXf b_host = Eigen::MatrixXf::Random(k, n);
    const Eigen::MatrixXf expected = a_host * b_host;

    Matrix<float> a = to_device(a_host);
    const Matrix<float> b = to_device(b_host);
    a.set_matmul_strategy(make_strategy(kind));
    const Matrix<float> c = a * b;

    ASSERT_EQ(c.nrows(), m);
    ASSERT_EQ(c.ncols(), n);

    const Eigen::MatrixXf actual = to_host(c);
    EXPECT_TRUE(actual.isApprox(expected, kPrecision))
        << "max |diff| = " << (actual - expected).cwiseAbs().maxCoeff();
}

std::string strategy_param_name(const ::testing::TestParamInfo<StrategyParams>& info) {
    const auto [kind, m, k, n] = info.param;
    return to_string(kind) + "_m" + std::to_string(m) + "_k" + std::to_string(k) + "_n" +
           std::to_string(n);
}

INSTANTIATE_TEST_SUITE_P(AllStrategies, MatmulStrategyParamTest,
                         ::testing::Combine(::testing::Values(StrategyKind::Naive,
                                                              StrategyKind::Shmem16,
                                                              StrategyKind::Shmem32),
                                            ::testing::ValuesIn(kSizes),
                                            ::testing::ValuesIn(kSizes),
                                            ::testing::ValuesIn(kSizes)),
                         strategy_param_name);

// ---------------------------------------------------------------------------
// Граничные случаи.
// ---------------------------------------------------------------------------

TEST(MatmulTest, KnownSmallProduct) {
    // [1 2 3]   [ 7  8]   [ 58  64]
    // [4 5 6] * [ 9 10] = [139 154]
    //           [11 12]
    const std::vector<float> a_host = {1, 2, 3, 4, 5, 6};
    const std::vector<float> b_host = {7, 8, 9, 10, 11, 12};
    Matrix<float> a(2, 3);
    Matrix<float> b(3, 2);
    a.data().copy_from_host(a_host.data());
    b.data().copy_from_host(b_host.data());

    const Matrix<float> c = a * b;

    std::vector<float> c_host(c.size());
    c.data().copy_to_host(c_host.data());
    EXPECT_EQ(c_host, (std::vector<float>{58, 64, 139, 154}));
}

TEST(MatmulTest, IdentityIsNeutral) {
    constexpr std::size_t n = 129;
    std::srand(7);
    const Eigen::MatrixXf a_host = Eigen::MatrixXf::Random(n, n);

    const Matrix<float> a = to_device(a_host);
    const Matrix<float> identity = to_device<float>(Eigen::MatrixXf::Identity(n, n));

    EXPECT_TRUE(to_host(a * identity).isApprox(a_host, kPrecision));
    EXPECT_TRUE(to_host(identity * a).isApprox(a_host, kPrecision));
}

// Вектор-строка на вектор-столбец и наоборот: сетка из одного блока и сетка
// с вырожденной по одной оси формой; внутренняя размерность пересекает много тайлов.
TEST(MatmulTest, VectorShapes) {
    std::srand(17);
    const Eigen::MatrixXf row = Eigen::MatrixXf::Random(1, 1000);
    const Eigen::MatrixXf col = Eigen::MatrixXf::Random(1000, 1);

    EXPECT_TRUE(to_host(to_device(row) * to_device(col)).isApprox(row * col, kPrecision));
    EXPECT_TRUE(to_host(to_device(col) * to_device(row)).isApprox(col * row, kPrecision));
}

// Размеры вокруг границ тайлов 16 и 32: проверяют загрузку неполных тайлов нулями.
TEST(MatmulTest, TileBoundaries) {
    const std::vector<std::size_t> sizes = {15, 16, 17, 31, 32, 33, 63, 65};
    for (const std::size_t n : sizes) {
        std::srand(static_cast<unsigned int>(n));
        const Eigen::MatrixXf a_host = Eigen::MatrixXf::Random(n, n + 1);
        const Eigen::MatrixXf b_host = Eigen::MatrixXf::Random(n + 1, n - 1);

        const Eigen::MatrixXf actual = to_host(to_device(a_host) * to_device(b_host));
        EXPECT_TRUE(actual.isApprox(a_host * b_host, kPrecision)) << "n = " << n;
    }
}

// Результат не зависит от количества запусков: повторное умножение не должно
// использовать данные, оставшиеся в разделяемой памяти от предыдущих запусков.
TEST(MatmulTest, RepeatedProductIsStable) {
    std::srand(19);
    const Eigen::MatrixXf a_host = Eigen::MatrixXf::Random(100, 70);
    const Eigen::MatrixXf b_host = Eigen::MatrixXf::Random(70, 90);
    const Matrix<float> a = to_device(a_host);
    const Matrix<float> b = to_device(b_host);

    const Eigen::MatrixXf first = to_host(a * b);
    const Eigen::MatrixXf second = to_host(a * b);

    EXPECT_EQ(first, second);
}

TEST(MatmulTest, OperandsAreNotModified) {
    std::srand(11);
    const Eigen::MatrixXf a_host = Eigen::MatrixXf::Random(64, 32);
    const Eigen::MatrixXf b_host = Eigen::MatrixXf::Random(32, 48);
    const Matrix<float> a = to_device(a_host);
    const Matrix<float> b = to_device(b_host);

    const Matrix<float> c = a * b;

    EXPECT_EQ(to_host(a), a_host);
    EXPECT_EQ(to_host(b), b_host);
    EXPECT_NE(c.data().data(), a.data().data());
    EXPECT_NE(c.data().data(), b.data().data());
}

TEST(MatmulTest, DimensionMismatchThrows) {
    const Matrix<float> a(3, 4);
    const Matrix<float> b(5, 3);
    EXPECT_THROW(a * b, std::invalid_argument);
}

TEST(MatmulTest, ZeroInnerDimensionGivesZeroMatrix) {
    const Matrix<float> a(4, 0);
    const Matrix<float> b(0, 5);

    const Matrix<float> c = a * b;

    ASSERT_EQ(c.nrows(), 4u);
    ASSERT_EQ(c.ncols(), 5u);
    EXPECT_TRUE(to_host(c).isZero());
}

TEST(MatmulTest, EmptyResult) {
    const Matrix<float> a(0, 3);
    const Matrix<float> b(3, 2);

    const Matrix<float> c = a * b;

    EXPECT_EQ(c.nrows(), 0u);
    EXPECT_EQ(c.ncols(), 2u);
    EXPECT_EQ(c.size(), 0u);
}

TEST(MatmulTest, DoublePrecision) {
    std::srand(13);
    const Eigen::MatrixXd a_host = Eigen::MatrixXd::Random(129, 127);
    const Eigen::MatrixXd b_host = Eigen::MatrixXd::Random(127, 3);

    const Matrix<double> c = to_device(a_host) * to_device(b_host);

    EXPECT_TRUE(to_host(c).isApprox(a_host * b_host, 1e-12));
}

TEST(MatmulTest, IntegerProductIsExact) {
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> a_host(3, 2);
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> b_host(2, 4);
    a_host << 1, -2, 3, 4, -5, 6;
    b_host << 1, 2, 3, 4, 5, 6, 7, 8;

    const Matrix<int> c = to_device(a_host) * to_device(b_host);

    EXPECT_EQ(to_host(c), (a_host * b_host).eval());
}

// Многократное умножение не должно приводить к утечке памяти устройства:
// после уничтожения всех матриц объём свободной памяти возвращается к исходному.
TEST(MatmulTest, NoDeviceMemoryLeak) {
    constexpr std::size_t n = 512;
    constexpr int kIterations = 50;

    // Первое обращение к CUDA и умножение инициализируют контекст и модуль заранее.
    { const Matrix<float> warm_up = Matrix<float>(n, n) * Matrix<float>(n, n); }

    std::size_t free_before = 0;
    std::size_t total = 0;
    check_cuda(cudaMemGetInfo(&free_before, &total), "cudaMemGetInfo");

    {
        const Matrix<float> a(n, n);
        const Matrix<float> b(n, n);
        for (int i = 0; i < kIterations; ++i) {
            const Matrix<float> c = a * b;
            const Matrix<float> copy = c;  // разделяет данные, не выделяет память
            Data<float> deep_copy(c.data());
        }
    }

    std::size_t free_after = 0;
    check_cuda(cudaMemGetInfo(&free_after, &total), "cudaMemGetInfo");
    EXPECT_EQ(free_after, free_before);
}

// ---------------------------------------------------------------------------
// Стратегии умножения.
// ---------------------------------------------------------------------------

TEST(MatmulStrategyTest, DefaultIsSharedMemory) {
    const Matrix<float> a(2, 2);
    EXPECT_NE(dynamic_cast<const SharedMemoryMatmul<float>*>(&a.matmul_strategy()), nullptr);
}

TEST(MatmulStrategyTest, ResultInheritsLhsStrategy) {
    const auto naive = std::make_shared<const NaiveMatmul<float>>();
    const Matrix<float> a(3, 4, naive);
    const Matrix<float> b(4, 5);

    const Matrix<float> c = a * b;

    EXPECT_EQ(&c.matmul_strategy(), naive.get());
}

TEST(MatmulStrategyTest, NullStrategyThrows) {
    EXPECT_THROW(Matrix<float>(2, 2, nullptr), std::invalid_argument);

    Matrix<float> a(2, 2);
    EXPECT_THROW(a.set_matmul_strategy(nullptr), std::invalid_argument);
}

TEST(MatmulStrategyTest, StrategiesAgree) {
    std::srand(23);
    const Eigen::MatrixXf a_host = Eigen::MatrixXf::Random(300, 200);
    const Eigen::MatrixXf b_host = Eigen::MatrixXf::Random(200, 100);
    Matrix<float> a = to_device(a_host);
    const Matrix<float> b = to_device(b_host);

    a.set_matmul_strategy(make_strategy(StrategyKind::Naive));
    const Eigen::MatrixXf naive = to_host(a * b);
    a.set_matmul_strategy(make_strategy(StrategyKind::Default));
    const Eigen::MatrixXf shmem = to_host(a * b);

    EXPECT_TRUE(shmem.isApprox(naive, kPrecision));
}

// ---------------------------------------------------------------------------
// Matrix / MatrixView / Data.
// ---------------------------------------------------------------------------

TEST(MatrixViewTest, IsTriviallyCopyable) {
    static_assert(std::is_trivially_copyable_v<MatrixView<float>>);
    static_assert(std::is_trivially_copyable_v<MatrixView<double>>);
    static_assert(std::is_trivially_copyable_v<MatrixView<int>>);
}

TEST(MatrixViewTest, RowMajorIndexing) {
    std::vector<int> storage(12);
    MatrixView<int> view(storage.data(), 3, 4);
    for (std::size_t i = 0; i < view.nrows(); ++i) {
        for (std::size_t j = 0; j < view.ncols(); ++j) {
            view(i, j) = static_cast<int>(i * 10 + j);
        }
    }

    EXPECT_EQ(view.size(), 12u);
    EXPECT_EQ(view[0], 0);
    EXPECT_EQ(view[3], 3);
    EXPECT_EQ(view[4], 10);
    EXPECT_EQ(view[11], 23);
}

TEST(MatrixTest, CopiesShareData) {
    Matrix<float> a(2, 2);
    const Matrix<float> b = a;  // NOLINT: проверяем именно копирование

    EXPECT_EQ(a.data().data(), b.data().data());
    EXPECT_NE(&a.view(), &b.view());  // собственное представление у каждого экземпляра
    EXPECT_EQ(&a.matmul_strategy(), &b.matmul_strategy());

    const std::vector<float> values = {1, 2, 3, 4};
    a.data().copy_from_host(values.data());
    std::vector<float> read_back(4);
    b.data().copy_to_host(read_back.data());
    EXPECT_EQ(read_back, values);
}

TEST(MatrixTest, DimensionsAndSize) {
    const Matrix<float> a(3, 7);
    EXPECT_EQ(a.nrows(), 3u);
    EXPECT_EQ(a.ncols(), 7u);
    EXPECT_EQ(a.size(), 21u);
    EXPECT_EQ(a.data().size(), 21u);
}

TEST(DataTest, CopyIsDeepAndMoveTransfersOwnership) {
    const std::vector<float> values = {1, 2, 3};
    Data<float> original(3);
    original.copy_from_host(values.data());

    Data<float> copy(original);
    EXPECT_NE(copy.data(), original.data());
    std::vector<float> read_back(3);
    copy.copy_to_host(read_back.data());
    EXPECT_EQ(read_back, values);

    const float* raw = copy.data();
    Data<float> moved(std::move(copy));
    EXPECT_EQ(moved.data(), raw);
    EXPECT_EQ(copy.data(), nullptr);  // NOLINT: проверяем состояние после перемещения
    EXPECT_EQ(copy.size(), 0u);
}
