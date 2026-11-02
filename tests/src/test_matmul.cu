#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "eigen_interop.cuh"
#include "matrix.cuh"

namespace {

constexpr double kPrecision = 1e-2;

const std::vector<std::size_t> kSizes = {16, 32, 64, 128, 256, 512};

// Проверяемые стратегии operator* для half.
enum class StrategyKind { Default, Naive, Shmem16, WmmaNarrow, WmmaWide, WmmaHuge };

std::shared_ptr<const MatmulStrategy<half>> make_strategy(StrategyKind kind) {
    switch (kind) {
        case StrategyKind::Naive:
            return std::make_shared<const NaiveMatmul<half>>();
        case StrategyKind::Shmem16:
            return std::make_shared<const SharedMemoryMatmul<half, 16>>();
        case StrategyKind::WmmaNarrow:
            return std::make_shared<const WmmaMatmul<32, 64, 16>>();
        case StrategyKind::WmmaWide:
            return std::make_shared<const WmmaMatmul<64, 128, 64>>();
        case StrategyKind::WmmaHuge:
            return std::make_shared<const WmmaMatmul<256, 256, 32>>();
        case StrategyKind::Default:
        default:
            return default_matmul_strategy<half>();
    }
}

std::string to_string(StrategyKind kind) {
    switch (kind) {
        case StrategyKind::Naive:
            return "Naive";
        case StrategyKind::Shmem16:
            return "Shmem16";
        case StrategyKind::WmmaNarrow:
            return "WmmaNarrow";
        case StrategyKind::WmmaWide:
            return "WmmaWide";
        case StrategyKind::WmmaHuge:
            return "WmmaHuge";
        case StrategyKind::Default:
        default:
            return "Default";
    }
}

// Случайная матрица half с элементами из [-1, 1].
HalfMatrix random_half(std::size_t rows, std::size_t cols) {
    return Eigen::MatrixXf::Random(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols))
        .cast<Eigen::half>();
}

// Сравнение с эталоном методом Eigen::MatrixXf::isApprox с точностью kPrecision.
::testing::AssertionResult is_approx(const HalfMatrix& actual, const HalfMatrix& expected) {
    const Eigen::MatrixXf actual_f = actual.cast<float>();
    const Eigen::MatrixXf expected_f = expected.cast<float>();
    if (actual_f.isApprox(expected_f, kPrecision)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
           << "max |diff| = " << (actual_f - expected_f).cwiseAbs().maxCoeff();
}

// isApprox сравнивает разность с нормой результата. При сокращении слагаемых результат малых
// матриц близок к нулю и этот критерий определяется округлением half, поэтому для вырожденных
// размеров дополнительно допускается абсолютное отклонение kPrecision по норме.
::testing::AssertionResult is_close(const HalfMatrix& actual, const HalfMatrix& expected) {
    const Eigen::MatrixXf difference = actual.cast<float>() - expected.cast<float>();
    if (difference.norm() <= kPrecision) {
        return ::testing::AssertionSuccess();
    }
    return is_approx(actual, expected);
}

// Точное произведение матриц half: разрядности double достаточно для сумм без округления.
Eigen::MatrixXd exact_product(const HalfMatrix& a, const HalfMatrix& b) {
    return a.cast<float>().cast<double>() * b.cast<float>().cast<double>();
}

double error_norm(const HalfMatrix& actual, const Eigen::MatrixXd& exact) {
    return (actual.cast<float>().cast<double>() - exact).norm();
}

}  // namespace

// ---------------------------------------------------------------------------
// Параметризованные тесты: A (m x k) * B (k x n) для всех m, n, k из kSizes.
// ---------------------------------------------------------------------------

using MatmulParams = std::tuple<std::size_t, std::size_t, std::size_t>;  // m, k, n

class MatmulParamTest : public ::testing::TestWithParam<MatmulParams> {};

// Основной набор: operator* со стратегией по умолчанию (kernel_matmul_wmma).
TEST_P(MatmulParamTest, MatchesEigen) {
    const auto [m, k, n] = GetParam();

    std::srand(static_cast<unsigned int>(m * 1'000'003 + k * 1'009 + n));
    const HalfMatrix a_host = random_half(m, k);
    const HalfMatrix b_host = random_half(k, n);
    const HalfMatrix expected = a_host * b_host;

    const Matrix<half> a = to_device(a_host);
    const Matrix<half> b = to_device(b_host);
    const Matrix<half> c = a * b;

    ASSERT_EQ(c.nrows(), m);
    ASSERT_EQ(c.ncols(), n);
    ASSERT_EQ(c.size(), m * n);

    EXPECT_TRUE(is_approx(to_host(c), expected));
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
// Те же размеры для каждой стратегии в отдельности (предыдущие версии operator* и
// другие геометрии блока kernel_matmul_wmma).
// ---------------------------------------------------------------------------

using StrategyParams = std::tuple<StrategyKind, std::size_t, std::size_t, std::size_t>;

class MatmulStrategyParamTest : public ::testing::TestWithParam<StrategyParams> {};

TEST_P(MatmulStrategyParamTest, MatchesEigen) {
    const auto [kind, m, k, n] = GetParam();

    std::srand(static_cast<unsigned int>(m * 1'000'003 + k * 1'009 + n));
    const HalfMatrix a_host = random_half(m, k);
    const HalfMatrix b_host = random_half(k, n);
    const HalfMatrix expected = a_host * b_host;

    Matrix<half> a = to_device(a_host);
    const Matrix<half> b = to_device(b_host);
    a.set_matmul_strategy(make_strategy(kind));
    const Matrix<half> c = a * b;

    ASSERT_EQ(c.nrows(), m);
    ASSERT_EQ(c.ncols(), n);

    EXPECT_TRUE(is_approx(to_host(c), expected));
}

std::string strategy_param_name(const ::testing::TestParamInfo<StrategyParams>& info) {
    const auto [kind, m, k, n] = info.param;
    return to_string(kind) + "_m" + std::to_string(m) + "_k" + std::to_string(k) + "_n" +
           std::to_string(n);
}

INSTANTIATE_TEST_SUITE_P(AllStrategies, MatmulStrategyParamTest,
                         ::testing::Combine(::testing::Values(StrategyKind::Naive,
                                                              StrategyKind::Shmem16,
                                                              StrategyKind::WmmaNarrow,
                                                              StrategyKind::WmmaWide,
                                                              StrategyKind::WmmaHuge),
                                            ::testing::ValuesIn(kSizes),
                                            ::testing::ValuesIn(kSizes),
                                            ::testing::ValuesIn(kSizes)),
                         strategy_param_name);

// ---------------------------------------------------------------------------
// Размеры, не кратные фрагменту и тайлам: неполные тайлы дополняются нулями, а строка C
// может быть не выровнена на 16 байт (ncols не кратно 8).
// ---------------------------------------------------------------------------

using UnalignedParams = std::tuple<StrategyKind, std::size_t, std::size_t, std::size_t>;

class MatmulUnalignedParamTest : public ::testing::TestWithParam<UnalignedParams> {};

TEST_P(MatmulUnalignedParamTest, MatchesEigen) {
    const auto [kind, m, k, n] = GetParam();

    std::srand(static_cast<unsigned int>(m * 1'000'003 + k * 1'009 + n));
    const HalfMatrix a_host = random_half(m, k);
    const HalfMatrix b_host = random_half(k, n);

    Matrix<half> a = to_device(a_host);
    const Matrix<half> b = to_device(b_host);
    a.set_matmul_strategy(make_strategy(kind));

    EXPECT_TRUE(is_close(to_host(a * b), a_host * b_host));
}

INSTANTIATE_TEST_SUITE_P(BoundarySizes, MatmulUnalignedParamTest,
                         ::testing::Combine(::testing::Values(StrategyKind::Default,
                                                              StrategyKind::WmmaNarrow,
                                                              StrategyKind::WmmaWide,
                                                              StrategyKind::WmmaHuge),
                                            ::testing::Values(1, 15, 17, 31, 33, 63, 129),
                                            ::testing::Values(1, 15, 17, 33, 129),
                                            ::testing::Values(1, 15, 17, 31, 33, 63, 129)),
                         strategy_param_name);

// ---------------------------------------------------------------------------
// Граничные случаи.
// ---------------------------------------------------------------------------

TEST(MatmulTest, KnownSmallProduct) {
    // [1 2 3]   [ 7  8]   [ 58  64]
    // [4 5 6] * [ 9 10] = [139 154]
    //           [11 12]
    HalfMatrix a_host(2, 3);
    HalfMatrix b_host(3, 2);
    HalfMatrix expected(2, 2);
    a_host << Eigen::half(1), Eigen::half(2), Eigen::half(3), Eigen::half(4), Eigen::half(5),
        Eigen::half(6);
    b_host << Eigen::half(7), Eigen::half(8), Eigen::half(9), Eigen::half(10), Eigen::half(11),
        Eigen::half(12);
    expected << Eigen::half(58), Eigen::half(64), Eigen::half(139), Eigen::half(154);

    EXPECT_EQ(to_host(to_device(a_host) * to_device(b_host)), expected);
}

TEST(MatmulTest, IdentityIsNeutral) {
    constexpr std::size_t n = 129;
    std::srand(7);
    const HalfMatrix a_host = random_half(n, n);
    const HalfMatrix identity_host = HalfMatrix::Identity(n, n);

    const Matrix<half> a = to_device(a_host);
    const Matrix<half> identity = to_device(identity_host);

    EXPECT_TRUE(is_approx(to_host(a * identity), a_host));
    EXPECT_TRUE(is_approx(to_host(identity * a), a_host));
}

// Вектор-строка на вектор-столбец и наоборот: сетка из одного блока и сетка
// с вырожденной по одной оси формой; внутренняя размерность пересекает много тайлов.
TEST(MatmulTest, VectorShapes) {
    std::srand(17);
    const HalfMatrix row = random_half(1, 1000);
    const HalfMatrix col = random_half(1000, 1);

    EXPECT_TRUE(is_approx(to_host(to_device(row) * to_device(col)), row * col));
    EXPECT_TRUE(is_approx(to_host(to_device(col) * to_device(row)), col * row));
}

// Результат не зависит от количества запусков: повторное умножение не должно
// использовать данные, оставшиеся в разделяемой памяти от предыдущих запусков.
TEST(MatmulTest, RepeatedProductIsStable) {
    std::srand(19);
    const Matrix<half> a = to_device(random_half(100, 70));
    const Matrix<half> b = to_device(random_half(70, 90));

    const HalfMatrix first = to_host(a * b);
    const HalfMatrix second = to_host(a * b);

    EXPECT_EQ(first, second);
}

// Представление с невыровненным начальным адресом: и загрузка тайлов, и запись результата
// должны перейти на поэлементный путь, иначе векторные инструкции получат невыровненный адрес.
TEST(MatmulTest, UnalignedViews) {
    constexpr std::size_t m = 48;
    constexpr std::size_t k = 64;
    constexpr std::size_t n = 32;

    std::srand(41);
    const HalfMatrix a_host = random_half(m, k);
    const HalfMatrix b_host = random_half(k, n);

    Data<half> storage_a(m * k + 1);
    Data<half> storage_b(k * n + 1);
    Data<half> storage_c(m * n + 1);
    check_cuda(cudaMemcpy(storage_a.data() + 1, a_host.data(), m * k * sizeof(half),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy (H2D)");
    check_cuda(cudaMemcpy(storage_b.data() + 1, b_host.data(), k * n * sizeof(half),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy (H2D)");

    default_matmul_strategy<half>()->multiply(MatrixView<half>(storage_a.data() + 1, m, k),
                                              MatrixView<half>(storage_b.data() + 1, k, n),
                                              MatrixView<half>(storage_c.data() + 1, m, n));
    check_cuda(cudaDeviceSynchronize(), "matmul execution");

    HalfMatrix actual(m, n);
    check_cuda(cudaMemcpy(actual.data(), storage_c.data() + 1, m * n * sizeof(half),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy (D2H)");
    EXPECT_TRUE(is_approx(actual, a_host * b_host));
}

// Дополнение нулями не должно приводить к записи за границу C: результат занимает начало
// буфера, остаток которого играет роль охранной зоны.
TEST(MatmulTest, DoesNotWriteOutsideResult) {
    constexpr std::size_t m = 33;
    constexpr std::size_t n = 17;
    constexpr std::size_t kGuardSize = 4096;

    std::srand(29);
    const Matrix<half> a = to_device(random_half(m, 65));
    const Matrix<half> b = to_device(random_half(65, n));

    const std::vector<Eigen::half> filled(m * n + kGuardSize, Eigen::half(1.5f));
    Data<half> storage(filled.size());
    storage.copy_from_host(reinterpret_cast<const half*>(filled.data()));

    a.matmul_strategy().multiply(a.view(), b.view(), MatrixView<half>(storage.data(), m, n));
    check_cuda(cudaDeviceSynchronize(), "matmul execution");

    std::vector<Eigen::half> actual(filled.size());
    storage.copy_to_host(reinterpret_cast<half*>(actual.data()));
    const std::vector<Eigen::half> guard(actual.begin() + m * n, actual.end());
    EXPECT_EQ(guard, std::vector<Eigen::half>(kGuardSize, Eigen::half(1.5f)));
}

TEST(MatmulTest, OperandsAreNotModified) {
    std::srand(11);
    const HalfMatrix a_host = random_half(64, 32);
    const HalfMatrix b_host = random_half(32, 48);
    const Matrix<half> a = to_device(a_host);
    const Matrix<half> b = to_device(b_host);

    const Matrix<half> c = a * b;

    EXPECT_EQ(to_host(a), a_host);
    EXPECT_EQ(to_host(b), b_host);
    EXPECT_NE(c.data().data(), a.data().data());
    EXPECT_NE(c.data().data(), b.data().data());
}

TEST(MatmulTest, DimensionMismatchThrows) {
    const Matrix<half> a(3, 4);
    const Matrix<half> b(5, 3);
    EXPECT_THROW(a * b, std::invalid_argument);
}

TEST(MatmulTest, ZeroInnerDimensionGivesZeroMatrix) {
    const Matrix<half> a(4, 0);
    const Matrix<half> b(0, 5);

    const Matrix<half> c = a * b;

    ASSERT_EQ(c.nrows(), 4u);
    ASSERT_EQ(c.ncols(), 5u);
    EXPECT_TRUE(to_host(c).cast<float>().isZero());
}

TEST(MatmulTest, EmptyResult) {
    const Matrix<half> a(0, 3);
    const Matrix<half> b(3, 2);

    const Matrix<half> c = a * b;

    EXPECT_EQ(c.nrows(), 0u);
    EXPECT_EQ(c.ncols(), 2u);
    EXPECT_EQ(c.size(), 0u);
}

// Для типов, которые не поддерживаются тензорными ядрами, operator* работает как раньше.
TEST(MatmulTest, FloatPrecision) {
    std::srand(31);
    const Eigen::MatrixXf a_host = Eigen::MatrixXf::Random(129, 127);
    const Eigen::MatrixXf b_host = Eigen::MatrixXf::Random(127, 65);

    const Matrix<float> c = to_device(a_host) * to_device(b_host);

    EXPECT_TRUE(to_host(c).isApprox(a_host * b_host, 1e-5));
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
    { const Matrix<half> warm_up = Matrix<half>(n, n) * Matrix<half>(n, n); }

    std::size_t free_before = 0;
    std::size_t total = 0;
    check_cuda(cudaMemGetInfo(&free_before, &total), "cudaMemGetInfo");

    {
        const Matrix<half> a(n, n);
        const Matrix<half> b(n, n);
        for (int i = 0; i < kIterations; ++i) {
            const Matrix<half> c = a * b;
            const Matrix<half> copy = c;  // разделяет данные, не выделяет память
            Data<half> deep_copy(c.data());
        }
    }

    std::size_t free_after = 0;
    check_cuda(cudaMemGetInfo(&free_after, &total), "cudaMemGetInfo");
    EXPECT_EQ(free_after, free_before);
}

// ---------------------------------------------------------------------------
// Точность: тензорные ядра округляют сумму реже, чем последовательное накопление в half.
// ---------------------------------------------------------------------------

TEST(MatmulAccuracyTest, WmmaIsMoreAccurateThanSequentialHalf) {
    std::srand(37);
    const HalfMatrix a_host = random_half(256, 256);
    const HalfMatrix b_host = random_half(256, 256);
    const Eigen::MatrixXd exact = exact_product(a_host, b_host);

    const HalfMatrix wmma = to_host(to_device(a_host) * to_device(b_host));
    const HalfMatrix sequential = a_host * b_host;

    EXPECT_LT(error_norm(wmma, exact), error_norm(sequential, exact));
}

// Таблица погрешностей для отчёта (относительно точного произведения в double):
// unit_tests --gtest_also_run_disabled_tests --gtest_filter=*AccuracyReport*
TEST(MatmulAccuracyTest, DISABLED_AccuracyReport) {
    std::printf("n,method,max_abs,rms,relative\n");
    for (const std::size_t n : {16, 64, 256, 512, 1024}) {
        std::srand(static_cast<unsigned int>(n));
        const HalfMatrix a_host = random_half(n, n);
        const HalfMatrix b_host = random_half(n, n);
        const Eigen::MatrixXd exact = exact_product(a_host, b_host);

        Matrix<half> shmem_lhs = to_device(a_host);
        shmem_lhs.set_matmul_strategy(make_strategy(StrategyKind::Shmem16));
        const Matrix<half> b = to_device(b_host);

        const std::vector<std::pair<std::string, HalfMatrix>> results = {
            {"eigen_half", a_host * b_host},
            {"shmem", to_host(shmem_lhs * b)},
            {"wmma", to_host(to_device(a_host) * b)},
        };
        for (const auto& [method, result] : results) {
            const Eigen::MatrixXd error = result.cast<float>().cast<double>() - exact;
            std::printf("%zu,%s,%.5f,%.5f,%.5f\n", n, method.c_str(), error.cwiseAbs().maxCoeff(),
                        std::sqrt(error.array().square().mean()), error.norm() / exact.norm());
        }
    }
}

// ---------------------------------------------------------------------------
// Стратегии умножения.
// ---------------------------------------------------------------------------

TEST(MatmulStrategyTest, DefaultForHalfIsWmma) {
    const Matrix<half> a(2, 2);
    EXPECT_NE(dynamic_cast<const WmmaMatmul<>*>(&a.matmul_strategy()), nullptr);
}

TEST(MatmulStrategyTest, DefaultForFloatIsSharedMemory) {
    const Matrix<float> a(2, 2);
    EXPECT_NE(dynamic_cast<const SharedMemoryMatmul<float>*>(&a.matmul_strategy()), nullptr);
}

TEST(MatmulStrategyTest, ResultInheritsLhsStrategy) {
    const auto naive = std::make_shared<const NaiveMatmul<half>>();
    const Matrix<half> a(3, 4, naive);
    const Matrix<half> b(4, 5);

    const Matrix<half> c = a * b;

    EXPECT_EQ(&c.matmul_strategy(), naive.get());
}

TEST(MatmulStrategyTest, NullStrategyThrows) {
    EXPECT_THROW(Matrix<half>(2, 2, nullptr), std::invalid_argument);

    Matrix<half> a(2, 2);
    EXPECT_THROW(a.set_matmul_strategy(nullptr), std::invalid_argument);
}

TEST(MatmulStrategyTest, StrategiesAgree) {
    std::srand(23);
    const HalfMatrix a_host = random_half(300, 200);
    const HalfMatrix b_host = random_half(200, 100);
    Matrix<half> a = to_device(a_host);
    const Matrix<half> b = to_device(b_host);

    a.set_matmul_strategy(make_strategy(StrategyKind::Shmem16));
    const HalfMatrix shmem = to_host(a * b);
    a.set_matmul_strategy(make_strategy(StrategyKind::Default));
    const HalfMatrix wmma = to_host(a * b);

    EXPECT_TRUE(is_approx(wmma, shmem));
}

// ---------------------------------------------------------------------------
// Matrix / MatrixView / Data.
// ---------------------------------------------------------------------------

TEST(MatrixViewTest, IsTriviallyCopyable) {
    static_assert(std::is_trivially_copyable_v<MatrixView<half>>);
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
    const Matrix<half> a(3, 7);
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
