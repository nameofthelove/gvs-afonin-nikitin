#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "cuda_event_timer.cuh"
#include "kernel.cuh"
#include "matrix.cuh"

namespace {

constexpr std::int64_t kMinSize = 16;
constexpr std::int64_t kMaxSize = 1024;
constexpr int kRepetitions = 5;

// Заполняет матрицу на устройстве случайными значениями из [-1, 1].
void fill_random(Matrix<float>& matrix) {
    using RowMajorXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const RowMajorXf host = RowMajorXf::Random(static_cast<Eigen::Index>(matrix.nrows()),
                                               static_cast<Eigen::Index>(matrix.ncols()));
    matrix.data().copy_from_host(host.data());
}

void set_counters(benchmark::State& state, std::size_t n) {
    state.counters["n"] = static_cast<double>(n);
    state.counters["flop"] = 2.0 * static_cast<double>(n) * n * n;
}

}  // namespace

// Matrix::operator* = выделение памяти под результат + launch_matmul_naive + синхронизация.
// Выделение/копирование/освобождение памяти вынесены за пределы цикла, а внутри цикла
// CUDA Events измеряют только время выполнения кёрнела на устройстве.
static void BM_Matrix_operator_mul(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));

    std::srand(42);
    Matrix<float> a(n, n);
    Matrix<float> b(n, n);
    Matrix<float> c(n, n);
    fill_random(a);
    fill_random(b);

    CudaEventTimer timer;

    // Прогрев: загрузка модуля, выход GPU из режима энергосбережения.
    launch_matmul_naive(a.view(), b.view(), c.view());
    check_cuda(cudaDeviceSynchronize(), "warm-up");

    for (auto _ : state) {
        timer.start();
        launch_matmul_naive(a.view(), b.view(), c.view());
        const float elapsed_ms = timer.stop();
        state.SetIterationTime(static_cast<double>(elapsed_ms) * 1e-3);
    }

    set_counters(state, n);
}

static void BM_Eigen_MatrixXf_operator_mul(benchmark::State& state) {
    const auto n = static_cast<Eigen::Index>(state.range(0));

    std::srand(42);
    const Eigen::MatrixXf a = Eigen::MatrixXf::Random(n, n);
    const Eigen::MatrixXf b = Eigen::MatrixXf::Random(n, n);
    Eigen::MatrixXf c(n, n);  // выделено заранее; noalias() исключает временный объект

    for (auto _ : state) {
        c.noalias() = a * b;
        benchmark::DoNotOptimize(c.data());
        benchmark::ClobberMemory();
    }

    set_counters(state, static_cast<std::size_t>(n));
}

BENCHMARK(BM_Matrix_operator_mul)
    ->RangeMultiplier(2)
    ->Range(kMinSize, kMaxSize)
    ->UseManualTime()
    ->Repetitions(kRepetitions)
    ->ReportAggregatesOnly(true)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Eigen_MatrixXf_operator_mul)
    ->RangeMultiplier(2)
    ->Range(kMinSize, kMaxSize)
    ->Repetitions(kRepetitions)
    ->ReportAggregatesOnly(true)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
