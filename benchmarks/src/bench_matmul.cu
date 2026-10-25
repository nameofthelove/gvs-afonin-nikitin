#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "cuda_event_timer.cuh"
#include "matrix.cuh"

namespace {

constexpr std::int64_t kMinSize = 16;
constexpr std::int64_t kMaxSize = 1024;
constexpr int kRepetitions = 10;

// Заполняет матрицу на устройстве случайными значениями из [-1, 1].
void fill_random(Matrix<float>& matrix) {
    using RowMajorXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const RowMajorXf host = RowMajorXf::Random(static_cast<Eigen::Index>(matrix.nrows()),
                                               static_cast<Eigen::Index>(matrix.ncols()));
    matrix.data().copy_from_host(host.data());
}

// Параметры устройства попадают в поле context JSON-отчёта.
void add_device_context() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    int clock_khz = 0;
    int memory_clock_khz = 0;
    check_cuda(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device),
               "cudaDeviceGetAttribute(ClockRate)");
    check_cuda(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device),
               "cudaDeviceGetAttribute(MemoryClockRate)");

    benchmark::AddCustomContext("gpu_name", prop.name);
    benchmark::AddCustomContext("gpu_compute_capability",
                                std::to_string(prop.major) + "." + std::to_string(prop.minor));
    benchmark::AddCustomContext("gpu_sm_count", std::to_string(prop.multiProcessorCount));
    benchmark::AddCustomContext("gpu_clock_mhz", std::to_string(clock_khz / 1000));
    benchmark::AddCustomContext("gpu_memory_clock_mhz", std::to_string(memory_clock_khz / 1000));
    benchmark::AddCustomContext("gpu_memory_bus_width_bit",
                                std::to_string(prop.memoryBusWidth));
    benchmark::AddCustomContext("gpu_l2_cache_bytes", std::to_string(prop.l2CacheSize));
    benchmark::AddCustomContext("gpu_shared_mem_per_block_bytes",
                                std::to_string(prop.sharedMemPerBlock));
}

}  // namespace

// Matrix::operator* = выделение памяти под результат + strategy.multiply + синхронизация.
// Выделение/копирование/освобождение памяти вынесены за пределы цикла, а внутри цикла
// CUDA Events измеряют только время выполнения кёрнела выбранной стратегии на устройстве.
template <typename Strategy>
static void BM_Matrix_operator_mul(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));

    std::srand(42);
    Matrix<float> a(n, n);
    Matrix<float> b(n, n);
    Matrix<float> c(n, n);
    fill_random(a);
    fill_random(b);

    const Strategy strategy{};
    CudaEventTimer timer;

    // Прогрев: загрузка модуля, выход GPU из режима энергосбережения.
    strategy.multiply(a.view(), b.view(), c.view());
    check_cuda(cudaDeviceSynchronize(), "warm-up");

    for (auto _ : state) {
        timer.start();
        strategy.multiply(a.view(), b.view(), c.view());
        const float elapsed_ms = timer.stop();
        state.SetIterationTime(static_cast<double>(elapsed_ms) * 1e-3);
    }

    state.counters["n"] = static_cast<double>(n);
    state.counters["flop"] = 2.0 * static_cast<double>(n) * n * n;
}

#define MATMUL_BENCHMARK(strategy)                                     \
    BENCHMARK_TEMPLATE(BM_Matrix_operator_mul, strategy)               \
        ->RangeMultiplier(2)                                           \
        ->Range(kMinSize, kMaxSize)                                    \
        ->UseManualTime()                                              \
        ->Repetitions(kRepetitions)                                    \
        ->ReportAggregatesOnly(true)                                   \
        ->Unit(benchmark::kMicrosecond)

// Предыдущая версия operator* (практическая работа №2).
using Naive = NaiveMatmul<float>;
// Новая версия operator*: тайлы 16 x 16 (по умолчанию) и 32 x 32.
using Shmem16 = SharedMemoryMatmul<float, 16>;
using Shmem32 = SharedMemoryMatmul<float, 32>;

MATMUL_BENCHMARK(Naive);
MATMUL_BENCHMARK(Shmem16);
MATMUL_BENCHMARK(Shmem32);

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    add_device_context();
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
