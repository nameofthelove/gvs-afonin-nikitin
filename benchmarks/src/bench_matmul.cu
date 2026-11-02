#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "cuda_event_timer.cuh"
#include "matrix.cuh"

namespace {

constexpr std::int64_t kMinSize = 16;
constexpr std::int64_t kMaxSize = 1024;
constexpr int kRepetitions = 10;

// Заполняет матрицу на устройстве случайными значениями из [-1, 1].
template <typename AtomT>
void fill_random(Matrix<AtomT>& matrix) {
    using RowMajorXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const RowMajorXf host = RowMajorXf::Random(static_cast<Eigen::Index>(matrix.nrows()),
                                               static_cast<Eigen::Index>(matrix.ncols()));
    std::vector<AtomT> atoms(static_cast<std::size_t>(host.size()));
    std::transform(host.data(), host.data() + host.size(), atoms.begin(),
                   [](float value) { return static_cast<AtomT>(value); });
    matrix.data().copy_from_host(atoms.data());
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
    using Atom = typename Strategy::Atom;
    const auto n = static_cast<std::size_t>(state.range(0));

    std::srand(42);
    Matrix<Atom> a(n, n);
    Matrix<Atom> b(n, n);
    Matrix<Atom> c(n, n);
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

// Предыдущая версия operator* (практическая работа №3): на float и на тех же данных half.
using ShmemFloat = SharedMemoryMatmul<float, 16>;
using ShmemHalf = SharedMemoryMatmul<half, 16>;
// Новая версия operator*.
using Wmma = WmmaMatmul<>;
// Другие геометрии блока: BlockItemsY, BlockItemsX, BlockItemsK.
using Wmma_32_64_16 = WmmaMatmul<32, 64, 16>;
using Wmma_32_64_32 = WmmaMatmul<32, 64, 32>;
using Wmma_64_64_32 = WmmaMatmul<64, 64, 32>;
using Wmma_64_128_32 = WmmaMatmul<64, 128, 32>;
using Wmma_128_128_16 = WmmaMatmul<128, 128, 16>;
using Wmma_128_128_64 = WmmaMatmul<128, 128, 64>;
using Wmma_128_256_32 = WmmaMatmul<128, 256, 32>;
using Wmma_256_256_32 = WmmaMatmul<256, 256, 32>;

MATMUL_BENCHMARK(ShmemFloat);
MATMUL_BENCHMARK(ShmemHalf);
MATMUL_BENCHMARK(Wmma);
MATMUL_BENCHMARK(Wmma_32_64_16);
MATMUL_BENCHMARK(Wmma_32_64_32);
MATMUL_BENCHMARK(Wmma_64_64_32);
MATMUL_BENCHMARK(Wmma_64_128_32);
MATMUL_BENCHMARK(Wmma_128_128_16);
MATMUL_BENCHMARK(Wmma_128_128_64);
MATMUL_BENCHMARK(Wmma_128_256_32);
MATMUL_BENCHMARK(Wmma_256_256_32);

// Контрольные размеры сверх задания: при n <= 512 блоков kernel_matmul_wmma меньше, чем SM,
// поэтому кубическую асимптотику можно проверить только на больших матрицах.
#define MATMUL_LARGE_BENCHMARK(strategy)                               \
    BENCHMARK_TEMPLATE(BM_Matrix_operator_mul, strategy)               \
        ->Arg(2048)                                                    \
        ->Arg(4096)                                                    \
        ->UseManualTime()                                              \
        ->Repetitions(kRepetitions)                                    \
        ->ReportAggregatesOnly(true)                                   \
        ->Unit(benchmark::kMicrosecond)

MATMUL_LARGE_BENCHMARK(ShmemFloat);
MATMUL_LARGE_BENCHMARK(ShmemHalf);
MATMUL_LARGE_BENCHMARK(Wmma);
MATMUL_LARGE_BENCHMARK(Wmma_32_64_32);

// Калибровка: предельный темп операций mma_sync без обращений к памяти. Сетка из 120 блоков
// занимает все слоты 20 SM (6 блоков по 256 нитей); накопитель записывается в sink, чтобы
// компилятор не удалил цикл.
__global__ void kernel_mma_peak(half* sink, int iterations) {
    namespace wmma = nvcuda::wmma;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc;
    wmma::fill_fragment(frag_a, __float2half(1e-3f));
    wmma::fill_fragment(frag_b, __float2half(1e-3f));
    wmma::fill_fragment(acc, half{});

    for (int i = 0; i < iterations; ++i) {
        wmma::mma_sync(acc, frag_a, frag_b, acc);
    }

    const unsigned int warp = blockIdx.x * kWmmaWarps + threadIdx.x / kWarpSize;
    wmma::store_matrix_sync(sink + warp * kWmmaFragItems, acc, kWmmaFragDim, wmma::mem_row_major);
}

static void BM_Wmma_peak(benchmark::State& state) {
    constexpr unsigned int kBlocks = 120;
    constexpr int kIterations = 16384;

    Data<half> sink(kBlocks * kWmmaWarps * kWmmaFragItems);
    CudaEventTimer timer;

    kernel_mma_peak<<<kBlocks, kWmmaBlockThreads>>>(sink.data(), kIterations);
    check_cuda(cudaDeviceSynchronize(), "warm-up");

    for (auto _ : state) {
        timer.start();
        kernel_mma_peak<<<kBlocks, kWmmaBlockThreads>>>(sink.data(), kIterations);
        const float elapsed_ms = timer.stop();
        check_cuda(cudaGetLastError(), "kernel_mma_peak launch");
        state.SetIterationTime(static_cast<double>(elapsed_ms) * 1e-3);
    }

    // Одна операция mma_sync — 16^3 умножений и столько же сложений.
    state.counters["flop"] = 2.0 * kBlocks * kWmmaWarps * kIterations * 4096.0;
}

BENCHMARK(BM_Wmma_peak)
    ->UseManualTime()
    ->Repetitions(kRepetitions)
    ->ReportAggregatesOnly(true)
    ->Unit(benchmark::kMicrosecond);

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
