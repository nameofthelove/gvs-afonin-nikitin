#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "cuda_event_timer.cuh"
#include "vector.cuh"

namespace {

constexpr std::int64_t kMinSize = 8;
constexpr int kMaxPower = 28;       // n = 8 * 2^p, p = 0..28: до 2^31 элементов.
constexpr int kMaxPowerFloat = 26;  // 2^29 float = 2 ГиБ; 2^30 float = 4 ГиБ не помещается в память.
constexpr int kRepetitions = 10;
constexpr std::int64_t kGeometrySize = std::int64_t{1} << 26;

// Заполняет вектор на устройстве случайными значениями.
template <typename AtomT>
void fill_random(Data<AtomT>& data) {
    std::mt19937 engine(42);
    std::vector<AtomT> atoms(data.size());
    if constexpr (std::is_floating_point_v<AtomT>) {
        std::uniform_real_distribution<AtomT> distribution(AtomT{-1}, AtomT{1});
        std::generate(atoms.begin(), atoms.end(), [&] { return distribution(engine); });
    } else {
        std::generate(atoms.begin(), atoms.end(), [&] { return static_cast<AtomT>(engine()); });
    }
    data.copy_from_host(atoms.data());
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
    benchmark::AddCustomContext("gpu_memory_bus_width_bit", std::to_string(prop.memoryBusWidth));
    benchmark::AddCustomContext("gpu_l2_cache_bytes", std::to_string(prop.l2CacheSize));
    benchmark::AddCustomContext("gpu_shared_mem_per_block_bytes",
                                std::to_string(prop.sharedMemPerBlock));
}

// Размеры n = 8 * 2^p, p = 0..max_power.
void add_sizes(benchmark::internal::Benchmark* bench, int max_power) {
    for (int power = 0; power <= max_power; ++power) {
        bench->Arg(kMinSize << power);
    }
}

}  // namespace

// Vector::sum = выделение буфера + strategy.reduce + синхронизация + копирование результата.
// Выделение/копирование/освобождение памяти вынесены за пределы цикла, а внутри цикла
// CUDA Events измеряют только время выполнения кёрнелов выбранной стратегии на устройстве.
template <typename Strategy>
static void BM_Vector_sum(benchmark::State& state) {
    using Atom = typename Strategy::Atom;
    using Sum = typename Strategy::Sum;
    const auto n = static_cast<std::size_t>(state.range(0));

    const Strategy strategy{};
    Data<Atom> data(n);
    Data<Sum> partial(strategy.partial_capacity());
    fill_random(data);

    const VectorView<Atom> in(data.data(), data.size());
    const VectorView<Sum> out(partial.data(), partial.size());
    CudaEventTimer timer;

    // Прогрев: загрузка модуля, выход GPU из режима энергосбережения.
    strategy.reduce(in, out);
    check_cuda(cudaDeviceSynchronize(), "warm-up");

    for (auto _ : state) {
        timer.start();
        strategy.reduce(in, out);
        const float elapsed_ms = timer.stop();
        state.SetIterationTime(static_cast<double>(elapsed_ms) * 1e-3);
    }

    state.counters["n"] = static_cast<double>(n);
    state.counters["bytes"] = static_cast<double>(n * sizeof(Atom));
}

#define VECRED_BENCHMARK(strategy, max_power)                                      \
    BENCHMARK_TEMPLATE(BM_Vector_sum, strategy)                                    \
        ->Apply([](benchmark::internal::Benchmark* b) { add_sizes(b, max_power); }) \
        ->UseManualTime()                                                          \
        ->Repetitions(kRepetitions)                                                \
        ->ReportAggregatesOnly(true)                                               \
        ->Unit(benchmark::kMicrosecond)

// float — основной тип; память GPU (4 ГБ) вмещает его до 2^29 элементов.
using NobrFloat = NoBroadcastVecred<float>;
using BrFloat = BroadcastVecred<float>;
// Байты с 64-разрядным накопителем — весь диапазон задания, до 2^31 элементов.
using NobrByte = NoBroadcastVecred<std::uint8_t, std::uint64_t>;
using BrByte = BroadcastVecred<std::uint8_t, std::uint64_t>;

VECRED_BENCHMARK(NobrFloat, kMaxPowerFloat);
VECRED_BENCHMARK(BrFloat, kMaxPowerFloat);
VECRED_BENCHMARK(NobrByte, kMaxPower);
VECRED_BENCHMARK(BrByte, kMaxPower);

// Выбор геометрии: BlockDim, MaxBlocks при n = 2^26 (float).
#define VECRED_GEOMETRY_BENCHMARK(block_dim, max_blocks)                                       \
    BENCHMARK_TEMPLATE(BM_Vector_sum, NoBroadcastVecred<float, float, block_dim, max_blocks>)  \
        ->Arg(kGeometrySize)                                                                   \
        ->UseManualTime()                                                                      \
        ->Repetitions(kRepetitions)                                                            \
        ->ReportAggregatesOnly(true)                                                           \
        ->Unit(benchmark::kMicrosecond);                                                       \
    BENCHMARK_TEMPLATE(BM_Vector_sum, BroadcastVecred<float, float, block_dim, max_blocks>)    \
        ->Arg(kGeometrySize)                                                                   \
        ->UseManualTime()                                                                      \
        ->Repetitions(kRepetitions)                                                            \
        ->ReportAggregatesOnly(true)                                                           \
        ->Unit(benchmark::kMicrosecond)

VECRED_GEOMETRY_BENCHMARK(64, 64);
VECRED_GEOMETRY_BENCHMARK(128, 120);
VECRED_GEOMETRY_BENCHMARK(256, 20);
VECRED_GEOMETRY_BENCHMARK(256, 60);
VECRED_GEOMETRY_BENCHMARK(256, 240);
VECRED_GEOMETRY_BENCHMARK(512, 60);
VECRED_GEOMETRY_BENCHMARK(1024, 20);

// Калибровка: накладные расходы запуска пустого кёрнела и записи событий.
__global__ void kernel_empty() {}

static void BM_Launch_overhead(benchmark::State& state) {
    const int launches = static_cast<int>(state.range(0));
    CudaEventTimer timer;

    kernel_empty<<<1, kVecredBlockDim>>>();
    check_cuda(cudaDeviceSynchronize(), "warm-up");

    for (auto _ : state) {
        timer.start();
        for (int i = 0; i < launches; ++i) {
            kernel_empty<<<1, kVecredBlockDim>>>();
        }
        const float elapsed_ms = timer.stop();
        check_cuda(cudaGetLastError(), "kernel_empty launch");
        state.SetIterationTime(static_cast<double>(elapsed_ms) * 1e-3);
    }
}

BENCHMARK(BM_Launch_overhead)
    ->Arg(1)
    ->Arg(2)
    ->UseManualTime()
    ->Repetitions(kRepetitions)
    ->ReportAggregatesOnly(true)
    ->Unit(benchmark::kMicrosecond);

// Калибровка: стоимость самой редукции блока без обращений к глобальной памяти.
// Блок много раз сворачивает значения нитей, результат записывается в sink, чтобы компилятор
// не удалил цикл.
template <bool Broadcast>
__global__ void kernel_block_reduce_peak(float* sink, int iterations) {
    __shared__ float partial[kVecredBlockDim];

    float value = static_cast<float>(threadIdx.x);
    for (int i = 0; i < iterations; ++i) {
        if constexpr (Broadcast) {
            value = 1e-3f * block_reduce_warp<float, kVecredBlockDim>(partial, value) + threadIdx.x;
        } else {
            value = 1e-3f * block_reduce_shared<float, kVecredBlockDim>(partial, value) + threadIdx.x;
        }
    }

    if (threadIdx.x == 0) {
        sink[blockIdx.x] = value;
    }
}

template <bool Broadcast>
static void BM_Block_reduce(benchmark::State& state) {
    constexpr int kIterations = 4096;
    const auto blocks = static_cast<unsigned int>(state.range(0));

    Data<float> sink(blocks);
    CudaEventTimer timer;

    kernel_block_reduce_peak<Broadcast><<<blocks, kVecredBlockDim>>>(sink.data(), kIterations);
    check_cuda(cudaDeviceSynchronize(), "warm-up");

    for (auto _ : state) {
        timer.start();
        kernel_block_reduce_peak<Broadcast><<<blocks, kVecredBlockDim>>>(sink.data(), kIterations);
        const float elapsed_ms = timer.stop();
        check_cuda(cudaGetLastError(), "kernel_block_reduce_peak launch");
        state.SetIterationTime(static_cast<double>(elapsed_ms) * 1e-3);
    }

    state.counters["iterations_per_launch"] = kIterations;
}

BENCHMARK_TEMPLATE(BM_Block_reduce, false)
    ->Arg(1)
    ->Arg(120)
    ->UseManualTime()
    ->Repetitions(kRepetitions)
    ->ReportAggregatesOnly(true)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_Block_reduce, true)
    ->Arg(1)
    ->Arg(120)
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
