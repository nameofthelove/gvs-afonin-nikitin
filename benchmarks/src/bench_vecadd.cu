#include <cstddef>
#include <vector>

#define EIGEN_NO_CUDA
#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include "kernel.cuh"
#include "vector.cuh"

namespace {

std::vector<std::size_t> benchmark_sizes() {
    // n in {8, 8^2, ..., 8^8}
    return {8, 64, 512, 4096, 32768, 262144, 2097152, 16777216};
}

}  // namespace

static void BM_Vector_operator_plus(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    std::vector<float> host_a(n, 1.0f);
    std::vector<float> host_b(n, 2.0f);

    Vector<float> a(n);
    Vector<float> b(n);
    Vector<float> result(n);
    a.data().copy_from_host(host_a.data());
    b.data().copy_from_host(host_b.data());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up (not timed): same compute path as operator+.
    launch_vecadd(a.view(), b.view(), result.view());
    cudaDeviceSynchronize();

    for (auto _ : state) {
        // Measure only kernel time (ignore alloc / H2D / D2H / free).
        cudaEventRecord(start);
        launch_vecadd(a.view(), b.view(), result.view());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        state.SetIterationTime(static_cast<double>(elapsed_ms) * 1e-3);
        benchmark::DoNotOptimize(result.view().size());
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
    state.SetBytesProcessed(
        state.iterations() * static_cast<int64_t>(n) * 3 * static_cast<int64_t>(sizeof(float)));
}

static void BM_Eigen_VectorXf_operator_plus(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    Eigen::VectorXf a = Eigen::VectorXf::Constant(static_cast<Eigen::Index>(n), 1.0f);
    Eigen::VectorXf b = Eigen::VectorXf::Constant(static_cast<Eigen::Index>(n), 2.0f);
    Eigen::VectorXf result(static_cast<Eigen::Index>(n));

    for (auto _ : state) {
        result = a + b;
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
    state.SetBytesProcessed(
        state.iterations() * static_cast<int64_t>(n) * 3 * static_cast<int64_t>(sizeof(float)));
}

static void RegisterBenchmarks() {
    for (const std::size_t n : benchmark_sizes()) {
        benchmark::RegisterBenchmark("BM_Vector_operator_plus", BM_Vector_operator_plus)
            ->Arg(static_cast<int64_t>(n))
            ->UseManualTime()
            ->Unit(benchmark::kMicrosecond);

        benchmark::RegisterBenchmark("BM_Eigen_VectorXf_operator_plus",
                                     BM_Eigen_VectorXf_operator_plus)
            ->Arg(static_cast<int64_t>(n))
            ->Unit(benchmark::kMicrosecond);
    }
}

int main(int argc, char** argv) {
    RegisterBenchmarks();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
