#include <benchmark/benchmark.h>
#include "vector.cuh"
#include <vector>

// «амер производительности сложени€ на CPU
static void BM_VectorAdd_CPU(benchmark::State& state) {
    const size_t N = state.range(0);
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    std::vector<float> c(N, 0.0f);

    for (auto _ : state) {
        for (size_t i = 0; i < N; ++i) {
            c[i] = a[i] + b[i];
        }
        benchmark::DoNotOptimize(c.data());
    }

    state.SetItemsProcessed(state.iterations() * N);
    state.SetBytesProcessed(state.iterations() * N * 3 * sizeof(float));
}
BENCHMARK(BM_VectorAdd_CPU)->RangeMultiplier(8)->Range(1024, 1 << 24);

// «амер производительности сложени€ на GPU (CUDA)
static void BM_VectorAdd_GPU(benchmark::State& state) {
    const size_t N = state.range(0);
    Vector<float> a(N);
    Vector<float> b(N);

    for (size_t i = 0; i < N; ++i) {
        a.view()[i] = 1.0f;
        b.view()[i] = 2.0f;
    }

    for (auto _ : state) {
        Vector<float> c = a + b;
        benchmark::DoNotOptimize(c.view().size());
    }

    state.SetItemsProcessed(state.iterations() * N);
    state.SetBytesProcessed(state.iterations() * N * 3 * sizeof(float));
}
BENCHMARK(BM_VectorAdd_GPU)->RangeMultiplier(8)->Range(1024, 1 << 24);

BENCHMARK_MAIN();