# Практическая работа №5 — параллельная редукция вектора на CUDA

Дисциплина «Гибридные вычислительные системы», ИКИТ СФУ.

Реализованы два кёрнела суммирования элементов вектора, принимающие `VectorView` по значению:

- `kernel_vecred_nobr` — частичные суммы нитей блока сворачиваются в разделяемой памяти
  (log2(BlockDim) шагов с барьером `__syncthreads` после каждого);
- `kernel_vecred_br` — варпы сворачивают значения широковещанием регистров `__shfl_down_sync`,
  в разделяемой памяти хранятся только суммы варпов.

Редукция двухпроходная: первый проход записывает суммы блоков (не более 120 блоков по 256 нитей),
второй одним блоком сворачивает их на месте. Алгоритм подключается к `Vector::sum` по паттерну
«Стратегия»: `NoBroadcastVecred` и `BroadcastVecred` (по умолчанию).

## Структура

```
core/        библиотека: Data, VectorView, Vector, стратегии и кёрнелы
  include/kernel.cuh           кёрнелы nobr, br, редукции блока и функции запуска
  include/vecred_strategy.cuh  VecredStrategy, NoBroadcastVecred, BroadcastVecred
  include/vector.cuh           фасад Vector и Vector::sum
tests/       модульные тесты (Google Test, эталон — Eigen::Matrix::sum, EXPECT_NEAR 1e-4)
benchmarks/  бенчмарки (Google Benchmark + CUDA Events)
scripts/     построение графиков (plotly)
results/     результаты бенчмарков (JSON)
docs/        отчёт и графики
```

## Сборка и запуск

Требуется CUDA Toolkit (проверено на 13.3), CMake ≥ 3.24, Ninja, компилятор C++20
(MSVC 19.42 / GCC). Google Test, Google Benchmark и Eigen загружаются через FetchContent.
Архитектура GPU по умолчанию — `sm_86`, переопределяется через `-DCMAKE_CUDA_ARCHITECTURES`.

```bash
cmake --preset release
cmake --build build
ctest --test-dir build
./build/tests/unit_tests --gtest_also_run_disabled_tests --gtest_filter='*MaxBenchmarkSize*'
./build/benchmarks/benchmarks --benchmark_filter='BM_Vector_sum<(Nobr|Br)|BM_Launch|BM_Block' --benchmark_enable_random_interleaving=true --benchmark_out=results/benchmark_results.json --benchmark_out_format=json
./build/benchmarks/benchmarks --benchmark_filter='BM_Vector_sum<.*Vecred<float.*/67108864' --benchmark_out=results/sweep_geometry.json --benchmark_out_format=json
```

Размеры векторов в бенчмарках: n = 8·2^p, p = 0..28. Тип `float` измеряется до 2^29 элементов
(2 ГиБ; 2^30 float не помещается в 4 ГБ памяти GPU), тип `uint8_t` с 64-разрядным накопителем —
во всём диапазоне до 2^31 элементов.

## Графики

```bash
pip install -r scripts/requirements.txt
python scripts/complexity_chart.py
python scripts/speedup_chart.py
```

Графики сохраняются в `docs/images` (PNG и интерактивный HTML).
