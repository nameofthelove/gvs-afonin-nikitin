# Практическая работа №4 — умножение матриц на тензорных ядрах CUDA

Дисциплина «Гибридные вычислительные системы», ИКИТ СФУ.

Реализован кёрнел `kernel_matmul_wmma` — умножение матриц `half` на тензорных ядрах (CUDA WMMA).
Блок из 8 варпов (решётка 2×4) вычисляет тайл C в регистрах, тайлы A и B буферизуются в разделяемой
памяти. Алгоритм подключается к `Matrix::operator*` по паттерну «Стратегия»: для `half` по умолчанию
используется `WmmaMatmul` (тайлы 128×128×32), для остальных типов — `SharedMemoryMatmul` из работы №3.

## Структура

```
core/        библиотека: Data, MatrixView, Matrix, стратегии и кёрнелы
  include/kernel.cuh           кёрнелы naive, shmem, wmma и функции запуска
  include/matmul_strategy.cuh  MatmulStrategy, NaiveMatmul, SharedMemoryMatmul, WmmaMatmul
  include/matrix.cuh           фасад Matrix и operator*
tests/       модульные тесты (Google Test, эталон — Eigen::half)
benchmarks/  бенчмарки (Google Benchmark + CUDA Events)
scripts/     построение графиков (plotly)
results/     результаты бенчмарков (JSON) и таблица погрешностей (CSV)
docs/        отчёт и графики
```

## Сборка и запуск

Требуется CUDA Toolkit (проверено на 13.3) и GPU не ниже Volta, CMake ≥ 3.24, Ninja, компилятор
C++20 (MSVC 19.42 / GCC). Google Test, Google Benchmark и Eigen загружаются через FetchContent.
Архитектура GPU по умолчанию — `sm_86`, переопределяется через `-DCMAKE_CUDA_ARCHITECTURES`.
Аппаратное асинхронное копирование `__pipeline_memcpy_async` доступно начиная с Ampere (`sm_80`).

```bash
cmake --preset release
cmake --build build
ctest --test-dir build
./build/benchmarks/benchmarks --benchmark_filter='-/(2048|4096)/' --benchmark_out=results/benchmark_results.json --benchmark_out_format=json
./build/benchmarks/benchmarks --benchmark_filter='/(2048|4096)/' --benchmark_out=results/benchmark_results_large.json --benchmark_out_format=json
./build/tests/unit_tests --gtest_also_run_disabled_tests --gtest_filter='*AccuracyReport*'
```

## Графики

```bash
pip install -r scripts/requirements.txt
python scripts/complexity_chart.py
python scripts/speedup_chart.py
python scripts/performance_chart.py
python scripts/geometry_chart.py
```

Графики сохраняются в `docs/images` (PNG и интерактивный HTML).

## Результаты (RTX 3050 Ti Laptop, медиана из 10 повторов)

| n    | shmem float, мкс | shmem half, мкс | wmma 128×128×32, мкс | ускорение (half) |
|-----:|-----------------:|----------------:|---------------------:|-----------------:|
| 256  | 54,1             | 54,1            | 26,0                 | 2,08             |
| 512  | 392,8            | 384,1           | 42,3                 | 9,08             |
| 1024 | 3280,8           | 3059,5          | 237,1                | 12,90            |
| 2048 | 25769,3          | 25259,2         | 1588,4               | 15,90            |
| 4096 | 207837,0         | 204341,4        | 12741,7              | 16,04            |

При n ≤ 128 время всех версий определяется постоянными накладными расходами запуска (около 18 мкс).
Предел тензорных ядер, измеренный калибровочным бенчмарком, — 37 TFLOP/s; достигнуто 10,8 TFLOP/s:
около 70 % времени уходит на загрузку тайлов из глобальной памяти.
