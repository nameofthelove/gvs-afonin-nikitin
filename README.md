# Практическая работа №3 — умножение матриц с разделяемой памятью CUDA

Дисциплина «Гибридные вычислительные системы», ИКИТ СФУ.

Реализован кёрнел `kernel_matmul_shmem` — блочное умножение матриц с буферизацией тайлов
в разделяемой памяти и синхронизацией нитей через `__syncthreads`. Алгоритм умножения
подключается к `Matrix::operator*` по паттерну «Стратегия»: по умолчанию используется
`SharedMemoryMatmul` (тайл 16×16), предыдущая версия из работы №2 доступна как `NaiveMatmul`.

## Структура

```
core/        библиотека: Data, MatrixView, Matrix, стратегии и кёрнелы
  include/kernel.cuh           kernel_matmul_naive, kernel_matmul_shmem и функции запуска
  include/matmul_strategy.cuh  MatmulStrategy, NaiveMatmul, SharedMemoryMatmul
  include/matrix.cuh           фасад Matrix и operator*
tests/       модульные тесты (Google Test, эталон — Eigen)
benchmarks/  бенчмарки (Google Benchmark + CUDA Events)
scripts/     построение графиков (plotly)
results/     результаты бенчмарков в JSON
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
./build/benchmarks/benchmarks --benchmark_out=results/benchmark_results.json --benchmark_out_format=json
```

## Графики

```bash
pip install -r scripts/requirements.txt
python scripts/complexity_chart.py
python scripts/speedup_chart.py
python scripts/performance_chart.py
```

Графики сохраняются в `docs/images` (PNG и интерактивный HTML).

## Результаты (RTX 3050 Ti Laptop, медиана из 10 повторов)

| n    | naive, мкс | shmem 16×16, мкс | ускорение |
|-----:|-----------:|-----------------:|----------:|
| 128  | 19,2       | 15,9             | 1,21      |
| 256  | 74,5       | 54,1             | 1,38      |
| 512  | 543,0      | 407,3            | 1,33      |
| 1024 | 4583,3     | 3565,4           | 1,29      |
