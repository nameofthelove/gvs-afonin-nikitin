"""Загрузка результатов Google Benchmark (JSON) для построения графиков."""

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = ROOT / "results" / "benchmark_results.json"
LARGE_JSON = ROOT / "results" / "benchmark_results_large.json"
IMAGES_DIR = ROOT / "docs" / "images"

SHMEM_FLOAT = "BM_Matrix_operator_mul<ShmemFloat>"
SHMEM_HALF = "BM_Matrix_operator_mul<ShmemHalf>"
WMMA = "BM_Matrix_operator_mul<Wmma>"
WMMA_MIN = "BM_Matrix_operator_mul<Wmma_32_64_32>"
PEAK = "BM_Wmma_peak"

LABELS = {
    SHMEM_FLOAT: "kernel_matmul_shmem, float (работа №3)",
    SHMEM_HALF: "kernel_matmul_shmem, half (работа №3)",
    WMMA: "kernel_matmul_wmma, тайл 128×128×32",
    WMMA_MIN: "kernel_matmul_wmma, тайл 32×64×32",
}

COLORS = {SHMEM_FLOAT: "#7570b3", SHMEM_HALF: "#d95f02", WMMA: "#1b9e77", WMMA_MIN: "#66a61e"}
SYMBOLS = {SHMEM_FLOAT: "diamond", SHMEM_HALF: "circle", WMMA: "square", WMMA_MIN: "triangle-up"}

# Типичная частота GPU под нагрузкой (nvidia-smi --query-gpu=clocks.gr), МГц:
# cudaDevAttrClockRate возвращает 1485 МГц, а во время измерений частота составляла 1845–1950 МГц.
LOADED_CLOCK_MHZ = 1900

# Постоянные накладные расходы на запуск кёрнела и запись событий, с (медиана T при n <= 64).
OVERHEAD_SIZES = (16, 32, 64)

# Множители перевода в секунды для поля time_unit.
_TO_SECONDS = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


@dataclass
class Series:
    sizes: list[int]
    median_s: list[float]
    stddev_s: list[float]


def load_json(json_path: Path = DEFAULT_JSON) -> dict:
    # На Windows путь к исполняемому файлу в "context" пишется в ANSI-кодировке.
    with open(json_path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


def load(json_path: Path = DEFAULT_JSON, large: bool = False) -> dict[str, Series]:
    """Возвращает {имя бенчмарка: Series} по агрегатам median и stddev.

    large=True добавляет контрольные размеры сверх задания (2048 и 4096).
    """
    benchmarks = load_json(json_path)["benchmarks"]
    if large:
        benchmarks = benchmarks + load_json(LARGE_JSON)["benchmarks"]

    stats: dict[str, dict[int, dict[str, float]]] = {}
    for bench in benchmarks:
        aggregate = bench.get("aggregate_name")
        if aggregate not in ("median", "stddev"):
            continue
        parts = bench["run_name"].split("/")
        if not parts[1].isdigit():  # калибровочный бенчмарк без размера матрицы
            continue
        seconds = bench["real_time"] * _TO_SECONDS[bench["time_unit"]]
        stats.setdefault(parts[0], {}).setdefault(int(parts[1]), {})[aggregate] = seconds

    result = {}
    for family, by_size in stats.items():
        sizes = sorted(by_size)
        result[family] = Series(
            sizes=sizes,
            median_s=[by_size[n]["median"] for n in sizes],
            stddev_s=[by_size[n].get("stddev", 0.0) for n in sizes],
        )
    return result


def peak_gflops(json_path: Path = DEFAULT_JSON) -> float:
    """Измеренный предел тензорных ядер, GFLOP/s (калибровочный бенчмарк BM_Wmma_peak)."""
    for bench in load_json(json_path)["benchmarks"]:
        if bench["run_name"].startswith(PEAK) and bench.get("aggregate_name") == "median":
            seconds = bench["real_time"] * _TO_SECONDS[bench["time_unit"]]
            return bench["flop"] / seconds / 1e9
    raise ValueError(f"{PEAK} not found in {json_path}")


def geometry(family: str) -> tuple[int, int, int]:
    """(BlockItemsY, BlockItemsX, BlockItemsK) для бенчмарка kernel_matmul_wmma."""
    if family == WMMA:
        return 128, 128, 32
    y, x, k = family.split("<")[1].rstrip(">").split("_")[1:]
    return int(y), int(x), int(k)


def is_wmma(family: str) -> bool:
    return "<Wmma" in family


def gflops(n: int, seconds: float) -> float:
    """Производительность умножения n x n матриц: 2 n^3 операций."""
    return 2.0 * n**3 / seconds / 1e9


def save(fig, name: str) -> None:
    """Сохраняет график в docs/images в форматах PNG и HTML (интерактивный)."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(IMAGES_DIR / f"{name}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(IMAGES_DIR / f"{name}.png", width=1000, height=600, scale=2)
        print(f"Saved {IMAGES_DIR / name}.png / .html")
    except Exception as exc:  # kaleido требует установленный Chrome
        print(f"PNG export failed ({exc}); saved {IMAGES_DIR / name}.html only")
