"""Загрузка результатов Google Benchmark (JSON) для построения графиков."""

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = ROOT / "results" / "benchmark_results.json"
IMAGES_DIR = ROOT / "docs" / "images"

MATRIX = "BM_Matrix_operator_mul"
EIGEN = "BM_Eigen_MatrixXf_operator_mul"

LABELS = {
    MATRIX: "Matrix (CUDA, kernel_matmul_naive)",
    EIGEN: "Eigen::MatrixXf (CPU)",
}

# Множители перевода в секунды для поля time_unit.
_TO_SECONDS = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


@dataclass
class Series:
    sizes: list[int]
    median_s: list[float]
    stddev_s: list[float]


def load(json_path: Path = DEFAULT_JSON) -> dict[str, Series]:
    """Возвращает {имя бенчмарка: Series} по агрегатам median и stddev."""
    # На Windows путь к исполняемому файлу в "context" пишется в ANSI-кодировке.
    with open(json_path, encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    stats: dict[str, dict[int, dict[str, float]]] = {}
    for bench in data["benchmarks"]:
        aggregate = bench.get("aggregate_name")
        if aggregate not in ("median", "stddev"):
            continue
        family, size = bench["run_name"].split("/")[:2]
        seconds = bench["real_time"] * _TO_SECONDS[bench["time_unit"]]
        stats.setdefault(family, {}).setdefault(int(size), {})[aggregate] = seconds

    result = {}
    for family, by_size in stats.items():
        sizes = sorted(by_size)
        result[family] = Series(
            sizes=sizes,
            median_s=[by_size[n]["median"] for n in sizes],
            stddev_s=[by_size[n].get("stddev", 0.0) for n in sizes],
        )
    return result


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
