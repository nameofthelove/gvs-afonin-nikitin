"""Загрузка результатов Google Benchmark (JSON) для построения графиков."""

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = ROOT / "results" / "benchmark_results.json"
GEOMETRY_JSON = ROOT / "results" / "sweep_geometry.json"
IMAGES_DIR = ROOT / "docs" / "images"

NOBR_FLOAT = "BM_Vector_sum<NobrFloat>"
BR_FLOAT = "BM_Vector_sum<BrFloat>"
NOBR_BYTE = "BM_Vector_sum<NobrByte>"
BR_BYTE = "BM_Vector_sum<BrByte>"

LABELS = {
    NOBR_FLOAT: "kernel_vecred_nobr, float",
    BR_FLOAT: "kernel_vecred_br, float",
    NOBR_BYTE: "kernel_vecred_nobr, uint8",
    BR_BYTE: "kernel_vecred_br, uint8",
}

COLORS = {NOBR_FLOAT: "#d95f02", BR_FLOAT: "#1b9e77", NOBR_BYTE: "#e6ab02", BR_BYTE: "#7570b3"}
SYMBOLS = {NOBR_FLOAT: "circle", BR_FLOAT: "square", NOBR_BYTE: "diamond", BR_BYTE: "triangle-up"}

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


def load(json_path: Path = DEFAULT_JSON) -> dict[str, Series]:
    """Возвращает {имя бенчмарка: Series} по агрегатам median и stddev."""
    stats: dict[str, dict[int, dict[str, float]]] = {}
    for bench in load_json(json_path)["benchmarks"]:
        aggregate = bench.get("aggregate_name")
        if aggregate not in ("median", "stddev"):
            continue
        family, arg = bench["run_name"].split("/")[:2]
        seconds = bench["real_time"] * _TO_SECONDS[bench["time_unit"]]
        stats.setdefault(family, {}).setdefault(int(arg), {})[aggregate] = seconds

    result = {}
    for family, by_size in stats.items():
        sizes = sorted(by_size)
        result[family] = Series(
            sizes=sizes,
            median_s=[by_size[n]["median"] for n in sizes],
            stddev_s=[by_size[n].get("stddev", 0.0) for n in sizes],
        )
    return result


def save(fig, name: str) -> None:
    """Сохраняет график в docs/images в форматах PNG и HTML (интерактивный)."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(IMAGES_DIR / f"{name}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(IMAGES_DIR / f"{name}.png", width=1000, height=600, scale=2)
        print(f"Saved {IMAGES_DIR / name}.png / .html")
    except Exception as exc:  # kaleido требует установленный Chrome
        print(f"PNG export failed ({exc}); saved {IMAGES_DIR / name}.html only")
