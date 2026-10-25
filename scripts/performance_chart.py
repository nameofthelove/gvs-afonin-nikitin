"""График производительности P(n) = 2n^3 / T(n) с теоретическими верхними оценками.

Оценки строятся по параметрам GPU из поля context JSON-отчёта (число SM, частота):
  * пиковая производительность FP32: 128 FMA/такт на SM (Ampere, sm_86);
  * предел kernel_matmul_naive: на одну FMA варп выполняет 2 загрузки из кэша L1
    (строка A — broadcast, столбец B — объединённая), L1 выдаёт 128 байт/такт на SM,
    т.е. одну загрузку варпа за такт -> 0.5 варп-FMA/такт;
  * предел kernel_matmul_shmem (тайл 16): на 16 FMA варп выполняет 16 LDS (тайл B)
    и 4 LDS.128 (тайл A, broadcast) -> 16/20 = 0.8 варп-FMA/такт.

Запуск из корня проекта: python scripts/performance_chart.py [path/to/benchmark_results.json]
"""

import sys
from pathlib import Path

import plotly.graph_objects as go

from bench_data import (COLORS, DEFAULT_JSON, LABELS, NAIVE, SHMEM16, SHMEM32, SYMBOLS,
                        gflops, load, load_json, save)

WARP_SIZE = 32
FP32_LANES_PER_SM = 128


def bounds_gflops(context: dict) -> dict[str, float]:
    sm_count = int(context["gpu_sm_count"])
    clock_ghz = int(context["gpu_clock_mhz"]) / 1e3

    def per_warp_fma_rate(warp_fma_per_clock: float) -> float:
        return 2.0 * warp_fma_per_clock * WARP_SIZE * sm_count * clock_ghz

    return {
        "peak": 2.0 * FP32_LANES_PER_SM * sm_count * clock_ghz,
        NAIVE: per_warp_fma_rate(1 / 2),
        SHMEM16: per_warp_fma_rate(16 / 20),
    }


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    series = load(json_path)
    bounds = bounds_gflops(load_json(json_path)["context"])

    fig = go.Figure()
    for family in (NAIVE, SHMEM16, SHMEM32):
        s = series[family]
        fig.add_trace(go.Scatter(
            x=s.sizes, y=[gflops(n, t) for n, t in zip(s.sizes, s.median_s)],
            mode="lines+markers", name=LABELS[family],
            line=dict(color=COLORS[family], width=2),
            marker=dict(symbol=SYMBOLS[family], size=9),
        ))

    for family, text in ((NAIVE, "предел L1 для naive"),
                         (SHMEM16, "предел shared memory для shmem 16×16")):
        fig.add_hline(y=bounds[family], line=dict(color=COLORS[family], dash="dash", width=1),
                      annotation_text=f"{text}: {bounds[family]:.0f} GFLOP/s",
                      annotation_position="top left")

    fig.update_layout(
        title="Производительность умножения матриц n×n "
              f"(пиковая FP32 ≈ {bounds['peak']:.0f} GFLOP/s)",
        xaxis=dict(title="Размер матрицы n", type="log", tickvals=series[NAIVE].sizes),
        yaxis=dict(title="Производительность P(n), GFLOP/s",
                   range=[0, bounds[SHMEM16] * 1.12]),
        template="plotly_white", legend=dict(x=0.01, y=0.75),
    )
    save(fig, "performance_chart")

    for name, value in bounds.items():
        print(f"{name}: {value:.0f} GFLOP/s")


if __name__ == "__main__":
    main()
