"""График производительности P(n) = 2n^3 / T(n) с измеренным пределом тензорных ядер.

Предел получен калибровочным бенчмарком BM_Wmma_peak (mma_sync без обращений к памяти),
предел FP32 — расчётный: 128 FMA/такт на SM для kernel_matmul_shmem.

Запуск из корня проекта: python scripts/performance_chart.py [path/to/benchmark_results.json]
"""

import math
import sys
from pathlib import Path

import plotly.graph_objects as go

from bench_data import (COLORS, DEFAULT_JSON, LABELS, LOADED_CLOCK_MHZ, SHMEM_FLOAT, SHMEM_HALF,
                        SYMBOLS, WMMA, WMMA_MIN, gflops, load, load_json, peak_gflops, save)

FP32_LANES_PER_SM = 128


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    series = load(json_path)
    peak = peak_gflops(json_path)

    sm_count = int(load_json(json_path)["context"]["gpu_sm_count"])
    clock_ghz = LOADED_CLOCK_MHZ / 1e3
    fp32_peak = 2.0 * FP32_LANES_PER_SM * sm_count * clock_ghz

    fig = go.Figure()
    for family in (SHMEM_FLOAT, SHMEM_HALF, WMMA, WMMA_MIN):
        s = series[family]
        fig.add_trace(go.Scatter(
            x=s.sizes, y=[gflops(n, t) for n, t in zip(s.sizes, s.median_s)],
            mode="lines+markers", name=LABELS[family],
            line=dict(color=COLORS[family], width=2),
            marker=dict(symbol=SYMBOLS[family], size=9),
        ))

    for value, family, text in ((peak, WMMA, "предел тензорных ядер (измерен)"),
                                (fp32_peak, SHMEM_FLOAT, "предел FP32 (расчёт)")):
        fig.add_hline(y=value, line=dict(color=COLORS[family], dash="dash", width=1))
        # На логарифмической оси координата подписи задаётся десятичным логарифмом.
        fig.add_annotation(xref="paper", x=0.01, y=math.log10(value), yshift=11, showarrow=False,
                           xanchor="left", text=f"{text}: {value:.0f} GFLOP/s")

    fig.update_layout(
        title="Производительность умножения матриц n×n",
        xaxis=dict(title="Размер матрицы n", type="log", tickvals=series[WMMA].sizes),
        yaxis=dict(title="Производительность P(n), GFLOP/s", type="log", exponentformat="power",
                   range=[math.log10(0.2), math.log10(peak * 2.5)]),
        template="plotly_white", legend=dict(x=0.62, y=0.05, yanchor="bottom"),
    )
    save(fig, "performance_chart")

    print(f"wmma peak: {peak:.0f} GFLOP/s = "
          f"{peak / 2 / sm_count / clock_ghz:.0f} FMA/clock/SM at {LOADED_CLOCK_MHZ} MHz")
    print(f"fp32 peak: {fp32_peak:.0f} GFLOP/s")


if __name__ == "__main__":
    main()
