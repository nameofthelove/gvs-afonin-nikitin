"""График ускорения S(n) = T_shmem(n) / T_wmma(n) новой реализации operator*.

Запуск из корня проекта: python scripts/speedup_chart.py [path/to/benchmark_results.json]
"""

import math
import sys
from pathlib import Path

import plotly.graph_objects as go

from bench_data import (COLORS, DEFAULT_JSON, SHMEM_FLOAT, SHMEM_HALF, SYMBOLS, WMMA, load, save)

BASELINES = {
    SHMEM_HALF: ("относительно kernel_matmul_shmem на half", "top left"),
    SHMEM_FLOAT: ("относительно kernel_matmul_shmem на float", "bottom right"),
}


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    series = load(json_path)

    wmma = dict(zip(series[WMMA].sizes, series[WMMA].median_s))

    fig = go.Figure()
    for family, (label, position) in BASELINES.items():
        baseline = dict(zip(series[family].sizes, series[family].median_s))
        sizes = sorted(wmma.keys() & baseline.keys())
        speedup = [baseline[n] / wmma[n] for n in sizes]

        fig.add_trace(go.Scatter(
            x=sizes, y=speedup, mode="lines+markers+text", name=label,
            text=[f"{s:.2f}×" for s in speedup], textposition=position,
            line=dict(color=COLORS[family], width=2),
            marker=dict(symbol=SYMBOLS[family], size=9),
        ))

        print(label)
        for n, s in zip(sizes, speedup):
            print(f"  n = {n:>5}: S = {s:6.2f}")

    fig.add_hline(y=1.0, line=dict(color="red", dash="dash"))  # граница S = 1

    fig.update_layout(
        title="Ускорение kernel_matmul_wmma относительно предыдущей версии operator*",
        xaxis=dict(title="Размер матрицы n", type="log", tickvals=series[WMMA].sizes,
                   range=[math.log10(11), math.log10(1800)]),
        yaxis=dict(title="Ускорение S(n) = T_shmem / T_wmma, раз"),
        template="plotly_white", legend=dict(x=0.01, y=0.99),
    )
    save(fig, "speedup_chart")


if __name__ == "__main__":
    main()
