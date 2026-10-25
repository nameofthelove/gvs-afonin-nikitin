"""График ускорения S(n) = T_naive(n) / T_shmem(n) новой реализации operator*.

Запуск из корня проекта: python scripts/speedup_chart.py [path/to/benchmark_results.json]
"""

import sys
from pathlib import Path

import plotly.graph_objects as go

from bench_data import COLORS, DEFAULT_JSON, LABELS, NAIVE, SHMEM16, SHMEM32, SYMBOLS, load, save


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    series = load(json_path)

    naive = dict(zip(series[NAIVE].sizes, series[NAIVE].median_s))

    fig = go.Figure()
    for family in (SHMEM16, SHMEM32):
        shmem = dict(zip(series[family].sizes, series[family].median_s))
        sizes = sorted(naive.keys() & shmem.keys())
        speedup = [naive[n] / shmem[n] for n in sizes]

        fig.add_trace(go.Scatter(
            x=sizes, y=speedup, mode="lines+markers+text", name=LABELS[family],
            text=[f"{s:.2f}×" for s in speedup],
            textposition="top left" if family == SHMEM16 else "bottom right",
            line=dict(color=COLORS[family], width=2),
            marker=dict(symbol=SYMBOLS[family], size=9),
        ))

        print(LABELS[family])
        for n, s in zip(sizes, speedup):
            print(f"  n = {n:>5}: S = {s:6.2f}")

    fig.add_hline(y=1.0, line=dict(color="red", dash="dash"))  # граница S = 1

    fig.update_layout(
        title="Ускорение kernel_matmul_shmem относительно kernel_matmul_naive",
        xaxis=dict(title="Размер матрицы n", type="log", tickvals=series[NAIVE].sizes),
        yaxis=dict(title="Ускорение S(n) = T_naive / T_shmem, раз"),
        template="plotly_white", legend=dict(x=0.01, y=0.99),
    )
    save(fig, "speedup_chart")


if __name__ == "__main__":
    main()
