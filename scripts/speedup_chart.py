"""График ускорения S(n) = T_Eigen(n) / T_Matrix(n).

Запуск из корня проекта: python scripts/speedup_chart.py [path/to/benchmark_results.json]
"""

import sys
from pathlib import Path

import plotly.graph_objects as go

from bench_data import DEFAULT_JSON, EIGEN, MATRIX, load, save


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    series = load(json_path)

    gpu = dict(zip(series[MATRIX].sizes, series[MATRIX].median_s))
    cpu = dict(zip(series[EIGEN].sizes, series[EIGEN].median_s))
    sizes = sorted(gpu.keys() & cpu.keys())
    speedup = [cpu[n] / gpu[n] for n in sizes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=speedup, mode="lines+markers+text", name="S(n) = T_Eigen / T_Matrix",
        text=[f"{s:.2f}×" for s in speedup], textposition="top left",
        line=dict(color="#7570b3", width=2), marker=dict(size=9),
    ))
    fig.add_hline(y=1.0, line=dict(color="red", dash="dash"))  # граница S = 1

    fig.update_layout(
        title="Ускорение Matrix::operator* (GPU) относительно Eigen::MatrixXf (CPU)",
        xaxis=dict(title="Размер матрицы n", type="log", tickvals=sizes),
        yaxis=dict(title="Ускорение S(n), раз", type="log",
                   tickvals=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]),
        template="plotly_white", showlegend=False,
    )
    save(fig, "speedup_chart")

    for n, s in zip(sizes, speedup):
        print(f"n = {n:>5}: S = {s:8.2f}")


if __name__ == "__main__":
    main()
