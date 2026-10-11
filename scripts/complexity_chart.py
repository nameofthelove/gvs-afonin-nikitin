"""График реальной вычислительной сложности T(n) для умножения n x n матриц.

Запуск из корня проекта: python scripts/complexity_chart.py [path/to/benchmark_results.json]
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from bench_data import DEFAULT_JSON, EIGEN, LABELS, MATRIX, gflops, load, save

COLORS = {MATRIX: "#1b9e77", EIGEN: "#d95f02"}
SYMBOLS = {MATRIX: "square", EIGEN: "circle"}


def fit_exponent(sizes: list[int], times: list[float], min_size: int) -> float:
    """Показатель степени p в T(n) ~ c * n^p (МНК в логарифмических координатах)."""
    pts = [(n, t) for n, t in zip(sizes, times) if n >= min_size]
    log_n, log_t = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    return float(np.polyfit(log_n, log_t, 1)[0])


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    series = load(json_path)

    fig = go.Figure()
    for family in (MATRIX, EIGEN):
        s = series[family]
        fig.add_trace(go.Scatter(
            x=s.sizes, y=[t * 1e3 for t in s.median_s],
            error_y=dict(type="data", array=[d * 1e3 for d in s.stddev_s], visible=True),
            mode="lines+markers", name=LABELS[family],
            line=dict(color=COLORS[family], width=2),
            marker=dict(symbol=SYMBOLS[family], size=9),
        ))

        # Теоретическая оценка O(n^3), привязанная к последней точке измерений.
        n_max, t_max = s.sizes[-1], s.median_s[-1]
        fig.add_trace(go.Scatter(
            x=s.sizes, y=[t_max * (n / n_max) ** 3 * 1e3 for n in s.sizes],
            mode="lines", name=f"c·n³ ({LABELS[family].split(' ')[0]})",
            line=dict(color=COLORS[family], width=1, dash="dash"),
        ))

    fig.update_layout(
        title="Реальная вычислительная сложность умножения матриц n×n",
        xaxis=dict(title="Размер матрицы n", type="log", tickvals=series[MATRIX].sizes),
        yaxis=dict(title="Время T(n), мс", type="log", exponentformat="power"),
        template="plotly_white", legend=dict(x=0.01, y=0.99),
    )
    save(fig, "complexity_chart")

    # Таблица для отчёта.
    print(f"\n{'n':>6} | {'T_Matrix, мс':>12} | {'GFLOP/s':>8} | {'T_Eigen, мс':>11} | {'GFLOP/s':>8}")
    m, e = series[MATRIX], series[EIGEN]
    for n, tm, te in zip(m.sizes, m.median_s, e.median_s):
        print(f"{n:>6} | {tm * 1e3:>12.4f} | {gflops(n, tm):>8.1f} | {te * 1e3:>11.4f} | {gflops(n, te):>8.1f}")

    for family in (MATRIX, EIGEN):
        s = series[family]
        print(f"{family}: p(n >= 16) = {fit_exponent(s.sizes, s.median_s, 16):.2f}, "
              f"p(n >= 256) = {fit_exponent(s.sizes, s.median_s, 256):.2f}")


if __name__ == "__main__":
    main()
