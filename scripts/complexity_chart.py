"""График реальной вычислительной сложности T(n) суммирования вектора из n элементов.

Запуск из корня проекта: python scripts/complexity_chart.py [path/to/benchmark_results.json]
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from bench_data import (BR_BYTE, BR_FLOAT, COLORS, DEFAULT_JSON, LABELS, NOBR_BYTE, NOBR_FLOAT,
                        SYMBOLS, load, save)

FAMILIES = (NOBR_FLOAT, BR_FLOAT, NOBR_BYTE, BR_BYTE)
MIN_FIT_SIZE = 2**24  # накладные расходы запуска меньше 2 % времени


def fit_exponent(sizes: list[int], times: list[float], min_size: int) -> float:
    """Показатель степени p в T(n) ~ c * n^p (МНК в логарифмических координатах)."""
    pts = [(n, t) for n, t in zip(sizes, times) if n >= min_size]
    log_n, log_t = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    return float(np.polyfit(log_n, log_t, 1)[0])


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    series = load(json_path)

    fig = go.Figure()
    for family in FAMILIES:
        s = series[family]
        fig.add_trace(go.Scatter(
            x=s.sizes, y=[t * 1e6 for t in s.median_s],
            error_y=dict(type="data", array=[d * 1e6 for d in s.stddev_s], visible=True),
            mode="lines+markers", name=LABELS[family],
            line=dict(color=COLORS[family], width=2),
            marker=dict(symbol=SYMBOLS[family], size=7),
        ))

    # Теоретическая оценка O(n), привязанная к последней точке измерений.
    for family, label in ((BR_FLOAT, "float"), (BR_BYTE, "uint8")):
        s = series[family]
        n_max, t_max = s.sizes[-1], s.median_s[-1]
        fig.add_trace(go.Scatter(
            x=s.sizes, y=[t_max * n / n_max * 1e6 for n in s.sizes],
            mode="lines", name=f"c·n ({label})",
            line=dict(color=COLORS[family], width=1, dash="dash"),
        ))

    fig.update_layout(
        title="Реальная вычислительная сложность суммирования вектора",
        xaxis=dict(title="Размер вектора n", type="log", exponentformat="power", dtick=np.log10(8)),
        yaxis=dict(title="Время T(n), мкс", type="log", exponentformat="power",
                   range=[np.log10(5.0), np.log10(2e5)]),
        template="plotly_white", legend=dict(x=0.01, y=0.99),
    )
    save(fig, "complexity_chart")

    print(f"\n{'n':>11}" + "".join(f" | {f.split('<')[1][:-1]:>10}" for f in FAMILIES))
    for n in series[BR_BYTE].sizes:
        row = f"{n:>11}"
        for family in FAMILIES:
            times = dict(zip(series[family].sizes, series[family].median_s))
            row += f" | {times[n] * 1e6:>10.1f}" if n in times else f" | {'—':>10}"
        print(row)

    for family in FAMILIES:
        s = series[family]
        print(f"{family}: p(n >= 2^24) = {fit_exponent(s.sizes, s.median_s, MIN_FIT_SIZE):.3f}, "
              f"last step ratio = {s.median_s[-1] / s.median_s[-2]:.3f}")


if __name__ == "__main__":
    main()
