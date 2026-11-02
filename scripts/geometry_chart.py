"""Влияние геометрии блока kernel_matmul_wmma: модель T(n) = t0 + 2n^3/P + V/B.

t0 — накладные расходы запуска, 2n^3/P — работа тензорных ядер с измеренным пределом P,
V = 2 n^3 (1/X + 1/Y) байт — объём загрузки тайлов из глобальной памяти (X, Y — стороны тайла C).
Эффективная пропускная способность B оценивается МНК по геометриям с полной загрузкой SM.

Запуск из корня проекта: python scripts/geometry_chart.py [path/to/benchmark_results.json]
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from bench_data import (COLORS, DEFAULT_JSON, OVERHEAD_SIZES, WMMA, geometry, is_wmma, load,
                        peak_gflops, save)

FIT_SIZE = 1024
MIN_BLOCKS = 40  # не меньше двух блоков на SM: сетка занимает все 20 SM
MAX_SHARED_KB = 128 / 6  # разделяемая память не ограничивает число блоков на SM (6 слотов)


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    series = load(json_path)
    peak_flops = peak_gflops(json_path) * 1e9

    overhead = float(np.median([t for s in series.values()
                                for n, t in zip(s.sizes, s.median_s) if n in OVERHEAD_SIZES]))

    rows = []
    for family, s in series.items():
        if not is_wmma(family):
            continue
        y, x, k = geometry(family)
        t = dict(zip(s.sizes, s.median_s))[FIT_SIZE]
        blocks = (FIT_SIZE // x) * (FIT_SIZE // y)
        shared_kb = 2.0 * (k * (x + y) + 8 * 256) / 1024  # тайлы A, B и буфер записи C
        rows.append(dict(family=family, y=y, x=x, k=k, t=t, blocks=blocks, shared_kb=shared_kb,
                         volume=2.0 * FIT_SIZE**3 * (1 / x + 1 / y)))
    rows.sort(key=lambda r: r["volume"])

    # T - t0 - 2n^3/P = V / B: прямая через начало координат.
    compute_s = 2.0 * FIT_SIZE**3 / peak_flops
    fit = [r for r in rows if r["blocks"] >= MIN_BLOCKS and r["shared_kb"] <= MAX_SHARED_KB]
    volume = np.array([r["volume"] for r in fit])
    load_s = np.array([r["t"] - overhead - compute_s for r in fit])
    bandwidth = float(volume @ volume / (volume @ load_s))

    names = [f"{r['y']}×{r['x']}×{r['k']}" for r in rows]
    modelled = [overhead + compute_s + r["volume"] / bandwidth for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[r["t"] * 1e6 for r in rows], name="Измерено",
                         marker_color=[COLORS[WMMA] if r["family"] == WMMA else "#a6a6a6"
                                       for r in rows]))
    fig.add_trace(go.Scatter(x=names, y=[t * 1e6 for t in modelled], name="Модель t₀ + 2n³/P + V/B",
                             mode="lines+markers", line=dict(color="#d95f02", width=2)))
    fig.update_layout(
        title=f"Время умножения матриц {FIT_SIZE}×{FIT_SIZE} при разной геометрии блока",
        xaxis=dict(title="Тайлы BlockItemsY × BlockItemsX × BlockItemsK"),
        yaxis=dict(title="Время T, мкс", range=[0, 620]),
        template="plotly_white", legend=dict(x=0.01, y=0.99),
    )
    save(fig, "geometry_chart")

    print(f"t0 = {overhead * 1e6:.1f} us, 2n^3/P = {compute_s * 1e6:.1f} us, "
          f"B = {bandwidth / 1e9:.1f} GB/s")
    print(f"{'geometry':>14} {'blocks':>7} {'shared':>7} {'V, MB':>8} {'T, us':>8} {'model':>8} "
          f"{'P_kernel':>9}")
    for name, r, m in zip(names, rows, modelled):
        print(f"{name:>14} {r['blocks']:>7} {r['shared_kb']:>7.1f} {r['volume'] / 1e6:>8.1f} "
              f"{r['t'] * 1e6:>8.1f} {m * 1e6:>8.1f} {2.0 * FIT_SIZE**3 / r['t'] / 1e9:>9.0f}")


if __name__ == "__main__":
    main()
