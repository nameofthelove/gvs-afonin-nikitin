"""График ускорения S(n) = T_nobr(n) / T_br(n) кёрнела с широковещанием.

Запуск из корня проекта: python scripts/speedup_chart.py [path/to/benchmark_results.json]
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from bench_data import (BR_BYTE, BR_FLOAT, COLORS, DEFAULT_JSON, NOBR_BYTE, NOBR_FLOAT, SYMBOLS,
                        load, save)

PAIRS = ((NOBR_FLOAT, BR_FLOAT, "float"), (NOBR_BYTE, BR_BYTE, "uint8"))


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    series = load(json_path)

    fig = go.Figure()
    for nobr, br, label in PAIRS:
        t_nobr = dict(zip(series[nobr].sizes, series[nobr].median_s))
        t_br = dict(zip(series[br].sizes, series[br].median_s))
        sizes = sorted(t_nobr.keys() & t_br.keys())
        speedup = [t_nobr[n] / t_br[n] for n in sizes]

        fig.add_trace(go.Scatter(
            x=sizes, y=speedup, mode="lines+markers", name=f"{label}",
            line=dict(color=COLORS[br], width=2),
            marker=dict(symbol=SYMBOLS[br], size=7),
        ))

        print(label)
        for n, s in zip(sizes, speedup):
            print(f"  n = {n:>10}: S = {s:6.3f}")

    fig.add_hline(y=1.0, line=dict(color="red", dash="dash"))  # граница S = 1

    fig.update_layout(
        title="Ускорение kernel_vecred_br относительно kernel_vecred_nobr",
        xaxis=dict(title="Размер вектора n", type="log", exponentformat="power", dtick=np.log10(8)),
        yaxis=dict(title="Ускорение S(n) = T_nobr / T_br, раз"),
        template="plotly_white", legend=dict(x=0.85, y=0.99),
    )
    save(fig, "speedup_chart")


if __name__ == "__main__":
    main()
