"""Build real-complexity chart (Fig. 2 style): Eigen CPU vs CUDA GPU.

Uses only the Python standard library. Plotly.js is loaded from CDN in the HTML.
"""

from __future__ import annotations

import json
import os
import re


def _bench_size(bench: dict) -> int | None:
    run_name = bench.get("run_name") or bench.get("name", "")
    parts = run_name.split("/")
    if len(parts) < 2:
        return None
    match = re.match(r"^(\d+)", parts[1])
    if not match:
        return None
    return int(match.group(1))


def _bench_family(bench: dict) -> str:
    name = bench.get("run_name") or bench.get("name", "")
    return name.split("/")[0]


def _collect_times(data: dict) -> tuple[list[int], list[float], list[int], list[float]]:
    vector_sizes: list[int] = []
    vector_times_ms: list[float] = []
    eigen_sizes: list[int] = []
    eigen_times_ms: list[float] = []

    to_ms = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}

    for bench in data.get("benchmarks", []):
        run_type = bench.get("run_type", "iteration")
        if run_type == "aggregate" and bench.get("aggregate_name") != "mean":
            continue

        size = _bench_size(bench)
        if size is None:
            continue

        time_ms = float(bench["real_time"]) * to_ms.get(bench.get("time_unit", "ns"), 1e-6)
        family = _bench_family(bench)

        if family == "BM_Vector_operator_plus":
            vector_sizes.append(size)
            vector_times_ms.append(time_ms)
        elif family == "BM_Eigen_VectorXf_operator_plus":
            eigen_sizes.append(size)
            eigen_times_ms.append(time_ms)

    return vector_sizes, vector_times_ms, eigen_sizes, eigen_times_ms


def _write_plotly_html(path: str, traces: list[dict]) -> None:
    # Match course Fig. 2: log-log, Time in ms with µ labels for small values.
    tick_vals = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    tick_text = ["1µ", "10µ", "100µ", "0.001", "0.01", "0.1", "1", "10", "100"]
    layout = {
        "title": "Real Complexity",
        "xaxis": {"title": "N", "type": "log"},
        "yaxis": {
            "title": "Time, ms",
            "type": "log",
            "tickvals": tick_vals,
            "ticktext": tick_text,
        },
        "template": "plotly_white",
        "legend": {"x": 0.02, "y": 0.98},
    }
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <title>Real Complexity</title>
</head>
<body>
  <div id="chart" style="width:100%;height:600px;"></div>
  <script>
    const data = {json.dumps(traces)};
    const layout = {json.dumps(layout)};
    Plotly.newPlot('chart', data, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    json_path = "benchmark_results.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run benchmarks with --benchmark_out first.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vector_sizes, vector_times_ms, eigen_sizes, eigen_times_ms = _collect_times(data)
    print(f"CUDA points: {len(vector_sizes)}, Eigen points: {len(eigen_sizes)}")
    if not vector_sizes or not eigen_sizes:
        print("Warning: missing series — check benchmark_results.json names.")

    # Same order/colors as the course example: Eigen=blue, CUDA=red.
    traces = [
        {
            "x": eigen_sizes,
            "y": eigen_times_ms,
            "mode": "lines+markers",
            "name": "Eigen Vector Addition (CPU)",
            "type": "scatter",
            "line": {"color": "#1f77b4"},
            "marker": {"color": "#1f77b4"},
        },
        {
            "x": vector_sizes,
            "y": vector_times_ms,
            "mode": "lines+markers",
            "name": "CUDA Vector Addition (GPU)",
            "type": "scatter",
            "line": {"color": "#d62728"},
            "marker": {"color": "#d62728"},
        },
    ]

    os.makedirs("docs/images", exist_ok=True)
    out_html = "docs/images/complexity_chart.html"
    _write_plotly_html(out_html, traces)
    print(f"Saved: {out_html}")


if __name__ == "__main__":
    main()
