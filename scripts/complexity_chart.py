"""Build real-complexity chart for Vector vs Eigen::VectorXf operator+.

Uses only the Python standard library. Plotly.js is loaded from CDN in the HTML.
"""

from __future__ import annotations

import json
import os


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

        parts = bench["name"].split("/")
        if len(parts) < 2:
            continue

        try:
            size = int(parts[1])
        except ValueError:
            continue

        time_ms = float(bench["real_time"]) * to_ms.get(bench.get("time_unit", "ns"), 1e-6)

        if parts[0] == "BM_Vector_operator_plus":
            vector_sizes.append(size)
            vector_times_ms.append(time_ms)
        elif parts[0] == "BM_Eigen_VectorXf_operator_plus":
            eigen_sizes.append(size)
            eigen_times_ms.append(time_ms)

    return vector_sizes, vector_times_ms, eigen_sizes, eigen_times_ms


def _write_plotly_html(
    path: str,
    title: str,
    x_title: str,
    y_title: str,
    traces: list[dict],
    x_log: bool = True,
    y_log: bool = True,
) -> None:
    layout = {
        "title": title,
        "xaxis": {"title": x_title, "type": "log" if x_log else "linear"},
        "yaxis": {"title": y_title, "type": "log" if y_log else "linear"},
        "template": "plotly_white",
        "legend": {"x": 0.02, "y": 0.98},
    }
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <title>{title}</title>
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

    traces = [
        {
            "x": vector_sizes,
            "y": vector_times_ms,
            "mode": "lines+markers",
            "name": "Vector (CUDA operator+)",
            "type": "scatter",
        },
        {
            "x": eigen_sizes,
            "y": eigen_times_ms,
            "mode": "lines+markers",
            "name": "Eigen::VectorXf (operator+)",
            "type": "scatter",
        },
    ]

    os.makedirs("docs/images", exist_ok=True)
    out_html = "docs/images/complexity_chart.html"
    _write_plotly_html(
        out_html,
        title="Real complexity of operator+",
        x_title="Vector size N",
        y_title="Time T(N), ms",
        traces=traces,
        x_log=True,
        y_log=True,
    )
    print(f"Saved: {out_html}")
    print("Open the HTML in a browser (needs network once for Plotly CDN).")


if __name__ == "__main__":
    main()
