"""Build speedup chart: Vector operator+ relative to Eigen::VectorXf.

Uses only the Python standard library. Plotly.js is loaded from CDN in the HTML.
"""

from __future__ import annotations

import json
import os
import re


def _bench_size(bench: dict) -> int | None:
    """Parse N from Google Benchmark names.

    Examples:
      BM_Vector_operator_plus/8/manual_time_mean
      BM_Eigen_VectorXf_operator_plus/8_mean
      BM_Eigen_VectorXf_operator_plus/8
    """
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


def _collect_times(data: dict) -> tuple[dict[int, float], dict[int, float]]:
    vector_times: dict[int, float] = {}
    eigen_times: dict[int, float] = {}
    to_ns = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}

    for bench in data.get("benchmarks", []):
        run_type = bench.get("run_type", "iteration")
        if run_type == "aggregate" and bench.get("aggregate_name") != "mean":
            continue

        size = _bench_size(bench)
        if size is None:
            continue

        time_ns = float(bench["real_time"]) * to_ns.get(bench.get("time_unit", "ns"), 1.0)
        family = _bench_family(bench)

        if family == "BM_Vector_operator_plus":
            vector_times[size] = time_ns
        elif family == "BM_Eigen_VectorXf_operator_plus":
            eigen_times[size] = time_ns

    return vector_times, eigen_times


def _write_plotly_html(path: str, title: str, sizes: list[int], speedups: list[float]) -> None:
    traces = [
        {
            "x": sizes,
            "y": speedups,
            "mode": "lines+markers",
            "name": "Speedup S(N) = T_Eigen / T_Vector",
            "type": "scatter",
        }
    ]
    layout = {
        "title": title,
        "xaxis": {"title": "Vector size N", "type": "log"},
        "yaxis": {"title": "Speedup S(N)"},
        "template": "plotly_white",
        "legend": {"x": 0.02, "y": 0.98},
        "shapes": [
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "y0": 1,
                "y1": 1,
                "line": {"color": "red", "dash": "dash"},
            }
        ],
        "annotations": [
            {
                "xref": "paper",
                "x": 1,
                "y": 1,
                "text": "parity (1x)",
                "showarrow": False,
                "yshift": 10,
                "font": {"color": "red"},
            }
        ],
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

    vector_times, eigen_times = _collect_times(data)
    common_sizes = sorted(set(vector_times) & set(eigen_times))
    speedups = [eigen_times[s] / vector_times[s] for s in common_sizes]

    print(f"Speedup points: {len(common_sizes)}")
    if not common_sizes:
        print("Warning: no overlapping sizes — chart would be empty.")
        print(f"  Vector sizes: {sorted(vector_times)}")
        print(f"  Eigen sizes:  {sorted(eigen_times)}")
        return

    os.makedirs("docs/images", exist_ok=True)
    out_html = "docs/images/speedup_chart.html"
    _write_plotly_html(
        out_html,
        title="Speedup of Vector operator+ vs Eigen::VectorXf",
        sizes=common_sizes,
        speedups=speedups,
    )
    print(f"Saved: {out_html}")
    for s, sp in zip(common_sizes, speedups):
        print(f"  N={s}: S={sp:.3f}")


if __name__ == "__main__":
    main()
