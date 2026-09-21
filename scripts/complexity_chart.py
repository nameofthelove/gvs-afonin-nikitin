"""Build real-complexity chart for Vector vs Eigen::VectorXf operator+."""

from __future__ import annotations

import json
import os

import plotly.graph_objects as go


def main() -> None:
    json_path = "benchmark_results.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run benchmarks with --benchmark_out first.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vector_sizes: list[int] = []
    vector_times_ms: list[float] = []
    eigen_sizes: list[int] = []
    eigen_times_ms: list[float] = []

    for bench in data.get("benchmarks", []):
        run_type = bench.get("run_type", "iteration")
        if run_type == "aggregate" and bench.get("aggregate_name") != "mean":
            continue

        name = bench["name"]
        parts = name.split("/")
        if len(parts) < 2:
            continue

        try:
            size = int(parts[1])
        except ValueError:
            continue

        unit = bench.get("time_unit", "ns")
        to_ms = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
        time_ms = float(bench["real_time"]) * to_ms.get(unit, 1e-6)

        if parts[0] == "BM_Vector_operator_plus":
            vector_sizes.append(size)
            vector_times_ms.append(time_ms)
        elif parts[0] == "BM_Eigen_VectorXf_operator_plus":
            eigen_sizes.append(size)
            eigen_times_ms.append(time_ms)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vector_sizes,
            y=vector_times_ms,
            mode="lines+markers",
            name="Vector (CUDA operator+)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eigen_sizes,
            y=eigen_times_ms,
            mode="lines+markers",
            name="Eigen::VectorXf (operator+)",
        )
    )

    fig.update_layout(
        title="Real complexity of operator+",
        xaxis_title="Vector size N",
        yaxis_title="Time T(N), ms",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )

    os.makedirs("docs/images", exist_ok=True)
    out_html = "docs/images/complexity_chart.html"
    out_png = "docs/images/complexity_chart.png"
    fig.write_html(out_html)
    try:
        fig.write_image(out_png, scale=2)
        print(f"Saved: {out_html}, {out_png}")
    except Exception as exc:
        print(f"Saved: {out_html} (PNG skipped: {exc})")


if __name__ == "__main__":
    main()
