"""Build speedup chart: Vector operator+ relative to Eigen::VectorXf."""

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

    vector_times: dict[int, float] = {}
    eigen_times: dict[int, float] = {}

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
        to_ns = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}
        time_ns = float(bench["real_time"]) * to_ns.get(unit, 1.0)

        if parts[0] == "BM_Vector_operator_plus":
            vector_times[size] = time_ns
        elif parts[0] == "BM_Eigen_VectorXf_operator_plus":
            eigen_times[size] = time_ns

    common_sizes = sorted(set(vector_times) & set(eigen_times))
    speedups = [eigen_times[s] / vector_times[s] for s in common_sizes]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=common_sizes,
            y=speedups,
            mode="lines+markers",
            name="Speedup S(N) = T_Eigen / T_Vector",
        )
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="parity (1x)")

    fig.update_layout(
        title="Speedup of Vector operator+ vs Eigen::VectorXf",
        xaxis_title="Vector size N",
        yaxis_title="Speedup S(N)",
        xaxis_type="log",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )

    os.makedirs("docs/images", exist_ok=True)
    out_html = "docs/images/speedup_chart.html"
    out_png = "docs/images/speedup_chart.png"
    fig.write_html(out_html)
    try:
        fig.write_image(out_png, scale=2)
        print(f"Saved: {out_html}, {out_png}")
    except Exception as exc:
        print(f"Saved: {out_html} (PNG skipped: {exc})")


if __name__ == "__main__":
    main()
