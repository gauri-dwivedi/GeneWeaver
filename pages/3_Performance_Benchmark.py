"""
GeneWeaver: Performance Benchmarking Page
=========================================
Compares CPU and GPU sequence alignment performance, measures execution latency,
calculates throughput (windows/sec), and evaluates speedup scaling.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from src.ui_components import (
    apply_custom_theme,
    render_hero_banner,
    render_device_status_sidebar,
)
from src.benchmark import BenchmarkSuite, export_benchmark_to_csv
from src.gpu_engine import GPUEngine

st.set_page_config(page_title="Performance Benchmark – GeneWeaver", page_icon="⚡", layout="wide")
apply_custom_theme()
render_device_status_sidebar()

render_hero_banner(
    title="CPU vs GPU Performance Benchmark",
    subtitle="Profile execution latency, calculate scanning throughput, and analyze speedup scaling across dataset sizes.",
    icon="⚡",
)

gpu_engine = GPUEngine()
device_info = gpu_engine.get_device_info()

# Benchmark Suite initialization
suite = BenchmarkSuite()

# Benchmark mode selection
bench_mode = st.radio(
    "Choose Benchmark Profiling Mode:",
    ["Multi-Scale Scaling Suite (Comprehensive)", "Single Sequence Direct Comparison (Quick)"],
    horizontal=True,
)

if bench_mode == "Single Sequence Direct Comparison (Quick)":
    st.markdown("### 🔬 Single Sequence Latency & Throughput Comparison")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        seq_len_choice = st.select_slider(
            "Select Test Sequence Size (bp):",
            options=[500, 1000, 2500, 5000, 10000, 20000, 50000],
            value=5000,
        )
    with col_s2:
        max_mm_bench = st.slider("Max Mismatches:", min_value=0, max_value=4, value=3)
    with col_s3:
        iterations_bench = st.slider("Timing Repetitions (Median):", min_value=1, max_value=5, value=3)

    if st.button("🚀 Run Single Benchmark Comparison", type="primary"):
        with st.spinner(f"Generating synthetic DNA ({seq_len_choice:,} bp) and profiling engines..."):
            # Synthetic DNA generation
            import numpy as np
            rng = np.random.default_rng(123)
            bases = np.array(["A", "C", "G", "T"])
            test_dna = "".join(rng.choice(bases, size=seq_len_choice))

            metric = suite.run_single_comparison(
                reference_seq=test_dna,
                guide_seq="GAGTCCGAGCAGAAGAAGAA",
                pam_pattern="NGG",
                max_mismatches=max_mm_bench,
                scan_both_strands=True,
                warmup=True,
                iterations=iterations_bench,
            )

            st.session_state["latest_single_metric"] = metric

    if "latest_single_metric" in st.session_state:
        m = st.session_state["latest_single_metric"]

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CPU Reference Time</div>
                <div class="metric-value">{m.cpu_time_ms:.2f} ms</div>
                <span style="color:#94A3B8; font-size:0.85rem;">{m.cpu_throughput_wps:,.0f} win/s</span>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">GPU Engine Time</div>
                <div class="metric-value" style="color:#38BDF8;">{m.gpu_time_ms:.2f} ms</div>
                <span style="color:#94A3B8; font-size:0.85rem;">{m.gpu_throughput_wps:,.0f} win/s</span>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            speedup_color = "#10B981" if m.speedup_ratio >= 1.0 else "#F59E0B"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Measured Speedup</div>
                <div class="metric-value" style="color:{speedup_color};">{m.speedup_ratio:.2f}x</div>
                <span style="color:#94A3B8; font-size:0.85rem;">(CPU Latency / GPU Latency)</span>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Active Compute Device</div>
                <div class="metric-value" style="font-size:1.15rem;">{m.active_device[:22]}</div>
                <span style="color:#94A3B8; font-size:0.85rem;">{m.windows_evaluated:,} windows</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"💡 **Engineering Diagnostic:** {m.notes}")

        # Quick comparison chart
        df_comp = pd.DataFrame({
            "Engine": ["CPU (Python Reference)", "GPU Engine (PyTorch Tensors)"],
            "Latency (ms)": [m.cpu_time_ms, m.gpu_time_ms],
            "Throughput (win/s)": [m.cpu_throughput_wps, m.gpu_throughput_wps],
        })

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_lat = px.bar(
                df_comp,
                x="Engine",
                y="Latency (ms)",
                text_auto=".2f",
                color="Engine",
                color_discrete_sequence=["#38BDF8", "#10B981"],
                title="Execution Latency (Lower is Better)",
            )
            fig_lat.update_layout(showlegend=False, template="plotly_dark")
            st.plotly_chart(fig_lat, use_container_width=True)

        with col_g2:
            fig_tp = px.bar(
                df_comp,
                x="Engine",
                y="Throughput (win/s)",
                text_auto=",.0f",
                color="Engine",
                color_discrete_sequence=["#38BDF8", "#10B981"],
                title="Throughput (Windows/Sec - Higher is Better)",
            )
            fig_tp.update_layout(showlegend=False, template="plotly_dark")
            st.plotly_chart(fig_tp, use_container_width=True)

else:
    # Multi-Scale Scaling Suite
    st.markdown("### 📊 Multi-Scale Dataset Scaling Suite")
    st.write(
        "Evaluate how CPU and GPU engines scale across increasing sequence lengths. "
        "Observe where tensor operations begin outperforming pure iterative Python code."
    )

    col_cfg1, col_cfg2 = st.columns([2, 1])
    with col_cfg1:
        lengths_options = st.multiselect(
            "Select Sequence Lengths (bp) to Benchmark:",
            options=[500, 1000, 2500, 5000, 10000, 20000, 40000, 80000],
            default=[500, 2000, 5000, 15000],
        )
    with col_cfg2:
        iterations_scaling = st.slider("Iterations per Dataset Size:", min_value=1, max_value=3, value=2)

    if st.button("⚡ Run Multi-Scale Scaling Benchmark", type="primary"):
        if not lengths_options:
            st.error("Please select at least one sequence length.")
        else:
            sorted_lengths = sorted(lengths_options)
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            results_rows = []
            total_steps = len(sorted_lengths)

            for step_idx, slen in enumerate(sorted_lengths):
                status_text.text(f"Profiling sequence length: {slen:,} bp (step {step_idx+1}/{total_steps})...")

                # Generate DNA
                import numpy as np
                rng = np.random.default_rng(42)
                bases = np.array(["A", "C", "G", "T"])
                test_dna = "".join(rng.choice(bases, size=slen))

                metric = suite.run_single_comparison(
                    reference_seq=test_dna,
                    guide_seq="GAGTCCGAGCAGAAGAAGAA",
                    pam_pattern="NGG",
                    max_mismatches=3,
                    scan_both_strands=True,
                    warmup=True,
                    iterations=iterations_scaling,
                )

                results_rows.append({
                    "Sequence Length (bp)": metric.sequence_length,
                    "Windows Evaluated": metric.windows_evaluated,
                    "CPU Time (ms)": metric.cpu_time_ms,
                    "GPU Engine Time (ms)": metric.gpu_time_ms,
                    "CPU Throughput (win/s)": metric.cpu_throughput_wps,
                    "GPU Engine Throughput (win/s)": metric.gpu_throughput_wps,
                    "Speedup (CPU / GPU)": metric.speedup_ratio,
                    "Active Device": metric.active_device,
                    "CUDA Enabled": metric.is_cuda_accelerated,
                    "Notes": metric.notes,
                })

                progress_bar.progress((step_idx + 1) / total_steps)

            status_text.success("Benchmarking complete!")
            df_scaling = pd.DataFrame(results_rows)
            st.session_state["latest_scaling_df"] = df_scaling

    if "latest_scaling_df" in st.session_state:
        df_bench = st.session_state["latest_scaling_df"]

        st.markdown("### 📈 Scaling Performance Visualizations")

        tab_charts, tab_data = st.tabs(["📊 Performance Charts", "📋 Raw Benchmark Data"])

        with tab_charts:
            # Latency Scaling Chart
            fig_scaling_lat = go.Figure()
            fig_scaling_lat.add_trace(go.Scatter(
                x=df_bench["Sequence Length (bp)"],
                y=df_bench["CPU Time (ms)"],
                mode="lines+markers",
                name="CPU Reference",
                line=dict(color="#38BDF8", width=3),
                marker=dict(size=8),
            ))
            fig_scaling_lat.add_trace(go.Scatter(
                x=df_bench["Sequence Length (bp)"],
                y=df_bench["GPU Engine Time (ms)"],
                mode="lines+markers",
                name="GPU Engine",
                line=dict(color="#10B981", width=3),
                marker=dict(size=8),
            ))
            fig_scaling_lat.update_layout(
                title="Execution Latency Scaling (ms) vs Sequence Length",
                xaxis_title="Sequence Length (base pairs)",
                yaxis_title="Execution Time (ms)",
                template="plotly_dark",
                hovermode="x unified",
            )
            st.plotly_chart(fig_scaling_lat, use_container_width=True)

            col_ch1, col_ch2 = st.columns(2)

            with col_ch1:
                # Speedup Curve
                fig_speedup = px.line(
                    df_bench,
                    x="Sequence Length (bp)",
                    y="Speedup (CPU / GPU)",
                    markers=True,
                    title="Speedup Factor (CPU Latency / GPU Latency)",
                )
                fig_speedup.add_hline(y=1.0, line_dash="dash", line_color="#F59E0B", annotation_text="Parity (1.0x)")
                fig_speedup.update_traces(line_color="#10B981", marker=dict(size=8))
                fig_speedup.update_layout(template="plotly_dark")
                st.plotly_chart(fig_speedup, use_container_width=True)

            with col_ch2:
                # Throughput Comparison
                fig_tp_bar = go.Figure()
                fig_tp_bar.add_trace(go.Bar(
                    x=df_bench["Sequence Length (bp)"].astype(str),
                    y=df_bench["CPU Throughput (win/s)"],
                    name="CPU Throughput",
                    marker_color="#38BDF8",
                ))
                fig_tp_bar.add_trace(go.Bar(
                    x=df_bench["Sequence Length (bp)"].astype(str),
                    y=df_bench["GPU Engine Throughput (win/s)"],
                    name="GPU Engine Throughput",
                    marker_color="#10B981",
                ))
                fig_tp_bar.update_layout(
                    title="Scanning Throughput (Windows / Sec)",
                    xaxis_title="Sequence Length (bp)",
                    yaxis_title="Windows / Second",
                    barmode="group",
                    template="plotly_dark",
                )
                st.plotly_chart(fig_tp_bar, use_container_width=True)

        with tab_data:
            st.dataframe(df_bench, use_container_width=True, hide_index=True)

            # Export CSV
            csv_bytes = df_bench.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Benchmark Results (CSV)",
                data=csv_bytes,
                file_name=f"geneweaver_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
            saved_csv = export_benchmark_to_csv(df_bench)
            st.caption(f"Results automatically persisted locally to: `{saved_csv}`")

        # Student Viva / Mentor Explanation Box
        st.markdown("---")
        with st.expander("🎓 Mentor / Viva Explanation: Why does GPU speedup behave this way?"):
            st.markdown("""
            ### Computer Science Principles Behind the Results:
            1. **Amdahl's Law and Kernel Launch Latency:**
               Whenever PyTorch launches a CUDA kernel or sets up tensor memory on Windows, there is an unavoidable
               constant time cost ($T_{overhead} \\approx 0.5 - 2.0\\text{ ms}$). On small inputs (e.g. 500 bp),
               the raw computational work is so minuscule that this overhead dwarfs the arithmetic runtime.
            2. **Memory Bandwidth & Vector Striding:**
               For sequences beyond 10,000–50,000 bp, the thousands of parallel streaming multiprocessors on a GPU
               or vector SIMD units on modern CPU cores evaluate sliding windows in parallel batches, overcoming
               the launch latency and demonstrating high throughput.
            3. **CPU Tensor Fallback Behavior:**
               On machines without dedicated NVIDIA hardware, GeneWeaver switches to CPU vectorized tensors.
               This still benefits from PyTorch's optimized C++ backend compared to interpreted Python `for` loops,
               achieving modest speedups without fabricating fake GPU execution.
            """)

