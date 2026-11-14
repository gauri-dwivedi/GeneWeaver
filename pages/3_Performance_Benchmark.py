"""
GeneWeaver: Performance Benchmarking Page
=========================================
Compares CPU and GPU sequence alignment performance, measures execution latency,
calculates throughput (windows/sec and Mbp/s), and evaluates speedup scaling.
"""

import time
import torch
import streamlit as st
import pandas as pd
import numpy as np
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
from src.input_handler import load_sample_reference_fasta
from src.validator import (
    IUPAC_NUCLEOTIDE_MAP,
    validate_guide_sequence,
    validate_pam_pattern,
)

st.set_page_config(page_title="Performance Benchmark – GeneWeaver", page_icon="⚡", layout="wide")
apply_custom_theme()
render_device_status_sidebar()

render_hero_banner(
    title="CPU vs GPU Performance Benchmark",
    subtitle="Profile execution latency, calculate scanning throughput (win/s and Mbp/s), and evaluate speedup scaling across datasets.",
    icon="⚡",
)

gpu_engine = GPUEngine()
device_info = gpu_engine.get_device_info()


def get_device_status_display():
    """
    Dynamically query PyTorch execution device and CUDA availability.
    Strictly avoids hardcoded assumptions.
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA GPU"
        device_label = f"GPU ({gpu_name})"
        cuda_status = "Available"
        explanation = f"PyTorch is executing tensor batch operations on physical NVIDIA GPU: {device_label}."
    else:
        device_label = "CPU (PyTorch CPU Fallback)"
        cuda_status = "Not Available"
        explanation = (
            "CUDA hardware is not available on this system. PyTorch is running on CPU "
            "(PyTorch CPU Fallback). All operations use vectorized C++ tensors executed on host CPU SIMD cores. "
            "Measured speedups reflect vector batching versus scalar Python loops, not physical GPU hardware."
        )
    return device_label, cuda_status, cuda_available, explanation


def render_compute_and_cuda_status_cards():
    """
    Render two clearly visible status cards for Compute Device and CUDA Status
    directly above the benchmark performance visualizations using Streamlit UI components.
    """
    device_label, cuda_status, cuda_available, explanation = get_device_status_display()

    st.markdown("### 🖥️ Compute Device & CUDA Status")
    col_dev, col_cuda = st.columns(2)

    with col_dev:
        with st.container(border=True):
            st.metric(
                label="Compute Device",
                value=device_label,
            )
            st.caption("Actual PyTorch tensor execution hardware.")

    with col_cuda:
        with st.container(border=True):
            st.metric(
                label="CUDA Status",
                value=cuda_status,
            )
            if not cuda_available:
                st.caption("CUDA is not available. PyTorch is running on CPU.")
            else:
                st.caption("Physical NVIDIA CUDA acceleration active.")

    if not cuda_available:
        st.info(f"ℹ️ **Hardware Note:** {explanation}")
    else:
        st.success(f"🚀 **Hardware Note:** {explanation}")


# Benchmark Suite initialization
suite = BenchmarkSuite()

# Benchmark mode selection
bench_mode = st.radio(
    "Choose Benchmark Profiling Mode:",
    [
        "Multi-Scale Scaling Suite (Comprehensive)",
        "Genomic FASTA Benchmark (Sample Datasets)",
        "Single Sequence Direct Comparison (Quick)",
    ],
    horizontal=True,
)

# Persistent Hardware Status Cards (Visible across page lifecycle)
render_compute_and_cuda_status_cards()

st.markdown("---")

# ==============================================================================
# MODE 1: Multi-Scale Dataset Scaling Suite
# ==============================================================================
if bench_mode == "Multi-Scale Scaling Suite (Comprehensive)":
    st.markdown("### 📊 Multi-Scale Dataset Scaling Suite")
    st.write(
        "Evaluate how CPU scalar loops and PyTorch vectorized tensor operations scale across increasing sequence lengths. "
        "Observe the crossover point where tensor parallelism overcomes setup overhead."
    )

    col_cfg1, col_cfg2, col_cfg3 = st.columns([2, 1, 1])
    with col_cfg1:
        lengths_options = st.multiselect(
            "Select Sequence Lengths (bp) to Benchmark:",
            options=[500, 1000, 2000, 2500, 5000, 10000, 15000, 20000, 40000, 80000],
            default=[500, 2000, 5000, 15000],
        )
    with col_cfg2:
        max_mm_scaling = st.slider("Max Mismatches:", min_value=0, max_value=4, value=3, key="scaling_mm")
    with col_cfg3:
        iterations_scaling = st.slider("Timing Repetitions (Median):", min_value=1, max_value=5, value=2, key="scaling_iter")

    col_g_cfg1, col_g_cfg2 = st.columns([3, 1])
    with col_g_cfg1:
        guide_seq_scaling = st.text_input(
            "CRISPR Guide Sequence (5' to 3'):",
            value="GAGTCCGAGCAGAAGAAGAA",
            key="scaling_guide",
        ).strip().upper()
    with col_g_cfg2:
        pam_scaling = st.text_input("PAM Motif:", value="NGG", key="scaling_pam").strip().upper()

        
    if st.button("⚡ Run Benchmark", type="primary", key="run_scaling_bench"):
        guide_validation = validate_guide_sequence(guide_seq_scaling)
        pam_validation = validate_pam_pattern(pam_scaling)

        if not lengths_options:
            st.error("Please select at least one sequence length.")
        elif not guide_validation.is_valid:
            for error in guide_validation.errors:
                st.error(error)
        elif not pam_validation.is_valid:
            for error in pam_validation.errors:
                st.error(error)
        else:
            guide_seq_scaling = guide_validation.cleaned_sequence
            pam_scaling = pam_validation.cleaned_sequence

            # Convert each IUPAC PAM code into a valid DNA base.
            concrete_pam = "".join(
                sorted(IUPAC_NUCLEOTIDE_MAP[code])[0]
                for code in pam_scaling
            )

            sorted_lengths = sorted(lengths_options)
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            results_rows = []
            total_steps = len(sorted_lengths)

            try:
                start_benchmark_wall = time.perf_counter()
                rng = np.random.default_rng(42)
                bases = np.array(["A", "C", "G", "T"])

                for step_idx, slen in enumerate(sorted_lengths):
                    status_text.text(f"Profiling sequence length: {slen:,} bp (step {step_idx + 1}/{total_steps})...")

                    random_bases = rng.choice(bases, size=slen)
                    test_seq = "".join(random_bases)
                    insert_pos = slen // 2
                    target_insert = guide_seq_scaling + concrete_pam
                    if slen >= len(target_insert):
                        test_seq = test_seq[:insert_pos] + target_insert + test_seq[insert_pos + len(target_insert):]

                    metric = suite.run_single_comparison(
                        reference_seq=test_seq,
                        guide_seq=guide_seq_scaling,
                        pam_pattern=pam_scaling,
                        max_mismatches=max_mm_scaling,
                        scan_both_strands=True,
                        warmup=True,
                        iterations=iterations_scaling,
                    )

                    results_rows.append({
                        "Sequence Length (bp)": metric.sequence_length,
                        "Windows Evaluated": metric.windows_evaluated,
                        "CPU Time (ms)": metric.cpu_time_ms,
                        "CPU Std Dev (ms)": metric.cpu_std_ms,
                        "GPU Engine Time (ms)": metric.gpu_time_ms,
                        "GPU Std Dev (ms)": metric.gpu_std_ms,
                        "CPU Throughput (win/s)": metric.cpu_throughput_wps,
                        "GPU Engine Throughput (win/s)": metric.gpu_throughput_wps,
                        "CPU Throughput (Mbp/s)": metric.cpu_mbps,
                        "GPU Engine Throughput (Mbp/s)": metric.gpu_mbps,
                        "Speedup (CPU / GPU)": metric.speedup_ratio,
                        "Active Device": metric.active_device,
                        "CUDA Enabled": metric.is_cuda_accelerated,
                        "Notes": metric.notes,
                    })

                    progress_bar.progress((step_idx + 1) / total_steps)

                elapsed_benchmark_wall = time.perf_counter() - start_benchmark_wall
                status_text.success("Multi-scale benchmark complete!")
                df_scaling = pd.DataFrame(results_rows)
                st.session_state["latest_scaling_df"] = df_scaling
                st.session_state["scaling_summary"] = {
                    "total_time_seconds": elapsed_benchmark_wall,
                    "lengths": sorted_lengths,
                    "iterations": iterations_scaling,
                    "warmup_runs": 1,
                    "active_device": results_rows[0]["Active Device"] if results_rows else gpu_engine.device_name,
                    "is_cuda": results_rows[0]["CUDA Enabled"] if results_rows else device_info["cuda_available"],
                }
            except Exception as e:
                st.error(f"Scaling benchmark failed: {str(e)}")

    has_scaling = "latest_scaling_df" in st.session_state
    if not has_scaling:
        st.info("ℹ️ **Status:** Ready to profile. Select sequence lengths above and click **⚡ Run Benchmark** to evaluate performance.")
    else:
        st.caption(f"🔧 **Benchmark Diagnostic:** `latest_scaling_df` active ({len(st.session_state['latest_scaling_df'])} rows evaluated).")

    if "latest_scaling_df" in st.session_state:
        df_bench = st.session_state["latest_scaling_df"]
        summary = st.session_state.get("scaling_summary", {})
        is_cuda = bool(df_bench["CUDA Enabled"].any()) if "CUDA Enabled" in df_bench.columns else False

        st.markdown("<br>", unsafe_allow_html=True)

        # 1. Benchmark Summary
        st.markdown("### ⏱️ Benchmark Summary")
        tot_time = summary.get("total_time_seconds", 0.0)
        tot_time_display = f"{tot_time:.2f} s" if tot_time >= 1.0 else f"{tot_time * 1000:.0f} ms"
        seq_lengths_list = summary.get("lengths", list(df_bench["Sequence Length (bp)"]))
        seq_str = ", ".join(f"{l:,}" for l in seq_lengths_list) if len(seq_lengths_list) <= 6 else f"{len(seq_lengths_list)} sizes ({min(seq_lengths_list):,} to {max(seq_lengths_list):,} bp)"
        num_runs = summary.get("iterations", 2)
        warmup_runs = summary.get("warmup_runs", 1)

        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Benchmark Time</div>
                <div class="metric-value" style="font-size:1.35rem; color:#38BDF8;">{tot_time_display}</div>
                <span style="color:#94A3B8; font-size:0.85rem;">Total wall-clock duration</span>
            </div>
            """, unsafe_allow_html=True)
        with sm2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Selected Sequences</div>
                <div class="metric-value" style="font-size:1.15rem;">{seq_str} bp</div>
                <span style="color:#94A3B8; font-size:0.85rem;">{len(seq_lengths_list)} test dataset sizes</span>
            </div>
            """, unsafe_allow_html=True)
        with sm3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Number of Runs</div>
                <div class="metric-value" style="font-size:1.35rem;">{num_runs} runs</div>
                <span style="color:#94A3B8; font-size:0.85rem;">Timed repetitions per size</span>
            </div>
            """, unsafe_allow_html=True)
        with sm4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Warmup Runs</div>
                <div class="metric-value" style="font-size:1.35rem;">{warmup_runs} run</div>
                <span style="color:#94A3B8; font-size:0.85rem;">Untimed pass (excluded)</span>
            </div>
            """, unsafe_allow_html=True)

        # 2. Compute Device and CUDA Status Cards (Directly above the performance graph)
        render_compute_and_cuda_status_cards()

        # 3. Performance Scaling Visualizations
        st.markdown("### 📈 Scaling Performance Visualizations")
        gpu_label = "CUDA GPU Hardware" if is_cuda else "PyTorch CPU Fallback"

        fig_scaling_lat = go.Figure()
        fig_scaling_lat.add_trace(go.Scatter(
            x=df_bench["Sequence Length (bp)"],
            y=df_bench["CPU Time (ms)"],
            error_y=dict(
                type="data",
                array=df_bench["CPU Std Dev (ms)"],
                visible=True,
                thickness=1.5,
                width=4,
            ),
            mode="lines+markers",
            name="CPU Reference (Scalar Loop)",
            line=dict(color="#38BDF8", width=3),
            marker=dict(size=8),
        ))
        fig_scaling_lat.add_trace(go.Scatter(
            x=df_bench["Sequence Length (bp)"],
            y=df_bench["GPU Engine Time (ms)"],
            error_y=dict(
                type="data",
                array=df_bench["GPU Std Dev (ms)"],
                visible=True,
                thickness=1.5,
                width=4,
            ),
            mode="lines+markers",
            name=f"Tensor Engine ({gpu_label})",
            line=dict(color="#10B981", width=3),
            marker=dict(size=8),
        ))
        fig_scaling_lat.update_layout(
            title="Execution Latency Scaling vs Sequence Length (with ±1 Sample SD Error Bars)",
            xaxis_title="Sequence Length (base pairs)",
            yaxis_title="Execution Latency (ms)",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig_scaling_lat, use_container_width=True)

        col_ch1, col_ch2 = st.columns(2)

        with col_ch1:
            fig_speedup = px.line(
                df_bench,
                x="Sequence Length (bp)",
                y="Speedup (CPU / GPU)",
                markers=True,
                title="Speedup Factor (CPU Latency / Tensor Engine Latency)",
            )
            fig_speedup.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="#F59E0B",
                annotation_text="Parity (1.0x)",
                annotation_position="bottom right",
            )
            fig_speedup.update_traces(line_color="#10B981", marker=dict(size=8))
            fig_speedup.update_layout(
                xaxis_title="Sequence Length (base pairs)",
                yaxis_title="Speedup (x)",
                template="plotly_dark",
            )
            st.plotly_chart(fig_speedup, use_container_width=True)

        with col_ch2:
            fig_mbps_bar = go.Figure()
            fig_mbps_bar.add_trace(go.Bar(
                x=df_bench["Sequence Length (bp)"].astype(str),
                y=df_bench["CPU Throughput (Mbp/s)"],
                name="CPU Reference (Mbp/s)",
                marker_color="#38BDF8",
            ))
            fig_mbps_bar.add_trace(go.Bar(
                x=df_bench["Sequence Length (bp)"].astype(str),
                y=df_bench["GPU Engine Throughput (Mbp/s)"],
                name=f"Tensor Engine ({gpu_label})",
                marker_color="#10B981",
            ))
            fig_mbps_bar.update_layout(
                title="Processing Throughput (Megabases / Second — Mbp/s)",
                xaxis_title="Sequence Length (bp)",
                yaxis_title="Throughput (Mbp/s)",
                barmode="group",
                template="plotly_dark",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig_mbps_bar, use_container_width=True)

        fig_tp_bar = go.Figure()
        fig_tp_bar.add_trace(go.Bar(
            x=df_bench["Sequence Length (bp)"].astype(str),
            y=df_bench["CPU Throughput (win/s)"],
            name="CPU Reference (win/s)",
            marker_color="#38BDF8",
        ))
        fig_tp_bar.add_trace(go.Bar(
            x=df_bench["Sequence Length (bp)"].astype(str),
            y=df_bench["GPU Engine Throughput (win/s)"],
            name=f"Tensor Engine ({gpu_label})",
            marker_color="#10B981",
        ))
        fig_tp_bar.update_layout(
            title="Scanning Throughput (Candidate Windows / Second)",
            xaxis_title="Sequence Length (bp)",
            yaxis_title="Windows / Second",
            barmode="group",
            template="plotly_dark",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig_tp_bar, use_container_width=True)

        # 4. Detailed Results Table
        st.markdown("### 📋 Detailed Results Table")
        detailed_table = pd.DataFrame({
            "Sequence Length": [f"{int(r['Sequence Length (bp)']):,} bp" for _, r in df_bench.iterrows()],
            "CPU Scalar Mean Time": [f"{r['CPU Time (ms)']:.3f} ms" for _, r in df_bench.iterrows()],
            "CPU Scalar Standard Deviation": [f"{r['CPU Std Dev (ms)']:.3f} ms" for _, r in df_bench.iterrows()],
            "PyTorch Vectorized Mean Time": [f"{r['GPU Engine Time (ms)']:.3f} ms" for _, r in df_bench.iterrows()],
            "PyTorch Vectorized Standard Deviation": [f"{r['GPU Std Dev (ms)']:.3f} ms" for _, r in df_bench.iterrows()],
            "Speedup": [f"{r['Speedup (CPU / GPU)']:.2f}x" for _, r in df_bench.iterrows()],
        })
        st.dataframe(detailed_table, use_container_width=True, hide_index=True)

        csv_bytes = df_bench.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Detailed Benchmark Results (CSV)",
            data=csv_bytes,
            file_name=f"geneweaver_scaling_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        try:
            saved_csv = export_benchmark_to_csv(df_bench)
            st.caption(f"Results automatically persisted locally to: `{saved_csv}`")
        except Exception as e:
            st.caption(f"Local file persistence notice: {e}")

# ==============================================================================
# MODE 2: Genomic FASTA Benchmark (Sample Datasets)
# ==============================================================================
elif bench_mode == "Genomic FASTA Benchmark (Sample Datasets)":
    st.markdown("### 🧬 Genomic Reference FASTA Benchmark")
    st.write(
        "Benchmark the CPU reference engine against PyTorch tensor operations on multi-record reference FASTA files."
    )

    # Scientific Transparency Notice
    st.warning(
        "⚠️ **Synthetic Demonstration Provenance:** The sample reference FASTA dataset (`sample_reference.fasta`) "
        "contains synthetic demonstration loci modeled after human targets (EMX1, VEGFA) with inserted "
        "CRISPR on-target and off-target sites. These sequences are designed specifically for computational verification, "
        "reproducibility, and performance benchmarking — **not** for clinical, therapeutic, or biological validation."
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        guide_seq_fasta = st.text_input(
            "CRISPR Guide Sequence (5' to 3'):",
            value="GAGTCCGAGCAGAAGAAGAA",
            key="fasta_guide",
        ).strip().upper()
        pam_fasta = st.text_input("PAM Motif:", value="NGG", key="fasta_pam").strip().upper()
    with col_f2:
        max_mm_fasta = st.slider("Max Mismatches:", min_value=0, max_value=4, value=3, key="fasta_mm")
        iterations_fasta = st.slider("Timing Repetitions per Record:", min_value=1, max_value=5, value=2, key="fasta_iter")

    if st.button("⚡ Run Benchmark", type="primary", key="run_fasta_bench"):
        if not guide_seq_fasta:
            st.error("Please enter a valid CRISPR guide sequence.")
        else:
            with st.spinner("Loading reference FASTA records and profiling engines..."):
                try:
                    start_fasta_wall = time.perf_counter()
                    df_fasta = suite.run_fasta_benchmark(
                        guide_seq=guide_seq_fasta,
                        pam_pattern=pam_fasta,
                        max_mismatches=max_mm_fasta,
                        iterations=iterations_fasta,
                    )
                    elapsed_fasta_wall = time.perf_counter() - start_fasta_wall
                    st.session_state["latest_fasta_df"] = df_fasta
                    st.session_state["fasta_summary"] = {
                        "total_time_seconds": elapsed_fasta_wall,
                        "num_records": len(df_fasta),
                        "record_names": list(df_fasta["Record ID"]),
                        "iterations": iterations_fasta,
                        "warmup_runs": 1,
                        "active_device": df_fasta["Active Device"].iloc[0] if not df_fasta.empty else gpu_engine.device_name,
                        "is_cuda": bool(df_fasta["CUDA Enabled"].any()) if "CUDA Enabled" in df_fasta.columns else False,
                    }
                except Exception as e:
                    st.error(f"FASTA benchmark failed: {str(e)}")

    has_fasta = "latest_fasta_df" in st.session_state
    if not has_fasta:
        st.info("ℹ️ **Status:** Ready to profile. Click **⚡ Run Benchmark** to evaluate reference FASTA records.")
    else:
        st.caption(f"🔧 **Benchmark Diagnostic:** `latest_fasta_df` active ({len(st.session_state['latest_fasta_df'])} records evaluated).")

    if "latest_fasta_df" in st.session_state:
        df_fasta = st.session_state["latest_fasta_df"]
        f_summary = st.session_state.get("fasta_summary", {})
        is_cuda_fasta = bool(df_fasta["CUDA Enabled"].any()) if "CUDA Enabled" in df_fasta.columns else False

        st.markdown("<br>", unsafe_allow_html=True)

        # 1. Benchmark Summary
        st.markdown("### ⏱️ Benchmark Summary")
        f_tot_time = f_summary.get("total_time_seconds", 0.0)
        f_tot_display = f"{f_tot_time:.2f} s" if f_tot_time >= 1.0 else f"{f_tot_time * 1000:.0f} ms"
        f_records = f_summary.get("record_names", list(df_fasta["Record ID"]))
        f_runs = f_summary.get("iterations", 2)
        f_warmup = f_summary.get("warmup_runs", 1)

        fsm1, fsm2, fsm3, fsm4 = st.columns(4)
        with fsm1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Benchmark Time</div>
                <div class="metric-value" style="font-size:1.35rem; color:#38BDF8;">{f_tot_display}</div>
                <span style="color:#94A3B8; font-size:0.85rem;">Total wall-clock duration</span>
            </div>
            """, unsafe_allow_html=True)
        with fsm2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Selected FASTA Records</div>
                <div class="metric-value" style="font-size:1.15rem;">{len(f_records)} records</div>
                <span style="color:#94A3B8; font-size:0.85rem;">{', '.join(f_records)}</span>
            </div>
            """, unsafe_allow_html=True)
        with fsm3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Number of Runs</div>
                <div class="metric-value" style="font-size:1.35rem;">{f_runs} runs</div>
                <span style="color:#94A3B8; font-size:0.85rem;">Timed repetitions per record</span>
            </div>
            """, unsafe_allow_html=True)
        with fsm4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Warmup Runs</div>
                <div class="metric-value" style="font-size:1.35rem;">{f_warmup} run</div>
                <span style="color:#94A3B8; font-size:0.85rem;">Untimed pass (excluded)</span>
            </div>
            """, unsafe_allow_html=True)

        # 2. Compute Device and CUDA Status Cards (Directly above the performance graph)
        render_compute_and_cuda_status_cards()

        # 3. Performance Visualizations
        st.markdown("### 📈 FASTA Benchmark Visualizations")
        gpu_label_fasta = "CUDA GPU Hardware" if is_cuda_fasta else "PyTorch CPU Fallback"

        fig_fasta_lat = go.Figure()
        fig_fasta_lat.add_trace(go.Bar(
            x=df_fasta["Record ID"],
            y=df_fasta["CPU Time (ms)"],
            error_y=dict(
                type="data",
                array=df_fasta["CPU Std Dev (ms)"],
                visible=True,
                thickness=1.5,
                width=4,
            ),
            name="CPU Reference",
            marker_color="#38BDF8",
        ))
        fig_fasta_lat.add_trace(go.Bar(
            x=df_fasta["Record ID"],
            y=df_fasta["GPU Engine Time (ms)"],
            error_y=dict(
                type="data",
                array=df_fasta["GPU Std Dev (ms)"],
                visible=True,
                thickness=1.5,
                width=4,
            ),
            name=f"Tensor Engine ({gpu_label_fasta})",
            marker_color="#10B981",
        ))
        fig_fasta_lat.update_layout(
            title="Execution Latency per FASTA Record (ms, with ±1 Sample SD Error Bars)",
            xaxis_title="FASTA Record ID",
            yaxis_title="Execution Time (ms)",
            barmode="group",
            template="plotly_dark",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig_fasta_lat, use_container_width=True)

        col_fg1, col_fg2 = st.columns(2)
        with col_fg1:
            fig_fasta_tp = go.Figure()
            fig_fasta_tp.add_trace(go.Bar(
                x=df_fasta["Record ID"],
                y=df_fasta["CPU Throughput (Mbp/s)"],
                name="CPU Reference (Mbp/s)",
                marker_color="#38BDF8",
            ))
            fig_fasta_tp.add_trace(go.Bar(
                x=df_fasta["Record ID"],
                y=df_fasta["GPU Engine Throughput (Mbp/s)"],
                name=f"Tensor Engine ({gpu_label_fasta})",
                marker_color="#10B981",
            ))
            fig_fasta_tp.update_layout(
                title="Processing Throughput (Megabases / Second — Mbp/s)",
                xaxis_title="FASTA Record ID",
                yaxis_title="Throughput (Mbp/s)",
                barmode="group",
                template="plotly_dark",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig_fasta_tp, use_container_width=True)

        with col_fg2:
            fig_fasta_spd = px.bar(
                df_fasta,
                x="Record ID",
                y="Speedup (CPU / GPU)",
                text_auto=".2f",
                title="Speedup Factor per Record (CPU Latency / Tensor Engine Latency)",
                color_discrete_sequence=["#10B981"],
            )
            fig_fasta_spd.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="#F59E0B",
                annotation_text="Parity (1.0x)",
                annotation_position="bottom right",
            )
            fig_fasta_spd.update_layout(
                xaxis_title="FASTA Record ID",
                yaxis_title="Speedup (x)",
                template="plotly_dark",
            )
            st.plotly_chart(fig_fasta_spd, use_container_width=True)

        # 4. Detailed Results Table
        st.markdown("### 📋 Detailed Results Table")
        detailed_fasta_table = pd.DataFrame({
            "Record ID": df_fasta["Record ID"],
            "Sequence Length": [f"{int(r['Sequence Length (bp)']):,} bp" for _, r in df_fasta.iterrows()],
            "CPU Scalar Mean Time": [f"{r['CPU Time (ms)']:.3f} ms" for _, r in df_fasta.iterrows()],
            "CPU Scalar Standard Deviation": [f"{r['CPU Std Dev (ms)']:.3f} ms" for _, r in df_fasta.iterrows()],
            "PyTorch Vectorized Mean Time": [f"{r['GPU Engine Time (ms)']:.3f} ms" for _, r in df_fasta.iterrows()],
            "PyTorch Vectorized Standard Deviation": [f"{r['GPU Std Dev (ms)']:.3f} ms" for _, r in df_fasta.iterrows()],
            "Speedup": [f"{r['Speedup (CPU / GPU)']:.2f}x" for _, r in df_fasta.iterrows()],
        })
        st.dataframe(detailed_fasta_table, use_container_width=True, hide_index=True)

        csv_fasta_bytes = df_fasta.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download FASTA Benchmark Results (CSV)",
            data=csv_fasta_bytes,
            file_name=f"geneweaver_fasta_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        try:
            saved_fasta_csv = export_benchmark_to_csv(df_fasta)
            st.caption(f"Results automatically persisted locally to: `{saved_fasta_csv}`")
        except Exception as e:
            st.caption(f"Local file persistence notice: {e}")

# ==============================================================================
# MODE 3: Single Sequence Direct Comparison (Quick)
# ==============================================================================
else:
    st.markdown("### 🔬 Single Sequence Latency & Throughput Comparison")
    st.write(
        "Run an isolated side-by-side benchmark on a single synthetic sequence to inspect latency, "
        "sample standard deviation variability, and scanning throughput."
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        seq_len_choice = st.select_slider(
            "Select Test Sequence Size (bp):",
            options=[500, 1000, 2500, 5000, 10000, 20000, 50000],
            value=5000,
        )
    with col_s2:
        max_mm_bench = st.slider("Max Mismatches:", min_value=0, max_value=4, value=3, key="single_mm")
    with col_s3:
        iterations_bench = st.slider("Timing Repetitions (Median):", min_value=1, max_value=5, value=3, key="single_iter")

    col_sq1, col_sq2 = st.columns([3, 1])
    with col_sq1:
        guide_seq_single = st.text_input(
            "CRISPR Guide Sequence (5' to 3'):",
            value="GAGTCCGAGCAGAAGAAGAA",
            key="single_guide",
        ).strip().upper()
    with col_sq2:
        pam_single = st.text_input("PAM Motif:", value="NGG", key="single_pam").strip().upper()

    if st.button("⚡ Run Benchmark", type="primary", key="run_single_bench"):
        if not guide_seq_single:
            st.error("Please enter a valid CRISPR guide sequence.")
        else:
            with st.spinner(f"Generating synthetic DNA ({seq_len_choice:,} bp) and profiling engines..."):
                try:
                    start_single_wall = time.perf_counter()
                    rng = np.random.default_rng(123)
                    bases = np.array(["A", "C", "G", "T"])
                    test_dna = "".join(rng.choice(bases, size=seq_len_choice))

                    metric = suite.run_single_comparison(
                        reference_seq=test_dna,
                        guide_seq=guide_seq_single,
                        pam_pattern=pam_single,
                        max_mismatches=max_mm_bench,
                        scan_both_strands=True,
                        warmup=True,
                        iterations=iterations_bench,
                    )
                    elapsed_single_wall = time.perf_counter() - start_single_wall

                    st.session_state["latest_single_metric"] = metric
                    st.session_state["single_summary"] = {
                        "total_time_seconds": elapsed_single_wall,
                        "seq_len": seq_len_choice,
                        "iterations": iterations_bench,
                        "warmup_runs": 1,
                    }
                except Exception as e:
                    st.error(f"Single benchmark failed: {str(e)}")

    has_single = "latest_single_metric" in st.session_state
    if not has_single:
        st.info("ℹ️ **Status:** Ready to profile. Select sequence size above and click **⚡ Run Benchmark** to evaluate performance.")
    else:
        st.caption(f"🔧 **Benchmark Diagnostic:** `latest_single_metric` active (evaluated {st.session_state['latest_single_metric'].sequence_length:,} bp).")

    if "latest_single_metric" in st.session_state:
        m = st.session_state["latest_single_metric"]
        s_summary = st.session_state.get("single_summary", {})

        st.markdown("<br>", unsafe_allow_html=True)

        # 1. Benchmark Summary
        st.markdown("### ⏱️ Benchmark Summary")
        s_tot_time = s_summary.get("total_time_seconds", 0.0)
        s_tot_display = f"{s_tot_time:.2f} s" if s_tot_time >= 1.0 else f"{s_tot_time * 1000:.0f} ms"
        s_runs = s_summary.get("iterations", 3)
        s_warmup = s_summary.get("warmup_runs", 1)

        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Benchmark Time</div>
                <div class="metric-value" style="font-size:1.35rem; color:#38BDF8;">{s_tot_display}</div>
                <span style="color:#94A3B8; font-size:0.85rem;">Total wall-clock duration</span>
            </div>
            """, unsafe_allow_html=True)
        with sm2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Selected Sequence Length</div>
                <div class="metric-value" style="font-size:1.35rem;">{m.sequence_length:,} bp</div>
                <span style="color:#94A3B8; font-size:0.85rem;">{m.windows_evaluated:,} candidate windows</span>
            </div>
            """, unsafe_allow_html=True)
        with sm3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Number of Runs</div>
                <div class="metric-value" style="font-size:1.35rem;">{s_runs} runs</div>
                <span style="color:#94A3B8; font-size:0.85rem;">Timed repetitions</span>
            </div>
            """, unsafe_allow_html=True)
        with sm4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Warmup Runs</div>
                <div class="metric-value" style="font-size:1.35rem;">{s_warmup} run</div>
                <span style="color:#94A3B8; font-size:0.85rem;">Untimed pass (excluded)</span>
            </div>
            """, unsafe_allow_html=True)

        # 2. Compute Device and CUDA Status Cards (Directly above the performance graph)
        render_compute_and_cuda_status_cards()

        # 3. Performance Visualizations
        st.markdown("### 📈 Single Comparison Visualizations")
        single_gpu_name = "CUDA GPU" if m.is_cuda_accelerated else "PyTorch CPU Fallback"

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Bar(
                x=["CPU Reference (Scalar)", f"Tensor Engine ({single_gpu_name})"],
                y=[m.cpu_time_ms, m.gpu_time_ms],
                error_y=dict(
                    type="data",
                    array=[m.cpu_std_ms, m.gpu_std_ms],
                    visible=True,
                    thickness=1.5,
                    width=6,
                ),
                text=[f"{m.cpu_time_ms:.2f} ms", f"{m.gpu_time_ms:.2f} ms"],
                textposition="auto",
                marker_color=["#38BDF8", "#10B981"],
            ))
            fig_lat.update_layout(
                title="Execution Latency (ms, with ±1 Sample SD Error Bars — Lower is Better)",
                xaxis_title="Alignment Engine",
                yaxis_title="Execution Latency (ms)",
                template="plotly_dark",
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        with col_g2:
            fig_tp = go.Figure()
            fig_tp.add_trace(go.Bar(
                x=["CPU Reference (Scalar)", f"Tensor Engine ({single_gpu_name})"],
                y=[m.cpu_throughput_wps, m.gpu_throughput_wps],
                text=[f"{m.cpu_throughput_wps:,.0f}", f"{m.gpu_throughput_wps:,.0f}"],
                textposition="auto",
                marker_color=["#38BDF8", "#10B981"],
            ))
            fig_tp.update_layout(
                title="Scanning Throughput (Windows / Sec — Higher is Better)",
                xaxis_title="Alignment Engine",
                yaxis_title="Windows / Second",
                template="plotly_dark",
            )
            st.plotly_chart(fig_tp, use_container_width=True)

        # 4. Detailed Results Table
        st.markdown("### 📋 Detailed Results Table")
        detailed_single_table = pd.DataFrame([{
            "Sequence Length": f"{m.sequence_length:,} bp",
            "CPU Scalar Mean Time": f"{m.cpu_time_ms:.3f} ms",
            "CPU Scalar Standard Deviation": f"{m.cpu_std_ms:.3f} ms",
            "PyTorch Vectorized Mean Time": f"{m.gpu_time_ms:.3f} ms",
            "PyTorch Vectorized Standard Deviation": f"{m.gpu_std_ms:.3f} ms",
            "Speedup": f"{m.speedup_ratio:.2f}x",
        }])
        st.dataframe(detailed_single_table, use_container_width=True, hide_index=True)

        st.info(f"💡 **Engineering Diagnostic:** {m.notes}")

# ==============================================================================
# Student Viva / Mentor Explanation Section
# ==============================================================================
st.markdown("---")
with st.expander("🎓 Mentor / Viva Explanation: Architectural Principles & Benchmark Methodology"):
    st.markdown("""
    ### Key Engineering & Statistical Concepts Behind the Benchmarks:

    1. **Amdahl's Law and Kernel Launch Overhead:**
       - Every PyTorch tensor allocation or GPU CUDA kernel launch involves an unavoidable constant overhead ($T_{overhead} \\approx 0.5 - 2.0\\text{ ms}$).
       - On short DNA sequences ($<1{,}000\\text{ bp}$), the computational work is so small that this initialization overhead dominates the runtime. Consequently, the CPU scalar loop often appears faster on small inputs.
       - As sequence length scales ($>10{,}000\\text{ bp}$), the parallel matrix operations ($O(N)$ vector batching) dominate, yielding significant speedups.

    2. **Sample Standard Deviation vs Population Standard Deviation:**
       - Repetitions measure operating system scheduling variability and memory bus contention.
       - GeneWeaver calculates **sample standard deviation** with Bessel's correction ($N - 1$ degrees of freedom, `ddof=1`):
         $$s = \\sqrt{\\frac{1}{N - 1} \\sum_{i=1}^N (x_i - \\bar{x})^2}$$
       - This provides an unbiased estimator of latency variance across timing iterations.

    3. **Untimed Warmup Passes:**
       - Monitored executions run an initial untimed warmup pass over the sequence before recording timings.
       - This primes Python bytecode caches, initializes PyTorch threadpools, and loads CUDA drivers, preventing cold-start timing distortion.

    4. **CPU Vectorized Fallback vs Physical CUDA Execution:**
       - When NVIDIA CUDA hardware is unavailable (`torch.cuda.is_available() == False`), GeneWeaver runs PyTorch tensors on CPU cores.
       - Speedup in this mode reflects vector SIMD execution and C++ backend efficiency compared to pure Python loops, **not** physical GPU hardware acceleration.

    5. **Dataset Provenance & Scientific Honesty:**
       - Demonstration sequences in `sample_reference.fasta` are synthetic models engineered for reproducible algorithmic validation, **not** clinical or biological in vivo trial results.
    """)
