"""
GeneWeaver: GPU-Accelerated CRISPR Alignment Engine
===================================================
Main Application Entry Point and Project Overview.
"""

import os
import tempfile

# Ensure Matplotlib writes cache to a safe temp folder on Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_gw"))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.ui_components import (
    apply_custom_theme,
    render_hero_banner,
    render_device_status_sidebar,
)
from src.gpu_engine import GPUEngine
from src.input_handler import list_sample_datasets

# Page Configuration
st.set_page_config(
    page_title="GeneWeaver – CRISPR Alignment Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_theme()
render_device_status_sidebar()

# Hero Header
render_hero_banner(
    title="GeneWeaver: GPU-Accelerated CRISPR Alignment Engine",
    subtitle="High-performance biological sequence matching, mismatch scoring, and hardware-accelerated bioinformatics.",
    icon="🧬",
)

# Quick System Status Metric Cards
gpu_engine = GPUEngine()
device_info = gpu_engine.get_device_info()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Execution Engine</div>
        <div class="metric-value">Dual Engine</div>
        <span style="color:#94A3B8; font-size:0.85rem;">CPU Reference + GPU Tensor</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    status_label = "CUDA Active" if device_info["cuda_available"] else "CPU Fallback"
    status_color = "#10B981" if device_info["cuda_available"] else "#38BDF8"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active Compute Device</div>
        <div class="metric-value" style="color:{status_color}; font-size:1.35rem;">{status_label}</div>
        <span style="color:#94A3B8; font-size:0.85rem;">{device_info['device_used'][:35]}</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Default PAM Motif</div>
        <div class="metric-value">SpCas9 (NGG)</div>
        <span style="color:#94A3B8; font-size:0.85rem;">Configurable IUPAC codes</span>
    </div>
    """, unsafe_allow_html=True)

with col4:
    datasets = list_sample_datasets()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Sample Datasets</div>
        <div class="metric-value">{len(datasets)} Files</div>
        <span style="color:#94A3B8; font-size:0.85rem;">Pre-packaged and ready</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Page Tabs: Overview, Architecture, How to Use, About
tab_overview, tab_architecture, tab_guide, tab_about = st.tabs([
    "📖 Project Overview",
    "🏗️ System Architecture",
    "🚀 How to Use GeneWeaver",
    "ℹ️ About & Evaluation Guide",
])

with tab_overview:
    st.markdown("### What is GeneWeaver?")
    st.write(
        "GeneWeaver is an open-source, dual-engine bioinformatics application developed to address "
        "the computational bottlenecks of **CRISPR-Cas9 target site search and off-target screening**. "
        "By pairing a deterministic Python reference algorithm on the CPU with a vectorized tensor execution "
        "engine implemented in PyTorch, GeneWeaver bridges rigorous biological sequence validation with "
        "modern parallel compute hardware."
    )

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🔬 Core Bioinformatics Capabilities")
        st.markdown("""
        - **DNA Sequence Analysis**: Paste or upload FASTA sequences, validate canonical nucleotides, and compute GC/AT composition.
        - **Dual-Strand Sliding Window**: Scans both the forward (sense) and reverse-complement (antisense) strands of reference sequences.
        - **Configurable PAM Recognition**: Evaluates candidate targets against standard (SpCas9 `NGG`) or custom IUPAC PAM patterns (`NAG`, `NNGRRT`).
        - **Seed-Weighted Scoring**: Penalizes mismatches in the PAM-proximal seed region (bases 11–20) more heavily than PAM-distal bases, matching CRISPR biology.
        - **Mismatch Visualization**: Provides a visual graphical breakdown of exact matches (`|`) and base substitutions (`.`).
        """)

    with col_right:
        st.markdown("#### ⚡ High-Performance Computing Principles")
        st.markdown("""
        - **PyTorch Tensor Batching**: Vectorizes nucleotide characters into integer tensors ($A=0, C=1, G=2, T=3$) and unrolls sliding windows using zero-copy tensor strides (`unfold()`).
        - **Automatic Hardware Sensing**: Detects NVIDIA CUDA availability at runtime. If absent, falls back gracefully to CPU tensor operations without crashing.
        - **Zero Fabrication Guarantee**: Execution times and speedup ratios are measured directly using high-precision wall-clock counters (`time.perf_counter()`).
        - **Interactive Benchmarking**: Empirically explores CPU vs GPU performance across varied dataset scales, exposing real-world GPU transfer overhead on small inputs.
        """)

    st.info(
        "💡 **Important Scientific Caveat:** GeneWeaver calculates sequence similarity and mismatch counts. "
        "Computational sequence matches identify candidate homology sites; they do not prove actual in vivo "
        "cleavage efficiency or biological editing activity, which depend on cellular and epigenetic factors."
    )

with tab_architecture:
    st.markdown("### GeneWeaver Modular Architecture")
    st.write(
        "The application is engineered with a modular separation of concerns. "
        "The diagram below illustrates how raw sequence data flows from user input through validation, "
        "sliding-window extraction, dual-engine alignment, scoring, and dashboard visualization."
    )

    st.markdown("""
```
 +-------------------------------------------------------------------------+
 |                              USER INPUT                                 |
 |        Pasted DNA Sequence  /  FASTA Upload  /  Sample Datasets         |
 +-----------------------------------+-------------------------------------+
                                     |
                                     v
 +-------------------------------------------------------------------------+
 |                  MODULE 1 & 2: INPUT & VALIDATION                       |
 |  clean_sequence() -> validate_dna_sequence() -> validate_pam_pattern()  |
 |       - Canonical ATGC checks       - Uppercase normalisation           |
 |       - IUPAC PAM motif parsing     - Nucleotide GC/AT statistics       |
 +-----------------------------------+-------------------------------------+
                                     |
                                     v
 +-------------------------------------------------------------------------+
 |                 MODULE 3: DUAL ALIGNMENT COORDINATOR                    |
 |     - Target Protospacer (20bp) + PAM (3bp) window calculation          |
 |     - Forward Strand + Reverse-Complement (Antisense) strand scan       |
 +-------------------+---------------------------------+-------------------+
                     |                                 |
                     v                                 v
   +---------------------------------+   +---------------------------------+
   |    MODULE 4: CPU REFERENCE      |   |    MODULE 5: GPU TENSOR ENGINE  |
   |   - Python sliding window       |   |   - Integer Tensor Encoding     |
   |   - Character-by-character scan |   |   - PyTorch unfold() Striding   |
   |   - time.perf_counter() latency |   |   - CUDA / CPU Tensor Fallback  |
   +-----------------+---------------+   +-----------------+---------------+
                     |                                     |
                     +------------------+------------------+
                                        |
                                        v
 +-------------------------------------------------------------------------+
 |             MODULE 3 (CONTINUED): SCORING & VISUALIZATION               |
 |    - Mismatch counts (1-based positions)                                |
 |    - Seed (bases 11-20) vs Distal (bases 1-10) weighted scoring         |
 |    - Visual match connector generation (||||||.||||||)                  |
 +--------------------------------------+----------------------------------+
                                        |
                                        v
 +-------------------------------------------------------------------------+
 |               MODULE 6 & 7: BENCHMARK & STREAMLIT DASHBOARD             |
 |    - Real-time CPU vs GPU charts    - Alignment match inspection table  |
 |    - Performance speedup profiling  - CSV & Markdown report downloads   |
 +-------------------------------------------------------------------------+
```
    """)

with tab_guide:
    st.markdown("### Step-by-Step Workflow Guide")

    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        st.markdown("#### 1️⃣ Analyze Sequence")
        st.write(
            "Navigate to **Sequence Analyzer** (`pages/1_Sequence_Analyzer.py`). "
            "Paste your target sequence or load a sample reference. "
            "Inspect sequence length, nucleotide distribution, and GC content."
        )
    with col_g2:
        st.markdown("#### 2️⃣ Run CRISPR Alignment")
        st.write(
            "Open **CRISPR Alignment** (`pages/2_CRISPR_Alignment.py`). "
            "Configure your guide RNA (e.g., `GAGTCCGAGCAGAAGAAGAA`) and PAM (`NGG`). "
            "Select CPU or GPU engine and click **Execute Alignment**."
        )
    with col_g3:
        st.markdown("#### 3️⃣ Profile Performance")
        st.write(
            "Visit **Performance Benchmark** (`pages/3_Performance_Benchmark.py`). "
            "Run single or scaling tests across sequence sizes to observe real hardware "
            "throughput and speedup metrics."
        )

with tab_about:
    st.markdown("### Project & Evaluation Metadata")
    st.markdown("""
    - **Project Name:** GeneWeaver
    - **Project Title:** GPU-Accelerated CRISPR Alignment Engine
    - **Domain:** Bioinformatics & Python High-Performance Computing
    - **Developer:** Individual Internship Project (Infotact Solutions)
    - **Programming Language:** Python 3.13
    - **Key Libraries:** Biopython, PyTorch, Streamlit, Pandas, NumPy, Plotly, Pytest
    - **License:** MIT Open-Source
    """)

st.markdown("---")
st.caption("GeneWeaver CRISPR Alignment Engine • Developed with Python, PyTorch & Biopython")
