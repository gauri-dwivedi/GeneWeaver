"""
GeneWeaver: DNA Sequence Analyzer
=================================
Allows users to input, validate, and compute composition metrics for DNA sequences.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.ui_components import (
    apply_custom_theme,
    render_hero_banner,
    render_device_status_sidebar,
    format_sequence_html,
)
from src.validator import validate_dna_sequence
from src.sequence_utils import reverse_complement, transcribe_to_rna
from src.input_handler import (
    parse_pasted_text,
    parse_fasta_content,
    load_sample_reference_fasta,
    SequenceRecord,
)

st.set_page_config(page_title="Sequence Analyzer – GeneWeaver", page_icon="🧬", layout="wide")
apply_custom_theme()
render_device_status_sidebar()

render_hero_banner(
    title="DNA Sequence Analyzer",
    subtitle="Input, validate, and explore nucleotide composition, GC content, and biological statistics.",
    icon="🔬",
)

# Input Method Selection
input_mode = st.radio(
    "Choose Sequence Input Source:",
    ["Use Built-in Sample Reference", "Paste DNA / FASTA Text", "Upload FASTA File (.fasta / .fa)"],
    horizontal=True,
)

active_record: SequenceRecord | None = None

if input_mode == "Use Built-in Sample Reference":
    try:
        sample_records = load_sample_reference_fasta()
        record_map = {f"{r.id} ({r.length:,} bp) - {r.description[:50]}...": r for r in sample_records}
        selected_key = st.selectbox("Select Sample Locus:", list(record_map.keys()))
        active_record = record_map[selected_key]
    except Exception as e:
        st.error(f"Error loading sample datasets: {e}")

elif input_mode == "Paste DNA / FASTA Text":
    default_text = (
        ">custom_sample_target\n"
        "TCAGTTCTTGCCCCTCAGGACGAGGATGTCACAGCCCAGGGCCTTGCCCAGCCCTCCAGG\n"
        "ACTCCCACCCCGCCCCCTCTCCCCAGGCACCTGAGGACCACATCACAAAGGGAGAGCCAG\n"
        "TGCTCTTTCCAAACTCCCACACCGCCACCCCCAGCCAGTGAGTGAGTGAGTGAGTCCCAG\n"
        "GCTCAGGGCTCAGAGTCCGAGCAGAAGAAGAAGGGCTCCCATCACATCAACCGGTGGCGC\n"
    )
    pasted_input = st.text_area("Paste DNA or FASTA formatted sequence here:", value=default_text, height=180)
    if pasted_input.strip():
        try:
            parsed_list = parse_pasted_text(pasted_input)
            if parsed_list:
                active_record = parsed_list[0]
                if len(parsed_list) > 1:
                    st.info(f"Multi-FASTA input detected with {len(parsed_list)} sequences. Analyzing first sequence: `{active_record.id}`.")
        except Exception as e:
            st.error(f"Error parsing pasted sequence: {e}")

elif input_mode == "Upload FASTA File (.fasta / .fa)":
    uploaded_file = st.file_uploader("Upload reference sequence file", type=["fasta", "fa", "fna", "txt"])
    if uploaded_file is not None:
        try:
            records = parse_fasta_content(uploaded_file)
            if records:
                if len(records) == 1:
                    active_record = records[0]
                else:
                    rec_choice = st.selectbox("Select Record from Multi-FASTA:", [r.id for r in records])
                    active_record = next(r for r in records if r.id == rec_choice)
        except Exception as e:
            st.error(f"Could not parse uploaded FASTA file: {e}")

st.markdown("---")

if active_record is not None and active_record.sequence:
    # Validate sequence
    validation = validate_dna_sequence(
        active_record.sequence,
        allow_ambiguous=False,
        min_length=10,
        sequence_label=f"Sequence '{active_record.id}'"
    )

    if validation.is_valid:
        st.success(f"✅ Sequence `{active_record.id}` is valid canonical DNA ({validation.stats['length']:,} base pairs).")
        if validation.warnings:
            for w in validation.warnings:
                st.warning(f"⚠️ {w}")
    else:
        st.error(f"❌ Validation Failed for `{active_record.id}`:")
        for err in validation.errors:
            st.markdown(f"- {err}")

    stats = validation.stats

    # Key Nucleotide Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sequence Length</div>
            <div class="metric-value">{stats['length']:,} bp</div>
            <span style="color:#94A3B8; font-size:0.85rem;">Total base pairs</span>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">GC Content</div>
            <div class="metric-value" style="color:#38BDF8;">{stats['gc_content']}%</div>
            <span style="color:#94A3B8; font-size:0.85rem;">Guanine + Cytosine ratio</span>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">AT Content</div>
            <div class="metric-value" style="color:#F59E0B;">{stats['at_content']}%</div>
            <span style="color:#94A3B8; font-size:0.85rem;">Adenine + Thymine ratio</span>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        ratio_str = f"{stats['purine_pyrimidine_ratio']}" if stats['purines'] > 0 else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Purine / Pyrimidine</div>
            <div class="metric-value">{ratio_str}</div>
            <span style="color:#94A3B8; font-size:0.85rem;">(A + G) / (C + T)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for Visualization, Composition, Transformations
    t_viz, t_chart, t_transforms = st.tabs(["🧬 Sequence Visualizer", "📊 Nucleotide Chart", "🔄 Transformations"])

    with t_viz:
        st.markdown("#### Syntax-Highlighted Sequence Viewer")
        st.caption("Color Key: Adenine (A)=Green, Cytosine (C)=Blue, Guanine (G)=Yellow, Thymine (T)=Red")
        st.markdown(format_sequence_html(active_record.sequence), unsafe_allow_html=True)

    with t_chart:
        st.markdown("#### Base Composition Breakdown")
        df_comp = pd.DataFrame({
            "Base": ["Adenine (A)", "Cytosine (C)", "Guanine (G)", "Thymine (T)"],
            "Count": [stats["count_A"], stats["count_C"], stats["count_G"], stats["count_T"]],
            "Percentage (%)": [stats["pct_A"], stats["pct_C"], stats["pct_G"], stats["pct_T"]],
            "Color": ["#4ADE80", "#60A5FA", "#FBBF24", "#F87171"],
        })

        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            fig_bar = px.bar(
                df_comp,
                x="Base",
                y="Count",
                text="Percentage (%)",
                color="Base",
                color_discrete_map={
                    "Adenine (A)": "#4ADE80",
                    "Cytosine (C)": "#60A5FA",
                    "Guanine (G)": "#FBBF24",
                    "Thymine (T)": "#F87171",
                },
                title="Nucleotide Counts & Frequency",
            )
            fig_bar.update_layout(showlegend=False, template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_c2:
            fig_pie = px.pie(
                df_comp,
                names="Base",
                values="Count",
                color="Base",
                color_discrete_map={
                    "Adenine (A)": "#4ADE80",
                    "Cytosine (C)": "#60A5FA",
                    "Guanine (G)": "#FBBF24",
                    "Thymine (T)": "#F87171",
                },
                title="Proportional Base Distribution",
                hole=0.4,
            )
            fig_pie.update_layout(template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)

    with t_transforms:
        st.markdown("#### Biological Transformations")
        col_tr1, col_tr2 = st.columns(2)
        with col_tr1:
            st.markdown("##### 5' -> 3' Reverse-Complement (Antisense Strand)")
            rc_seq = reverse_complement(active_record.sequence)
            st.text_area("Reverse Complement Strand", value=rc_seq[:500], height=120)
            st.caption(f"Full reverse complement length: {len(rc_seq):,} bp")

        with col_tr2:
            st.markdown("##### mRNA Transcript (T substituted with U)")
            rna_seq = transcribe_to_rna(active_record.sequence)
            st.text_area("RNA Transcript", value=rna_seq[:500], height=120)
            st.caption(f"Full transcript length: {len(rna_seq):,} nt")

    # Send sequence to CRISPR Alignment Page via session state
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎯 Use This Sequence for CRISPR Alignment"):
        st.session_state["loaded_reference_seq"] = active_record.sequence
        st.session_state["loaded_reference_id"] = active_record.id
        st.success(f"Sequence `{active_record.id}` loaded into session! Go to the 'CRISPR Alignment' page to align guides against it.")
else:
    st.info("Please select or upload a sequence above to begin analysis.")

