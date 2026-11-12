"""
GeneWeaver: CRISPR Alignment Engine Page
========================================
Interactive CRISPR target scanning, dual-strand sliding window matching,
mismatch positioning, seed-weighted scoring, and visual alignment breakdown.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from src.ui_components import (
    apply_custom_theme,
    render_hero_banner,
    render_device_status_sidebar,
    render_visual_alignment,
)
from src.validator import (
    validate_dna_sequence,
    validate_guide_sequence,
    validate_pam_pattern,
)
from src.input_handler import (
    load_sample_reference_fasta,
    load_sample_guides_csv,
    parse_pasted_text,
    SequenceRecord,
)
from src.cpu_engine import CPUEngine
from src.gpu_engine import GPUEngine
from src.report_generator import matches_to_dataframe, export_alignment_csv

st.set_page_config(page_title="CRISPR Alignment – GeneWeaver", page_icon="🎯", layout="wide")
apply_custom_theme()
render_device_status_sidebar()

render_hero_banner(
    title="CRISPR Target Alignment Engine",
    subtitle="Scan double-stranded DNA for candidate CRISPR target sites, calculate mismatches, and inspect off-targets.",
    icon="🎯",
)

# Initialize Session State collections if missing
if "alignment_history" not in st.session_state:
    st.session_state["alignment_history"] = []

# Sidebar Configuration Settings
st.sidebar.markdown("### ⚙️ Alignment Parameters")

# 1. Guide Selection
guide_input_mode = st.sidebar.radio("Guide Source:", ["Preset Sample Guide", "Custom Guide Entry"])
guide_seq = "GAGTCCGAGCAGAAGAAGAA"

if guide_input_mode == "Preset Sample Guide":
    try:
        sample_guides_df = load_sample_guides_csv()
        guide_options = {
            f"{row['guide_id']} ({row['sequence']}) - {row['notes'][:40]}...": row['sequence']
            for _, row in sample_guides_df.iterrows()
        }
        selected_guide_key = st.sidebar.selectbox("Select Sample Guide:", list(guide_options.keys()))
        guide_seq = guide_options[selected_guide_key]
    except Exception as e:
        st.sidebar.error(f"Error loading sample guides: {e}")
else:
    guide_seq = st.sidebar.text_input("Custom Guide Sequence (5'->3'):", value="GAGTCCGAGCAGAAGAAGAA").strip().upper()

# 2. PAM Motif Pattern
pam_pattern = st.sidebar.text_input("PAM Motif Pattern (IUPAC):", value="NGG").strip().upper()
st.sidebar.caption("Common PAMs: `NGG` (SpCas9), `NAG` (alternative), `NNGRRT` (SaCas9)")

# 3. Mismatch Threshold & Strand
max_mismatches = st.sidebar.slider("Maximum Allowed Mismatches:", min_value=0, max_value=6, value=3)
scan_both_strands = st.sidebar.checkbox("Scan Both Strands (Forward & Reverse-Complement)", value=True)

# 4. Engine Selection
engine_choice = st.sidebar.radio("Compute Engine:", ["GPU (PyTorch Vectorized Batching)", "CPU (Native Reference Engine)"])

# Main Column Layout
col_ref, col_info = st.columns([2, 1])

with col_ref:
    st.markdown("### 1. Reference Sequence")
    ref_source = st.radio(
        "Choose Reference Source:",
        ["From Session / Built-in Samples", "Paste Custom Reference DNA"],
        horizontal=True
    )

    ref_seq = ""
    ref_id = "target_locus"

    if ref_source == "From Session / Built-in Samples":
        # Check if user forwarded a sequence from Sequence Analyzer page
        session_ref = st.session_state.get("loaded_reference_seq", "")
        session_id = st.session_state.get("loaded_reference_id", "")

        sample_records = load_sample_reference_fasta()
        sample_options = {f"{r.id} ({r.length:,} bp)": r for r in sample_records}

        if session_ref and session_id:
            st.info(f"Loaded active reference sequence from Session: `{session_id}` ({len(session_ref):,} bp).")
            ref_seq = session_ref
            ref_id = session_id
        else:
            choice = st.selectbox("Select Sample Reference Sequence:", list(sample_options.keys()))
            chosen_rec = sample_options[choice]
            ref_seq = chosen_rec.sequence
            ref_id = chosen_rec.id
    else:
        raw_pasted = st.text_area(
            "Paste Reference DNA Sequence:",
            value="TCAGTTCTTGCCCCTCAGGACGAGGATGTCACAGCCCAGGGCCTTGCCCAGCCCTCCAGGACTCCCACCCCGCCCCCTCTCCCCAGGCACCTGAGGACCACATCACAAAGGGAGAGCCAGTGCTCTTTCCAAACTCCCACACCGCCACCCCCAGCCAGTGAGTGAGTGAGTGAGTCCCAGGCTCAGGGCTCAGAGTCCGAGCAGAAGAAGAAGGGCTCCCATCACATCAACCGGTGGCGC",
            height=120
        )
        parsed = parse_pasted_text(raw_pasted, default_id="pasted_locus")
        if parsed:
            ref_seq = parsed[0].sequence
            ref_id = parsed[0].id

with col_info:
    st.markdown("### 2. Validation & Sanity Check")
    guide_val = validate_guide_sequence(guide_seq, expected_length=len(guide_seq))
    pam_val = validate_pam_pattern(pam_pattern)
    ref_val = validate_dna_sequence(ref_seq, min_length=len(guide_seq) + len(pam_pattern))

    all_valid = guide_val.is_valid and pam_val.is_valid and ref_val.is_valid

    if not all_valid:
        if not guide_val.is_valid:
            for err in guide_val.errors:
                st.error(f"Guide Error: {err}")
        if not pam_val.is_valid:
            for err in pam_val.errors:
                st.error(f"PAM Error: {err}")
        if not ref_val.is_valid:
            for err in ref_val.errors:
                st.error(f"Reference Error: {err}")
    else:
        st.success(f"Guide: `{guide_seq}` ({len(guide_seq)} bp) ✅")
        st.success(f"PAM: `{pam_pattern}` ({len(pam_pattern)} bp) ✅")
        st.success(f"Reference: `{ref_id}` ({len(ref_seq):,} bp) ✅")

        if guide_val.warnings:
            for w in guide_val.warnings:
                st.warning(f"⚠️ {w}")

st.markdown("---")

# Execution Button
if st.button("🚀 Execute CRISPR Alignment", disabled=not all_valid, type="primary"):
    with st.spinner("Executing sliding-window sequence alignment..."):
        if "GPU" in engine_choice:
            engine = GPUEngine()
        else:
            engine = CPUEngine()

        result = engine.align(
            reference_seq=ref_seq,
            reference_id=ref_id,
            guide_seq=guide_seq,
            pam_pattern=pam_pattern,
            max_mismatches=max_mismatches,
            scan_both_strands=scan_both_strands,
        )

        # Store in session state for Results page
        st.session_state["latest_alignment_result"] = result
        st.session_state["latest_guide_seq"] = guide_seq
        st.session_state["latest_ref_id"] = ref_id
        st.session_state["latest_ref_len"] = len(ref_seq)

        # Append to historical runs
        st.session_state["alignment_history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reference_id": ref_id,
            "guide": guide_seq,
            "matches_found": len(result.matches),
            "execution_time_ms": result.execution_time_ms,
            "device": result.device_used,
            "result_object": result,
        })

# Render Alignment Results if available in session state
if "latest_alignment_result" in st.session_state:
    result = st.session_state["latest_alignment_result"]
    guide_seq = st.session_state["latest_guide_seq"]
    ref_id = st.session_state["latest_ref_id"]
    match_count = len(result.matches)

    st.markdown("### 🎯 Alignment Execution Results")

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Target Matches Found</div>
            <div class="metric-value" style="color:#10B981;">{match_count}</div>
            <span style="color:#94A3B8; font-size:0.85rem;">≤ {result.max_mismatches} mismatches</span>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Execution Latency</div>
            <div class="metric-value">{result.execution_time_ms:.2f} ms</div>
            <span style="color:#94A3B8; font-size:0.85rem;">Wall-clock processing time</span>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Throughput</div>
            <div class="metric-value" style="color:#38BDF8;">{result.windows_per_second:,.0f}</div>
            <span style="color:#94A3B8; font-size:0.85rem;">Windows scanned / sec</span>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Processing Device</div>
            <div class="metric-value" style="font-size:1.15rem;">{result.device_used[:22]}</div>
            <span style="color:#94A3B8; font-size:0.85rem;">{result.total_windows_evaluated:,} windows evaluated</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if match_count == 0:
        st.warning(
            f"No matching target sites found matching guide `{guide_seq}` with PAM `{result.pam_pattern}` "
            f"within ≤ {result.max_mismatches} mismatches. Try increasing the mismatch limit or checking the PAM motif."
        )
    else:
        df_matches = matches_to_dataframe(result.matches)

        # Filter and Sorting widgets
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            strand_filter = st.multiselect("Filter by Strand:", options=["+", "-"], default=["+", "-"])
        with col_f2:
            sort_by = st.selectbox(
                "Sort Results By:",
                ["Mismatches (Lowest first)", "Alignment Score (Highest first)", "Start Coordinate"]
            )

        filtered_df = df_matches[df_matches["Strand"].isin(strand_filter)]
        if sort_by == "Mismatches (Lowest first)":
            filtered_df = filtered_df.sort_values(by=["Mismatches", "Alignment Score"], ascending=[True, False])
        elif sort_by == "Alignment Score (Highest first)":
            filtered_df = filtered_df.sort_values(by=["Alignment Score"], ascending=False)
        else:
            filtered_df = filtered_df.sort_values(by=["Start (1-based)"], ascending=True)

        st.dataframe(
            filtered_df[[
                "Strand", "Start (1-based)", "End (1-based)", "Target DNA (5'->3')",
                "PAM", "Mismatches", "Seed Mismatches (11-20)", "Alignment Score", "Visual Alignment"
            ]],
            use_container_width=True,
            hide_index=True,
        )

        # Detailed Individual Match Inspector
        st.markdown("#### 🔍 Individual Match Inspector & Visual Alignment")
        selected_index = st.selectbox(
            "Select Candidate Site to Inspect in Detail:",
            range(len(result.matches)),
            format_func=lambda idx: (
                f"Rank #{idx+1} | Strand {result.matches[idx].strand} | "
                f"Pos {result.matches[idx].start_pos}–{result.matches[idx].end_pos} | "
                f"{result.matches[idx].mismatch_count} Mismatches | Score: {result.matches[idx].alignment_score}"
            )
        )

        inspect_match = result.matches[selected_index]

        col_vis1, col_vis2 = st.columns([2, 1])
        with col_vis1:
            st.markdown("##### Visual Alignment Breakdown")
            render_visual_alignment(
                guide=guide_seq,
                target=inspect_match.target_dna_sequence,
                pam=inspect_match.pam_sequence,
                mismatches=inspect_match.mismatch_positions,
            )
            st.caption("Visual Key: Green letters and `|` denote exact matches. Red letters and `.` denote mismatches. Orange underlined text is the PAM motif.")

        with col_vis2:
            st.markdown("##### Biological Site Metrics")
            st.markdown(f"- **Strand:** `{'Sense / Forward (+)' if inspect_match.strand == '+' else 'Antisense / Reverse-Complement (-)'}`")
            st.markdown(f"- **Coordinates:** {inspect_match.start_pos} to {inspect_match.end_pos} bp")
            st.markdown(f"- **Seed Mismatches (Bases 11–20):** `{inspect_match.seed_mismatches}` *(High impact on Cas9)*")
            st.markdown(f"- **Distal Mismatches (Bases 1–10):** `{inspect_match.distal_mismatches}` *(Lower impact on Cas9)*")
            st.markdown(f"- **Calculated Specificity Score:** `{inspect_match.alignment_score:.2f} / 100.0`")

        # Download CSV
        st.markdown("<br>", unsafe_allow_html=True)
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Alignment Matches (CSV)",
            data=csv_data,
            file_name=f"crispr_matches_{ref_id}_{guide_seq[:10]}.csv",
            mime="text/csv",
        )

