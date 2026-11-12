"""
GeneWeaver: Results and Downloads Page
======================================
View historical alignment executions, inspect benchmark runs, and download
detailed CSV tables and formatted Markdown summary reports.
"""

import os
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd

from src.ui_components import (
    apply_custom_theme,
    render_hero_banner,
    render_device_status_sidebar,
)
from src.report_generator import (
    matches_to_dataframe,
    generate_markdown_alignment_report,
)

st.set_page_config(page_title="Results & Downloads – GeneWeaver", page_icon="📥", layout="wide")
apply_custom_theme()
render_device_status_sidebar()

render_hero_banner(
    title="Results & Report Downloads",
    subtitle="Access execution logs, download alignment tables (CSV), and export formatted Markdown evaluation reports.",
    icon="📥",
)

tab_alignment_reports, tab_persisted_files = st.tabs([
    "📄 Alignment Runs & Markdown Reports",
    "💾 Persisted CSV Results in `results/`",
])

# -----------------------------------------------------------------------------
# TAB 1: Alignment Runs & Markdown Reports
# -----------------------------------------------------------------------------
with tab_alignment_reports:
    history = st.session_state.get("alignment_history", [])

    if not history:
        st.info("No alignment runs recorded yet during this session. Run an alignment on the 'CRISPR Alignment' page to view reports here.")
    else:
        st.markdown(f"### Session Alignment History ({len(history)} Runs)")

        run_labels = [
            f"Run #{i+1} | {run['timestamp']} | Guide: {run['guide'][:10]}... | Matches: {run['matches_found']} ({run['device'][:20]})"
            for i, run in enumerate(history)
        ]

        selected_run_idx = st.selectbox("Select Alignment Run:", range(len(history)), format_func=lambda i: run_labels[i])
        selected_run = history[selected_run_idx]
        res_obj = selected_run["result_object"]

        # Run summary cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Candidate Matches Found", selected_run["matches_found"])
        with c2:
            st.metric("Execution Latency", f"{selected_run['execution_time_ms']:.2f} ms")
        with c3:
            st.metric("Engine Device", selected_run["device"][:25])

        # Generate Markdown Report
        md_report = generate_markdown_alignment_report(
            result=res_obj,
            guide_seq=selected_run["guide"],
            reference_id=selected_run["reference_id"],
            reference_len=st.session_state.get("latest_ref_len", 0),
        )

        st.markdown("#### Formatted Markdown Report Preview")
        st.markdown(md_report)

        st.markdown("<br>", unsafe_allow_html=True)
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.download_button(
                label="📥 Download Full Markdown Report (.md)",
                data=md_report.encode("utf-8"),
                file_name=f"alignment_report_{selected_run['reference_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )

        with col_d2:
            df_export = matches_to_dataframe(res_obj.matches)
            st.download_button(
                label="📥 Download Matches Table (.csv)",
                data=df_export.to_csv(index=False).encode("utf-8"),
                file_name=f"crispr_matches_{selected_run['reference_id']}.csv",
                mime="text/csv",
            )

        if st.button("🗑️ Clear Session History"):
            st.session_state["alignment_history"] = []
            st.rerun()

# -----------------------------------------------------------------------------
# TAB 2: Persisted CSV Results in results/
# -----------------------------------------------------------------------------
with tab_persisted_files:
    st.markdown("### Exported Result Files (`results/`)")

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    all_files = list(results_dir.glob("*.csv"))

    if not all_files:
        st.info("No persisted CSV files found in `results/`. Export an alignment or benchmark run to see it here.")
    else:
        st.write(f"Found {len(all_files)} saved result CSV file(s) on disk:")

        file_choices = {f.name: f for f in sorted(all_files, reverse=True)}
        selected_file_name = st.selectbox("Select File to View:", list(file_choices.keys()))
        selected_file_path = file_choices[selected_file_name]

        try:
            df_disk = pd.read_csv(selected_file_path)
            st.caption(f"File size: {selected_file_path.stat().st_size:,} bytes | Rows: {len(df_disk):,}")
            st.dataframe(df_disk, use_container_width=True)

            with open(selected_file_path, "rb") as f:
                st.download_button(
                    label=f"📥 Download {selected_file_name}",
                    data=f.read(),
                    file_name=selected_file_name,
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error reading file {selected_file_name}: {e}")
