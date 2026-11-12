"""
GeneWeaver: Dataset Explorer
============================
Browse pre-packaged synthetic reference sequences, inspect guide libraries,
and review testing scenarios, biological provenance, and data dictionaries.
"""

import streamlit as st
import pandas as pd
from src.ui_components import (
    apply_custom_theme,
    render_hero_banner,
    render_device_status_sidebar,
    format_sequence_html,
)
from src.input_handler import (
    load_sample_reference_fasta,
    load_sample_guides_csv,
    get_data_dir,
)

st.set_page_config(page_title="Dataset Explorer – GeneWeaver", page_icon="📁", layout="wide")
apply_custom_theme()
render_device_status_sidebar()

render_hero_banner(
    title="Dataset Explorer & Data Dictionary",
    subtitle="Inspect pre-packaged synthetic reference genomes, curated CRISPR guide libraries, and test scenarios.",
    icon="📁",
)

tab_fasta, tab_guides, tab_dictionary, tab_provenance = st.tabs([
    "🧬 Reference FASTA Datasets",
    "🎯 Sample Guide RNA Library",
    "📖 Data Dictionary",
    "🔬 Provenance & Test Scenarios",
])

# -----------------------------------------------------------------------------
# TAB 1: Reference FASTA Datasets
# -----------------------------------------------------------------------------
with tab_fasta:
    st.markdown("### Pre-Packaged Reference Sequences (`data/sample_reference.fasta`)")
    st.caption("All pre-packaged sequences are clearly labeled as `[organism=synthetic]`.")

    try:
        fasta_records = load_sample_reference_fasta()
        rec_names = [f"{r.id} ({r.length:,} bp)" for r in fasta_records]
        selected_rec_name = st.selectbox("Select Reference Locus to Inspect:", rec_names)
        active_rec = next(r for r in fasta_records if f"{r.id} ({r.length:,} bp)" == selected_rec_name)

        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            st.metric("Sequence Length", f"{active_rec.length:,} bp")
        with col_f2:
            st.metric("GC Content", f"{active_rec.stats['gc_content']}%")
        with col_f3:
            st.metric("AT Content", f"{active_rec.stats['at_content']}%")
        with col_f4:
            st.metric("Purine / Pyrimidine", active_rec.stats["purine_pyrimidine_ratio"])

        st.markdown(f"**Header Description:** `{active_rec.description}`")
        st.markdown("#### Sequence Preview")
        st.markdown(format_sequence_html(active_rec.sequence, max_display_chars=1200), unsafe_allow_html=True)

        fasta_bytes = f">{active_rec.id} {active_rec.description}\n{active_rec.sequence}\n".encode("utf-8")
        st.download_button(
            label=f"📥 Download {active_rec.id}.fasta",
            data=fasta_bytes,
            file_name=f"{active_rec.id}.fasta",
            mime="text/plain",
        )
    except Exception as e:
        st.error(f"Error loading reference datasets: {e}")

# -----------------------------------------------------------------------------
# TAB 2: Sample Guide RNA Library
# -----------------------------------------------------------------------------
with tab_guides:
    st.markdown("### Curated Guide Sequences (`data/sample_guides.csv`)")
    st.write(
        "A multi-scenario guide library covering exact matches, single-nucleotide mismatches, "
        "multiple mismatches across seed and distal regions, reverse-strand targets, and negative controls."
    )

    try:
        guides_df = load_sample_guides_csv()

        # Filtering
        target_filter = st.selectbox("Filter by Expected Target Locus:", ["All Loci"] + list(guides_df["target_locus"].unique()))
        if target_filter != "All Loci":
            filtered_guides = guides_df[guides_df["target_locus"] == target_filter]
        else:
            filtered_guides = guides_df

        st.dataframe(filtered_guides, use_container_width=True, hide_index=True)

        csv_data = guides_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download sample_guides.csv",
            data=csv_data,
            file_name="sample_guides.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Error loading guides dataset: {e}")

# -----------------------------------------------------------------------------
# TAB 3: Data Dictionary
# -----------------------------------------------------------------------------
with tab_dictionary:
    st.markdown("### Formal Data Dictionary for Datasets")

    st.markdown("#### Reference FASTA Attributes")
    df_dict_fasta = pd.DataFrame([
        {"Field": "Record ID", "Type": "String", "Description": "Unique identifier for the chromosome or scaffold (e.g. synthetic_ref_01_emx1_locus)."},
        {"Field": "Description", "Type": "String", "Description": "Full FASTA header line containing metadata tags like organism=synthetic and homolog target."},
        {"Field": "Sequence", "Type": "String", "Description": "Canonical DNA base pairs (A, C, G, T) in 5' to 3' orientation."},
        {"Field": "Length", "Type": "Integer", "Description": "Total base pair count of the locus."},
    ])
    st.table(df_dict_fasta)

    st.markdown("#### Guide Library Attributes (`sample_guides.csv`)")
    df_dict_guides = pd.DataFrame([
        {"Column": "guide_id", "Type": "String", "Description": "Unique identifier for the guide RNA (e.g. G01_EMX1_Exact)."},
        {"Column": "sequence", "Type": "String", "Description": "5' to 3' nucleotide sequence of the guide spacer (typically 20 bp)."},
        {"Column": "target_locus", "Type": "String", "Description": "Expected matching locus identifier in sample_reference.fasta."},
        {"Column": "expected_strand", "Type": "Categorical", "Description": "Strand containing the target: forward (+), reverse (-), or none."},
        {"Column": "expected_mismatches", "Type": "Integer", "Description": "Ground-truth number of mismatches (0 for exact, 99 for negative control)."},
        {"Column": "pam_motif", "Type": "String", "Description": "Expected Protospacer Adjacent Motif (e.g. NGG for SpCas9)."},
        {"Column": "notes", "Type": "String", "Description": "Scientific context, literature reference, or specific validation role."},
    ])
    st.table(df_dict_guides)

# -----------------------------------------------------------------------------
# TAB 4: Provenance & Test Scenarios
# -----------------------------------------------------------------------------
with tab_provenance:
    st.markdown("### Biological Provenance and Scenario Mapping")
    st.markdown("""
    The sample datasets in GeneWeaver were designed following canonical benchmarks in CRISPR literature:

    1. **Human *EMX1* Target Site 1** (Cong et al., *Science* 2013):
       - Target: `GAGTCCGAGCAGAAGAAGAA GGG`
       - One of the first genomic sites targeted by CRISPR-Cas9 in human cells.
       - Used in GeneWeaver as the primary exact-match ground truth (`G01`).
    2. **Single-Mismatch Variant** (`G02`):
       - Target: `GAGTCCGAGCAGTAGAAGAA` (A $\\rightarrow$ T substitution at position 14).
       - Tests algorithm sensitivity to single base substitutions in the proximal region.
    3. **Three-Mismatch Off-Target Variant** (`G03`):
       - Target: `GAGTCCATGCAATAGAAGAA` (Substitutions across distal and proximal regions).
       - Simulates candidate off-target homolog searching.
    4. **Human *VEGFA* Target Site 1** (Fu et al., *Nature Biotechnology* 2013):
       - Target: `GGGTGGGGGGAGTTTGCTCC TGG`
       - A challenging, GC-rich (>70% GC) target sequence with multiple G-repeats.
    5. **Antisense Strand Target** (`G05`):
       - Evaluates reverse-complement matching and coordinate mapping back to the forward strand.
    6. **Negative Control** (`G06`):
       - Poly-C sequence (`CCCCCCCCCCCCCCCCCCCC`) that shares 0% homology with the EMX1 locus.
    """)

