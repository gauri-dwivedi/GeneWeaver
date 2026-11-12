"""
GeneWeaver: Report Generator Module
===================================
Produces structured reports and export formats (CSV, Markdown, JSON)
for alignment results and performance benchmarks.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from src.crispr_alignment import AlignmentEngineResult, CRISPRTargetMatch


def matches_to_dataframe(matches: List[CRISPRTargetMatch]) -> pd.DataFrame:
    """
    Convert a list of CRISPRTargetMatch objects into a pandas DataFrame.

    Parameters:
        matches (List[CRISPRTargetMatch]): Target matches.

    Returns:
        pd.DataFrame: Formatted table with human-readable column names.
    """
    if not matches:
        return pd.DataFrame(columns=[
            "Reference ID", "Strand", "Start (1-based)", "End (1-based)",
            "Target DNA (5'->3')", "PAM", "Full Window", "Mismatches",
            "Mismatch Positions", "Seed Mismatches (11-20)",
            "Distal Mismatches (1-10)", "Alignment Score", "Visual Alignment"
        ])
    return pd.DataFrame([m.to_dict() for m in matches])


def generate_markdown_alignment_report(
    result: AlignmentEngineResult,
    guide_seq: str,
    reference_id: str,
    reference_len: int,
) -> str:
    """
    Generate a formatted Markdown report of an alignment run.

    Parameters:
        result (AlignmentEngineResult): Alignment run results.
        guide_seq (str): Guide sequence evaluated.
        reference_id (str): Reference sequence ID.
        reference_len (int): Length of reference sequence.

    Returns:
        str: Markdown document string.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    match_count = len(result.matches)

    md = [
        f"# GeneWeaver CRISPR Alignment Report",
        f"**Generated:** {now_str}",
        "",
        "## 1. Run Configuration & Parameters",
        f"- **Reference Target:** `{reference_id}` ({reference_len:,} bp)",
        f"- **Guide Sequence (5'->3'):** `{guide_seq}` ({len(guide_seq)} bp)",
        f"- **PAM Motif Filter:** `{result.pam_pattern}`",
        f"- **Max Allowed Mismatches:** `{result.max_mismatches}`",
        f"- **Compute Engine / Device:** `{result.device_used}`",
        f"- **Execution Time:** `{result.execution_time_ms:.2f} ms`",
        f"- **Throughput:** `{result.windows_per_second:,.1f} windows/sec`",
        f"- **Total Windows Evaluated:** `{result.total_windows_evaluated:,}`",
        "",
        "## 2. Summary of Identified Target Sites",
        f"- **Total Candidate Targets Found:** `{match_count}`",
    ]

    if match_count == 0:
        md.append("")
        md.append("> No candidate target sites matched the given guide within the configured mismatch threshold.")
    else:
        exact_matches = sum(1 for m in result.matches if m.mismatch_count == 0)
        one_mm = sum(1 for m in result.matches if m.mismatch_count == 1)
        two_plus_mm = sum(1 for m in result.matches if m.mismatch_count >= 2)

        md.extend([
            f"- **Exact Matches (0 Mismatches):** {exact_matches}",
            f"- **1-Mismatch Candidates:** {one_mm}",
            f"- **Multi-Mismatch (>= 2 Mismatches):** {two_plus_mm}",
            "",
            "## 3. Top Candidate Target Sites",
            "",
            "| Rank | Strand | Coordinates (1-based) | Target DNA (5'->3') | PAM | Mismatches | Seed MM | Score | Visual |",
            "| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |",
        ])

        for idx, m in enumerate(result.matches[:25], start=1):
            md.append(
                f"| {idx} | `{m.strand}` | {m.start_pos}–{m.end_pos} | `{m.target_dna_sequence}` | `{m.pam_sequence}` | "
                f"{m.mismatch_count} | {m.seed_mismatches} | {m.alignment_score:.1f} | `{m.alignment_visual}` |"
            )

        if match_count > 25:
            md.append(f"\n*(Showing top 25 of {match_count} total candidate target sites)*")

    md.extend([
        "",
        "---",
        "### Scientific Caveat",
        "GeneWeaver calculates computational sequence matching and mismatch penalty scores. "
        "Computational homology does not guarantee in vivo cleavage or editing efficiency.",
    ])

    return "\n".join(md)


def export_alignment_csv(
    matches: List[CRISPRTargetMatch],
    custom_path: Optional[str] = None
) -> Path:
    """
    Save alignment matches table to a CSV file in results/.

    Parameters:
        matches (List[CRISPRTargetMatch]): Matches to export.
        custom_path (Optional[str]): Custom file path.

    Returns:
        Path: Path to saved file.
    """
    df = matches_to_dataframe(matches)
    if custom_path:
        target_path = Path(custom_path)
    else:
        results_dir = Path(__file__).resolve().parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = results_dir / f"alignment_results_{timestamp}.csv"

    df.to_csv(target_path, index=False)
    return target_path
