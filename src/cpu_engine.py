"""
GeneWeaver: CPU Alignment Engine
================================
Provides the deterministic reference implementation for CRISPR target scanning.

Design & Scientific Principles:
- Employs a sliding-window string comparison algorithm.
- Serves as the ground-truth reference implementation against which GPU results are verified.
- Measures real-world execution latency using time.perf_counter().
- Scans both forward (sense) and reverse-complement (antisense) strands.
"""

import time
from typing import List, Optional
from src.crispr_alignment import (
    CRISPRTargetMatch,
    AlignmentEngineResult,
    match_pam,
    compute_mismatches,
    calculate_alignment_score,
)
from src.sequence_utils import clean_sequence, reverse_complement


class CPUEngine:
    """
    CPU-based sequence alignment engine for CRISPR-Cas9 target identification.
    """

    def __init__(self):
        self.device_name = "CPU (Python Native Reference)"

    def align(
        self,
        reference_seq: str,
        reference_id: str,
        guide_seq: str,
        pam_pattern: str = "NGG",
        max_mismatches: int = 3,
        scan_both_strands: bool = True,
    ) -> AlignmentEngineResult:
        """
        Scan a reference DNA sequence against a CRISPR guide sequence using the CPU.

        Parameters:
            reference_seq (str): Reference DNA sequence string.
            reference_id (str): Identifier of the reference sequence.
            guide_seq (str): 5' to 3' CRISPR guide sequence.
            pam_pattern (str): PAM motif (e.g. 'NGG').
            max_mismatches (int): Maximum allowable mismatches.
            scan_both_strands (bool): If True, scans both forward and reverse strands.

        Returns:
            AlignmentEngineResult: Complete alignment results including timing metrics.
        """
        clean_ref = clean_sequence(reference_seq)
        clean_guide = clean_sequence(guide_seq)
        clean_pam = clean_sequence(pam_pattern)

        guide_len = len(clean_guide)
        pam_len = len(clean_pam)
        window_len = guide_len + pam_len
        ref_len = len(clean_ref)

        matches: List[CRISPRTargetMatch] = []
        total_windows = 0

        # High-resolution start timer
        start_time = time.perf_counter()

        if ref_len >= window_len:
            # -------------------------------------------------------------
            # 1. Forward Strand Scanning (Sense Strand: 5' -> 3')
            # -------------------------------------------------------------
            num_forward_windows = ref_len - window_len + 1
            total_windows += num_forward_windows

            for i in range(num_forward_windows):
                # Window: [Protospacer: guide_len] + [PAM: pam_len]
                cand_target = clean_ref[i : i + guide_len]
                cand_pam = clean_ref[i + guide_len : i + window_len]

                if match_pam(cand_pam, clean_pam):
                    mm_count, mm_pos, seed_mm, distal_mm, visual = compute_mismatches(clean_guide, cand_target)
                    if mm_count <= max_mismatches:
                        score = calculate_alignment_score(guide_len, mm_count, seed_mm, distal_mm)
                        match = CRISPRTargetMatch(
                            reference_id=reference_id,
                            strand="+",
                            start_pos=i + 1,  # 1-based biological coordinate
                            end_pos=i + window_len,
                            target_dna_sequence=cand_target,
                            pam_sequence=cand_pam,
                            full_window=clean_ref[i : i + window_len],
                            mismatch_count=mm_count,
                            mismatch_positions=mm_pos,
                            seed_mismatches=seed_mm,
                            distal_mismatches=distal_mm,
                            alignment_score=score,
                            alignment_visual=visual,
                        )
                        matches.append(match)

            # -------------------------------------------------------------
            # 2. Reverse Complement Strand Scanning (Antisense Strand)
            # -------------------------------------------------------------
            if scan_both_strands:
                ref_rc = reverse_complement(clean_ref)
                num_reverse_windows = len(ref_rc) - window_len + 1
                total_windows += num_reverse_windows

                for i_rc in range(num_reverse_windows):
                    cand_target = ref_rc[i_rc : i_rc + guide_len]
                    cand_pam = ref_rc[i_rc + guide_len : i_rc + window_len]

                    if match_pam(cand_pam, clean_pam):
                        mm_count, mm_pos, seed_mm, distal_mm, visual = compute_mismatches(clean_guide, cand_target)
                        if mm_count <= max_mismatches:
                            score = calculate_alignment_score(guide_len, mm_count, seed_mm, distal_mm)

                            # Coordinate mapping back to forward strand:
                            # In 1-based coordinates on forward strand:
                            # Start = ref_len - (i_rc + window_len) + 1
                            # End = ref_len - i_rc
                            fwd_start = ref_len - (i_rc + window_len) + 1
                            fwd_end = ref_len - i_rc

                            match = CRISPRTargetMatch(
                                reference_id=reference_id,
                                strand="-",
                                start_pos=fwd_start,
                                end_pos=fwd_end,
                                target_dna_sequence=cand_target,
                                pam_sequence=cand_pam,
                                full_window=ref_rc[i_rc : i_rc + window_len],
                                mismatch_count=mm_count,
                                mismatch_positions=mm_pos,
                                seed_mismatches=seed_mm,
                                distal_mismatches=distal_mm,
                                alignment_score=score,
                                alignment_visual=visual,
                            )
                            matches.append(match)

        # High-resolution end timer
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        elapsed_ms = elapsed_seconds * 1000.0

        # Rank candidate matches:
        # Primary sort: Mismatches ascending (fewest mismatches first)
        # Secondary sort: Alignment Score descending (highest score first)
        matches.sort(key=lambda m: (m.mismatch_count, -m.alignment_score))

        throughput = (total_windows / elapsed_seconds) if elapsed_seconds > 0 else float("inf")

        return AlignmentEngineResult(
            matches=matches,
            total_windows_evaluated=total_windows,
            execution_time_seconds=elapsed_seconds,
            execution_time_ms=elapsed_ms,
            device_used=self.device_name,
            windows_per_second=round(throughput, 2),
            guide_length=guide_len,
            pam_pattern=clean_pam,
            max_mismatches=max_mismatches,
        )

