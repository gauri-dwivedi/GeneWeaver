"""
GeneWeaver: CRISPR Alignment Core Architecture
==============================================
Contains the domain models, PAM matching algorithms, mismatch calculation,
and biological scoring logic for CRISPR-Cas9 target identification.

Educational Guide for Students:
-------------------------------
In CRISPR-Cas9 genome editing:
1. The Cas9 endonuclease is guided by a ~20 nucleotide single guide RNA (sgRNA).
2. Cas9 first interrogates DNA by binding to a short motif called the
   Protospacer Adjacent Motif (PAM). For SpCas9, the canonical PAM is 5'-NGG-3'.
3. If a compatible PAM is found, Cas9 initiates local DNA unwinding from the PAM-proximal
   end (bases 11-20, called the 'Seed Region') towards the PAM-distal end (bases 1-10).
4. Mismatches in the seed region severely inhibit cleavage activity, while mismatches
   in the distal region are often tolerated (leading to potential off-target effects).
5. GeneWeaver calculates both the raw mismatch count and a weighted specificity score
   calibrated on empirical seed-weighting principles (Hsu et al., Nature Biotech 2013).

IMPORTANT SCIENTIFIC CAVEAT:
Computational sequence homology does NOT guarantee biological cleavage.
Actual in vivo editing depends on chromatin state, epigenetics, and Cas9 kinetics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.validator import IUPAC_NUCLEOTIDE_MAP
from src.sequence_utils import clean_sequence, reverse_complement


@dataclass
class CRISPRTargetMatch:
    """
    Detailed information for a single candidate CRISPR target site identified on DNA.

    Attributes:
        reference_id (str): Identifier of the reference chromosome/scaffold.
        strand (str): '+' (forward) or '-' (reverse-complement).
        start_pos (int): 1-based start coordinate on the forward reference.
        end_pos (int): 1-based end coordinate on the forward reference.
        target_dna_sequence (str): The genomic protospacer sequence (matching guide).
        pam_sequence (str): The genomic PAM motif sequence adjacent to protospacer.
        full_window (str): Combined target + PAM sequence.
        mismatch_count (int): Total number of base mismatches against guide.
        mismatch_positions (List[int]): 1-based mismatch indices relative to 5' end of guide.
        seed_mismatches (int): Mismatches occurring in the PAM-proximal seed region (bases 11-20).
        distal_mismatches (int): Mismatches in the PAM-distal region (bases 1-10).
        alignment_score (float): Normalized alignment score (0 - 100), where 100 = perfect match.
        alignment_visual (str): Visual representation of matching ('|' = match, '.' = mismatch).
    """
    reference_id: str
    strand: str
    start_pos: int
    end_pos: int
    target_dna_sequence: str
    pam_sequence: str
    full_window: str
    mismatch_count: int
    mismatch_positions: List[int]
    seed_mismatches: int
    distal_mismatches: int
    alignment_score: float
    alignment_visual: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert match object into dictionary suitable for pandas DataFrame display."""
        return {
            "Reference ID": self.reference_id,
            "Strand": self.strand,
            "Start (1-based)": self.start_pos,
            "End (1-based)": self.end_pos,
            "Target DNA (5'->3')": self.target_dna_sequence,
            "PAM": self.pam_sequence,
            "Full Window": self.full_window,
            "Mismatches": self.mismatch_count,
            "Mismatch Positions": ", ".join(map(str, self.mismatch_positions)) if self.mismatch_positions else "None (Exact)",
            "Seed Mismatches (11-20)": self.seed_mismatches,
            "Distal Mismatches (1-10)": self.distal_mismatches,
            "Alignment Score": round(self.alignment_score, 2),
            "Visual Alignment": self.alignment_visual,
        }


@dataclass
class AlignmentEngineResult:
    """
    Container for the complete output of an alignment execution run.

    Attributes:
        matches (List[CRISPRTargetMatch]): All candidate targets within mismatch threshold.
        total_windows_evaluated (int): Number of candidate sliding windows examined.
        execution_time_seconds (float): Wall-clock processing time in seconds.
        execution_time_ms (float): Wall-clock processing time in milliseconds.
        device_used (str): Processing device ('CPU' or 'GPU: <device_name>').
        windows_per_second (float): Processing throughput (windows scanned / second).
        guide_length (int): Length of the evaluated guide RNA.
        pam_pattern (str): The PAM motif filter applied.
        max_mismatches (int): Configured mismatch cutoff threshold.
    """
    matches: List[CRISPRTargetMatch]
    total_windows_evaluated: int
    execution_time_seconds: float
    execution_time_ms: float
    device_used: str
    windows_per_second: float
    guide_length: int
    pam_pattern: str
    max_mismatches: int


def match_pam(candidate_pam: str, pam_pattern: str) -> bool:
    """
    Determine whether a genomic candidate PAM sequence matches a PAM motif pattern,
    accounting for IUPAC degenerate nucleotide codes (e.g., 'N' = any base, 'R' = A or G).

    Parameters:
        candidate_pam (str): Genomic sequence slice (e.g. 'CGG', 'TGG').
        pam_pattern (str): Target PAM motif (e.g. 'NGG', 'NNGRRT').

    Returns:
        bool: True if candidate satisfies the PAM pattern.

    Example:
        >>> match_pam("AGG", "NGG")
        True
        >>> match_pam("AGT", "NGG")
        False
    """
    if len(candidate_pam) != len(pam_pattern):
        return False

    cand_upper = candidate_pam.upper()
    pattern_upper = pam_pattern.upper()

    for cand_char, pat_code in zip(cand_upper, pattern_upper):
        allowed_bases = IUPAC_NUCLEOTIDE_MAP.get(pat_code, {pat_code})
        if cand_char not in allowed_bases:
            return False

    return True


def compute_mismatches(guide_seq: str, target_seq: str) -> tuple[int, List[int], int, int, str]:
    """
    Compute base-by-base mismatches between a CRISPR guide and candidate genomic target.

    Parameters:
        guide_seq (str): 5' to 3' guide sequence (length L).
        target_seq (str): 5' to 3' target sequence (length L).

    Returns:
        tuple containing:
            - mismatch_count (int): Total mismatch count.
            - mismatch_positions (List[int]): 1-based mismatch indices (1 = 5' base).
            - seed_mismatches (int): Mismatch count in positions 11-20 (for standard 20bp guide).
            - distal_mismatches (int): Mismatch count in positions 1-10.
            - alignment_visual (str): String with '|' for match and '.' for mismatch.
    """
    mismatches = 0
    positions: List[int] = []
    visual_chars: List[str] = []

    guide_len = len(guide_seq)
    # Seed boundary: last 10 bases adjacent to PAM (e.g., positions 11-20 for 20bp guide)
    seed_start_1based = max(1, guide_len - 10 + 1)

    seed_mismatches = 0
    distal_mismatches = 0

    for idx, (g_base, t_base) in enumerate(zip(guide_seq, target_seq)):
        pos_1based = idx + 1
        if g_base != t_base:
            mismatches += 1
            positions.append(pos_1based)
            visual_chars.append(".")
            if pos_1based >= seed_start_1based:
                seed_mismatches += 1
            else:
                distal_mismatches += 1
        else:
            visual_chars.append("|")

    return (
        mismatches,
        positions,
        seed_mismatches,
        distal_mismatches,
        "".join(visual_chars),
    )


def calculate_alignment_score(
    guide_len: int,
    mismatch_count: int,
    seed_mismatches: int,
    distal_mismatches: int
) -> float:
    """
    Calculate an alignment specificity score (0.0 to 100.0).

    Scoring Logic:
    - Perfect match (0 mismatches) = 100.0.
    - Seed region mismatches carry a heavier penalty (weight = 2.5x)
      because mismatches close to the PAM disrupt Cas9 R-loop formation.
    - Distal region mismatches carry standard penalty (weight = 1.0x).
    - Score is normalized between 0.0 and 100.0:
        penalty = (distal_mm * 1.0 + seed_mm * 2.5) / (guide_len * 1.5)
        score = max(0.0, 100.0 * (1.0 - penalty))

    Parameters:
        guide_len (int): Total guide length.
        mismatch_count (int): Total mismatches.
        seed_mismatches (int): Seed region mismatches.
        distal_mismatches (int): Distal region mismatches.

    Returns:
        float: Score ranging from 0.0 to 100.0.
    """
    if mismatch_count == 0:
        return 100.0

    weighted_penalty = (distal_mismatches * 1.0) + (seed_mismatches * 2.5)
    max_possible_penalty = max(1.0, guide_len * 1.75)
    score = 100.0 * max(0.0, 1.0 - (weighted_penalty / max_possible_penalty))
    return round(score, 2)
