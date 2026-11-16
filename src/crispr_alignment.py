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

    Genomic Coordinate Convention:
    ------------------------------
    All genomic coordinates in GeneWeaver are strictly 1-based and fully inclusive,
    referring directly to positions on the original forward reference sequence
    (matching GFF3, NCBI, and Ensembl standard conventions).
    - Length of interval = end_pos - start_pos + 1.
    - Python slice on forward reference = clean_ref[start_pos - 1 : end_pos].

    Forward Strand (+) Orientation:
    - Target protospacer is at the 5' end: [target_start .. target_end].
    - PAM motif is at the 3' end: [pam_start .. pam_end].
    - target_start = start_pos
    - target_end = start_pos + len(target) - 1
    - pam_start = start_pos + len(target)
    - pam_end = end_pos

    Reverse-Complement Strand (-) Orientation:
    - The candidate target lies on the antisense strand (5'->3' right-to-left).
    - When mapped onto the forward reference sequence (5'->3' left-to-right):
      The PAM's complement occupies the lower coordinates (left), and the target's
      complement occupies the higher coordinates (right):
      - pam_start = start_pos
      - pam_end = start_pos + len(pam) - 1
      - target_start = start_pos + len(pam)
      - target_end = end_pos
    - Sequence verification:
      clean_ref[pam_start - 1 : pam_end] == reverse_complement(pam_sequence)
      clean_ref[target_start - 1 : target_end] == reverse_complement(target_dna_sequence)

    Attributes:
        reference_id (str): Identifier of the reference chromosome/scaffold.
        strand (str): '+' (forward) or '-' (reverse-complement).
        start_pos (int): 1-based inclusive start coordinate on the forward reference.
        end_pos (int): 1-based inclusive end coordinate on the forward reference.
        target_dna_sequence (str): The genomic protospacer sequence (5'->3' on the matching strand).
        pam_sequence (str): The genomic PAM motif sequence (5'->3' on the matching strand).
        full_window (str): Combined target + PAM sequence (5'->3' on matching strand).
        mismatch_count (int): Total number of base mismatches against guide.
        mismatch_positions (List[int]): 1-based mismatch indices relative to 5' end of guide.
        seed_mismatches (int): Mismatches occurring in the PAM-proximal seed region.
        distal_mismatches (int): Mismatches in the PAM-distal region.
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

    @property
    def target_start(self) -> int:
        """1-based inclusive start coordinate of target protospacer on forward reference."""
        if self.strand == "+":
            return self.start_pos
        return self.start_pos + len(self.pam_sequence)

    @property
    def target_end(self) -> int:
        """1-based inclusive end coordinate of target protospacer on forward reference."""
        if self.strand == "+":
            return self.start_pos + len(self.target_dna_sequence) - 1
        return self.end_pos

    @property
    def pam_start(self) -> int:
        """1-based inclusive start coordinate of PAM motif on forward reference."""
        if self.strand == "+":
            return self.start_pos + len(self.target_dna_sequence)
        return self.start_pos

    @property
    def pam_end(self) -> int:
        """1-based inclusive end coordinate of PAM motif on forward reference."""
        if self.strand == "+":
            return self.end_pos
        return self.start_pos + len(self.pam_sequence) - 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert match object into dictionary suitable for pandas DataFrame display."""
        return {
            "Reference ID": self.reference_id,
            "Strand": self.strand,
            "Start (1-based)": self.start_pos,
            "End (1-based)": self.end_pos,
            "Target Start": self.target_start,
            "Target End": self.target_end,
            "PAM Start": self.pam_start,
            "PAM End": self.pam_end,
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
    distal_mismatches: int,
) -> float:
    """
    Calculate an alignment specificity score (0.0 to 100.0) with dynamic penalty
    scaling for supported guide lengths (typically 15-30 bp in GeneWeaver).

    Scoring Logic & Scientific Calibration:
    ---------------------------------------
    - Exact match (0 mismatches) = 100.0.
    - Cas9 target interrogation initiates at the PAM-proximal end (seed region,
      defined as the last min(10, guide_len) nucleotides adjacent to the 3' PAM).
    - Mismatches in the seed region carry a heavier penalty (weight = 2.5x)
      because they disrupt initial R-loop nucleation (Hsu et al., Nat Biotechnol 2013).
    - Mismatches in the PAM-distal region carry standard penalty (weight = 1.0x).
    - Dynamic Maximum Penalty:
        seed_len = min(10, max(1, guide_len))
        distal_len = max(0, guide_len - seed_len)
        max_possible_penalty = (distal_len * 1.0) + (seed_len * 2.5)
      For a canonical 20 bp guide:
        max_possible_penalty = (10 * 1.0) + (10 * 2.5) = 35.0
        (Identical to the 20 * 1.75 = 35.0 baseline, preserving exact backward compatibility).
    - Normalized Score:
        weighted_penalty = (distal_mismatches * 1.0) + (seed_mismatches * 2.5)
        score = 100.0 * max(0.0, 1.0 - (weighted_penalty / max_possible_penalty))

    Biological Scope & Modeling Boundary:
    -------------------------------------
    The 15-30 bp range and 10 bp seed partition represent GeneWeaver's software
    validation and modeling bounds for SpCas9-like Type II CRISPR systems (accommodating
    standard 20 bp guides, truncated 17-19 bp tru-gRNAs, and extended 21-25 bp guides).
    These bounds are engineering and heuristic modeling parameters, NOT universal
    biological laws. Nature features diverse CRISPR-Cas orthologs (e.g. Cas12a/Cpf1,
    Cas13, Cas14, and compact Cas12f) with different guide architectures, seed lengths,
    and cleavage kinetics.

    Numerical Example 1 (Standard 20 bp Guide):
    -------------------------------------------
    - Guide length: 20 bp (seed_len = 10, distal_len = 10)
    - Target has 1 distal mismatch and 1 seed mismatch (total 2 mismatches):
        weighted_penalty = (1 * 1.0) + (1 * 2.5) = 3.5
        max_possible_penalty = (10 * 1.0) + (10 * 2.5) = 35.0
        score = 100.0 * (1.0 - 3.5 / 35.0) = 100.0 * 0.90 = 90.00

    Numerical Example 2 (Truncated 17 bp Guide, tru-gRNA):
    ------------------------------------------------------
    - Guide length: 17 bp (seed_len = 10, distal_len = 7)
    - Target has 2 distal mismatches and 0 seed mismatches (total 2 mismatches):
        weighted_penalty = (2 * 1.0) + (0 * 2.5) = 2.0
        max_possible_penalty = (7 * 1.0) + (10 * 2.5) = 32.0
        score = 100.0 * (1.0 - 2.0 / 32.0) = 100.0 * 0.9375 = 93.75

    Parameters:
        guide_len (int): Total guide length.
        mismatch_count (int): Total base mismatches.
        seed_mismatches (int): Seed region mismatches.
        distal_mismatches (int): Distal region mismatches.

    Returns:
        float: Score ranging from 0.0 to 100.0 rounded to 2 decimal places.
    """
    if mismatch_count == 0:
        return 100.0

    seed_len = min(10, max(1, guide_len))
    distal_len = max(0, guide_len - seed_len)
    max_possible_penalty = max(1.0, (distal_len * 1.0) + (seed_len * 2.5))

    weighted_penalty = (distal_mismatches * 1.0) + (seed_mismatches * 2.5)
    score = 100.0 * max(0.0, 1.0 - (weighted_penalty / max_possible_penalty))
    return round(score, 2)

