"""
GeneWeaver: Sequence Utilities Module
=====================================
Contains core biological sequence operations:
1. Cleaning and normalisation (stripping whitespace, newlines, numbers, uppercase conversion).
2. Reverse complementation (crucial for scanning the antisense / negative DNA strand).
3. Nucleotide composition statistics (A, C, G, T counts, GC content %, AT content %).
4. RNA transcription (T -> U).

Educational Note for Students:
- In double-stranded DNA, Cas9 can bind to either the forward (5'->3') strand or the
  reverse complement (antisense) strand.
- Complementary base pairing rules:
    Adenine (A) pairs with Thymine (T)
    Cytosine (C) pairs with Guanine (G)
"""

from typing import Dict, Any


def clean_sequence(seq: str) -> str:
    """
    Remove all whitespace, newlines, carriage returns, and numeric indices
    from raw sequence input and convert to uppercase.

    Parameters:
        seq (str): Raw DNA sequence string.

    Returns:
        str: Cleaned, uppercase sequence string.

    Example:
        >>> clean_sequence("  atg c\\n123  ")
        'ATGC'
    """
    if not seq:
        return ""
    # Filter out whitespace, digits (common in GenBank downloads), and formatting characters
    cleaned = "".join(ch for ch in seq if not ch.isspace() and not ch.isdigit())
    return cleaned.upper()


# Standard DNA base complement lookup table
DNA_COMPLEMENT_TABLE = str.maketrans({
    "A": "T", "a": "T",
    "T": "A", "t": "A",
    "C": "G", "c": "G",
    "G": "C", "g": "C",
    "N": "N", "n": "N",
    "R": "Y", "Y": "R",
    "S": "S", "W": "W",
    "K": "M", "M": "K",
    "B": "V", "V": "B",
    "D": "H", "H": "D",
})


def reverse_complement(seq: str) -> str:
    """
    Calculate the reverse complement of a DNA sequence in 5' to 3' orientation.

    Bioinformatics Explanation:
    - To search the reverse (antisense) strand of DNA in the same 5'->3' direction
      as the forward strand, we must BOTH complement each base (A<->T, C<->G)
      AND reverse the order of bases.

    Parameters:
        seq (str): DNA sequence string.

    Returns:
        str: Reverse complement sequence (5' to 3').

    Example:
        >>> reverse_complement("ATGCG")
        'CGCAT'
    """
    cleaned = clean_sequence(seq)
    if not cleaned:
        return ""
    complemented = cleaned.translate(DNA_COMPLEMENT_TABLE)
    return complemented[::-1]


def calculate_nucleotide_stats(seq: str) -> Dict[str, Any]:
    """
    Compute detailed nucleotide composition and statistical metrics for a sequence.

    Calculates:
    - Total length
    - Counts and percentages of A, C, G, T
    - Ambiguous / other base counts
    - GC content percentage: (G + C) / Total * 100
    - AT content percentage: (A + T) / Total * 100
    - Purine (A + G) vs Pyrimidine (C + T) counts and ratio

    Parameters:
        seq (str): Raw or cleaned DNA sequence.

    Returns:
        dict: Dictionary containing all computed statistics.
    """
    cleaned = clean_sequence(seq)
    total_len = len(cleaned)

    if total_len == 0:
        return {
            "length": 0,
            "count_A": 0, "count_C": 0, "count_G": 0, "count_T": 0, "count_other": 0,
            "pct_A": 0.0, "pct_C": 0.0, "pct_G": 0.0, "pct_T": 0.0,
            "gc_content": 0.0,
            "at_content": 0.0,
            "purines": 0,
            "pyrimidines": 0,
            "purine_pyrimidine_ratio": 0.0,
        }

    count_a = cleaned.count("A")
    count_c = cleaned.count("C")
    count_g = cleaned.count("G")
    count_t = cleaned.count("T")
    count_other = total_len - (count_a + count_c + count_g + count_t)

    gc_count = count_g + count_c
    at_count = count_a + count_t

    purines = count_a + count_g
    pyrimidines = count_c + count_t

    purine_pyr_ratio = (purines / pyrimidines) if pyrimidines > 0 else float("inf")

    return {
        "length": total_len,
        "count_A": count_a,
        "count_C": count_c,
        "count_G": count_g,
        "count_T": count_t,
        "count_other": count_other,
        "pct_A": round((count_a / total_len) * 100.0, 2),
        "pct_C": round((count_c / total_len) * 100.0, 2),
        "pct_G": round((count_g / total_len) * 100.0, 2),
        "pct_T": round((count_t / total_len) * 100.0, 2),
        "gc_content": round((gc_count / total_len) * 100.0, 2),
        "at_content": round((at_count / total_len) * 100.0, 2),
        "purines": purines,
        "pyrimidines": pyrimidines,
        "purine_pyrimidine_ratio": round(purine_pyr_ratio, 2) if purine_pyr_ratio != float("inf") else "N/A",
    }


def transcribe_to_rna(seq: str) -> str:
    """
    Transcribe a DNA coding sequence to RNA by substituting Thymine (T) with Uracil (U).

    Parameters:
        seq (str): DNA sequence.

    Returns:
        str: RNA sequence.
    """
    return clean_sequence(seq).replace("T", "U")
