"""
GeneWeaver: Sequence Validation Module
======================================
Provides thorough validation and defensive sanity checks for:
1. DNA Reference sequences.
2. CRISPR Guide sequences (protospacer targets).
3. PAM (Protospacer Adjacent Motif) patterns.

Features:
- Character-level analysis identifying exact positions of invalid characters.
- Comprehensive IUPAC nucleotide code handling for degenerate PAM motifs.
- Clear, beginner-friendly error messages that explain *why* an input failed.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from src.sequence_utils import clean_sequence, calculate_nucleotide_stats

# Canonical DNA bases
STANDARD_DNA_BASES: Set[str] = {"A", "C", "G", "T"}

# IUPAC Degenerate Nucleotide Code Definitions
# Used in biology when a position can tolerate multiple bases (e.g., 'N' = any base, 'R' = A or G)
IUPAC_NUCLEOTIDE_MAP: Dict[str, Set[str]] = {
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "T": {"T"},
    "U": {"T"},  # Support RNA transcript equivalents
    "R": {"A", "G"},        # Purine
    "Y": {"C", "T"},        # Pyrimidine
    "S": {"G", "C"},        # Strong interaction (3 H-bonds)
    "W": {"A", "T"},        # Weak interaction (2 H-bonds)
    "K": {"G", "T"},        # Keto
    "M": {"A", "C"},        # Amino
    "B": {"C", "G", "T"},   # Not A
    "D": {"A", "G", "T"},   # Not C
    "H": {"A", "C", "T"},   # Not G
    "V": {"A", "C", "G"},   # Not T
    "N": {"A", "C", "G", "T"},  # Any base
}


@dataclass
class ValidationResult:
    """
    Data container holding validation status and human-readable feedback.

    Attributes:
        is_valid (bool): True if sequence passed all checks.
        cleaned_sequence (str): Whitespace-stripped, uppercase sequence.
        errors (List[str]): Critical error messages preventing processing.
        warnings (List[str]): Informational advisories that don't block execution.
        invalid_characters (List[Dict[str, Any]]): List of invalid chars with index and character.
        stats (Dict[str, Any]): Nucleotide metrics (length, GC content, etc.).
    """
    is_valid: bool
    cleaned_sequence: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    invalid_characters: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


def validate_dna_sequence(
    raw_seq: str,
    allow_ambiguous: bool = False,
    min_length: int = 1,
    max_length: Optional[int] = None,
    sequence_label: str = "DNA sequence"
) -> ValidationResult:
    """
    Validate a DNA sequence against standard biological rules.

    Parameters:
        raw_seq (str): User-provided sequence string.
        allow_ambiguous (bool): If True, allows IUPAC degenerate codes (N, R, Y, etc.).
                                If False, strictly enforces canonical A, C, G, T.
        min_length (int): Minimum permissible sequence length.
        max_length (Optional[int]): Maximum permissible sequence length (or None).
        sequence_label (str): Human-friendly name for reporting in error messages.

    Returns:
        ValidationResult: Full validation analysis.
    """
    errors: List[str] = []
    warnings: List[str] = []
    invalid_chars: List[Dict[str, Any]] = []

    if raw_seq is None or not str(raw_seq).strip():
        errors.append(f"{sequence_label} is empty. Please enter or upload a valid sequence.")
        return ValidationResult(
            is_valid=False,
            cleaned_sequence="",
            errors=errors,
            warnings=warnings,
            invalid_characters=[],
            stats=calculate_nucleotide_stats("")
        )

    cleaned = clean_sequence(raw_seq)
    stats = calculate_nucleotide_stats(cleaned)
    seq_len = len(cleaned)

    # 1. Length checks
    if seq_len < min_length:
        errors.append(
            f"{sequence_label} length ({seq_len} bp) is shorter than the minimum required length of {min_length} bp."
        )

    if max_length is not None and seq_len > max_length:
        errors.append(
            f"{sequence_label} length ({seq_len} bp) exceeds maximum allowable limit of {max_length} bp."
        )

    # 2. Character validity check
    allowed_bases = set(IUPAC_NUCLEOTIDE_MAP.keys()) if allow_ambiguous else STANDARD_DNA_BASES

    for idx, char in enumerate(cleaned):
        if char not in allowed_bases:
            invalid_chars.append({"index": idx, "character": char})

    if invalid_chars:
        # Group unique invalid characters for clear error feedback
        unique_invalids = sorted(list({item["character"] for item in invalid_chars}))
        sample_positions = [str(item["index"] + 1) for item in invalid_chars[:5]]
        pos_str = ", ".join(sample_positions)
        if len(invalid_chars) > 5:
            pos_str += f" ... (+{len(invalid_chars) - 5} more)"

        if allow_ambiguous:
            errors.append(
                f"{sequence_label} contains invalid character(s): {', '.join(unique_invalids)}. "
                f"Positions (1-indexed): [{pos_str}]. Allowed: standard bases (A, C, G, T) or IUPAC codes."
            )
        else:
            errors.append(
                f"{sequence_label} contains non-standard base character(s): {', '.join(unique_invalids)}. "
                f"Positions (1-indexed): [{pos_str}]. Only canonical DNA bases (A, C, G, T) are permitted."
            )

    # 3. Informational warnings
    if stats["gc_content"] < 20.0:
        warnings.append(
            f"Low GC content detected ({stats['gc_content']}%). High AT regions may have lower binding stability."
        )
    elif stats["gc_content"] > 80.0:
        warnings.append(
            f"High GC content detected ({stats['gc_content']}%). High GC regions can form stable secondary structures."
        )

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        cleaned_sequence=cleaned,
        errors=errors,
        warnings=warnings,
        invalid_characters=invalid_chars,
        stats=stats
    )


def validate_guide_sequence(
    raw_guide: str,
    expected_length: int = 20
) -> ValidationResult:
    """
    Validate a CRISPR sgRNA / crRNA protospacer guide sequence.

    Biological Context:
    - Standard SpCas9 (Streptococcus pyogenes) guides typically have a 20 bp spacer length.
    - Some engineered Cas variants use shorter (17-19 bp tru-gRNAs) or longer guides.
    - Guides must only contain standard canonical bases (A, C, G, T).

    Parameters:
        raw_guide (str): Raw guide sequence input.
        expected_length (int): Target length (default: 20 bp).

    Returns:
        ValidationResult: Detailed validation outcome.
    """
    result = validate_dna_sequence(
        raw_seq=raw_guide,
        allow_ambiguous=False,
        min_length=10,
        max_length=40,
        sequence_label="CRISPR guide sequence"
    )

    if not result.is_valid:
        return result

    # Check if length matches the configured guide length
    guide_len = len(result.cleaned_sequence)
    if guide_len != expected_length:
        result.warnings.append(
            f"Guide length is {guide_len} bp, while configured guide length is {expected_length} bp. "
            f"The alignment engine will evaluate matching windows of length {guide_len} bp."
        )

    # Check for homopolymer tracts (e.g., poly-T causes premature termination in U6 pol III promoters)
    if "TTTT" in result.cleaned_sequence:
        result.warnings.append(
            "Poly-T tract (>= 4 consecutive Ts) detected. In mammalian expression vectors using "
            "U6 or H1 RNA Polymerase III promoters, >= 4 Ts acts as an intrinsic transcription terminator."
        )

    return result


def validate_pam_pattern(raw_pam: str) -> ValidationResult:
    """
    Validate a CRISPR PAM (Protospacer Adjacent Motif) motif pattern.

    Biological Context:
    - SpCas9 canonical PAM is 'NGG' (Any base followed by two Guanines).
    - Other enzymes use different motifs:
        - SaCas9: NNGRRT
        - Cas12a (Cpf1): TTTV (5' PAM)
        - xCas9 / SpCas9-NG: NGN

    Parameters:
        raw_pam (str): Raw PAM motif string (e.g. 'NGG').

    Returns:
        ValidationResult: Validation outcome.
    """
    if raw_pam is None or not str(raw_pam).strip():
        return ValidationResult(
            is_valid=False,
            cleaned_sequence="",
            errors=["PAM motif pattern cannot be empty. (Default for SpCas9 is 'NGG')."],
            warnings=[],
            invalid_characters=[],
            stats={}
        )

    cleaned = clean_sequence(raw_pam)
    errors: List[str] = []
    invalid_chars: List[Dict[str, Any]] = []

    # Check against IUPAC codes
    valid_iupac = set(IUPAC_NUCLEOTIDE_MAP.keys())
    for idx, ch in enumerate(cleaned):
        if ch not in valid_iupac:
            invalid_chars.append({"index": idx, "character": ch})

    if invalid_chars:
        bad_chars = sorted(list({item["character"] for item in invalid_chars}))
        errors.append(
            f"PAM motif contains invalid code(s): {', '.join(bad_chars)}. "
            "PAM motifs must use IUPAC nucleotide codes (e.g., A, C, G, T, N, R, Y, etc.)."
        )

    if len(cleaned) < 2 or len(cleaned) > 10:
        errors.append(
            f"PAM motif length ({len(cleaned)}) is unusual. Standard CRISPR PAM motifs are 2 to 6 bases."
        )

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        cleaned_sequence=cleaned,
        errors=errors,
        warnings=[],
        invalid_characters=invalid_chars,
        stats={"length": len(cleaned)}
    )
