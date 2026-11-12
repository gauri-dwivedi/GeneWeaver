"""
Tests for Module 1 (DNA Sequence Input) & Module 2 (Sequence Validation)
"""

import pytest
import io
from src.sequence_utils import (
    clean_sequence,
    reverse_complement,
    calculate_nucleotide_stats,
    transcribe_to_rna,
)
from src.validator import (
    validate_dna_sequence,
    validate_guide_sequence,
    validate_pam_pattern,
)
from src.input_handler import (
    parse_pasted_text,
    parse_fasta_content,
    load_sample_reference_fasta,
    load_sample_guides_csv,
    SequenceRecord,
)


# ============================================================================
# Sequence Utilities Tests
# ============================================================================

def test_clean_sequence():
    """Verify whitespace, newlines, and numbers are stripped, and text uppercased."""
    raw = "  atg c\n\r123 \t  "
    assert clean_sequence(raw) == "ATGC"
    assert clean_sequence("") == ""
    assert clean_sequence(None) == ""


def test_reverse_complement_canonical():
    """Verify 5' to 3' reverse complementation for canonical DNA bases."""
    # A <-> T, C <-> G reversed
    assert reverse_complement("ATGC") == "GCAT"
    assert reverse_complement("GAGTCC") == "GGACTC"
    assert reverse_complement("AAAA") == "TTTT"
    assert reverse_complement("") == ""


def test_reverse_complement_symmetry():
    """Reverse complement of a reverse complement must return the original sequence."""
    seq = "ATGCGATCGATCGATCG"
    assert reverse_complement(reverse_complement(seq)) == seq


def test_calculate_nucleotide_stats():
    """Verify composition counts and GC/AT percentages."""
    seq = "GGCCAA"  # G:2, C:2, A:2, T:0 -> Total 6
    stats = calculate_nucleotide_stats(seq)
    assert stats["length"] == 6
    assert stats["count_G"] == 2
    assert stats["count_C"] == 2
    assert stats["count_A"] == 2
    assert stats["count_T"] == 0
    assert stats["gc_content"] == round((4 / 6) * 100, 2)
    assert stats["at_content"] == round((2 / 6) * 100, 2)


def test_calculate_nucleotide_stats_empty():
    """Empty sequence should yield 0 for all metrics without raising ZeroDivisionError."""
    stats = calculate_nucleotide_stats("")
    assert stats["length"] == 0
    assert stats["gc_content"] == 0.0


def test_transcribe_to_rna():
    """Verify T -> U replacement."""
    assert transcribe_to_rna("ATGCAT") == "AUGCAU"


# ============================================================================
# Validation Tests
# ============================================================================

def test_validate_dna_sequence_valid():
    """Valid DNA sequence passes without errors."""
    result = validate_dna_sequence("ATGCGATCGATC")
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.cleaned_sequence == "ATGCGATCGATC"


def test_validate_dna_sequence_case_and_whitespace():
    """Lowercase and whitespace should be cleaned automatically and pass."""
    result = validate_dna_sequence("  atgc gatc \n ")
    assert result.is_valid is True
    assert result.cleaned_sequence == "ATGCGATC"


def test_validate_dna_sequence_empty():
    """Empty sequence must fail validation with helpful message."""
    result = validate_dna_sequence("")
    assert result.is_valid is False
    assert any("empty" in err.lower() for err in result.errors)


def test_validate_dna_sequence_invalid_characters():
    """Non-DNA characters (e.g. X, Z, 9) must trigger validation failure."""
    result = validate_dna_sequence("ATGCXZGATC")
    assert result.is_valid is False
    assert len(result.invalid_characters) == 2
    assert any("non-standard base" in err.lower() for err in result.errors)


def test_validate_dna_sequence_min_length():
    """Sequences below min_length must fail."""
    result = validate_dna_sequence("ATG", min_length=10)
    assert result.is_valid is False
    assert any("shorter than" in err.lower() for err in result.errors)


def test_validate_guide_sequence_valid():
    """Standard 20 bp guide passes cleanly."""
    result = validate_guide_sequence("GAGTCCGAGCAGAAGAAGAA", expected_length=20)
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validate_guide_sequence_poly_t_warning():
    """Poly-T tract (>= 4 Ts) triggers an advisory warning."""
    result = validate_guide_sequence("GAGTCCGTTTTAGAAGAAGAA")
    assert result.is_valid is True
    assert any("poly-t" in warn.lower() for warn in result.warnings)


def test_validate_pam_pattern():
    """Validate canonical SpCas9 'NGG' and invalid PAM motifs."""
    res_ngg = validate_pam_pattern("NGG")
    assert res_ngg.is_valid is True
    assert res_ngg.cleaned_sequence == "NGG"

    res_invalid = validate_pam_pattern("N12")
    assert res_invalid.is_valid is False

    res_empty = validate_pam_pattern("")
    assert res_empty.is_valid is False


# ============================================================================
# Input Handler & Dataset Loading Tests
# ============================================================================

def test_parse_pasted_text_plain():
    """Plain sequence string is parsed as a single SequenceRecord."""
    records = parse_pasted_text("ATGCGATCGATC", default_id="my_seq")
    assert len(records) == 1
    assert records[0].id == "my_seq"
    assert records[0].sequence == "ATGCGATCGATC"
    assert records[0].length == 12


def test_parse_pasted_text_fasta():
    """Pasted FASTA formatted text is parsed correctly."""
    raw = ">seq_alpha\nATGC\n>seq_beta\nCGTA\n"
    records = parse_pasted_text(raw)
    assert len(records) == 2
    assert records[0].id == "seq_alpha"
    assert records[0].sequence == "ATGC"
    assert records[1].id == "seq_beta"
    assert records[1].sequence == "CGTA"


def test_load_sample_reference_fasta():
    """Verify built-in sample reference FASTA loads cleanly."""
    records = load_sample_reference_fasta()
    assert len(records) >= 3
    assert any("emx1" in r.id.lower() for r in records)


def test_load_sample_guides_csv():
    """Verify built-in sample guides CSV loads correctly."""
    df = load_sample_guides_csv()
    assert len(df) >= 5
    assert "guide_id" in df.columns
    assert "sequence" in df.columns
    assert "pam_motif" in df.columns

