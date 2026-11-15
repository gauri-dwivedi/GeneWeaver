"""
Tests for Module 1 (DNA Sequence Input) & Module 2 (Sequence Validation)
========================================================================
Validates all Phase 2 and Phase 3 requirements:
- Harmless formatting cleanup (whitespace, newlines, uppercase)
- Preservation and detection of digits and punctuation symbols
- Validation of pasted DNA sequences and FASTA file uploads
- Detection of empty records and malformed FASTA files
- Validation of guide CSV structure, required columns, and sequence content
- Preservation of synthetic vs. experimental dataset distinctions
"""

import pytest
import io
import pandas as pd
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
    validate_guides_dataframe,
    SequenceRecord,
    REQUIRED_GUIDE_COLUMNS,
)


# ============================================================================
# Sequence Utilities & Formatting Cleanup Tests
# ============================================================================

def test_clean_sequence_whitespace_and_casing():
    """Verify whitespace, newlines, and tabs are removed, and lowercase converted to uppercase."""
    raw = "  atg c \t \n\r "
    assert clean_sequence(raw) == "ATGC"
    assert clean_sequence("") == ""
    assert clean_sequence(None) == ""


def test_clean_sequence_preserves_digits_and_symbols_for_validation():
    """
    CRITICAL CHECK:
    Formatting cleanup must NOT silently drop numeric characters or symbols.
    Non-base characters must be preserved so the validator can detect and report them.
    """
    assert clean_sequence("atgc123") == "ATGC123"
    assert clean_sequence("atgc-gatc") == "ATGC-GATC"
    assert clean_sequence("atgc*gatc") == "ATGC*GATC"


def test_reverse_complement_canonical():
    """Verify 5' to 3' reverse complementation for canonical DNA bases."""
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
# DNA Sequence Validation Tests (Module 2)
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
    """Empty sequence must fail validation with a helpful error message."""
    result = validate_dna_sequence("")
    assert result.is_valid is False
    assert any("empty" in err.lower() for err in result.errors)


def test_validate_dna_sequence_digits_detected_and_reported():
    """
    CRITICAL CHECK:
    Digits must NOT be silently stripped; they must trigger validation failure,
    identifying exact character identities and 1-based coordinates.
    """
    result = validate_dna_sequence("ATGC123GATC")
    assert result.is_valid is False
    assert len(result.invalid_characters) == 3
    # Check that characters 1, 2, 3 were recorded
    bad_chars = {item["character"] for item in result.invalid_characters}
    assert bad_chars == {"1", "2", "3"}
    # Positions 5, 6, 7 in 1-based biological indexing
    bad_indices = [item["index"] + 1 for item in result.invalid_characters]
    assert bad_indices == [5, 6, 7]
    assert any("numeric digits" in err.lower() for err in result.errors)


def test_validate_dna_sequence_punctuation_and_symbols():
    """
    CRITICAL CHECK:
    Punctuation (dashes, asterisks, periods) must trigger validation failure.
    """
    result = validate_dna_sequence("ATGC-GAT*C.TAG")
    assert result.is_valid is False
    bad_chars = {item["character"] for item in result.invalid_characters}
    assert {"-", "*", "."}.issubset(bad_chars)
    assert any("punctuation" in err.lower() for err in result.errors)


def test_validate_dna_sequence_min_length():
    """Sequences below min_length must fail."""
    result = validate_dna_sequence("ATG", min_length=10)
    assert result.is_valid is False
    assert any("shorter than" in err.lower() for err in result.errors)


def test_validate_dna_sequence_max_length():
    """Sequence exceeding max_length constraint must fail validation."""
    result = validate_dna_sequence("ATGCATGC", max_length=5)
    assert result.is_valid is False
    assert any("exceeds maximum" in err.lower() for err in result.errors)


def test_validate_dna_sequence_ambiguous_toggle():
    """Verify allow_ambiguous flag handles IUPAC degenerate codes."""
    # Strict mode (default) rejects 'N'
    strict_res = validate_dna_sequence("ATGCNGATC", allow_ambiguous=False)
    assert strict_res.is_valid is False

    # Ambiguous mode accepts standard IUPAC codes like 'N', 'R', 'Y'
    ambig_res = validate_dna_sequence("ATGCNRYGATC", allow_ambiguous=True)
    assert ambig_res.is_valid is True


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
# Input Handler & FASTA Upload Tests (Module 1)
# ============================================================================

def test_sequence_record_automatic_validation():
    """SequenceRecord auto-validates upon instantiation."""
    valid_rec = SequenceRecord(id="seq1", description="desc", sequence="ATGCGATC")
    assert valid_rec.is_valid is True
    assert len(valid_rec.validation_errors) == 0

    bad_rec = SequenceRecord(id="seq2", description="desc", sequence="ATGC123")
    assert bad_rec.is_valid is False
    assert len(bad_rec.validation_errors) > 0


def test_parse_pasted_text_plain():
    """Plain sequence string is parsed and validated as a single SequenceRecord."""
    records = parse_pasted_text("ATGCGATCGATC", default_id="my_seq")
    assert len(records) == 1
    assert records[0].id == "my_seq"
    assert records[0].sequence == "ATGCGATCGATC"
    assert records[0].length == 12
    assert records[0].is_valid is True


def test_parse_pasted_text_fasta():
    """Pasted FASTA formatted text is parsed and validated correctly."""
    raw = ">seq_alpha\nATGC\n>seq_beta\nCGTA\n"
    records = parse_pasted_text(raw)
    assert len(records) == 2
    assert records[0].id == "seq_alpha"
    assert records[0].sequence == "ATGC"
    assert records[0].is_valid is True
    assert records[1].id == "seq_beta"
    assert records[1].sequence == "CGTA"
    assert records[1].is_valid is True


def test_parse_pasted_text_strict_invalid_raises():
    """Strict mode raises ValueError on invalid pasted sequence."""
    with pytest.raises(ValueError, match="Validation failed for pasted sequence"):
        parse_pasted_text("ATGC123GATC", strict=True)

    with pytest.raises(ValueError, match="empty"):
        parse_pasted_text("", strict=True)


def test_parse_fasta_bytes_io_upload():
    """Simulate file upload buffer via io.BytesIO (Streamlit file_uploader)."""
    upload_bytes = io.BytesIO(b">uploaded_gene_target [organism=synthetic]\nATGCATGCATGC\n")
    records = parse_fasta_content(upload_bytes)
    assert len(records) == 1
    assert records[0].id == "uploaded_gene_target"
    assert records[0].sequence == "ATGCATGCATGC"
    assert records[0].length == 12
    assert records[0].is_valid is True


def test_parse_fasta_empty_record_handled():
    """A FASTA record with an empty sequence must be flagged as invalid."""
    empty_rec_fasta = io.StringIO(">empty_seq\n\n>valid_seq\nATGCATGC\n")
    records = parse_fasta_content(empty_rec_fasta, strict=False)
    assert len(records) == 2
    assert records[0].id == "empty_seq"
    assert records[0].is_valid is False
    assert any("empty" in err for err in records[0].validation_errors)
    assert records[1].id == "valid_seq"
    assert records[1].is_valid is True


def test_parse_fasta_empty_record_strict_raises():
    """Strict parsing raises ValueError if any record in FASTA has an empty sequence."""
    empty_rec_fasta = io.StringIO(">empty_seq\n\n>valid_seq\nATGCATGC\n")
    with pytest.raises(ValueError, match="Validation failed for FASTA record 'empty_seq'"):
        parse_fasta_content(empty_rec_fasta, strict=True)


def test_parse_fasta_corrupted_raises_value_error():
    """FASTA content without standard '>' headers must raise ValueError."""
    corrupted_data = io.StringIO("This is not a fasta header\nATGCATGC\n")
    with pytest.raises(ValueError, match="No valid FASTA records"):
        parse_fasta_content(corrupted_data)


def test_parse_fasta_empty_upload_raises():
    """Uploading an empty file (0 bytes) must raise a clear ValueError."""
    empty_upload = io.BytesIO(b"")
    with pytest.raises(ValueError, match="empty"):
        parse_fasta_content(empty_upload)


# ============================================================================
# Guide CSV Structure and Validation Tests
# ============================================================================

def test_load_sample_reference_fasta():
    """Verify built-in sample reference FASTA loads cleanly and records are valid."""
    records = load_sample_reference_fasta(strict=True)
    assert len(records) >= 3
    assert any("emx1" in r.id.lower() for r in records)
    for r in records:
        assert r.is_valid is True
        assert r.length > 0


def test_load_sample_guides_csv():
    """Verify built-in sample guides CSV loads and required columns are present."""
    df = load_sample_guides_csv()
    assert len(df) >= 5
    for col in REQUIRED_GUIDE_COLUMNS:
        assert col in df.columns
    assert "is_valid" in df.columns


def test_validate_guides_dataframe_missing_columns():
    """Guide CSV missing required columns must raise ValueError."""
    bad_df = pd.DataFrame([{"guide_id": "G01", "sequence": "ATGC"}])  # Missing other columns
    with pytest.raises(ValueError, match="Missing required column"):
        validate_guides_dataframe(bad_df)


def test_validate_guides_dataframe_empty_raises():
    """Empty guide DataFrame must raise ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Guide CSV is empty"):
        validate_guides_dataframe(empty_df)


def test_validate_guides_dataframe_invalid_sequence_in_strict_mode():
    """Invalid guide sequence (e.g. containing digits) in strict mode raises ValueError."""
    bad_data_df = pd.DataFrame([{
        "guide_id": "G_BAD",
        "sequence": "GAGTCC123AAGAAGAAGAA",  # Contains digits
        "target_locus": "locus_1",
        "expected_strand": "forward",
        "expected_mismatches": 0,
        "pam_motif": "NGG",
    }])

    # Strict mode raises error
    with pytest.raises(ValueError, match="Validation failed for guide 'G_BAD'"):
        validate_guides_dataframe(bad_data_df, strict=True)

    # Non-strict mode flags it gracefully
    result_df = validate_guides_dataframe(bad_data_df, strict=False)
    assert bool(result_df.iloc[0]["is_valid"]) is False
    assert "non-standard base" in result_df.iloc[0]["validation_notes"]
def test_validate_guides_dataframe_missing_sequence_and_pam():
    """Missing sequence and PAM values should be flagged as invalid."""
    df = pd.DataFrame([{
        "guide_id": "G_MISSING",
        "sequence": None,
        "target_locus": "locus_1",
        "expected_strand": "forward",
        "expected_mismatches": 0,
        "pam_motif": None,
    }])

    result = validate_guides_dataframe(df)

    assert bool(result.iloc[0]["is_valid"]) is False


def test_validate_guides_dataframe_invalid_strand():
    """Invalid expected strand should be rejected."""
    df = pd.DataFrame([{
        "guide_id": "G_STRAND",
        "sequence": "GAGTCCGAGCAGAAGAAGAA",
        "target_locus": "locus_1",
        "expected_strand": "sideways",
        "expected_mismatches": 0,
        "pam_motif": "NGG",
    }])

    result = validate_guides_dataframe(df)

    assert bool(result.iloc[0]["is_valid"]) is False
    assert "Invalid expected_strand" in result.iloc[0]["validation_notes"]


def test_validate_guides_dataframe_invalid_mismatches():
    """Negative and decimal mismatch values should be rejected."""
    for value in [-1, 1.5, "abc"]:
        df = pd.DataFrame([{
            "guide_id": "G_MISMATCH",
            "sequence": "GAGTCCGAGCAGAAGAAGAA",
            "target_locus": "locus_1",
            "expected_strand": "forward",
            "expected_mismatches": value,
            "pam_motif": "NGG",
        }])

        result = validate_guides_dataframe(df)

        assert bool(result.iloc[0]["is_valid"]) is False
        assert "expected_mismatches" in result.iloc[0]["validation_notes"]