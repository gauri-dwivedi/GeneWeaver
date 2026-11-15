"""
GeneWeaver: Input Handler Module
================================
Responsible for handling user inputs, parsing FASTA files, and loading sample datasets.

Features:
- Handles raw pasted text and pasted multi-FASTA formats.
- Parses FASTA files using Biopython's SeqIO parser.
- Validates sequence records upon construction.
- Validates guide CSV structure, required columns, and sequence content.
- Provides built-in loader functions for pre-packaged sample reference genomes and guide libraries.
"""

import os
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Union, Dict, Any, Optional
import pandas as pd
from Bio import SeqIO
from src.sequence_utils import clean_sequence, calculate_nucleotide_stats
from src.validator import (
    validate_dna_sequence,
    validate_guide_sequence,
    validate_pam_pattern,
)

# Standard required columns for CRISPR guide libraries
REQUIRED_GUIDE_COLUMNS: List[str] = [
    "guide_id",
    "sequence",
    "target_locus",
    "expected_strand",
    "expected_mismatches",
    "pam_motif",
]


@dataclass
class SequenceRecord:
    """
    Structured representation of an individual biological sequence.

    Attributes:
        id (str): Short sequence identifier (e.g. 'chr1', 'emx1_locus').
        description (str): Full descriptive metadata or FASTA header text.
        sequence (str): Cleaned, uppercase DNA nucleotide string.
        length (int): Total nucleotide base pair count.
        stats (Dict[str, Any]): Nucleotide metrics (A/C/G/T counts, GC content, etc.).
        is_valid (bool): Whether the sequence contains exclusively valid canonical DNA.
        validation_errors (List[str]): Error messages if invalid characters or length violations exist.
        validation_warnings (List[str]): Non-blocking biological advisories (e.g. extreme GC content).
    """
    id: str
    description: str
    sequence: str
    length: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.sequence:
            self.sequence = ""
        # Normalization: strip whitespace and uppercase without dropping digits/symbols
        self.sequence = clean_sequence(self.sequence)
        self.length = len(self.sequence)
        if not self.stats:
            self.stats = calculate_nucleotide_stats(self.sequence)

        # Automatic validation upon record instantiation
        if self.length == 0:
            self.is_valid = False
            self.validation_errors = [f"Record '{self.id}' contains an empty sequence."]
        else:
            val_res = validate_dna_sequence(
                self.sequence,
                allow_ambiguous=False,
                sequence_label=f"Record '{self.id}'"
            )
            self.is_valid = val_res.is_valid
            self.validation_errors = val_res.errors
            self.validation_warnings = val_res.warnings


def parse_pasted_text(
    raw_text: str,
    default_id: str = "custom_pasted_seq",
    strict: bool = False,
) -> List[SequenceRecord]:
    """
    Parse a user-pasted text string which may be either:
    1. A single plain DNA sequence string (e.g. 'ATGCGATCG...').
    2. One or more FASTA-formatted records starting with '>header'.

    Parameters:
        raw_text (str): Raw string from text area or clipboard.
        default_id (str): Default ID assigned if not in FASTA format.
        strict (bool): If True, raises ValueError if the input is empty or invalid.

    Returns:
        List[SequenceRecord]: One or more parsed SequenceRecords.

    Raises:
        ValueError: If strict=True and input is empty or fails validation.
    """
    if raw_text is None or not str(raw_text).strip():
        if strict:
            raise ValueError("Pasted sequence input is empty. Please provide a valid DNA sequence.")
        return []

    stripped = str(raw_text).strip()

    # Check if this is FASTA formatted (starts with '>')
    if stripped.startswith(">"):
        return parse_fasta_content(io.StringIO(stripped), strict=strict)

    # Plain sequence string
    cleaned = clean_sequence(stripped)
    record = SequenceRecord(
        id=default_id,
        description="Directly pasted DNA sequence",
        sequence=cleaned,
    )

    if strict and not record.is_valid:
        raise ValueError(
            f"Validation failed for pasted sequence: {'; '.join(record.validation_errors)}"
        )

    return [record]


def parse_fasta_content(
    source: Union[str, io.StringIO, io.BytesIO, Path],
    strict: bool = False,
) -> List[SequenceRecord]:
    """
    Parse FASTA formatted sequences from a file path, StringIO, or BytesIO buffer.
    Utilizes Biopython SeqIO for biological standards compliance and validates
    all extracted sequence records.

    Parameters:
        source: File path string, Path object, or file-like buffer.
        strict (bool): If True, raises ValueError if any record contains invalid characters or is empty.

    Returns:
        List[SequenceRecord]: List of parsed SequenceRecord objects.

    Raises:
        FileNotFoundError: If a file path is provided that does not exist.
        ValueError: If file content is corrupt, empty, or fails strict validation.
    """
    records: List[SequenceRecord] = []

    # Handle BytesIO if uploaded via Streamlit st.file_uploader
    if isinstance(source, io.BytesIO):
        raw_bytes = source.getvalue()
        if not raw_bytes or not raw_bytes.strip():
            raise ValueError("Uploaded FASTA file is empty (0 bytes).")
        text_stream = io.StringIO(raw_bytes.decode("utf-8", errors="replace"))
    elif isinstance(source, io.StringIO):
        text_stream = source
    elif isinstance(source, (str, Path)):
        if not os.path.exists(source):
            raise FileNotFoundError(f"FASTA file not found at: {source}")
        if os.path.getsize(source) == 0:
            raise ValueError(f"FASTA file at '{source}' is empty (0 bytes).")
        text_stream = open(source, "r", encoding="utf-8", errors="replace")
    else:
        text_stream = source

    try:
        try:
            for bio_rec in SeqIO.parse(text_stream, "fasta"):
                clean_seq = clean_sequence(str(bio_rec.seq))
                record = SequenceRecord(
                    id=str(bio_rec.id),
                    description=str(bio_rec.description),
                    sequence=clean_seq,
                )
                if strict and not record.is_valid:
                    raise ValueError(
                        f"Validation failed for FASTA record '{record.id}': {'; '.join(record.validation_errors)}"
                    )
                records.append(record)
        except Exception as parse_err:
            if isinstance(parse_err, ValueError) and "Validation failed" in str(parse_err):
                raise
            raise ValueError(
                f"No valid FASTA records were found. Ensure each record begins with a '>' header. (Details: {parse_err})"
            ) from parse_err
    finally:
        if isinstance(source, (str, Path)) and hasattr(text_stream, "close"):
            text_stream.close()

    if not records:
        raise ValueError("No valid FASTA records were found. Ensure each record begins with a '>' header.")

    return records


def validate_guides_dataframe(
    df: pd.DataFrame,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Validate a CRISPR guide DataFrame structure, required columns, and sequence content.

    Parameters:
        df (pd.DataFrame): DataFrame loaded from CSV.
        strict (bool): If True, raises ValueError if columns are missing or if any row contains invalid sequences.

    Returns:
        pd.DataFrame: Cleaned DataFrame with 'is_valid' and 'validation_notes' columns.

    Raises:
        ValueError: If CSV is empty, missing required columns, or contains invalid data in strict mode.
    """
    if df is None or df.empty:
        raise ValueError("Guide CSV is empty. A valid guide table must contain at least one row.")

    # 1. Check required columns
    missing_cols = [col for col in REQUIRED_GUIDE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Malformed guide CSV: Missing required column(s): {', '.join(missing_cols)}. "
            f"Required columns are: {', '.join(REQUIRED_GUIDE_COLUMNS)}"
        )

    validated_df = df.copy()
    valid_flags: List[bool] = []
    error_notes: List[str] = []

    for idx, row in validated_df.iterrows():
        guide_id = str(row.get("guide_id", f"row_{idx}"))
        raw_guide = "" if pd.isna(row.get("sequence")) else str(row.get("sequence"))
        raw_pam = "" if pd.isna(row.get("pam_motif")) else str(row.get("pam_motif"))

        row_errors: List[str] = []

        # Validate expected strand
        expected_strand = row.get("expected_strand")

        if pd.isna(expected_strand) or str(expected_strand).strip().lower() not in {"forward", "reverse", "none"}:
            row_errors.append(
                f"Invalid expected_strand: {expected_strand}. "
                "Allowed values: forward, reverse, none."
            )

        # Validate expected mismatches
        expected_mismatches = row.get("expected_mismatches")

        if pd.isna(expected_mismatches):
            row_errors.append("expected_mismatches cannot be empty.")
        else:
            try:
                mismatch_value = float(expected_mismatches)

                if not mismatch_value.is_integer() or mismatch_value < 0:
                    row_errors.append(
                        "expected_mismatches must be a non-negative integer."
                    )

            except (ValueError, TypeError):
                row_errors.append(
                    "expected_mismatches must be a non-negative integer."
                )
        # Validate sequence
        g_val = validate_guide_sequence(raw_guide)
        if not g_val.is_valid:
            row_errors.extend(g_val.errors)

        # Validate PAM
        p_val = validate_pam_pattern(raw_pam)
        if not p_val.is_valid:
            row_errors.extend(p_val.errors)

        is_row_valid = len(row_errors) == 0
        valid_flags.append(is_row_valid)
        error_notes.append("; ".join(row_errors) if row_errors else "Valid")

        if strict and not is_row_valid:
            raise ValueError(
                f"Validation failed for guide '{guide_id}' at row {idx + 1}: {'; '.join(row_errors)}"
            )

    validated_df["is_valid"] = valid_flags
    validated_df["validation_notes"] = error_notes
    return validated_df


def get_data_dir() -> Path:
    """Return the absolute path to the data directory."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    return project_root / "data"


def list_sample_datasets() -> Dict[str, Path]:
    """
    Enumerate all sample FASTA and CSV datasets available in the data directory.

    Returns:
        Dict[str, Path]: Mapping of dataset label to absolute file path.
    """
    data_dir = get_data_dir()
    datasets: Dict[str, Path] = {}
    if data_dir.exists():
        for file in data_dir.iterdir():
            if file.suffix.lower() in [".fasta", ".fa", ".fna", ".csv"]:
                datasets[file.name] = file
    return datasets


def load_sample_reference_fasta(
    filename: str = "sample_reference.fasta",
    strict: bool = False,
) -> List[SequenceRecord]:
    """
    Load a pre-packaged sample reference FASTA dataset.

    Parameters:
        filename (str): Name of the file inside data/ directory.
        strict (bool): If True, validates records strictly.

    Returns:
        List[SequenceRecord]: Parsed and validated sequence records.
    """
    target_path = get_data_dir() / filename
    if not target_path.exists():
        raise FileNotFoundError(f"Sample reference FASTA not found: {target_path}")
    return parse_fasta_content(target_path, strict=strict)


def load_sample_guides_csv(
    filename: str = "sample_guides.csv",
    strict: bool = False,
) -> pd.DataFrame:
    """
    Load and validate a CRISPR guide sequence CSV dataset.

    Parameters:
        filename (str): Name of the CSV file inside data/ directory, or custom file path.
        strict (bool): If True, raises ValueError on any malformed or invalid row.

    Returns:
        pd.DataFrame: Loaded and validated guide library table.
    """
    if os.path.isabs(filename) or os.path.exists(filename):
        target_path = Path(filename)
    else:
        target_path = get_data_dir() / filename

    if not target_path.exists():
        raise FileNotFoundError(f"Sample guides CSV not found: {target_path}")

    if target_path.stat().st_size == 0:
        raise ValueError(f"Guide CSV file '{target_path.name}' is empty (0 bytes).")

    df = pd.read_csv(target_path)
    return validate_guides_dataframe(df, strict=strict)
