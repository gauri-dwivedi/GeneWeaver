"""
GeneWeaver: Input Handler Module
================================
Responsible for handling user inputs, parsing FASTA files, and loading sample datasets.

Features:
- Handles raw pasted text and pasted multi-FASTA formats.
- Parses FASTA files using Biopython's SeqIO parser.
- Auto-detects single sequence vs multi-FASTA files.
- Provides built-in loader functions for pre-packaged sample reference genomes and guide libraries.
"""

import os
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Union, Dict, Any
import pandas as pd
from Bio import SeqIO
from src.sequence_utils import clean_sequence, calculate_nucleotide_stats
from src.validator import validate_dna_sequence


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
    """
    id: str
    description: str
    sequence: str
    length: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.sequence:
            self.sequence = ""
        self.cleaned_sequence = clean_sequence(self.sequence)
        self.sequence = self.cleaned_sequence
        self.length = len(self.sequence)
        if not self.stats:
            self.stats = calculate_nucleotide_stats(self.sequence)


def parse_pasted_text(raw_text: str, default_id: str = "custom_pasted_seq") -> List[SequenceRecord]:
    """
    Parse a user-pasted text string which may be either:
    1. A single plain DNA sequence string (e.g. 'ATGCGATCG...').
    2. One or more FASTA-formatted records starting with '>header'.

    Parameters:
        raw_text (str): Raw string from text area or clipboard.
        default_id (str): Default ID assigned if not in FASTA format.

    Returns:
        List[SequenceRecord]: One or more parsed SequenceRecords.
    """
    if not raw_text or not raw_text.strip():
        return []

    stripped = raw_text.strip()

    # Check if this is FASTA formatted (starts with '>')
    if stripped.startswith(">"):
        return parse_fasta_content(io.StringIO(stripped))

    # Plain sequence string
    cleaned = clean_sequence(stripped)
    record = SequenceRecord(
        id=default_id,
        description="Directly pasted DNA sequence",
        sequence=cleaned,
    )
    return [record]


def parse_fasta_content(source: Union[str, io.StringIO, io.BytesIO, Path]) -> List[SequenceRecord]:
    """
    Parse FASTA formatted sequences from a file path, StringIO, or BytesIO buffer.
    Utilizes Biopython SeqIO for robust biological standards compliance.

    Parameters:
        source: File path string, Path object, or file-like buffer.

    Returns:
        List[SequenceRecord]: List of parsed SequenceRecord objects.

    Raises:
        ValueError: If file content is corrupt or cannot be parsed.
    """
    records: List[SequenceRecord] = []

    # Handle BytesIO if uploaded via Streamlit st.file_uploader
    if isinstance(source, io.BytesIO):
        text_stream = io.StringIO(source.getvalue().decode("utf-8", errors="replace"))
    elif isinstance(source, io.StringIO):
        text_stream = source
    elif isinstance(source, (str, Path)):
        if not os.path.exists(source):
            raise FileNotFoundError(f"FASTA file not found at: {source}")
        text_stream = open(source, "r", encoding="utf-8", errors="replace")
    else:
        text_stream = source

    try:
        for bio_rec in SeqIO.parse(text_stream, "fasta"):
            clean_seq = clean_sequence(str(bio_rec.seq))
            record = SequenceRecord(
                id=str(bio_rec.id),
                description=str(bio_rec.description),
                sequence=clean_seq,
            )
            records.append(record)
    finally:
        if isinstance(source, (str, Path)) and hasattr(text_stream, "close"):
            text_stream.close()

    if not records:
        raise ValueError("No valid FASTA records were found. Ensure each record begins with a '>' header.")

    return records


def get_data_dir() -> Path:
    """Return the absolute path to the data directory."""
    # GeneWeaver/data
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


def load_sample_reference_fasta(filename: str = "sample_reference.fasta") -> List[SequenceRecord]:
    """
    Load a pre-packaged sample reference FASTA dataset.

    Parameters:
        filename (str): Name of the file inside data/ directory.

    Returns:
        List[SequenceRecord]: Parsed sequence records.
    """
    target_path = get_data_dir() / filename
    if not target_path.exists():
        raise FileNotFoundError(f"Sample reference FASTA not found: {target_path}")
    return parse_fasta_content(target_path)


def load_sample_guides_csv(filename: str = "sample_guides.csv") -> pd.DataFrame:
    """
    Load pre-packaged sample guide sequences CSV.

    Parameters:
        filename (str): Name of the CSV file inside data/ directory.

    Returns:
        pd.DataFrame: Loaded guide library table.
    """
    target_path = get_data_dir() / filename
    if not target_path.exists():
        raise FileNotFoundError(f"Sample guides CSV not found: {target_path}")
    df = pd.read_csv(target_path)
    # Clean guide sequence column
    if "sequence" in df.columns:
        df["sequence"] = df["sequence"].astype(str).apply(clean_sequence)
    return df
