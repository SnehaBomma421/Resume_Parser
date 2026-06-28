"""
Resume file parser module.

Handles extraction of text from PDF and DOCX file formats.
Provides a unified interface for file parsing with robust error handling.
"""

import os
from typing import Optional, Tuple

import pdfplumber
from docx import Document


# Maximum file size in bytes (10 MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Supported file formats
SUPPORTED_FORMATS = {
    ".pdf": "PDF Document",
    ".docx": "Word Document",
}


def validate_file(filename: str, file_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate the uploaded file for supported format and size constraints.

    Args:
        filename: Original filename of the uploaded file.
        file_bytes: Raw byte content of the file.

    Returns:
        A tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    if not filename:
        return False, "No file was provided."

    # Extract and check the file extension
    _, ext = os.path.splitext(filename.lower())
    if ext not in SUPPORTED_FORMATS:
        supported = ", ".join(SUPPORTED_FORMATS.keys())
        return (
            False,
            f"Unsupported file format '{ext}'. Supported formats: {supported}",
        )

    # Check if the file is empty
    if len(file_bytes) == 0:
        return False, "The uploaded file is empty. Please upload a valid resume."

    # Check file size
    if len(file_bytes) > MAX_FILE_SIZE:
        max_mb = MAX_FILE_SIZE // (1024 * 1024)
        return (
            False,
            f"File size exceeds the maximum limit of {max_mb} MB. "
            f"Uploaded file is {len(file_bytes) / (1024 * 1024):.1f} MB.",
        )

    return True, ""


def parse_pdf(file_path: str) -> Optional[str]:
    """
    Extract text content from a PDF file.

    Args:
        file_path: Absolute path to the PDF file.

    Returns:
        Extracted text as a single string, or None if no text was found.

    Raises:
        ValueError: If the PDF file is corrupted or cannot be parsed.
    """
    try:
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text.strip())

        if not text_parts:
            return None

        return "\n".join(text_parts)

    except Exception as exc:
        raise ValueError(
            f"Failed to parse the PDF file. The file may be corrupted or "
            f"password-protected. Error: {exc}"
        )


def parse_docx(file_path: str) -> Optional[str]:
    """
    Extract text content from a DOCX file.

    Args:
        file_path: Absolute path to the DOCX file.

    Returns:
        Extracted text as a single string, or None if no text was found.

    Raises:
        ValueError: If the DOCX file is corrupted or cannot be parsed.
    """
    try:
        doc = Document(file_path)
        text_parts = []

        # Extract text from all paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                text_parts.append(text)

        if not text_parts:
            return None

        return "\n".join(text_parts)

    except Exception as exc:
        raise ValueError(
            f"Failed to parse the DOCX file. The file may be corrupted. "
            f"Error: {exc}"
        )


def parse_file(file_path: str) -> str:
    """
    Parse a resume file and return its text content.

    Automatically detects the file format from the extension and dispatches
    to the appropriate parser.

    Args:
        file_path: Absolute path to the resume file.

    Returns:
        Extracted text content as a single string.

    Raises:
        ValueError: If the file format is unsupported, the file is corrupted,
                    or no text content could be extracted.
    """
    _, ext = os.path.splitext(file_path.lower())

    if ext == ".pdf":
        text = parse_pdf(file_path)
    elif ext == ".docx":
        text = parse_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    if text is None:
        raise ValueError(
            "No text content could be extracted from the file. "
            "The file may be empty or contain only images."
        )

    return text
