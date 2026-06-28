"""
Utility functions for the AI Resume Parser.

Provides helpers for data export, text cleaning, and formatting.
"""

import json
from typing import Any, Dict, List, Optional

import pandas as pd


def prepare_json_download(data: Dict[str, Any]) -> str:
    """
    Prepare extracted resume data as a formatted JSON string.

    Removes the raw_text field (too large) and serializes the rest.

    Args:
        data: The full extracted data dictionary.

    Returns:
        Pretty-printed JSON string suitable for download.
    """
    download_data = _sanitize_for_json(data)
    return json.dumps(download_data, indent=2, ensure_ascii=False, default=str)


def prepare_csv_download(data: Dict[str, Any]) -> str:
    """
    Prepare extracted resume data as a CSV string.

    Flattens the nested structure into a single-row CSV with descriptive
    column names suitable for spreadsheets.

    Args:
        data: The full extracted data dictionary.

    Returns:
        CSV string with header row and one data row.
    """
    row: Dict[str, Any] = {}

    # --- Personal Information ---
    personal_info = data.get("personal_info", {})
    if isinstance(personal_info, dict):
        field_labels = {
            "name": "Full Name",
            "email": "Email Address",
            "phone": "Phone Number",
            "linkedin": "LinkedIn Profile",
            "github": "GitHub Profile",
            "portfolio": "Portfolio Website",
            "location": "Location",
        }
        for key, label in field_labels.items():
            row[label] = personal_info.get(key, "")

    # --- Professional Summary ---
    row["Professional Summary"] = data.get("summary", "")

    # --- Skills (flattened by category) ---
    skills = data.get("skills", {})
    if isinstance(skills, dict):
        # Create one column per skill category
        for category, skill_list in skills.items():
            if isinstance(skill_list, list):
                col_name = f"Skills - {category}"
                row[col_name] = ", ".join(skill_list)

    # --- Education ---
    education = data.get("education", [])
    if isinstance(education, list):
        for i, edu in enumerate(education, start=1):
            if isinstance(edu, dict):
                prefix = f"Education #{i}"
                field_map = {
                    "degree": "Degree",
                    "institution": "Institution",
                    "field": "Field",
                    "year": "Graduation Year",
                    "gpa": "GPA / Percentage",
                }
                for key, label in field_map.items():
                    val = edu.get(key, "")
                    if val:
                        row[f"{prefix} - {label}"] = val

    # --- Work Experience ---
    experience = data.get("experience", [])
    if isinstance(experience, list):
        for i, exp in enumerate(experience, start=1):
            if isinstance(exp, dict):
                prefix = f"Experience #{i}"
                row[f"{prefix} - Company"] = exp.get("company", "")
                row[f"{prefix} - Job Title"] = exp.get("title", "")
                row[f"{prefix} - Duration"] = exp.get("duration", "")
                responsibilities = exp.get("responsibilities", [])
                if isinstance(responsibilities, list):
                    row[f"{prefix} - Responsibilities"] = "; ".join(
                        r for r in responsibilities if r
                    )

    # --- Projects ---
    projects = data.get("projects", [])
    if isinstance(projects, list):
        for i, proj in enumerate(projects, start=1):
            if isinstance(proj, dict):
                prefix = f"Project #{i}"
                row[f"{prefix} - Name"] = proj.get("name", "")
                row[f"{prefix} - Description"] = proj.get("description", "")
                technologies = proj.get("technologies", [])
                if isinstance(technologies, list):
                    row[f"{prefix} - Technologies"] = ", ".join(technologies)

    # --- Certifications ---
    certifications = data.get("certifications", [])
    if isinstance(certifications, list):
        cert_names = []
        for cert in certifications:
            if isinstance(cert, dict):
                cert_names.append(cert.get("name", ""))
            elif isinstance(cert, str):
                cert_names.append(cert)
        row["Certifications"] = "; ".join(c for c in cert_names if c)

    # --- Languages ---
    languages = data.get("languages", [])
    if isinstance(languages, list):
        lang_names = []
        for lang in languages:
            if isinstance(lang, dict):
                lang_names.append(
                    f"{lang.get('name', '')} ({lang.get('proficiency', 'Not specified')})"
                )
            elif isinstance(lang, str):
                lang_names.append(lang)
        row["Languages"] = ", ".join(l for l in lang_names if l)

    # --- Score & Summary ---
    row["Resume Score"] = data.get("resume_score", 0)
    row["Candidate Summary"] = data.get("candidate_summary", "")

    df = pd.DataFrame([row])
    return df.to_csv(index=False, encoding="utf-8-sig")


def _sanitize_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove non-serializable fields and convert types for JSON export.

    Args:
        data: The extracted data dictionary.

    Returns:
        A JSON-safe copy of the data.
    """
    sanitized = {}
    for key, value in data.items():
        if key == "raw_text":
            continue
        if isinstance(value, dict):
            sanitized[key] = _sanitize_for_json(value)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_for_json(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    return sanitized


def get_file_extension(filename: str) -> str:
    """
    Extract the file extension from a filename.

    Args:
        filename: The original filename.

    Returns:
        The lowercased file extension including the dot (e.g. '.pdf').
    """
    return os.path.splitext(filename)[1].lower()


def estimate_reading_time(text: str) -> int:
    """
    Estimate the reading time of a text in seconds.

    Args:
        text: The text content.

    Returns:
        Estimated reading time in seconds.
    """
    words_per_second = 3.0  # ~180 WPM
    word_count = len(text.split())
    return max(1, int(word_count / words_per_second))
