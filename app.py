"""
AI Resume Parser — Streamlit Application.

Upload a resume in PDF or DOCX format and automatically extract structured
candidate information using AI / NLP techniques.
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import spacy
from spacy import Language as SpacyLanguage

from parser import parse_file, validate_file
from extractor import ResumeExtractor
from utils import prepare_json_download, prepare_csv_download


# Page configuration

st.set_page_config(
    page_title="AI Resume Parser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
 #Custom CSS for styling the app 

st.markdown(
    """
<style>
    /* ---- Base ---- */
    .stApp {
        background: #0E1117;
    }
    .main > div {
        padding: 1rem 2rem;
    }

    /* ---- Header ---- */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .app-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        color: #FAFAFA;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header .accent {
        color: #00C853;
    }
    .app-header p {
        font-size: 1.1rem;
        color: #9CA3AF;
        margin: 0.3rem 0 0 0;
    }

    /* ---- Upload area ---- */
    .upload-area {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        border: 2px dashed rgba(0, 200, 83, 0.4);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        margin: 1.5rem 0;
        transition: border-color 0.3s;
    }
    .upload-area:hover {
        border-color: #00C853;
    }
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .upload-area .formats {
        color: #9CA3AF;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .upload-area .formats span {
        display: inline-block;
        background: rgba(0, 200, 83, 0.15);
        color: #00C853;
        border-radius: 6px;
        padding: 0.15rem 0.6rem;
        margin: 0 0.2rem;
        font-size: 0.8rem;
    }

    /* ---- Extract button ---- */
    .extract-btn {
        text-align: center;
        margin: 1rem 0 2rem 0;
    }
    .extract-btn .stButton button {
        background: linear-gradient(135deg, #00C853 0%, #009624 100%);
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.6rem 3rem;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.3);
        transition: all 0.3s;
        width: 100%;
        max-width: 320px;
    }
    .extract-btn .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 200, 83, 0.45);
    }
    .extract-btn .stButton button:disabled {
        background: #374151;
        box-shadow: none;
    }

    /* ---- Cards ---- */
    .card {
        background: linear-gradient(135deg, #1E1E2E 0%, #2A2A3E 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        transition: border-color 0.2s;
    }
    .card:hover {
        border-color: rgba(0, 200, 83, 0.3);
    }
    .card h3 {
        font-size: 1.15rem;
        font-weight: 600;
        color: #00C853;
        margin: 0 0 1rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .card .field {
        display: flex;
        margin-bottom: 0.4rem;
    }
    .card .field-label {
        color: #9CA3AF;
        min-width: 110px;
        font-size: 0.9rem;
    }
    .card .field-value {
        color: #E5E7EB;
        font-weight: 400;
        word-break: break-word;
    }
    .card .skill-tag {
        display: inline-block;
        background: rgba(0, 200, 83, 0.12);
        color: #6EE7B7;
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        margin: 0.2rem 0.3rem;
        font-size: 0.85rem;
        border: 1px solid rgba(0, 200, 83, 0.2);
    }
    .card .skill-category {
        color: #D1D5DB;
        font-weight: 500;
        margin: 0.6rem 0 0.3rem 0;
        font-size: 0.9rem;
    }
    .card .bullet-list {
        padding-left: 1rem;
        margin: 0.3rem 0;
    }
    .card .bullet-list li {
        color: #E5E7EB;
        margin-bottom: 0.2rem;
        font-size: 0.9rem;
    }
    .card .entry {
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .card .entry:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }

    /* ---- Summary card ---- */
    .summary-card {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        border: 1px solid rgba(100, 200, 255, 0.2);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
    }
    .summary-card h3 {
        color: #60A5FA;
        margin: 0 0 0.8rem 0;
        font-size: 1.1rem;
    }
    .summary-card p {
        color: #E5E7EB;
        font-size: 1rem;
        line-height: 1.6;
        margin: 0;
    }

    /* ---- Score card ---- */
    .score-card {
        background: linear-gradient(135deg, #1E1E2E 0%, #0D1117 100%);
        border: 2px solid #00C853;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin: 1rem 0;
    }
    .score-card .score-value {
        font-size: 3.5rem;
        font-weight: 700;
        color: #00C853;
        line-height: 1;
    }
    .score-card .score-label {
        color: #9CA3AF;
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }
    .score-card .score-out-of {
        color: #6B7280;
        font-size: 1.5rem;
    }

    /* ---- Download buttons ---- */
    .download-section {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin: 1.5rem 0;
    }
    .download-section .stDownloadButton button {
        background: #1F2937;
        color: #E5E7EB;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 0.4rem 1.5rem;
        font-size: 0.9rem;
    }
    .download-section .stDownloadButton button:hover {
        border-color: #00C853;
    }

    /* ---- Footer ---- */
    .footer {
        text-align: center;
        color: #4B5563;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #1F2937;
        margin-top: 2rem;
    }

    /* ---- Error styling ---- */
    .stAlert {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 10px;
        color: #FCA5A5;
    }

    /* ---- Hide default Streamlit elements ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---- Progress bar overrides ---- */
    .stProgress > div > div {
        background: #00C853 !important;
    }

    /* ---- Responsive tweaks ---- */
    @media (max-width: 768px) {
        .app-header h1 { font-size: 2rem; }
        .card .field { flex-direction: column; }
        .card .field-label { min-width: auto; margin-bottom: 0.1rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)


# Session state

if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "extraction_error" not in st.session_state:
    st.session_state.extraction_error = None
if "extracted_filename" not in st.session_state:
    st.session_state.extracted_filename = None


# Helper: load / cache spaCy model
# NOTE: en_core_web_sm is installed via requirements.txt on Streamlit Cloud
# during the build phase. The filesystem is read-only at runtime, so we
# NEVER try to download the model here.

@st.cache_resource(show_spinner="Loading NLP model...")
def load_spacy_model() -> SpacyLanguage:
    """Load the spaCy English small model (pre-installed via requirements.txt)."""
    return spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# Helper: display functions
# ---------------------------------------------------------------------------

def _display_personal_info(data):
    """Render the Personal Information card."""
    pi = data.get("personal_info", {})
    if not isinstance(pi, dict):
        return
    has_data = any(pi.get(k) for k in pi)

    with st.container():
        st.markdown(
            f"""
            <div class="card">
                <h3>👤 Personal Information</h3>
            """,
            unsafe_allow_html=True,
        )
        if not has_data:
            st.markdown(
                '<p style="color: #6B7280;">No personal information detected.</p>',
                unsafe_allow_html=True,
            )
        else:
            fields = [
                ("Full Name", pi.get("name", "")),
                ("Email", pi.get("email", "")),
                ("Phone", pi.get("phone", "")),
                ("Location", pi.get("location", "")),
                ("LinkedIn", pi.get("linkedin", "")),
                ("GitHub", pi.get("github", "")),
                ("Portfolio", pi.get("portfolio", "")),
            ]
            for label, value in fields:
                if value:
                    st.markdown(
                        f'<div class="field"><span class="field-label">{label}:</span>'
                        f'<span class="field-value">{value}</span></div>',
                        unsafe_allow_html=True,
                    )
        st.markdown("</div>", unsafe_allow_html=True)


def _display_skills(data):
    """Render the Skills card with tags grouped by category."""
    skills = data.get("skills", {})
    if not isinstance(skills, dict) or not skills:
        with st.container():
            st.markdown(
                '<div class="card"><h3>🛠️ Skills</h3>'
                '<p style="color: #6B7280;">No skills detected.</p></div>',
                unsafe_allow_html=True,
            )
        return

    html = '<div class="card"><h3>🛠️ Skills</h3>'
    for category, skill_list in skills.items():
        if isinstance(skill_list, list) and skill_list:
            html += f'<div class="skill-category">{category}</div><div>'
            for skill in skill_list:
                html += f'<span class="skill-tag">{skill}</span>'
            html += "</div>"
    html += "</div>"

    with st.container():
        st.markdown(html, unsafe_allow_html=True)


def _display_education(data):
    """Render the Education card."""
    education = data.get("education", [])
    if not isinstance(education, list) or not education:
        with st.container():
            st.markdown(
                '<div class="card"><h3>🎓 Education</h3>'
                '<p style="color: #6B7280;">No education details detected.</p></div>',
                unsafe_allow_html=True,
            )
        return

    html = '<div class="card"><h3>🎓 Education</h3>'
    for entry in education:
        if not isinstance(entry, dict):
            continue
        html += '<div class="entry">'
        fields = [
            ("Degree", entry.get("degree", "")),
            ("Institution", entry.get("institution", "")),
            ("Field", entry.get("field", "")),
            ("Year", entry.get("year", "")),
            ("GPA", entry.get("gpa", "")),
        ]
        for label, value in fields:
            if value:
                html += (
                    f'<div class="field"><span class="field-label">{label}:</span>'
                    f'<span class="field-value">{value}</span></div>'
                )
        html += "</div>"
    html += "</div>"

    with st.container():
        st.markdown(html, unsafe_allow_html=True)


def _display_experience(data):
    """Render the Work Experience card."""
    experience = data.get("experience", [])
    if not isinstance(experience, list) or not experience:
        with st.container():
            st.markdown(
                '<div class="card"><h3>💼 Work Experience</h3>'
                '<p style="color: #6B7280;">No work experience detected.</p></div>',
                unsafe_allow_html=True,
            )
        return

    html = '<div class="card"><h3>💼 Work Experience</h3>'
    for entry in experience:
        if not isinstance(entry, dict):
            continue
        html += '<div class="entry">'
        title = entry.get("title", "")
        company = entry.get("company", "")
        duration = entry.get("duration", "")
        header = f"<strong>{title}</strong>" if title else ""
        if company:
            header += f" at {company}" if header else f"<strong>{company}</strong>"
        if duration:
            header += f" — {duration}"
        if header:
            html += f'<div style="color: #E5E7EB; margin-bottom: 0.4rem;">{header}</div>'

        responsibilities = entry.get("responsibilities", [])
        if isinstance(responsibilities, list) and responsibilities:
            html += '<ul class="bullet-list">'
            for resp in responsibilities:
                if resp:
                    html += f"<li>{resp}</li>"
            html += "</ul>"
        html += "</div>"
    html += "</div>"

    with st.container():
        st.markdown(html, unsafe_allow_html=True)


def _display_projects(data):
    """Render the Projects card."""
    projects = data.get("projects", [])
    if not isinstance(projects, list) or not projects:
        with st.container():
            st.markdown(
                '<div class="card"><h3>📁 Projects</h3>'
                '<p style="color: #6B7280;">No projects detected.</p></div>',
                unsafe_allow_html=True,
            )
        return

    html = '<div class="card"><h3>📁 Projects</h3>'
    for proj in projects:
        if not isinstance(proj, dict):
            continue
        html += '<div class="entry">'
        name = proj.get("name", "")
        if name:
            html += f'<div style="color: #E5E7EB; font-weight: 500; margin-bottom: 0.3rem;">{name}</div>'
        desc = proj.get("description", "")
        if desc:
            html += f'<div style="color: #9CA3AF; font-size: 0.9rem; margin-bottom: 0.3rem;">{desc}</div>'
        techs = proj.get("technologies", [])
        if isinstance(techs, list) and techs:
            html += '<div style="margin-top: 0.3rem;">'
            for t in techs:
                if t:
                    html += f'<span class="skill-tag">{t}</span>'
            html += "</div>"
        html += "</div>"
    html += "</div>"

    with st.container():
        st.markdown(html, unsafe_allow_html=True)


def _display_certifications(data):
    """Render the Certifications card."""
    certs = data.get("certifications", [])
    if not isinstance(certs, list) or not certs:
        with st.container():
            st.markdown(
                '<div class="card"><h3>🏅 Certifications</h3>'
                '<p style="color: #6B7280;">No certifications detected.</p></div>',
                unsafe_allow_html=True,
            )
        return

    html = '<div class="card"><h3>🏅 Certifications</h3>'
    for cert in certs:
        if not isinstance(cert, dict):
            continue
        name = cert.get("name", "")
        issuer = cert.get("issuer", "")
        if name:
            line = f"<strong>{name}</strong>"
            if issuer:
                line += f" — {issuer}"
            html += f'<div style="color: #E5E7EB; margin-bottom: 0.3rem;">{line}</div>'
    html += "</div>"

    with st.container():
        st.markdown(html, unsafe_allow_html=True)


def _display_languages(data):
    """Render the Languages card."""
    langs = data.get("languages", [])
    if not isinstance(langs, list) or not langs:
        with st.container():
            st.markdown(
                '<div class="card"><h3>🌐 Languages</h3>'
                '<p style="color: #6B7280;">No languages detected.</p></div>',
                unsafe_allow_html=True,
            )
        return

    html = '<div class="card"><h3>🌐 Languages</h3>'
    for lang in langs:
        if not isinstance(lang, dict):
            continue
        name = lang.get("name", "")
        prof = lang.get("proficiency", "")
        if name:
            line = f"<strong>{name}</strong>"
            if prof:
                line += f" — {prof}"
            html += f'<div style="color: #E5E7EB; margin-bottom: 0.2rem;">{line}</div>'
    html += "</div>"

    with st.container():
        st.markdown(html, unsafe_allow_html=True)


def _display_summary(summary_text: str):
    """Render the candidate summary card."""
    if not summary_text:
        return
    st.markdown(
        f"""
        <div class="summary-card">
            <h3>📋 Candidate Summary</h3>
            <p>{summary_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _display_score(score: int):
    """Render the resume score card with a progress bar."""
    bar_color = "#00C853" if score >= 60 else "#F59E0B" if score >= 30 else "#EF4444"

    st.markdown(
        f"""
        <div class="score-card">
            <div>
                <span class="score-value">{score}</span>
                <span class="score-out-of">/100</span>
            </div>
            <div class="score-label">Resume Completeness Score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(score / 100)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>AI Resume <span class="accent">Parser</span></h1>
        <p>Upload a resume and automatically extract structured candidate information
        using Artificial Intelligence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — File upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 0.5rem 0;">
            <span style="font-size: 2rem;">📄</span>
            <h3 style="color: #E5E7EB; margin: 0.3rem 0;">Upload Resume</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded_file:
        file_details = {
            "Filename": uploaded_file.name,
            "Size": f"{len(uploaded_file.getvalue()) / 1024:.1f} KB",
        }
        st.markdown(
            f"""
            <div style="background: #1E293B; border-radius: 10px; padding: 0.8rem; margin: 0.5rem 0;">
                <div style="color: #E5E7EB; font-weight: 500;">{uploaded_file.name}</div>
                <div style="color: #9CA3AF; font-size: 0.85rem;">{file_details['Size']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color: #1F2937;'>", unsafe_allow_html=True)

    extract_clicked = st.button(
        "🔍 Extract Resume",
        use_container_width=True,
        disabled=uploaded_file is None,
        type="primary",
    )

    st.markdown(
        """
        <div style="color: #6B7280; font-size: 0.8rem; text-align: center; margin-top: 1rem;">
            <div>Supported formats:</div>
            <div style="margin-top: 0.3rem;">
                <span style="background: rgba(0,200,83,0.15); color: #00C853; border-radius: 4px;
                             padding: 0.1rem 0.5rem; font-size: 0.75rem;">PDF</span>
                <span style="background: rgba(0,200,83,0.15); color: #00C853; border-radius: 4px;
                             padding: 0.1rem 0.5rem; font-size: 0.75rem; margin-left: 0.3rem;">DOCX</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

# Only show upload prompt if nothing processed yet
if st.session_state.extracted_data is None and not st.session_state.extraction_error:
    st.markdown(
        """
        <div style="text-align: center; padding: 4rem 1rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📄</div>
            <h2 style="color: #9CA3AF; font-weight: 400;">Upload a resume to get started</h2>
            <p style="color: #6B7280;">Use the sidebar to upload a PDF or DOCX file, then click
            <strong style="color: #00C853;">Extract Resume</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Process extraction ------------------------------------------------ #
if extract_clicked and uploaded_file is not None:
    with st.spinner("⏳ Parsing resume and extracting information..."):
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name

        # 1. Validate
        is_valid, error_msg = validate_file(filename, file_bytes)
        if not is_valid:
            st.session_state.extraction_error = error_msg
            st.session_state.extracted_data = None
            st.error(error_msg)
        else:
            try:
                # 2. Save to temporary file
                _, ext = os.path.splitext(filename.lower())
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=ext
                ) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                # 3. Parse
                text = parse_file(tmp_path)

                # 4. Extract using AI
                nlp = load_spacy_model()
                extractor = ResumeExtractor(nlp=nlp)
                data = extractor.extract(text)

                # 5. Calculate score and summary (these are fast)
                score = extractor.calculate_score(data)
                summary = extractor.generate_summary(data)

                data["resume_score"] = score
                data["candidate_summary"] = summary
                data["raw_text"] = text

                st.session_state.extracted_data = data
                st.session_state.extraction_error = None
                st.session_state.extracted_filename = filename

                # Clean up the temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            except ValueError as ve:
                st.session_state.extraction_error = str(ve)
                st.session_state.extracted_data = None
                st.error(str(ve))
            except Exception as exc:
                st.session_state.extraction_error = f"An unexpected error occurred: {exc}"
                st.session_state.extracted_data = None
                st.error(f"An unexpected error occurred: {exc}")

    # Force a re-run so the new session state renders immediately
    st.rerun()

# ---- Display error from previous attempt -------------------------------- #
if st.session_state.extraction_error and st.session_state.extracted_data is None:
    st.error(st.session_state.extraction_error)
    if st.button("Try Again"):
        st.session_state.extraction_error = None
        st.rerun()

# ---- Display results ---------------------------------------------------- #
data = st.session_state.extracted_data
if data is not None:
    filename = st.session_state.extracted_filename or "resume"

    # Layout: two columns for the cards
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        _display_personal_info(data)
        _display_education(data)
        _display_projects(data)
        _display_languages(data)

    with col_right:
        _display_skills(data)
        _display_experience(data)
        _display_certifications(data)

    # Full-width sections
    summary_text = data.get("candidate_summary", "")
    _display_summary(summary_text)

    score = data.get("resume_score", 0)
    _display_score(score)

    # ---- Download section ---- #
    st.markdown('<div class="download-section">', unsafe_allow_html=True)

    json_data = prepare_json_download(data)
    csv_data = prepare_csv_download(data)

    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 2])
    with dl_col1:
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"{Path(filename).stem}_parsed.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"{Path(filename).stem}_parsed.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="footer">AI Resume Parser — Built with Streamlit &amp; spaCy</div>',
    unsafe_allow_html=True,
)
