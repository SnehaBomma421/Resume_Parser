# AI Resume Parser

> An intelligent resume parsing application that automatically extracts and organises structured candidate information using AI/NLP techniques.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![spaCy](https://img.shields.io/badge/spaCy-3.7%2B-09A3D5)

---

## 📋 Overview

The **AI Resume Parser** is a production-ready web application that takes the pain out of resume screening. Upload a resume in PDF or DOCX format and the system automatically extracts structured information including personal details, skills, education, work experience, projects, certifications, and languages — all presented in a clean, professional HR dashboard.

No paid APIs or API keys are required. Everything runs locally using open-source NLP models.

---

## ✨ Features

| Feature | Description |
|---|---|
| **📄 Multi-format Upload** | Supports PDF and DOCX via drag-and-drop |
| **🧠 AI-Powered Extraction** | Uses spaCy Named Entity Recognition + intelligent heuristics |
| **📊 Section Detection** | Automatically identifies resume sections regardless of formatting |
| **🛠️ Structured Output** | Personal info, skills (categorised), education, experience, projects, certifications, languages |
| **📝 Candidate Summary** | AI-generated narrative summarising the candidate's profile |
| **⭐ Resume Score** | Completeness score out of 100 with visual progress bar |
| **📥 Download** | Export extracted data as JSON or CSV |
| **🎨 Dark Theme UI** | Professional HR dashboard appearance |
| **🛡️ Error Handling** | Graceful handling of empty, corrupted, or unsupported files |

---

## 🧰 Technologies

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **Streamlit** | Web application framework and UI |
| **spaCy** | NLP and Named Entity Recognition |
| **pdfplumber** | PDF text extraction |
| **python-docx** | DOCX text extraction |
| **pandas** | Data export and CSV generation |
| **Regex** | Phone, email, and structured pattern matching |

---

## 🔧 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd Resume_Parser

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download the spaCy model (automatic on first run, or manually)
python -m spacy download en_core_web_sm
```

---

## 🚀 Running

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### First Run

On the first launch, the app will automatically download the **en_core_web_sm** spaCy model (~12 MB) if it is not already installed. This is a one-time operation.

---

## 📁 Project Structure

```
Resume_Parser/
│
├── app.py                 # Main Streamlit application (UI + orchestration)
├── extractor.py           # AI extraction engine (NER, skill DB, scoring)
├── parser.py              # File parsing (PDF, DOCX text extraction)
├── utils.py               # Export helpers (JSON, CSV), text utilities
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
└── .streamlit/
    └── config.toml        # Streamlit theme and server configuration
```

---

## 🧠 How It Works

### 1. File Parsing
PDF files are processed with **pdfplumber** for text extraction. DOCX files are processed with **python-docx**. Both handle formatting variations and extract clean, structured text.

### 2. Section Detection
The system identifies common resume sections (Summary, Skills, Education, Experience, Projects, Certifications, Languages) using regex patterns and heuristics — even when section headers use different naming conventions or formatting.

### 3. Entity Extraction
- **spaCy NER** identifies person names, organisations, and locations
- **Regex** captures emails, phone numbers, and URLs
- **Comprehensive skill database** (~250 skills across 11 categories) identifies both technical and soft skills

### 4. Structured Parsing
Each section is parsed independently:
- **Education** — recognises degree types, institutions, fields of study, graduation years, and GPA
- **Experience** — extracts company names, job titles, date ranges, and responsibilities
- **Projects** — captures project names, descriptions, and technology stacks

### 5. Scoring & Summary
A completeness score (0–100) is calculated based on the presence and quality of each section. A natural-language summary is generated from the extracted information.

---

## 💡 Usage Tips

- **Well-structured resumes** produce the best results. Clear section headers help the parser identify boundaries.
- **File size limit** is 10 MB — most resumes are well under this.
- **Scanned PDFs** (image-only) are not supported because OCR is not included. Use a digital PDF.
- **Multiple pages** are handled correctly.

---

## 📸 Screenshots

<!-- Screenshots placeholder — add actual images here -->
```
[ Screenshot: Main interface with upload ]
[ Screenshot: Extracted results with cards  ]
[ Screenshot: Download options              ]
```

---

## 🚧 Future Enhancements

- [ ] **OCR support** for scanned/image-based PDFs
- [ ] **Batch processing** for multiple resumes
- [ ] **Database integration** to store candidate profiles
- [ ] **Advanced visualisations** for skills gap analysis
- [ ] **Improved NER** with fine-tuned transformer models
- [ ] **Multi-language support** for non-English resumes
- [ ] **API mode** for integration with ATS platforms

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [spaCy](https://spacy.io/) for the excellent NLP library
- [Streamlit](https://streamlit.io/) for the rapid web app framework
- [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF parsing

---

*Built with Python, Streamlit, and spaCy.*
