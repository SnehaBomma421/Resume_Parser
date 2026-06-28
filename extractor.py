"""
Intelligent resume information extractor.

Uses spaCy NER, regex, and keyword-based heuristics to extract structured
information from resume text. Detects sections, identifies skills, parses
education and experience entries, and generates a completeness score.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import spacy
from spacy import Language as SpacyLanguage


# ---------------------------------------------------------------------------
# Section header patterns
# ---------------------------------------------------------------------------
# Matched as full lines (re.fullmatch). Order matters: more specific patterns
# should come before generic ones within the same section group.
SECTION_PATTERNS: List[Tuple[str, str]] = [
    # Professional Summary / Objective
    (r"(?i)^(?:professional\s+)?summary\s*$", "summary"),
    (r"(?i)^(?:career\s+)?objective\s*$", "summary"),
    (r"(?i)^(?:profile|about\s+me)\s*$", "summary"),
    # Skills
    (r"(?i)^(?:technical\s+)?skills?\s*(?:&\s*competencies)?\s*$", "skills"),
    (r"(?i)^(?:core\s+)?competenc(?:y|ies)\s*$", "skills"),
    (r"(?i)^skills\s*&\s*(?:abilities|expertise)\s*$", "skills"),
    # Education
    (r"(?i)^education(?:\s*\(?\w+\)?)?\s*$", "education"),
    (r"(?i)^(?:academic\s+)?(?:background|qualifications?|training)\s*$", "education"),
    # Experience
    (r"(?i)^(?:professional\s+)?(?:work\s+)?(?:experience|history|background)\s*$", "experience"),
    (r"(?i)^(?:relevant\s+)?(?:employment|experience)\s*$", "experience"),
    (r"(?i)^work\s*$", "experience"),
    # Projects
    (r"(?i)^(?:academic\s+)?projects?\s*(?:\(\w+\))?\s*$", "projects"),
    (r"(?i)^(?:key\s+)?(?:projects?|initiatives)\s*$", "projects"),
    # Certifications
    (r"(?i)^(?:professional\s+)?certifications?\s*(?:&\s*licenses?)?\s*$", "certifications"),
    (r"(?i)^licenses?\s*(?:&\s*certifications?)?\s*$", "certifications"),
    (r"(?i)^(?:certifications?\s*&\s*)?(?:training|courses?)\s*$", "certifications"),
    # Languages
    (r"(?i)^languages?\s*(?::|spokens?|known)?\s*$", "languages"),
    # Publications
    (r"(?i)^(?:technical\s+)?publications?\s*$", "publications"),
    # Awards / Honors
    (r"(?i)^(?:honors?\s*&\s*)?awards?\s*$", "awards"),
    (r"(?i)^(?:achievements?|accolades)\s*$", "awards"),
    # Volunteer
    (r"(?i)^(?:volunteer\s+)?(?:experience|work|activities?)\s*$", "volunteer"),
    # Contact
    (r"(?i)^(?:contact|personal)\s*(?:information|details|info)?\s*$", "contact"),
]


# ---------------------------------------------------------------------------
# Skill database — categorised for structured presentation
# ---------------------------------------------------------------------------
SKILL_DATABASE: Dict[str, List[str]] = {
    "Programming Languages": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "C",
        "Ruby", "Go", "Golang", "Rust", "Swift", "Kotlin", "PHP", "R",
        "Scala", "Perl", "Haskell", "Dart", "Lua", "MATLAB", "Groovy",
        "Julia", "Elixir", "Clojure", "Solidity", "COBOL", "Fortran",
        "Assembly", "Ada", "Lisp", "Prolog", "Bash", "PowerShell",
        "Shell Scripting", "Delphi", "VB.NET", "Visual Basic", "SQL",
        "PL/SQL", "T-SQL", "GraphQL", "HTML", "CSS", "Sass", "SCSS",
        "Less", "XML", "JSON", "YAML", "TOML",
    ],
    "Frontend": [
        "React", "React.js", "Angular", "AngularJS", "Vue.js", "Vue",
        "Svelte", "Next.js", "Nuxt.js", "jQuery", "Bootstrap", "Tailwind",
        "Tailwind CSS", "Material UI", "MUI", "Chakra UI", "Ant Design",
        "Semantic UI", "D3.js", "Three.js", "Chart.js", "Redux", "MobX",
        "Webpack", "Vite", "Parcel", "Babel", "ESLint", "Prettier",
        "Gulp", "Grunt", "Storybook", "Cypress", "Jest", "Figma",
        "Adobe XD", "Sketch",
    ],
    "Backend Frameworks": [
        "Django", "Flask", "FastAPI", "Spring", "Spring Boot", "Spring MVC",
        "Hibernate", "JPA", "ASP.NET", "ASP.NET Core", "Node.js", "Express",
        "Express.js", "Ruby on Rails", "Rails", "Laravel", "Symfony",
        "CakePHP", "CodeIgniter", "Phalcon", "Yii", "Koa", "Koa.js",
        "Sails.js", "Meteor", "Nest.js", "NestJS", "Gin", "Echo",
        "Fiber", "Phoenix", "Play Framework", "Dropwizard", "Micronaut",
        "Quarkus", "Vert.x", "LoopBack",
    ],
    "Databases": [
        "MySQL", "PostgreSQL", "SQLite", "Oracle", "Oracle Database",
        "Microsoft SQL Server", "SQL Server", "MongoDB", "Redis", "Cassandra",
        "CouchDB", "MariaDB", "DynamoDB", "Firebase", "Firestore",
        "Supabase", "Neo4j", "InfluxDB", "Elasticsearch", "Memcached",
        "CockroachDB", "ScyllaDB", "ClickHouse", "Presto", "Trino",
        "Snowflake", "BigQuery", "Redshift", "Cosmos DB",
    ],
    "Cloud & DevOps": [
        "AWS", "Amazon Web Services", "Azure", "Microsoft Azure",
        "Google Cloud Platform", "GCP", "Google Cloud", "Heroku",
        "DigitalOcean", "Netlify", "Vercel", "Cloudflare", "OpenStack",
        "IBM Cloud", "Oracle Cloud", "EC2", "S3", "Lambda",
        "CloudFormation", "CloudWatch", "IAM", "Route53", "Azure Functions",
        "Azure DevOps", "Cloud Functions", "GKE", "AKS", "EKS",
        "Docker", "Kubernetes", "K8s", "Jenkins", "GitLab CI",
        "GitLab CI/CD", "GitHub Actions", "CircleCI", "Travis CI",
        "Ansible", "Chef", "Puppet", "Vagrant", "Terraform",
        "Helm", "ArgoCD", "Prometheus", "Grafana", "ELK Stack",
        "Datadog", "New Relic", "Splunk",
    ],
    "AI / ML": [
        "Machine Learning", "Deep Learning", "Natural Language Processing",
        "NLP", "Computer Vision", "LLM", "Large Language Model",
        "Neural Networks", "Reinforcement Learning", "Generative AI",
        "Gen AI", "GenAI", "PyTorch", "TensorFlow", "Keras",
        "Scikit-learn", "Hugging Face", "Transformers", "LangChain",
        "LlamaIndex", "Stable Diffusion", "OpenAI", "Ollama",
        "XGBoost", "LightGBM", "CatBoost", "Weights & Biases",
        "MLflow", "Kubeflow", "JAX", "Numba", "CUDA",
        "OpenCV", "NLTK", "SpaCy", "Gensim", "Sentence Transformers",
    ],
    "Data & Analytics": [
        "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn", "Plotly",
        "Tableau", "Power BI", "Looker", "Jupyter", "Jupyter Notebook",
        "Statistical Analysis", "Data Mining", "Data Visualization",
        "A/B Testing", "ETL", "Airflow", "Hadoop", "Spark", "Apache Spark",
        "Kafka", "Apache Kafka", "Hive", "Pig", "HBase",
        "Databricks", "dbt", "Flink", "Storm",
    ],
    "Testing": [
        "PyTest", "Selenium", "JUnit", "TestNG", "Mockito", "Cypress",
        "Jest", "Mocha", "Chai", "Jasmine", "Karma", "Postman",
        "SoapUI", "RestAssured", "Cucumber", "Gatling", "JMeter",
        "Playwright", "Puppeteer",
    ],
    "Tools & Platforms": [
        "Git", "GitHub", "GitLab", "Bitbucket", "SVN", "Mercurial",
        "Jira", "Confluence", "Slack", "Trello", "Asana", "Notion",
        "VS Code", "Visual Studio Code", "IntelliJ", "Eclipse", "Vim",
        "Nano", "Postman", "curl", "Docker Compose", "Make",
        "Gradle", "Maven", "npm", "yarn", "pnpm",
        "Linux", "Ubuntu", "CentOS", "Debian", "Fedora", "Red Hat",
        "Windows Server", "macOS", "Unix",
    ],
    "Soft Skills": [
        "Leadership", "Communication", "Teamwork", "Collaboration",
        "Problem Solving", "Critical Thinking", "Time Management",
        "Project Management", "Agile", "Scrum", "Kanban", "Waterfall",
        "Presentation", "Negotiation", "Conflict Resolution",
        "Decision Making", "Adaptability", "Creativity", "Innovation",
        "Analytical Skills", "Attention to Detail", "Organization",
        "Multitasking", "Interpersonal Skills", "Mentoring",
        "Cross-functional Collaboration", "Strategic Planning",
        "Customer Focus", "Emotional Intelligence",
    ],
}


# ---------------------------------------------------------------------------
# Education degree patterns — matched against text lines
# ---------------------------------------------------------------------------
DEGREE_PATTERNS: List[re.Pattern] = [
    re.compile(p)
    for p in [
        r"(?i)\bB\.?(?:Tech|E\.|Sc\.?|A\.?|Com\.?|BA\.?|Bus\.?|Eng\.?|"
        r"Arch\.?|Pharm\.?|Ed\.?|FA\.?)\b",
        r"(?i)\bBachelor(?:'s)?\s+(?:of\s+)?(?:Technology|Engineering|"
        r"Science|Arts|Commerce|Business|Education|Architecture|Pharmacy|"
        r"Computer\s+Applications|Law|Fine\s+Arts|Design)\b",
        r"(?i)\bB(?:\.\s*)?(?:Tech|E|Sc|A|Com|BA|Bus|Eng|Arch|Pharm|Ed|FA)\b",
        r"(?i)\bM\.?(?:Tech\.?|E\.?|Sc\.?|A\.?|Com\.?|Bus\.?|Eng\.?|Pharm\.?|"
        r"Ed\.?|BA\.?)\b",
        r"(?i)\bMaster(?:'s)?\s+(?:of\s+)?(?:Technology|Engineering|Science|"
        r"Arts|Commerce|Business|Education|Pharmacy|Computer\s+Applications|"
        r"Law|Public\s+Health|Fine\s+Arts|Design|Data\s+Science)\b",
        r"(?i)\bM(?:\.\s*)?(?:Tech|E|Sc|A|Com|Bus|Eng|Pharm|Ed)\b",
        r"(?i)\bMBA\b",
        r"(?i)\bEMBA\b",
        r"(?i)\bPGDM\b",
        r"(?i)\bPh\.?D\.?\b",
        r"(?i)\bPhD\b",
        r"(?i)\bDoctor(?:'s)?\s+(?:of\s+)?(?:Philosophy|Education|Law|"
        r"Medicine|Science|Engineering|Business)\b",
        r"(?i)\bDiploma\b",
        r"(?i)\bPG\s*Diploma\b",
        r"(?i)\bAssociate(?:'s)?\s+(?:Degree|of)\b",
        r"(?i)\bPost\s*Graduate\s*(?:Diploma|Certificate|Program)\b",
        r"(?i)\b(?:B|M)\s*\.\s*(?:Tech|Eng|Sc|A|Com|BA|Ed)\b",
        r"(?i)\b(?:Bachelor|Master|Doctor)\b",
    ]
]


# ---------------------------------------------------------------------------
# Date-range pattern used in experience extraction
# ---------------------------------------------------------------------------
DATE_RANGE_PATTERN = re.compile(
    r"(?P<start>"
    r"(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+)?\d{4})"
    r"\s*(?:[-–—to]+|–|—|to|until|present)\s*"
    r"(?P<end>(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+)?(?:\d{4}|Present|Current|Now))",
    re.IGNORECASE,
)

# Single year pattern for education
YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")


class ResumeExtractor:
    """Main class for extracting structured information from resume text."""

    def __init__(self, nlp: Optional[SpacyLanguage] = None):
        """
        Initialise the extractor and load the spaCy model.

        If no model is provided, ``en_core_web_sm`` is downloaded and loaded
        automatically.

        Args:
            nlp: A pre-loaded spaCy Language object (optional).
        """
        if nlp is not None:
            self.nlp = nlp
        else:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract all structured information from resume text.

        Args:
            text: Raw text extracted from a resume file.

        Returns:
            A dictionary with keys: personal_info, summary, skills,
            education, experience, projects, certifications, languages.
        """
        text = self._clean_text(text)
        if not text:
            return self._empty_result()

        # Truncate very long texts at 150 000 characters for performance
        doc = self.nlp(text[:150_000])

        sections = self._detect_sections(text, doc)

        return {
            "personal_info": self._extract_personal_info(text, doc, sections),
            "summary": self._extract_summary(text, sections),
            "skills": self._extract_skills(text, sections),
            "education": self._extract_education(text, sections),
            "experience": self._extract_experience(text, sections),
            "projects": self._extract_projects(text, sections),
            "certifications": self._extract_certifications(text, sections),
            "languages": self._extract_languages(text, sections),
        }

    def calculate_score(self, data: Dict[str, Any]) -> int:
        """
        Calculate a resume completeness score out of 100.

        Points are awarded for the presence and quality of each section.
        Each section has a defined maximum contribution.

        Args:
            data: The extracted resume data dictionary.

        Returns:
            An integer score from 0 to 100.
        """
        section_scores = []

        # Personal Information (max 20)
        pi_score = 0
        pi = data.get("personal_info", {})
        if isinstance(pi, dict):
            if pi.get("name"):
                pi_score += 5
            if pi.get("email"):
                pi_score += 4
            if pi.get("phone"):
                pi_score += 4
            if pi.get("location"):
                pi_score += 3
            if pi.get("linkedin") or pi.get("github") or pi.get("portfolio"):
                pi_score += 4
        pi_score = min(pi_score, 20)
        section_scores.append(pi_score)

        # Professional Summary (max 10)
        summary_text = data.get("summary", "")
        if summary_text and len(summary_text) > 20:
            section_scores.append(10)
        else:
            section_scores.append(0)

        # Skills (max 20)
        skills_score = 0
        skills = data.get("skills", {})
        if isinstance(skills, dict):
            total_skills = sum(len(v) for v in skills.values() if isinstance(v, list))
            if total_skills >= 10:
                skills_score = 20
            elif total_skills >= 5:
                skills_score = 15
            elif total_skills >= 3:
                skills_score = 10
            elif total_skills >= 1:
                skills_score = 5
        section_scores.append(skills_score)

        # Education (max 15)
        edu_score = 0
        education = data.get("education", [])
        if isinstance(education, list) and education:
            entry = education[0]
            if isinstance(entry, dict):
                if entry.get("degree"):
                    edu_score += 6
                if entry.get("institution"):
                    edu_score += 5
                if entry.get("year"):
                    edu_score += 4
            if len(education) > 1:
                edu_score += 3
        edu_score = min(edu_score, 15)
        section_scores.append(edu_score)

        # Work Experience (max 25)
        exp_score = 0
        experience = data.get("experience", [])
        if isinstance(experience, list) and experience:
            entry = experience[0]
            if isinstance(entry, dict):
                if entry.get("company") or entry.get("title"):
                    exp_score += 10
                if entry.get("duration"):
                    exp_score += 5
                if entry.get("responsibilities"):
                    exp_score += 5
            if len(experience) > 1:
                exp_score += 5
        exp_score = min(exp_score, 25)
        section_scores.append(exp_score)

        # Projects (max 10)
        proj_score = 0
        projects = data.get("projects", [])
        if isinstance(projects, list) and projects:
            proj_score += 5
            if len(projects) > 1:
                proj_score += 5
        section_scores.append(proj_score)

        # Certifications (max 5)
        certs = data.get("certifications", [])
        cert_score = 5 if (isinstance(certs, list) and certs) else 0
        section_scores.append(cert_score)

        # Languages (max 5)
        langs = data.get("languages", [])
        lang_score = 5 if (isinstance(langs, list) and langs) else 0
        section_scores.append(lang_score)

        total = sum(section_scores)
        return min(total, 100)

    def generate_summary(self, data: Dict[str, Any]) -> str:
        """
        Generate a short narrative summary of the candidate.

        Args:
            data: The extracted resume data dictionary.

        Returns:
            A human-readable summary paragraph.
        """
        parts = []
        pi = data.get("personal_info", {})

        # Name
        candidate_name = pi.get("name", "This candidate") if isinstance(pi, dict) else "This candidate"

        # Education
        edu = data.get("education", [])
        if isinstance(edu, list) and edu:
            first = edu[0]
            if isinstance(first, dict):
                degree = first.get("degree", "")
                field = first.get("field", "")
                institution = first.get("institution", "")
                edu_parts = []
                if degree and field:
                    edu_parts.append(f"a {degree} in {field}")
                elif degree:
                    edu_parts.append(f"a {degree}")
                elif field:
                    edu_parts.append(f"a background in {field}")
                if institution and edu_parts:
                    edu_parts.append(f"from {institution}")
                if edu_parts:
                    parts.append("is " + ", ".join(edu_parts))

        # Skills
        skills = data.get("skills", {})
        all_skills = []
        if isinstance(skills, dict):
            for cat_list in skills.values():
                if isinstance(cat_list, list):
                    all_skills.extend(cat_list[:3])

        if all_skills:
            top = all_skills[:6]
            if len(top) == 1:
                parts.append(f"with expertise in {top[0]}")
            elif len(top) <= 3:
                parts.append(f"with expertise in {', '.join(top)}")
            else:
                parts.append(f"with strong skills in {', '.join(top[:3])}, {top[3]} and {len(all_skills) - 4} other skills")

        # Experience
        exp = data.get("experience", [])
        if isinstance(exp, list) and exp:
            companies = [
                e.get("company", "") for e in exp if isinstance(e, dict) and e.get("company")
            ]
            titles = [
                e.get("title", "") for e in exp if isinstance(e, dict) and e.get("title")
            ]
            if companies:
                if len(companies) == 1:
                    parts.append(f"professional experience at {companies[0]}")
                else:
                    placed = ", ".join(companies[:-1]) + f" and {companies[-1]}"
                    parts.append(f"experience across {len(companies)} organizations including {placed}")
            elif titles:
                parts.append(f"experienced as {' / '.join(titles[:2])}")
            elif len(exp) > 0:
                parts.append(f"has held {len(exp)} professional position(s)")

        # Projects
        proj = data.get("projects", [])
        if isinstance(proj, list) and proj:
            techs = set()
            for p in proj:
                if isinstance(p, dict) and p.get("technologies"):
                    if isinstance(p["technologies"], list):
                        techs.update(p["technologies"])
            if techs:
                sample = list(techs)[:3]
                parts.append(
                    f"completed {len(proj)} project(s) using {', '.join(sample)}"
                )
            else:
                parts.append(f"completed {len(proj)} project(s)")

        # Certifications
        certs = data.get("certifications", [])
        if isinstance(certs, list) and certs:
            cert_count = len(certs)
            parts.append(f"holds {cert_count} professional certification(s)")

        if not parts:
            return (
                f"{candidate_name} is a professional with experience. "
                "Upload a complete resume for a detailed analysis."
            )

        # Build the summary sentence
        # Start with the candidate name and first part
        summary = f"{candidate_name} {parts[0]}"
        for part in parts[1:]:
            summary += f", {part}"

        summary += "."
        # Clean up awkward punctuation
        summary = summary.replace(",.", ".")
        summary = summary.replace(" ,", ",")
        # Ensure it ends with a period
        if not summary.endswith("."):
            summary += "."

        return summary

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalise whitespace and remove non-printable characters."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\t", " ", text)
        # Replace common bullet characters with a standard marker
        text = re.sub(r"[•‣◦⁃∙●○■]", "-", text)
        # Remove zero-width characters
        text = re.sub(r"[​-‍﻿]", "", text)
        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Section detection
    # ------------------------------------------------------------------

    def _detect_sections(
        self, text: str, doc: SpacyLanguage
    ) -> Dict[str, Tuple[int, int]]:
        """
        Identify resume section boundaries.

        Returns a dict mapping section names to ``(start_line, end_line)``
        in the line-split text. Lines are zero-indexed.
        """
        lines = text.split("\n")
        sections: Dict[str, int] = {}  # section_name -> start_line

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            # --- Check against known patterns (full-line match) ---
            for pattern, sec_name in SECTION_PATTERNS:
                if re.fullmatch(pattern, stripped):
                    if sec_name not in sections:
                        sections[sec_name] = i
                    break

            # --- ALL-CAPS heuristic ---
            # Lines that are short, all-caps, mostly alphabetic, and contain
            # a known keyword are likely section headers.
            if (
                2 <= len(stripped) <= 50
                and stripped.isupper()
                and sum(1 for c in stripped if c.isalpha()) >= 4
                and not re.search(r"\d", stripped)
            ):
                lower_stripped = stripped.lower().strip()
                for pattern, sec_name in SECTION_PATTERNS:
                    if re.fullmatch(pattern, lower_stripped):
                        if sec_name not in sections:
                            sections[sec_name] = i
                        break

        # --- Convert start positions to (start, end) ranges ---
        sorted_secs = sorted(sections.items(), key=lambda kv: kv[1])
        result: Dict[str, Tuple[int, int]] = {}
        for j, (name, start) in enumerate(sorted_secs):
            if j + 1 < len(sorted_secs):
                next_start = sorted_secs[j + 1][1]
                result[name] = (start, next_start - 1)
            else:
                result[name] = (start, len(lines) - 1)

        return result

    def _get_section_text(
        self,
        text: str,
        sections: Dict[str, Tuple[int, int]],
        section_name: str,
    ) -> str:
        """Return the text of a detected section (excluding the header)."""
        if section_name not in sections:
            return ""
        lines = text.split("\n")
        start, end = sections[section_name]
        # Skip the header line itself
        content_start = start + 1
        if content_start > end:
            return ""
        return "\n".join(lines[content_start : end + 1]).strip()

    def _section_contains(
        self,
        text: str,
        sections: Dict[str, Tuple[int, int]],
        section_name: str,
        keyword: str,
    ) -> bool:
        """Check if a section contains a keyword (case-insensitive)."""
        section_text = self._get_section_text(text, sections, section_name)
        return keyword.lower() in section_text.lower()

    # ------------------------------------------------------------------
    # Personal information
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_personal_info(
        text: str,
        doc: SpacyLanguage,
        sections: Dict[str, Tuple[int, int]],
    ) -> Dict[str, str]:
        """Extract name, email, phone, URLs and location."""
        info: Dict[str, str] = {
            "name": "",
            "email": "",
            "phone": "",
            "linkedin": "",
            "github": "",
            "portfolio": "",
            "location": "",
        }

        # Prefer contact section for email/phone if it exists
        contact_text = ""
        if "contact" in sections:
            lines = text.split("\n")
            start, end = sections["contact"]
            contact_text = "\n".join(lines[start + 1 : end + 1])
        search_text = text if not contact_text else contact_text

        # --- Email ---
        email_match = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", search_text)
        if email_match:
            info["email"] = email_match.group(0)

        # --- Phone ---
        phone_match = re.search(
            r"(?:(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})"
            r"(?!\s*[-.\s]?\d)",  # avoid matching 16+ digit numbers
            search_text,
        )
        if phone_match:
            info["phone"] = phone_match.group(0).strip("-.")

        # --- LinkedIn ---
        linkedin_match = re.search(
            r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?",
            text,
            re.IGNORECASE,
        )
        if linkedin_match:
            info["linkedin"] = linkedin_match.group(0).rstrip("/")

        # --- GitHub ---
        github_match = re.search(
            r"(?:https?://)?(?:www\.)?github\.com/[\w-]+/?",
            text,
            re.IGNORECASE,
        )
        if github_match:
            info["github"] = github_match.group(0).rstrip("/")

        # --- Portfolio / personal website ---
        # Look for common URL patterns not already captured.
        portfolio_match = re.search(
            r"(?:https?://)?(?:[\w-]+\.)+(?:com|dev|io|me|app|net|org|tech|"
            r"co|in|ai)(?:/[\w/-]*)?",
            text,
            re.IGNORECASE,
        )
        if portfolio_match:
            url = portfolio_match.group(0)
            match_start = portfolio_match.start()
            # Exclude if it looks like an email domain or is LinkedIn/GitHub
            if (
                "linkedin" not in url.lower()
                and "github" not in url.lower()
                and not re.search(r"[\w.+-]+@", text[max(0, match_start - 30):match_start])
            ):
                info["portfolio"] = url.rstrip("/")

        # --- Name ---
        # Known non-name words/phrases that spaCy may misclassify as PERSON
        _NON_NAME_WORDS = {
            "problem solving", "critical thinking", "teamwork", "leadership",
            "communication", "project management", "decision making",
            "analytical", "interpersonal", "creativity", "adaptability",
            "collaboration", "time management", "conflict resolution",
            "active listening", "negotiation", "presentation", "mentoring",
            "customer service", "detail oriented", "self motivated",
            "problem-solving", "team player", "quick learner",
        }

        # 1) Prefer the first non-empty, name-like line in the document.
        #    Resumes almost always start with the candidate's name on line 1.
        raw_lines = text.split("\n")
        for raw_line in raw_lines:
            candidate = raw_line.strip()
            if not candidate:
                continue
            # Must be 2-4 words, no special characters, not a URL/email/header
            words = candidate.split()
            if len(words) < 2 or len(words) > 4:
                continue
            if len(candidate) > 50:
                continue
            if candidate.lower() in _NON_NAME_WORDS:
                continue
            if re.search(r"[@#\$%\^&*+=<>\[\]{}|]", candidate):
                continue
            if re.match(r"^(?:https?://|www\.)", candidate, re.I):
                continue
            if re.fullmatch(r"(?i)^(?:contact|personal).*", candidate):
                continue
            if any(re.fullmatch(pat[0], candidate) for pat in SECTION_PATTERNS):
                continue
            # Looks like a valid name — accept it
            info["name"] = candidate
            break

        # 2) If the heuristic failed, fall back to spaCy PERSON entities.
        if not info["name"]:
            persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
            for p in persons:
                parts = p.split()
                if not (2 <= len(parts) <= 4 and len(p) <= 50):
                    continue
                if re.search(r"[\n\r@]", p):
                    continue
                if re.search(r"\.(com|io|net|org|dev)\b", p, re.I):
                    continue
                if p.lower() in _NON_NAME_WORDS:
                    continue
                info["name"] = p
                break

        # --- Location ---
        # Try GPE entities from spaCy
        gpe_entities = [
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ == "GPE" and len(ent.text.strip()) > 2
        ]
        if gpe_entities:
            # Prefer the first GPE near the top of the document
            info["location"] = gpe_entities[0]

        return info

    # ------------------------------------------------------------------
    # Professional summary
    # ------------------------------------------------------------------

    def _extract_summary(
        self,
        text: str,
        sections: Dict[str, Tuple[int, int]],
    ) -> str:
        """Extract the professional summary / objective paragraph."""
        summary_text = self._get_section_text(text, sections, "summary")
        if summary_text:
            # Take the first paragraph (up to ~500 chars)
            paragraphs = summary_text.split("\n\n")
            summary = paragraphs[0].strip()
            if len(summary) > 500:
                summary = summary[:500].rsplit(". ", 1)[0] + "."
            return summary

        # Fallback: look for summary-like patterns early in the text
        lines = text.split("\n")
        # Skip the first 1-2 lines (name, title) and look for a paragraph
        content_lines = []
        started = False
        for line in lines[2:]:
            stripped = line.strip()
            if not stripped:
                if started:
                    break
                continue
            # Skip if it looks like a section header
            if any(re.fullmatch(p[0], stripped) for p in SECTION_PATTERNS):
                break
            content_lines.append(stripped)
            started = True

        candidate = " ".join(content_lines)
        if 30 <= len(candidate) <= 500:
            return candidate

        return ""

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    def _extract_skills(
        self,
        text: str,
        sections: Dict[str, Tuple[int, int]],
    ) -> Dict[str, List[str]]:
        """
        Identify and categorise skills mentioned in the resume.

        Returns a dict mapping category names to lists of matched skills.
        """
        # Determine the text region to search
        if "skills" in sections:
            search_text = self._get_section_text(text, sections, "skills")
        else:
            search_text = text

        found: Dict[str, List[str]] = {}

        for category, skill_list in SKILL_DATABASE.items():
            matched: List[str] = []
            for skill in skill_list:
                # Build a pattern that matches the skill as a whole word
                # Escape special regex characters
                escaped = re.escape(skill)
                pattern = re.compile(
                    rf"(?<![a-zA-Z]){escaped}(?![a-zA-Z])", re.IGNORECASE
                )
                if pattern.search(search_text):
                    matched.append(skill)
            if matched:
                # Deduplicate while preserving order
                seen: set = set()
                deduped: List[str] = []
                for m in matched:
                    key = m.lower()
                    if key not in seen:
                        seen.add(key)
                        deduped.append(m)
                found[category] = deduped

        return found

    # ------------------------------------------------------------------
    # Education
    # ------------------------------------------------------------------

    def _extract_education(
        self,
        text: str,
        sections: Dict[str, Tuple[int, int]],
    ) -> List[Dict[str, str]]:
        """Extract education entries from the education section."""
        if "education" in sections:
            section_text = self._get_section_text(text, sections, "education")
        else:
            section_text = text

        if not section_text.strip():
            return []

        lines = section_text.split("\n")
        entries: List[Dict[str, str]] = []
        current: Dict[str, str] = {}
        edu_pattern = r"(?i)(?:degree|gpa|cgpa|grade|percentage|g\.?p\.?a)"

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current:
                    entries.append(current)
                    current = {}
                continue

            # Detect if a new entry is starting (line contains a degree keyword)
            is_degree_line = any(p.search(stripped) for p in DEGREE_PATTERNS)

            if is_degree_line:
                if current:
                    entries.append(current)
                current = {"degree": "", "institution": "", "field": "", "year": "", "gpa": ""}
                # Try to extract both degree and institution from one line
                current["degree"] = self._find_degree(stripped)

                # Everything after the degree name could be institution or field
                remaining = self._remove_degree(stripped)
                # Check if remaining contains institution keywords
                if re.search(r"(?i)(?:university|college|institute|school)", remaining):
                    current["institution"] = remaining
                else:
                    current["field"] = remaining

            else:
                if not current:
                    current = {"degree": "", "institution": "", "field": "", "year": "", "gpa": ""}

                # Check for institution name
                if re.search(r"(?i)(?:university|college|institute|school)", stripped):
                    current["institution"] = stripped

                # Check for year
                year_match = YEAR_PATTERN.search(stripped)
                if year_match:
                    year = year_match.group(0)
                    current_year = 2026
                    if 1950 <= int(year) <= current_year + 5:
                        current["year"] = year

                # Check for GPA / percentage
                gpa_match = re.search(
                    r"(?i)(?:gpa|cgpa|grade|percentage|score)[:\s]*([\d.]+(?:\s*/\s*[\d.]+)?)",
                    stripped,
                )
                if gpa_match:
                    current["gpa"] = gpa_match.group(1)
                else:
                    # Also match patterns like "8.5/10" or "85%"
                    gpa_raw = re.search(r"(\d+\.?\d*)\s*/\s*(\d+\.?\d*)", stripped)
                    if gpa_raw:
                        current["gpa"] = gpa_raw.group(0)
                    pct_match = re.search(r"(\d+\.?\d*)\s*%", stripped)
                    if pct_match and not current.get("gpa"):
                        current["gpa"] = pct_match.group(0) + "%"

                # If line has field of study keywords
                field_keywords = [
                    r"(?i)(?:computer|information|software|engineering|"
                    r"electronics|mechanical|civil|electrical|chemical|"
                    r"biotechnology|mathematics|physics|chemistry|biology|"
                    r"business|commerce|economics|finance|marketing|management|"
                    r"data\s+science|artificial|machine\s+learning|"
                    r"cyber|communication|design|architecture|education)",
                ]
                if re.search(field_keywords[0], stripped):
                    current["field"] = stripped

        if current:
            entries.append(current)

        # Clean institution fields: remove trailing years, GPA, commas
        for entry in entries:
            if entry.get("institution"):
                inst = entry["institution"]
                # Remove trailing year range (e.g. ", 2016 - 2020")
                inst = re.sub(r",?\s*\d{4}\s*[-–—]+\s*\d{4}\s*$", "", inst)
                # Remove trailing year (e.g. "2020")
                inst = re.sub(r",?\s*\d{4}\s*$", "", inst)
                # Remove trailing GPA (e.g. "| 3.7 GPA")
                inst = re.sub(r"[|,]\s*(?:gpa|cgpa|grade|percentage).*$", "", inst, flags=re.I)
                # Remove trailing " - " or commas
                inst = inst.strip().rstrip(",- \t")
                entry["institution"] = inst

        # Clean empty entries
        entries = [
            e for e in entries if any(v.strip() for v in e.values())
        ]

        return entries

    @staticmethod
    def _find_degree(text: str) -> str:
        """Return the first recognised degree string from text."""
        for pat in DEGREE_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(0).strip()
        return ""

    @staticmethod
    def _remove_degree(text: str) -> str:
        """Strip the recognised degree from text and return the remainder."""
        for pat in DEGREE_PATTERNS:
            text = pat.sub("", text, count=1)
        return text.strip().lstrip(",").strip()

    # ------------------------------------------------------------------
    # Work experience
    # ------------------------------------------------------------------

    def _extract_experience(
        self,
        text: str,
        sections: Dict[str, Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        """Extract work experience entries."""
        if "experience" in sections:
            section_text = self._get_section_text(text, sections, "experience")
        else:
            section_text = self._get_non_education_experience_text(text, sections)

        if not section_text.strip():
            return []

        return self._parse_experience_entries(section_text)

    def _get_non_education_experience_text(
        self,
        text: str,
        sections: Dict[str, Tuple[int, int]],
    ) -> str:
        """Fallback: use the entire text minus education sections for search."""
        if "experience" in sections:
            return self._get_section_text(text, sections, "experience")

        # Use all text but try to find date ranges as experience markers
        return text

    def _parse_experience_entries(self, section_text: str) -> List[Dict[str, Any]]:
        """Parse individual experience entries from a section."""
        lines = section_text.split("\n")
        entries: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {}
        in_bullets = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if "title" in current:
                    entries.append(current)
                current = {}
                in_bullets = False
                continue

            # Check if line contains a date range (start of a new entry)
            date_match = DATE_RANGE_PATTERN.search(stripped)

            if date_match and not current:
                # New entry starting with a date
                current = {
                    "title": "",
                    "company": "",
                    "duration": date_match.group(0).strip(),
                    "responsibilities": [],
                }
                # Try to extract title and company from this line.
                # Strip trailing delimiters that may remain after date removal.
                before_date = stripped[: date_match.start()].strip().rstrip("|·•,;- \t")
                current = self._parse_title_company(before_date, current)
            elif date_match and current:
                # This entry has a date but we're mid-bullet — close and start new
                if "title" in current and current.get("responsibilities"):
                    entries.append(current)
                current = {
                    "title": "",
                    "company": "",
                    "duration": date_match.group(0).strip(),
                    "responsibilities": [],
                }
                before_date = stripped[: date_match.start()].strip().rstrip("|·•,;- \t")
                current = self._parse_title_company(before_date, current)
            elif stripped.startswith("-") or stripped.startswith("•") or stripped.startswith("*"):
                # Bullet point / responsibility
                in_bullets = True
                if not current:
                    current = {
                        "title": "",
                        "company": "",
                        "duration": "",
                        "responsibilities": [],
                    }
                bullet = stripped.lstrip("-•* ").strip()
                if bullet:
                    current.setdefault("responsibilities", []).append(bullet)
            elif in_bullets:
                # Continuation of a bullet
                if current.get("responsibilities"):
                    current["responsibilities"][-1] += " " + stripped
            else:
                # Could be title/company line before any date
                if not current:
                    current = {
                        "title": "",
                        "company": "",
                        "duration": "",
                        "responsibilities": [],
                    }
                # Check for ORG entities via spaCy (skip; too slow per line)
                # Instead, separate by common delimiters
                current = self._parse_title_company(stripped, current)

        if current and current.get("title"):
            entries.append(current)

        # Second pass: try to use spaCy NER to identify companies
        # in entries that don't have one yet
        for entry in entries:
            if not entry.get("company") and entry.get("title"):
                # Use the first part of the title/company line
                pass  # already handled in _parse_title_company

        return entries

    @staticmethod
    def _parse_title_company(
        line: str,
        current: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Attempt to split a line into job title and company.
        Common delimiters: |  ·  -  at  @  (parentheses)
        """
        if not line:
            return current

        # Pattern 1: "Title | Company" or "Title · Company"
        delim_match = re.split(r"\s*[|·•]\s*", line, maxsplit=1)
        if len(delim_match) == 2:
            part_a, part_b = delim_match[0].strip(), delim_match[1].strip()
            # Guess which is title vs company
            if re.search(r"(?i)(?:engineer|developer|manager|analyst|designer|"
                         r"consultant|lead|architect|intern|associate|director|"
                         r"head|officer|specialist|coordinator|administrator)", part_a):
                current["title"] = part_a
                current["company"] = part_b
            else:
                current["company"] = part_a
                current["title"] = part_b
            return current

        # Pattern 2: "Title at Company" or "Title @ Company"
        at_match = re.split(r"\s+(?:at|@)\s+", line, maxsplit=1)
        if len(at_match) == 2:
            current["title"] = at_match[0].strip()
            current["company"] = at_match[1].strip()
            return current

        # Pattern 3: "Company — Title" or "Company - Title"
        dash_match = re.split(r"\s+[-–—]+\s+", line, maxsplit=1)
        if len(dash_match) == 2:
            part_a, part_b = dash_match[0].strip(), dash_match[1].strip()
            if re.search(r"(?i)(?:engineer|developer|manager|analyst|designer|"
                         r"consultant|lead|architect|intern|associate|director)", part_b):
                current["company"] = part_a
                current["title"] = part_b
            else:
                current["title"] = part_a
                current["company"] = part_b
            return current

        # Pattern 4: "Title (Company)"
        paren_match = re.match(r"^(.+?)\s*\((.+?)\)\s*$", line)
        if paren_match:
            current["title"] = paren_match.group(1).strip()
            current["company"] = paren_match.group(2).strip()
            return current

        # Fallback: if it looks like a job title, set as title
        if re.search(
            r"(?i)(?:engineer|developer|manager|analyst|designer|consultant|"
            r"lead|architect|intern|associate|director|head|officer|specialist)",
            line,
        ):
            current["title"] = line
        else:
            # Could be a company name on its own
            current["company"] = line

        return current

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def _extract_projects(
        self,
        text: str,
        sections: Dict[str, Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        """Extract project entries."""
        if "projects" in sections:
            section_text = self._get_section_text(text, sections, "projects")
        else:
            section_text = ""

        if not section_text.strip():
            return []

        return self._parse_project_entries(section_text)

    def _parse_project_entries(self, section_text: str) -> List[Dict[str, Any]]:
        """Parse individual project entries."""
        lines = section_text.split("\n")
        entries: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {}

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current and current.get("name"):
                    entries.append(current)
                    current = {}
                continue

            # Check for a project name pattern
            # "Project Name:" or "Project Name –" or bullet with name
            is_project_start = False

            # Pattern 1: "Project Name: description" or "Project Name – description"
            name_match = re.match(r"^(?:[•\-*]\s*)?(.+?)\s*[:–—]\s*", stripped)
            # Pattern 2: "Project Name (Tech1, Tech2)" — parens containing tech keywords
            tech_paren_match = re.match(
                r"^(?:[•\-*]\s*)?(.+?)\s*\((.+?)\)\s*$", stripped
            )
            # Pattern 3: Plain project name on its own line (starts with a bullet or stands alone,
            #            not a tech line, not a date range)
            bare_name_match = (
                not name_match
                and not tech_paren_match
                and not re.match(r"^[-•*]\s*", stripped)  # not a bullet continuation
                and len(stripped.split()) >= 2
                and len(stripped) <= 80
                and not re.search(r"(?i)(?:technolog(?:y|ies)|tech[- ]stack|tools?\s*used|built\s+(?:with|using))", stripped)
                and not DATE_RANGE_PATTERN.search(stripped)
                and not any(re.fullmatch(p[0], stripped) for p in SECTION_PATTERNS)
            )

            tech_match = re.search(
                r"(?i)(?:technolog(?:y|ies)|tech[- ]stack|tools?\s*used|"
                r"built\s+(?:with|using))[\s:]+(.+?)$",
                stripped,
            )

            if name_match and not tech_paren_match:
                if current and current.get("name"):
                    entries.append(current)
                current = {
                    "name": name_match.group(1).strip(),
                    "description": "",
                    "technologies": [],
                }
                # What follows the colon/dash could be description, tech list, or both
                after_colon = stripped[name_match.end():].strip()
                if after_colon:
                    # If it has tech keywords, parse as technologies
                    if re.search(r"(?i)(?:python|java|react|node|django|flask|tensorflow|aws|docker|kubernetes|angular|vue|spring|mongo|postgres|redis|typescript|javascript|html|css|git|linux|sql)", after_colon):
                        current["technologies"] = self._split_tech_list(after_colon)
                    else:
                        current["description"] = after_colon
                is_project_start = True

            elif tech_paren_match and not name_match:
                # "Project Name (Tech1, Tech2)"
                if current and current.get("name"):
                    entries.append(current)
                proj_name = tech_paren_match.group(1).strip()
                tech_part = tech_paren_match.group(2).strip()
                current = {
                    "name": proj_name,
                    "description": "",
                    "technologies": self._split_tech_list(tech_part),
                }
                is_project_start = True

            elif bare_name_match:
                # Standalone project name line (no colon, no parens)
                if current and current.get("name"):
                    entries.append(current)
                current = {
                    "name": stripped.strip(),
                    "description": "",
                    "technologies": [],
                }
                is_project_start = True

            if tech_match:
                tech_str = tech_match.group(1).strip()
                if not current:
                    current = {"name": "", "description": "", "technologies": []}
                current["technologies"] = self._split_tech_list(tech_str)
                # Remove the tech line from description if it was part of current
                continue

            if not is_project_start and current.get("name") is not None:
                # Continuation — description or tech
                if current["description"]:
                    current["description"] += " " + stripped
                else:
                    current["description"] = stripped

        if current and current.get("name"):
            entries.append(current)

        return entries

    @staticmethod
    def _split_tech_list(text: str) -> List[str]:
        """Split a comma/slash-separated technology list."""
        items = re.split(r"[,/•|;]\s*", text)
        return [item.strip() for item in items if item.strip() and len(item.strip()) > 1]

    # ------------------------------------------------------------------
    # Certifications
    # ------------------------------------------------------------------

    def _extract_certifications(
        self,
        text: str,
        sections: Dict[str, Tuple[int, int]],
    ) -> List[Dict[str, str]]:
        """Extract certification entries."""
        if "certifications" in sections:
            section_text = self._get_section_text(text, sections, "certifications")
        else:
            section_text = ""

        if not section_text.strip():
            return []

        certs: List[Dict[str, str]] = []
        lines = section_text.split("\n")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Remove bullet markers
            clean = stripped.lstrip("-•* ").strip()

            # Try to extract issuing organisation
            org_match = re.search(
                r"(?:issued\s+(?:by|from)|-|–|•|\[)\s*(.+?)$", clean, re.I
            )
            org = ""
            if org_match:
                org = org_match.group(1).strip()
                name = clean[: org_match.start()].strip()
            else:
                name = clean

            # Skip lines that look like section headers
            if any(re.fullmatch(p[0], name) for p in SECTION_PATTERNS):
                continue

            if len(name) > 3:
                certs.append({"name": name, "issuer": org})

        return certs

    # ------------------------------------------------------------------
    # Languages
    # ------------------------------------------------------------------

    def _extract_languages(
        self,
        text: str,
        sections: Dict[str, Tuple[int, int]],
    ) -> List[Dict[str, str]]:
        """Extract language entries."""
        if "languages" in sections:
            section_text = self._get_section_text(text, sections, "languages")
        else:
            section_text = ""

        if not section_text.strip():
            return []

        lines = section_text.split("\n")
        languages: List[Dict[str, str]] = []

        # Known languages for detection (spoken)
        known_languages = [
            "English", "Spanish", "French", "German", "Italian", "Portuguese",
            "Russian", "Chinese", "Mandarin", "Japanese", "Korean", "Arabic",
            "Hindi", "Bengali", "Urdu", "Turkish", "Dutch", "Polish",
            "Swedish", "Norwegian", "Danish", "Finnish", "Greek", "Hebrew",
            "Thai", "Vietnamese", "Indonesian", "Malay", "Filipino",
            "Czech", "Romanian", "Hungarian", "Ukrainian", "Tamil", "Telugu",
            "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi",
            "Persian", "Swahili", "Catalan", "Serbian", "Croatian", "Bulgarian",
            "Slovak", "Slovenian", "Lithuanian", "Latvian", "Estonian",
        ]

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            clean = stripped.lstrip("-•* ").strip()

            # Try to extract language and proficiency
            # Pattern: "Language (Proficiency)" or "Language - Proficiency"
            prof_match = re.match(
                r"^(.+?)\s*[(-]\s*(Native|Fluent|Advanced|Intermediate|"
                r"Basic|Beginner|Proficient|Bilingual|Professional)"
                r"\s*[)-]?\s*$",
                clean,
                re.I,
            )
            if prof_match:
                name = prof_match.group(1).strip()
                proficiency = prof_match.group(2).strip()
            else:
                # Just the language name
                name = clean
                proficiency = ""

            # Check against known languages or accept as-is if likely a language
            is_known = name.strip().lower() in [l.lower() for l in known_languages]
            if is_known or len(name.strip()) < 30:
                lang_name = name.strip()
                # Capitalise properly
                for known in known_languages:
                    if lang_name.lower() == known.lower():
                        lang_name = known
                        break
                languages.append({"name": lang_name, "proficiency": proficiency})

        return languages

    # ------------------------------------------------------------------
    # Empty result helper
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """Return a skeleton result dict with default values."""
        return {
            "personal_info": {
                "name": "",
                "email": "",
                "phone": "",
                "linkedin": "",
                "github": "",
                "portfolio": "",
                "location": "",
            },
            "summary": "",
            "skills": {},
            "education": [],
            "experience": [],
            "projects": [],
            "certifications": [],
            "languages": [],
        }
