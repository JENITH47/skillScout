"""
SkillScout v2.0 — Ranking & Feedback Engine
=============================================
Extracts features from resume text + JD, runs them through the stacking
ensemble, and generates ranked results with actionable feedback.
"""

import os
import re
import joblib
import pdfplumber
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from sentence_transformers import SentenceTransformer
from inference_pipeline import InferencePipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────

def pdf_to_text(path: str) -> str:
    """Extract text from a PDF, falling back to PyPDF2 if pdfplumber fails."""
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[pdfplumber] Failed on {path}: {e}")

    if not text.strip():
        try:
            import PyPDF2
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
        except Exception:
            pass

    return text.strip()


# ─────────────────────────────────────────────
# TEXT NORMALIZATION
# ─────────────────────────────────────────────

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower()
    # Fix split years like "20 24" → "2024"
    s = re.sub(r'\b(19|20)\s+(\d{2})\b', r'\1\2', s)
    # Normalize dashes
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", s.strip())


def tokenize_text(s: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9+#.]+", normalize_text(s)) if len(t) > 1]


def assess_text_quality(text: str) -> Dict[str, float]:
    """Estimate how trustworthy extracted text is before downstream scoring."""
    raw_text = text or ""
    normalized = normalize_text(raw_text)
    tokens = tokenize_text(normalized)
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    alpha_ratio = sum(ch.isalpha() for ch in raw_text) / max(len(raw_text), 1)
    line_count = raw_text.count("\n") + (1 if raw_text else 0)

    section_patterns = [
        r"\bexperience\b",
        r"\bskills?\b",
        r"\beducation\b",
        r"\bprojects?\b",
        r"\bsummary\b",
        r"\bcertifications?\b",
    ]
    section_hits = sum(1 for pat in section_patterns if re.search(pat, normalized))

    confidence = 0.0
    confidence += min(len(normalized) / 1800.0, 1.0) * 0.40
    confidence += min(len(tokens) / 280.0, 1.0) * 0.25
    confidence += min(section_hits / 4.0, 1.0) * 0.20
    confidence += min(max(unique_ratio - 0.15, 0.0) / 0.35, 1.0) * 0.10
    confidence += min(max(alpha_ratio - 0.35, 0.0) / 0.45, 1.0) * 0.05
    confidence = float(max(0.0, min(1.0, confidence)))

    return {
        "char_count": float(len(normalized)),
        "token_count": float(len(tokens)),
        "line_count": float(line_count),
        "unique_token_ratio": float(round(unique_ratio, 4)),
        "alpha_ratio": float(round(alpha_ratio, 4)),
        "section_hits": float(section_hits),
        "extraction_confidence": float(round(confidence, 4)),
    }


# ─────────────────────────────────────────────
# EXPERIENCE EXTRACTION (robust)
# ─────────────────────────────────────────────

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_PRESENT_WORDS = {"present", "current", "ongoing", "till date", "now", "today"}


def _parse_date_token(tok: str):
    """Return (year, month) from a date token, or (None, None)."""
    if not tok:
        return None, None
    t = tok.strip().lower()

    if any(x in t for x in _PRESENT_WORDS):
        now = datetime.now()
        return now.year, now.month

    # MM/YYYY or MM-YYYY
    m = re.match(r"(\d{1,2})[/\-](\d{4})", t)
    if m:
        return int(m.group(2)), int(m.group(1))

    # MonthName YYYY or MonthName-YYYY
    m = re.match(r"([a-z]{3})[a-z]*[^a-z0-9]*(\d{4})", t)
    if m:
        return int(m.group(2)), _MONTH_MAP.get(m.group(1), 1)

    # Just YYYY
    m = re.match(r"(\d{4})$", t)
    if m:
        return int(m.group(1)), 1

    return None, None


def _extract_date_ranges(text: str) -> List[Tuple[int, int]]:
    """Extract normalized month ranges from resume text."""
    months_re = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    sep_re = r"\s*(?:-|to)\s*"
    present_re = r"present|now|current|ongoing|till\s*date|today"

    patterns = [
        re.compile(
            rf"({months_re}[^a-z0-9]*\d{{4}}|\d{{1,2}}[/\-]\d{{4}}|\d{{4}})"
            rf"{sep_re}"
            rf"({months_re}[^a-z0-9]*\d{{4}}|\d{{1,2}}[/\-]\d{{4}}|\d{{4}}|{present_re})",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(20\d{{2}}|19\d{{2}}){sep_re}(20\d{{2}}|19\d{{2}}|{present_re})\b",
            re.IGNORECASE,
        ),
    ]

    ranges: Set[Tuple[int, int]] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            sy, sm = _parse_date_token(match.group(1))
            ey, em = _parse_date_token(match.group(2))
            if sy is None or ey is None:
                continue
            start_idx = sy * 12 + sm
            end_idx = ey * 12 + em
            if 0 < (end_idx - start_idx) < 600:
                ranges.add((start_idx, end_idx))
    return sorted(ranges)


def _sum_merged_month_ranges(ranges: List[Tuple[int, int]]) -> int:
    """Merge overlapping ranges so overlapping jobs are not double-counted."""
    if not ranges:
        return 0

    merged: List[List[int]] = [[ranges[0][0], ranges[0][1]]]
    for start_idx, end_idx in ranges[1:]:
        last = merged[-1]
        if start_idx <= last[1]:
            last[1] = max(last[1], end_idx)
        else:
            merged.append([start_idx, end_idx])

    return int(sum(end_idx - start_idx for start_idx, end_idx in merged))


def extract_experience_years_from_text(resume_text: str) -> float:
    """Extract total years of experience from resume text."""
    text = normalize_text(resume_text)

    # 1. Try explicit "X years of experience" statements
    explicit_patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:year|yr|yrs|years)\s+(?:of\s+)?experience",
        r"experience\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*\+?\s*(?:year|yr|yrs|years)",
        r"total\s+experience[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:year|yr|yrs|years)",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:year|yr|yrs|years)\s+(?:of\s+)?(?:relevant|professional|industry)\s+experience",
    ]
    explicit_years = 0.0
    for pat in explicit_patterns:
        for m in re.finditer(pat, text):
            val = float(m.group(1))
            if 0 < val < 50:
                explicit_years = max(explicit_years, val)
    if explicit_years > 0:
        return explicit_years

    # 2. Parse date ranges: "Month YYYY - Month YYYY" or "Month YYYY - present"
    months_re = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    sep_re = r"\s*(?:-|–|—|to)\s*"
    present_re = r"present|now|current|ongoing|till\s*date|today"

    date_range_pat = re.compile(
        rf"({months_re}[^a-z0-9]*\d{{4}}|\d{{1,2}}[/\-]\d{{4}}|\d{{4}})"
        rf"{sep_re}"
        rf"({months_re}[^a-z0-9]*\d{{4}}|\d{{1,2}}[/\-]\d{{4}}|\d{{4}}|{present_re})",
        re.IGNORECASE,
    )

    total_months = 0
    for m in date_range_pat.finditer(text):
        sy, sm = _parse_date_token(m.group(1))
        ey, em = _parse_date_token(m.group(2))

        if sy and ey:
            dur = (ey - sy) * 12 + (em - sm)
            if 0 < dur < 600:  # sanity: max 50 years
                total_months += dur

    # 3. Also try standalone "YYYY - YYYY" or "YYYY-YYYY" patterns
    year_range_pat = re.compile(r"\b(20\d{2}|19\d{2})\s*[-–—]\s*(20\d{2}|19\d{2}|present|current|now|ongoing)\b", re.IGNORECASE)
    for m in year_range_pat.finditer(text):
        sy, sm = _parse_date_token(m.group(1))
        ey, em = _parse_date_token(m.group(2))
        if sy and ey:
            dur = (ey - sy) * 12 + (em - sm)
            if 0 < dur < 600 and dur not in range(max(0, total_months - 12), total_months + 12):
                # Only add if not already captured by the more specific pattern
                pass  # The detailed pattern above should cover this

    return min(round(total_months / 12.0, 2), 50.0)


def extract_required_experience_years_from_jd(jd_text: str) -> float:
    if not jd_text:
        return 0.0
    text = jd_text.lower()
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:\d+(?:\.\d+)?)?\s*\+?\s*(?:year|yr|yrs|years)",
        re.IGNORECASE,
    )
    max_years = 0.0
    for m in pattern.finditer(text):
        val = float(m.group(1))
        if 0 < val < 50:
            max_years = max(max_years, val)
    return max_years


def extract_required_education_level_from_jd(jd_text: str) -> int:
    """Infer the minimum education level requested by the JD."""
    if not jd_text:
        return 0

    text = normalize_text(jd_text)
    if re.search(r"\b(?:ph\.?d|doctor of philosophy|d\.phil|doctoral)\b", text):
        return 5
    if re.search(r"\b(?:master|msc|m\.sc|m\.tech|mtech|m\.s\b|mba|m\.eng|m\.phil)\b", text):
        return 4
    if re.search(r"\b(?:bachelor|bsc|b\.sc|b\.tech|btech|b\.e\b|b\.eng|b\.s\b|undergraduate degree)\b", text):
        return 3
    if re.search(r"\b(?:diploma|associate degree|associate)\b", text):
        return 2
    return 0


def extract_experience_years_from_text(resume_text: str) -> float:
    """Override with merged-range logic to avoid double-counting overlapping jobs."""
    text = normalize_text(resume_text)

    explicit_patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:year|yr|yrs|years)\s+(?:of\s+)?experience",
        r"experience\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*\+?\s*(?:year|yr|yrs|years)",
        r"total\s+experience[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:year|yr|yrs|years)",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:year|yr|yrs|years)\s+(?:of\s+)?(?:relevant|professional|industry)\s+experience",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:year|yr|yrs|years)\s+(?:in|with)\b",
    ]
    explicit_years = 0.0
    for pat in explicit_patterns:
        for match in re.finditer(pat, text):
            val = float(match.group(1))
            if 0 < val < 50:
                explicit_years = max(explicit_years, val)

    duration_years = 0.0
    duration_patterns = [
        re.compile(
            r"(\d+(?:\.\d+)?)\s*(?:year|yr|yrs|years)\s*(?:and)?\s*(\d+)\s*(?:month|months|mo)\b",
            re.IGNORECASE,
        ),
        re.compile(r"(\d+)\s*(?:month|months|mo)\s+(?:of\s+)?experience", re.IGNORECASE),
    ]
    for pattern in duration_patterns:
        for match in pattern.finditer(text):
            if len(match.groups()) == 2:
                years = float(match.group(1))
                months = float(match.group(2))
                duration_years = max(duration_years, years + (months / 12.0))
            else:
                duration_years = max(duration_years, float(match.group(1)) / 12.0)

    total_months = _sum_merged_month_ranges(_extract_date_ranges(text))
    derived_years = min(round(total_months / 12.0, 2), 50.0)

    if explicit_years > 0 or duration_years > 0:
        return min(round(max(explicit_years, duration_years), 2), 50.0)
    return derived_years


# ─────────────────────────────────────────────
# SKILL VOCABULARIES
# ─────────────────────────────────────────────

# Skills that need word-boundary matching (short/ambiguous terms)
_BOUNDARY_SKILLS = {
    "c", "r", "go", "js", "ai", "bi", "ml", "dl", "qa", "ui",
    "ux", "sql", "css", "xml", "php", "git", "iot", "svm", "knn",
    "gru", "rnn", "cnn", "ann", "pca", "lda", "qda", "gpt", "rag",
    "etl", "nlp", "ocr", "nfc", "ble", "qr", "drf", "wsl", "svn",
    "gan", "hmm", "lstm", "bert", "rest", "html", "java", "oop",
    "dart", "rust",
}

CS_SKILLS = {
    # Programming languages
    "python", "c++", "c", "java", "javascript", "php", "html", "css",
    "html5", "css3", "xml", "oop", "assembly language",
    "c#", "kotlin", "r", "elixir", "typescript", "vba", "pl/sql",
    "c/c++", "golang", "go", "rust", "dart", "scala", "perl", "ruby",
    "swift", "objective-c",

    # Web / Frontend
    "web development", "react", "reactjs", "react native", "angular",
    "vue.js", "vuejs", "bootstrap", "jquery", "next.js", "nextjs",
    "tailwind", "sass", "webpack", "mern stack", "mean stack",

    # Mobile
    "android studio", "android development", "android jetpack",
    "flutter", "react native", "ios development", "xcode",

    # Game / AR-VR
    "unity", "unity 3d", "game development", "unreal engine",
    "ar/vr", "augmented reality", "virtual reality", "blender",

    # Backend / APIs
    "django", "flask", "rest api", "restful api", "spring boot",
    "node.js", "express.js", "fastapi", "fast api", "drf",
    ".net core", ".net mvc", "laravel", "ruby on rails",

    # Databases
    "mysql", "postgresql", "sqlite", "mongodb", "redis",
    "sql server", "elasticsearch", "nosql", "cassandra",
    "dynamodb", "neo4j", "firebase",

    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "devops", "mlops",
    "ci/cd", "jenkins", "terraform", "ansible", "apache airflow",
    "microservices", "serverless",

    # OS / Networking
    "linux", "unix", "windows server", "tcp/ip", "networking",
    "cybersecurity", "information security", "firewalls",

    # AI / ML / DL
    "machine learning", "deep learning", "nlp",
    "natural language processing", "computer vision",
    "neural networks", "cnn", "rnn", "lstm", "gru",
    "supervised learning", "unsupervised learning",
    "reinforcement learning", "generative ai",
    "transformers", "gpt", "bert", "llm",
    "artificial intelligence", "ai agents",

    # ML Frameworks & Tools
    "tensorflow", "keras", "pytorch", "scikit-learn",
    "xgboost", "lightgbm", "catboost",
    "huggingface", "hugging face", "openai api",
    "langchain", "llamaindex", "rag",
    "pinecone", "chromadb", "faiss", "vector databases",

    # CV / OCR
    "opencv", "image processing", "image classification", "ocr",
    "tesseract", "yolo",

    # Dev Tools
    "git", "github", "gitlab", "bitbucket", "svn",
    "vs code", "jupyter", "pycharm", "postman",

    # Algorithms & DS
    "data structures", "algorithms", "svm", "knn", "naive bayes",
    "decision tree", "random forest", "gradient boosting",
    "k-means", "dbscan", "pca", "lda",
    "linear regression", "logistic regression",
    "arima", "time series",

    # Other
    "regex", "web scraping", "api development",
    "software testing", "agile", "scrum",
    "data engineering", "etl", "big data", "iot",
    "blockchain", "cryptography",
}

DATA_SKILLS = {
    # Core data / analytics
    "data analysis", "data analytics", "data visualization",
    "business intelligence", "data mining", "data science",
    "statistical analysis", "statistics", "data wrangling",
    "data preprocessing", "data cleaning", "data modeling",
    "exploratory data analysis", "feature engineering",
    "feature selection", "predictive modeling",
    "time series analysis", "time series forecasting",

    # Tools
    "excel", "microsoft excel", "google sheets",
    "vlookup", "pivot tables",
    "sql", "power bi", "tableau", "looker",
    "google data studio", "qlik",
    "pandas", "numpy", "matplotlib", "seaborn",
    "scipy", "plotly", "altair",
    "jupyter notebook", "anaconda", "google colab",

    # Stats software
    "spss", "stata", "sas", "minitab",

    # Big data
    "pyspark", "spark", "apache hadoop", "hive",
    "kafka", "airflow", "databricks",

    # Data scraping
    "web scraping", "beautiful soup", "scrapy", "selenium",
}

SOFT_SKILLS = {
    "project management", "time management", "team leadership",
    "leadership", "communication", "problem solving",
    "critical thinking", "stakeholder management",
    "team management", "change management",
    "requirement analysis", "requirement gathering",
    "documentation", "strategic planning",
    "presentation skills", "collaboration",
    "decision making", "mentoring",
}

_MANDATORY_SENTENCE_HINTS = {
    "must",
    "required",
    "mandatory",
    "minimum",
    "qualification",
    "qualifications",
    "need to",
    "needs to",
    "should have",
    "strong in",
    "proficient in",
    "hands-on",
}

_EVIDENCE_ACTION_HINTS = {
    "built", "build", "developed", "develop", "implemented", "implement",
    "designed", "design", "created", "create", "deployed", "deploy",
    "integrated", "integrate", "used", "using", "worked", "work",
    "trained", "training", "optimized", "fine-tuned", "fine tuned",
}

_EXPERIENCE_SECTION_HINTS = {
    "experience", "employment", "work history", "professional experience",
    "career history", "internship", "role", "responsibilities",
}

_PROJECT_SECTION_HINTS = {
    "project", "projects", "academic projects", "client projects", "case study",
}

_SKILL_SECTION_HINTS = {
    "skills", "technical skills", "tools", "toolkit", "technologies",
    "libraries", "coding skill", "core competencies",
}

_JD_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "such",
    "have", "has", "will", "your", "our", "you", "are", "their", "real",
    "world", "applications", "application", "candidate", "ideal", "basic",
    "understanding", "familiarity", "knowledge", "experience", "strong",
    "proficiency", "hands", "hand", "skills", "skill", "concepts",
    "concept", "good", "team", "ability", "work", "build", "develop",
}

_SKILL_ALIASES = {
    "artificial intelligence": [r"\bai\b", r"artificial intelligence"],
    "machine learning": [r"\bml\b", r"machine learning"],
    "deep learning": [r"\bdl\b", r"deep learning"],
    "natural language processing": [r"natural language processing"],
    "nlp": [r"\bnlp\b"],
    "llm": [r"\bllm\b", r"large language models?", r"large language model based"],
    "langchain": [r"langchain", r"lang chain"],
    "scikit-learn": [r"scikit[\s-]?learn", r"sklearn"],
    "tensorflow": [r"tensorflow", r"\btf\b"],
    "pytorch": [r"pytorch", r"py[\s-]?torch", r"\btorch\b"],
    "rest api": [r"rest[\s-]?apis?", r"restful[\s-]apis?", r"restful services?"],
    "python": [r"\bpython\b", r"python programming"],
    "numpy": [r"\bnumpy\b"],
    "pandas": [r"\bpandas\b"],
    "docker": [r"\bdocker\b", r"containeri[sz]ation"],
    "generative ai": [r"generative ai", r"gen ai"],
}


def _skill_in_text(skill: str, text: str) -> bool:
    """Check if a skill appears in text while avoiding substring false positives."""
    aliases = _SKILL_ALIASES.get(skill, [])
    for alias_pattern in aliases:
        if re.search(alias_pattern, text):
            return True

    escaped = re.escape(skill)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\/", r"(?:/|\s+)")
    escaped = escaped.replace(r"\-", r"(?:-|\s+)")
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return bool(re.search(pattern, text))


def extract_skills_from_text(text: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Extract skills from text, returning (cs_skills, data_skills, soft_skills).
    Uses word-boundary matching for short/ambiguous terms.
    """
    text_n = normalize_text(text)

    cs_found = {s for s in CS_SKILLS if _skill_in_text(s, text_n)}
    data_found = {s for s in DATA_SKILLS if _skill_in_text(s, text_n)}
    soft_found = {s for s in SOFT_SKILLS if _skill_in_text(s, text_n)}

    return cs_found, data_found, soft_found


def extract_jd_skill_sections(jd_text: str) -> Dict[str, str]:
    """Split a JD into rough skill sections so required and optional skills are not mixed."""
    text = jd_text or ""
    patterns = {
        "required": r"(required skills?|must have skills?|minimum qualifications?)(.*?)(good to have skills?|preferred skills?|soft skills?|skills\s*:|$)",
        "preferred": r"(good to have skills?|preferred skills?)(.*?)(soft skills?|skills\s*:|$)",
        "soft": r"(soft skills?)(.*?)(skills\s*:|$)",
        "tagged": r"(skills\s*:)(.*)$",
    }

    sections = {"required": "", "preferred": "", "soft": "", "tagged": ""}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections[key] = normalize_text(match.group(2))
    return sections


def infer_requirement_weight(text: str) -> float:
    """Map JD wording to requirement strength."""
    line = normalize_text(text)
    if any(phrase in line for phrase in ["strong proficiency", "must have", "required", "hands-on experience", "hands on experience"]):
        return 1.0
    if any(phrase in line for phrase in ["proficiency in", "expertise in", "solid understanding"]):
        return 0.90
    if any(phrase in line for phrase in ["knowledge of", "experience with", "experience in"]):
        return 0.80
    if any(phrase in line for phrase in ["basic understanding", "familiarity with", "exposure to"]):
        return 0.55
    return 0.75


def extract_required_skill_group_specs_from_jd(jd_text: str, jd_skills: Set[str]) -> List[Dict]:
    """Build weighted requirement groups so softer JD wording is penalized less."""
    specs: List[Dict] = []
    raw_text = jd_text or ""

    required_match = re.search(
        r"(required skills?|must have skills?|minimum qualifications?)(.*?)(good to have skills?|preferred skills?|soft skills?|skills\s*:|$)",
        raw_text,
        re.IGNORECASE | re.DOTALL,
    )
    required_block = required_match.group(2) if required_match else ""

    title_context = raw_text
    for marker in ["Key Responsibilities", "Required Skills", "Must Have Skills"]:
        idx = title_context.lower().find(marker.lower())
        if idx != -1:
            title_context = title_context[:idx]
            break

    high_signal_title_skills = {
        "artificial intelligence",
        "machine learning",
        "python",
    }
    title_cs, title_data, _ = extract_skills_from_text(title_context)
    title_found = (title_cs | title_data) & high_signal_title_skills & jd_skills
    for skill in sorted(title_found):
        specs.append({"skills": {skill}, "weight": 0.90, "source": "title"})

    if not required_block:
        return specs

    raw_lines = [line.strip(" -\t\r") for line in required_block.splitlines() if line.strip()]
    if not raw_lines:
        raw_lines = [required_block]

    for raw_line in raw_lines:
        normalized_line = normalize_text(raw_line)
        if not normalized_line:
            continue

        cs_found, data_found, _ = extract_skills_from_text(normalized_line)
        line_skills = list(sorted((cs_found | data_found) & jd_skills))
        if not line_skills:
            continue

        weight = infer_requirement_weight(normalized_line)
        if " or " in normalized_line and len(line_skills) >= 2:
            specs.append({"skills": set(line_skills), "weight": weight, "source": normalized_line})
            continue

        if "such as" in normalized_line and len(line_skills) >= 2:
            specs.append({"skills": set(line_skills), "weight": weight, "source": normalized_line})
            continue

        for skill in line_skills:
            specs.append({"skills": {skill}, "weight": weight, "source": normalized_line})

    deduped_map: Dict[Tuple[str, ...], Dict] = {}
    for spec in specs:
        key = tuple(sorted(spec["skills"]))
        if not spec["skills"]:
            continue
        existing = deduped_map.get(key)
        if existing is None or float(spec["weight"]) > float(existing["weight"]):
            deduped_map[key] = {
                "skills": set(spec["skills"]),
                "weight": float(spec["weight"]),
                "source": spec["source"],
            }
    return list(deduped_map.values())


def extract_required_skill_groups_from_jd(jd_text: str, jd_skills: Set[str]) -> List[Set[str]]:
    """Build requirement groups so 'A or B' counts as one requirement unit."""
    return [set(spec["skills"]) for spec in extract_required_skill_group_specs_from_jd(jd_text, jd_skills)]


def summarize_missing_requirement_groups(
    requirement_groups: List[Set[str]],
    resume_skills: Set[str],
    skill_evidence_scores: Optional[Dict[str, float]] = None,
    threshold: float = 0.65,
) -> List[str]:
    """Create user-friendly missing labels for unresolved requirement groups."""
    missing_labels: List[str] = []
    for group in requirement_groups:
        if skill_evidence_scores:
            best_score = max(skill_evidence_scores.get(skill, 0.0) for skill in group)
            if best_score >= threshold:
                continue
        elif group & resume_skills:
            continue
        if len(group) == 1:
            missing_labels.append(next(iter(group)))
        else:
            missing_labels.append(" / ".join(sorted(group)))
    return missing_labels


def build_informative_jd_tokens(jd_text: str) -> Set[str]:
    """Extract informative JD tokens for lexical alignment scoring."""
    tokens = set(tokenize_text(jd_text))
    return {tok for tok in tokens if len(tok) > 2 and tok not in _JD_STOPWORDS}


def compute_hybrid_semantic_score(
    resume_text: str,
    jd_text: str,
    resume_emb: np.ndarray,
    jd_emb: np.ndarray,
    core_jd_skills: Set[str],
) -> float:
    """Combine embedding similarity with lexical alignment to avoid score saturation."""
    cos_sim = float(np.dot(resume_emb, jd_emb))
    embedding_component = max(0.0, min(1.0, (cos_sim - 0.985) / 0.015))

    jd_tokens = build_informative_jd_tokens(jd_text)
    resume_tokens = set(tokenize_text(resume_text))
    lexical_overlap = (len(jd_tokens & resume_tokens) / len(jd_tokens)) if jd_tokens else 0.0

    core_skill_overlap = (
        len(core_jd_skills & (extract_skills_from_text(resume_text)[0] | extract_skills_from_text(resume_text)[1]))
        / len(core_jd_skills)
    ) if core_jd_skills else 0.0

    hybrid = (
        0.25 * embedding_component
        + 0.45 * lexical_overlap
        + 0.30 * core_skill_overlap
    )
    return float(max(0.0, min(10.0, hybrid * 10.0)))


def infer_role_level(jd_text: str) -> str:
    """Infer role seniority from JD wording."""
    text = normalize_text(jd_text)
    if any(term in text for term in ["junior", "entry level", "entry-level", "fresher", "associate"]):
        return "junior"
    if any(term in text for term in ["senior", "lead", "principal", "staff"]):
        return "senior"
    return "mid"


def compute_role_fit_score(experience_years: float, role_level: str) -> float:
    """Estimate how well the candidate's experience fits the role seniority."""
    exp = max(0.0, float(experience_years))
    if role_level == "junior":
        if exp <= 4.0:
            return 1.0
        if exp <= 6.0:
            return 0.82
        if exp <= 9.0:
            return 0.62
        return 0.45
    if role_level == "senior":
        if exp >= 5.0:
            return 1.0
        if exp >= 3.0:
            return 0.75
        return 0.45
    if 1.0 <= exp <= 6.0:
        return 1.0
    if exp < 1.0:
        return 0.7
    return 0.8


def extract_skill_evidence_scores(resume_text: str, target_skills: Set[str]) -> Dict[str, float]:
    """Score how strongly each target skill is evidenced in the resume text."""
    lines = [line for line in resume_text.splitlines() if line and line.strip()]
    scores = {skill: 0.0 for skill in target_skills}
    current_section = "generic"

    for raw_line in lines:
        line = normalize_text(raw_line)
        if not line:
            continue

        if any(hint in line for hint in _PROJECT_SECTION_HINTS):
            current_section = "project"
        elif any(hint in line for hint in _EXPERIENCE_SECTION_HINTS):
            current_section = "experience"
        elif any(hint in line for hint in _SKILL_SECTION_HINTS):
            current_section = "skills"
        elif "summary" in line or "profile" in line or "objective" in line:
            current_section = "summary"

        matched_skills = [skill for skill in target_skills if _skill_in_text(skill, line)]
        if not matched_skills:
            continue

        action_present = any(hint in line for hint in _EVIDENCE_ACTION_HINTS)
        dense_skill_list = (line.count(",") >= 3) or (len(matched_skills) >= 3)

        base_score = 0.65
        if current_section == "skills":
            base_score = 0.55
        elif current_section == "summary":
            base_score = 0.72
        elif current_section == "experience":
            base_score = 0.85
        elif current_section == "project":
            base_score = 0.92

        if action_present:
            base_score = max(base_score, 0.88)
        if "years" in line or "experience" in line:
            base_score = max(base_score, 0.82)
        if dense_skill_list and current_section == "skills" and not action_present:
            base_score = min(base_score, 0.58)

        for skill in matched_skills:
            scores[skill] = max(scores[skill], min(base_score, 1.0))

    return scores


def extract_mandatory_skills_from_jd(jd_text: str, jd_skills: Set[str]) -> Set[str]:
    """Pull out the subset of JD skills that look like hard requirements."""
    text = normalize_text(jd_text)
    sections = extract_jd_skill_sections(jd_text)
    mandatory_skills: Set[str] = set()
    requirement_specs = extract_required_skill_group_specs_from_jd(jd_text, jd_skills)

    if requirement_specs:
        for spec in requirement_specs:
            if spec["weight"] >= 0.75:
                mandatory_skills |= spec["skills"]
        return mandatory_skills & jd_skills

    if sections["required"]:
        cs_found, data_found, _ = extract_skills_from_text(sections["required"])
        mandatory_skills |= (cs_found | data_found)

    if mandatory_skills:
        return mandatory_skills & jd_skills

    # If there is no explicit required section, fall back to strong requirement
    # wording in the main prose, but stop before optional / soft-skill sections.
    scan_text = text
    for marker in ["good to have skills", "preferred skills", "soft skills"]:
        idx = scan_text.find(marker)
        if idx != -1:
            scan_text = scan_text[:idx]
            break

    sentences = re.split(r"[\n.;]+", scan_text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if any(hint in sentence for hint in _MANDATORY_SENTENCE_HINTS):
            cs_found, data_found, _ = extract_skills_from_text(sentence)
            mandatory_skills |= (cs_found | data_found)

    if sections["tagged"]:
        cs_found, data_found, _ = extract_skills_from_text(sections["tagged"])
        mandatory_skills |= (cs_found | data_found)

    return mandatory_skills & jd_skills


def extract_core_jd_skills(jd_text: str, jd_skills: Set[str]) -> Set[str]:
    """Prefer required skills for matching; fall back to all JD skills when needed."""
    requirement_groups = extract_required_skill_groups_from_jd(jd_text, jd_skills)
    if requirement_groups:
        core_skills: Set[str] = set()
        for group in requirement_groups:
            core_skills |= group
        return core_skills

    sections = extract_jd_skill_sections(jd_text)
    if sections["required"]:
        cs_found, data_found, _ = extract_skills_from_text(sections["required"])
        required_skills = (cs_found | data_found) & jd_skills
        if required_skills:
            return required_skills

    mandatory_skills = extract_mandatory_skills_from_jd(jd_text, jd_skills)
    return mandatory_skills if mandatory_skills else jd_skills


# ─────────────────────────────────────────────
# EDUCATION EXTRACTION
# ─────────────────────────────────────────────

def extract_education_level_from_text(resume_text: str) -> int:
    """
    0 = unknown, 1 = high school, 2 = diploma, 3 = bachelor,
    4 = master, 5 = PhD, 6 = postdoc
    """
    text = normalize_text(resume_text)

    if re.search(r'\bpost\s*doc', text):
        return 6
    if re.search(r'\b(?:ph\.?d|doctor of philosophy|d\.phil)\b', text):
        return 5
    if re.search(r'\b(?:master|msc|m\.sc|m\.tech|mtech|m\.s\b|mba|m\.eng|m\.phil)\b', text):
        return 4
    if re.search(r'\b(?:bachelor|bsc|b\.sc|b\.tech|btech|b\.e\b|b\.eng|b\.s\b|bca|bba)\b', text):
        return 3
    if re.search(r'\b(?:diploma|associate)\b', text):
        return 2
    if re.search(r'\b(?:high school|intermediate|12th|hsc)\b', text):
        return 1
    return 0


# ─────────────────────────────────────────────
# BERT MODEL (singleton, cached)
# ─────────────────────────────────────────────

BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_bert_model: Optional[SentenceTransformer] = None
_bert_model_name: Optional[str] = None


def resolve_bert_model_name() -> str:
    """Prefer a local fine-tuned model when it is available."""
    candidates = [
        os.getenv("SKILLSCOUT_BERT_MODEL"),
        "bert-fine-tuned-triplet",
        "fine-tuned-bert",
        BERT_MODEL_NAME,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.exists(candidate) or candidate == BERT_MODEL_NAME:
            return candidate
    return BERT_MODEL_NAME


def load_bert() -> SentenceTransformer:
    global _bert_model, _bert_model_name
    if _bert_model is None:
        _bert_model_name = resolve_bert_model_name()
        print(f"Loading embedding model: {_bert_model_name}")
        _bert_model = SentenceTransformer(_bert_model_name)
    return _bert_model


def get_loaded_bert_model_name() -> str:
    global _bert_model_name
    if _bert_model_name is None:
        _bert_model_name = resolve_bert_model_name()
    return _bert_model_name


def get_bert_embeddings(resume_text: str, jd_text: str):
    """Compute BERT embeddings for resume and JD text. Returns (resume_emb, jd_emb)."""
    model = load_bert()
    # Truncate to first 2000 chars of normalized text
    resume_norm = normalize_text(resume_text)[:2000]
    jd_norm = normalize_text(jd_text)[:2000]
    # NOTE: Training used normalize_embeddings=True
    resume_emb = model.encode(resume_norm, convert_to_numpy=True, normalize_embeddings=True)
    jd_emb = model.encode(jd_norm, convert_to_numpy=True, normalize_embeddings=True)
    return resume_emb.astype(np.float32), jd_emb.astype(np.float32)


def bert_similarity_from_embeddings(resume_emb: np.ndarray, jd_emb: np.ndarray) -> float:
    """Compute BERT-based similarity score (0-10) from pre-computed embeddings."""
    cos_sim = float(np.dot(resume_emb, jd_emb))
    return (cos_sim + 1.0) / 2.0 * 10.0


# ─────────────────────────────────────────────
# JUSTIFICATION FEATURE EXTRACTION (at inference)
# ─────────────────────────────────────────────

def compute_justification_features(
    resume_text: str,
    jd_text: str,
    experience_years: float,
    required_experience: float,
    resume_skills: Set[str],
    jd_skills: Set[str],
) -> Dict[str, float]:
    """
    Compute justification features that were used during training.
    During training these came from HF dataset justification text.
    At inference we compute them from the actual resume/JD content.
    """
    # Experience penalty: candidate has less experience than required
    exp_gap = experience_years - required_experience
    just_exp_penalty = 1.0 if (required_experience > 0 and exp_gap < -1.0) else 0.0

    # Missing skill count: skills in JD but not in resume
    missing = jd_skills - resume_skills
    just_missing_skill_count = float(min(len(missing), 10))

    # Domain penalty: resume is in a very different domain than JD
    jd_lower = normalize_text(jd_text)
    resume_lower = normalize_text(resume_text)

    domain_keywords = {
        "data science": ["data scien", "machine learning", "deep learning", "neural"],
        "web development": ["web develop", "frontend", "backend", "full stack", "fullstack"],
        "mobile development": ["android", "ios", "mobile app", "flutter", "react native"],
        "devops": ["devops", "kubernetes", "docker", "ci/cd", "infrastructure"],
        "data analytics": ["data analy", "business intelligence", "power bi", "tableau"],
    }

    jd_domain = None
    for domain, keywords in domain_keywords.items():
        if any(kw in jd_lower for kw in keywords):
            jd_domain = domain
            break

    resume_domain = None
    for domain, keywords in domain_keywords.items():
        if any(kw in resume_lower for kw in keywords):
            resume_domain = domain
            break

    just_domain_penalty = 1.0 if (jd_domain and resume_domain and jd_domain != resume_domain) else 0.0

    return {
        "just_exp_penalty": just_exp_penalty,
        "just_missing_skill_count": just_missing_skill_count,
        "just_domain_penalty": just_domain_penalty,
    }


# ─────────────────────────────────────────────
# MICRO-MATCH COMPUTATION
# ─────────────────────────────────────────────

def compute_micro_match(skills_set: Set[str], micro_dict: Optional[Dict]) -> Tuple[int, float, float]:
    if not micro_dict:
        return 0, 0.0, 0.0
    matched_weight = 0.0
    total_weight = sum(float(w) for w in micro_dict.values())
    match_count = 0
    for crit, w in micro_dict.items():
        crit_lower = str(crit).lower()
        if any(crit_lower in s or s in crit_lower for s in skills_set):
            matched_weight += float(w)
            match_count += 1
    ratio = matched_weight / total_weight if total_weight > 0 else 0.0
    return match_count, matched_weight, ratio


# ─────────────────────────────────────────────
# JOB ROLE SUGGESTIONS
# ─────────────────────────────────────────────

_JOB_ROLE_MAP = {
    "Data Scientist": {"data science", "machine learning", "deep learning", "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn"},
    "ML Engineer": {"machine learning", "deep learning", "tensorflow", "pytorch", "mlops", "docker", "kubernetes"},
    "Data Analyst": {"data analysis", "sql", "excel", "power bi", "tableau", "pandas", "data visualization"},
    "Data Engineer": {"etl", "pyspark", "spark", "apache hadoop", "airflow", "data engineering", "sql"},
    "Full Stack Developer": {"react", "angular", "node.js", "django", "flask", "javascript", "html", "css", "mongodb", "postgresql"},
    "Backend Developer": {"django", "flask", "fastapi", "spring boot", "node.js", "express.js", "rest api", "sql", "mongodb"},
    "Frontend Developer": {"react", "angular", "vue.js", "javascript", "html", "css", "bootstrap", "typescript"},
    "Mobile App Developer": {"android development", "flutter", "react native", "ios development", "kotlin", "swift"},
    "DevOps Engineer": {"docker", "kubernetes", "aws", "azure", "gcp", "ci/cd", "terraform", "jenkins"},
    "Cloud Engineer": {"aws", "azure", "gcp", "docker", "kubernetes", "serverless", "terraform"},
    "NLP Engineer": {"nlp", "natural language processing", "bert", "gpt", "transformers", "huggingface", "langchain"},
    "LLM/GenAI Engineer": {"llm", "gpt", "langchain", "rag", "huggingface", "openai api", "vector databases", "generative ai"},
    "Cybersecurity Analyst": {"cybersecurity", "information security", "firewalls", "networking", "tcp/ip"},
    "QA / Test Engineer": {"software testing", "selenium", "qa", "agile", "ci/cd"},
    "Game Developer": {"unity", "unreal engine", "game development", "blender", "c++"},
}


def suggest_job_roles(all_skills: Set[str], top_n: int = 5) -> List[Tuple[str, float]]:
    """Suggest job roles based on skill overlap. Returns [(role, match_pct), ...]."""
    scores = []
    for role, role_skills in _JOB_ROLE_MAP.items():
        matched = all_skills & role_skills
        if role_skills:
            pct = len(matched) / len(role_skills)
            if pct > 0.15:  # at least ~15% match
                scores.append((role, round(pct * 100, 1)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


# ─────────────────────────────────────────────
# FEATURE BUILDING & SCORING
# ─────────────────────────────────────────────

def build_structured_features(
    resume_text: str,
    jd_text: str,
    micro_dict: Optional[Dict],
    college_tier: str,
) -> Tuple[Dict, Set[str], Set[str], Set[str], Set[str]]:
    """
    Build the structured feature dict from resume + JD text.
    Returns (features_dict, all_skills, cs_skills, data_skills, soft_skills).
    """
    cs_skills, data_skills, soft_skills = extract_skills_from_text(resume_text)
    all_skills = cs_skills | data_skills | soft_skills
    resume_quality = assess_text_quality(resume_text)

    # Extract JD skills for comparison
    jd_cs, jd_data, jd_soft = extract_skills_from_text(jd_text)
    jd_skills = jd_cs | jd_data | jd_soft
    requirement_specs = extract_required_skill_group_specs_from_jd(jd_text, jd_skills)
    requirement_groups = [set(spec["skills"]) for spec in requirement_specs]
    requirement_group_weights = [float(spec["weight"]) for spec in requirement_specs]
    core_jd_skills = extract_core_jd_skills(jd_text, jd_skills)
    mandatory_skills = extract_mandatory_skills_from_jd(jd_text, jd_skills)
    matched_skills = all_skills & core_jd_skills
    matched_mandatory_skills = all_skills & mandatory_skills
    skill_evidence_scores = extract_skill_evidence_scores(resume_text, core_jd_skills)
    group_evidence_scores = [max(skill_evidence_scores.get(skill, 0.0) for skill in group) for group in requirement_groups]
    matched_requirement_groups = sum(1 for score, weight in zip(group_evidence_scores, requirement_group_weights) if score >= max(0.55, min(0.80, weight - 0.05)))
    requirement_group_ratio = (
        (sum(score * weight for score, weight in zip(group_evidence_scores, requirement_group_weights)) / sum(requirement_group_weights))
        if group_evidence_scores else ((len(matched_skills) / len(core_jd_skills)) if core_jd_skills else 1.0)
    )

    edu_level = extract_education_level_from_text(resume_text)
    exp_years = extract_experience_years_from_text(resume_text)
    req_exp = extract_required_experience_years_from_jd(jd_text)
    req_edu = extract_required_education_level_from_jd(jd_text)
    role_level = infer_role_level(jd_text)
    role_fit_score = compute_role_fit_score(exp_years, role_level)

    micro_cnt, micro_w, micro_ratio = compute_micro_match(all_skills, micro_dict)

    # BERT embeddings (computed once)
    resume_emb, jd_emb = get_bert_embeddings(resume_text, jd_text)
    bert_sim = compute_hybrid_semantic_score(resume_text, jd_text, resume_emb, jd_emb, core_jd_skills)

    # Justification features (computed from actual content)
    just_feats = compute_justification_features(
        resume_text, jd_text, exp_years, req_exp, all_skills, jd_skills
    )

    features = {
        "college_tier": college_tier,
        "num_skills": len(all_skills),
        "cs_skill_count": len(cs_skills),
        "data_skill_count": len(data_skills),
        "education_level": edu_level,
        "experience_years": exp_years,
        "required_experience_years": req_exp,
        "experience_gap": exp_years - req_exp,
        "micro_match_weight": micro_w,
        "just_exp_penalty": just_feats["just_exp_penalty"],
        "just_missing_skill_count": just_feats["just_missing_skill_count"],
        "just_domain_penalty": just_feats["just_domain_penalty"],
        "bert_sim_score": bert_sim,
        "skill_match_ratio": requirement_group_ratio,
        "mandatory_skill_count": len(requirement_groups) if requirement_groups else len(mandatory_skills),
        "mandatory_skill_match_count": matched_requirement_groups if requirement_groups else len(matched_mandatory_skills),
        "mandatory_skill_match_ratio": requirement_group_ratio if requirement_groups else ((len(matched_mandatory_skills) / len(mandatory_skills)) if mandatory_skills else 1.0),
        "requirement_group_count": len(requirement_groups),
        "requirement_group_match_count": matched_requirement_groups,
        "requirement_group_match_ratio": requirement_group_ratio,
        "core_skill_evidence_ratio": (sum(skill_evidence_scores.values()) / len(skill_evidence_scores)) if skill_evidence_scores else 0.0,
        "role_level": role_level,
        "role_fit_score": role_fit_score,
        "required_education_level": req_edu,
        "education_gap": float(edu_level - req_edu) if req_edu else 0.0,
        "meets_experience_requirement": 1.0 if (req_exp <= 0 or exp_years + 0.25 >= req_exp) else 0.0,
        "meets_education_requirement": 1.0 if (req_edu <= 0 or edu_level >= req_edu) else 0.0,
        "text_quality_score": resume_quality["extraction_confidence"],
        "resume_char_count": resume_quality["char_count"],
        "resume_token_count": resume_quality["token_count"],
        "resume_section_hits": resume_quality["section_hits"],
        "embedding_model": get_loaded_bert_model_name(),
        # Store embeddings for pipeline
        "_resume_emb": resume_emb,
        "_jd_emb": jd_emb,
        # Store skill sets for feedback
        "_all_skills": all_skills,
        "_cs_skills": cs_skills,
        "_data_skills": data_skills,
        "_soft_skills": soft_skills,
        "_jd_skills": jd_skills,
        "_core_jd_skills": core_jd_skills,
        "_mandatory_skills": mandatory_skills,
        "_requirement_groups": requirement_groups,
        "_requirement_group_specs": requirement_specs,
        "_skill_evidence_scores": skill_evidence_scores,
    }
    return features, all_skills, cs_skills, data_skills, soft_skills


def compute_requirement_score(features: Dict) -> float:
    """Aggregate hard-requirement signals into a bounded 0-1 score."""
    components: List[float] = []

    req_exp = float(features.get("required_experience_years", 0.0) or 0.0)
    if req_exp > 0:
        exp_gap = float(features.get("experience_gap", 0.0) or 0.0)
        if exp_gap >= 0:
            components.append(1.0)
        else:
            components.append(max(0.0, 1.0 + (exp_gap / max(req_exp, 1.0))))

    req_edu = float(features.get("required_education_level", 0.0) or 0.0)
    if req_edu > 0:
        components.append(float(features.get("meets_education_requirement", 0.0) or 0.0))

    mandatory_count = float(features.get("mandatory_skill_count", 0.0) or 0.0)
    if mandatory_count > 0:
        components.append(float(features.get("mandatory_skill_match_ratio", 0.0) or 0.0))

    if not components:
        return 1.0
    return float(max(0.0, min(1.0, sum(components) / len(components))))


def rerank_prediction(model_score: float, features: Dict) -> Dict[str, float]:
    """Blend the trained model with robust inference-time signals."""
    semantic_score = float(features.get("bert_sim_score", 0.0) or 0.0)
    skill_match_ratio = float(features.get("skill_match_ratio", 0.0) or 0.0)
    mandatory_ratio = float(features.get("mandatory_skill_match_ratio", 1.0) or 1.0)
    requirement_group_ratio = float(features.get("requirement_group_match_ratio", mandatory_ratio) or mandatory_ratio)
    core_skill_evidence_ratio = float(features.get("core_skill_evidence_ratio", skill_match_ratio) or skill_match_ratio)
    role_fit_score = float(features.get("role_fit_score", 1.0) or 1.0)
    text_quality = float(features.get("text_quality_score", 0.0) or 0.0)

    heuristic_score = (
        0.24 * semantic_score
        + 0.33 * (requirement_group_ratio * 10.0)
        + 0.18 * (core_skill_evidence_ratio * 10.0)
        + 0.10 * (skill_match_ratio * 10.0)
        + 0.10 * (mandatory_ratio * 10.0)
        + 0.05 * (role_fit_score * 10.0)
        + 0.05 * (text_quality * 10.0)
    )
    blended_score = (0.35 * model_score) + (0.65 * heuristic_score)

    # Favor resumes that satisfy grouped required skills even if the legacy
    # regressor is conservative on newer JD styles.
    if requirement_group_ratio >= 0.70 and heuristic_score > model_score:
        blended_score += min(0.80, (heuristic_score - model_score) * 0.35)

    exp_gap = float(features.get("experience_gap", 0.0) or 0.0)
    req_exp = float(features.get("required_experience_years", 0.0) or 0.0)
    if req_exp > 0 and exp_gap < 0:
        blended_score += max(-1.25, exp_gap * 0.25)

    if float(features.get("required_education_level", 0.0) or 0.0) > 0 and not features.get("meets_education_requirement", 0.0):
        blended_score -= 0.50

    mandatory_count = float(features.get("mandatory_skill_count", 0.0) or 0.0)
    if mandatory_count > 0:
        blended_score -= (1.0 - mandatory_ratio) * 1.20

    requirement_group_count = float(features.get("requirement_group_count", 0.0) or 0.0)
    if requirement_group_count > 0:
        blended_score -= (1.0 - requirement_group_ratio) * 1.60

    # A resume that only mentions required tools in a flat skill list should rank
    # below one that shows hands-on evidence in projects or experience.
    blended_score -= max(0.0, 0.65 - core_skill_evidence_ratio) * 1.40

    # Lightly align experience band with role seniority, especially for junior JDs.
    blended_score -= max(0.0, 0.85 - role_fit_score) * 0.70

    blended_score *= (0.85 + (0.15 * text_quality))
    final_score = float(max(0.0, min(10.0, blended_score)))

    requirement_score = compute_requirement_score(features)
    signal_agreement = float(max(0.0, 1.0 - min(abs(model_score - heuristic_score) / 5.0, 1.0)))
    ranking_confidence = float(
        max(0.0, min(1.0, (0.50 * text_quality) + (0.25 * requirement_score) + (0.25 * signal_agreement)))
    )

    return {
        "heuristic_score": round(heuristic_score, 4),
        "final_score": round(final_score, 4),
        "requirement_score": round(requirement_score, 4),
        "signal_agreement": round(signal_agreement, 4),
        "ranking_confidence": round(ranking_confidence, 4),
    }


# ─────────────────────────────────────────────
# FEEDBACK GENERATION
# ─────────────────────────────────────────────

_EDU_LABELS = {
    0: "Not detected",
    1: "High School",
    2: "Diploma",
    3: "Bachelor's",
    4: "Master's",
    5: "PhD",
    6: "Post-Doctoral",
}


def generate_feedback(
    features: Dict,
    predicted_score: float,
    resume_text: str,
    jd_text: str,
) -> Dict:
    """Generate rich, multi-section feedback for a resume-JD pair."""

    all_skills = features.get("_all_skills", set())
    jd_skills = features.get("_jd_skills", set())
    core_jd_skills = features.get("_core_jd_skills", jd_skills)
    cs_skills = features.get("_cs_skills", set())
    data_skills = features.get("_data_skills", set())
    soft_skills = features.get("_soft_skills", set())
    mandatory_skills = features.get("_mandatory_skills", set())
    requirement_groups = features.get("_requirement_groups", [])
    requirement_group_specs = features.get("_requirement_group_specs", [])
    skill_evidence_scores = features.get("_skill_evidence_scores", {})

    matched_skills = all_skills & core_jd_skills
    missing_skills = core_jd_skills - all_skills
    extra_skills = all_skills - jd_skills
    high_priority_groups = [spec["skills"] for spec in requirement_group_specs if float(spec.get("weight", 0.0)) >= 0.75]
    missing_mandatory_skills = summarize_missing_requirement_groups(
        high_priority_groups,
        all_skills,
        skill_evidence_scores,
    ) if high_priority_groups else sorted(mandatory_skills - all_skills)

    exp_years = features["experience_years"]
    req_exp = features["required_experience_years"]
    edu_level = features["education_level"]
    bert_sim = features["bert_sim_score"]
    text_quality = float(features.get("text_quality_score", 0.0) or 0.0)
    ranking_confidence = float(features.get("ranking_confidence", 0.0) or 0.0)
    requirement_score = float(features.get("requirement_score", 1.0) or 1.0)
    role_fit_score = float(features.get("role_fit_score", 1.0) or 1.0)

    # Overall verdict
    if predicted_score >= 8.0:
        verdict = "🟢 Excellent Match"
        verdict_msg = "This candidate is a strong fit for the role."
    elif predicted_score >= 6.5:
        verdict = "🔵 Good Match"
        verdict_msg = "This candidate meets most of the requirements."
    elif predicted_score >= 5.0:
        verdict = "🟡 Moderate Match"
        verdict_msg = "This candidate has relevant skills but has gaps to address."
    else:
        verdict = "🔴 Weak Match"
        verdict_msg = "This candidate has significant gaps for this role."

    # Strengths
    strengths = []
    requirement_group_count = int(features.get("requirement_group_count", 0) or 0)
    requirement_group_match_count = int(features.get("requirement_group_match_count", 0) or 0)
    if requirement_group_count:
        skill_match_pct = (requirement_group_match_count / requirement_group_count) * 100
    else:
        skill_match_pct = (len(matched_skills) / len(core_jd_skills) * 100) if core_jd_skills else 100
    if skill_match_pct >= 70:
        if requirement_group_count:
            strengths.append(f"✅ Strong requirement match: {requirement_group_match_count}/{requirement_group_count} required skill groups ({skill_match_pct:.0f}%)")
        else:
            strengths.append(f"✅ Strong skill match: {len(matched_skills)}/{len(core_jd_skills)} core JD skills ({skill_match_pct:.0f}%)")
    elif skill_match_pct >= 40:
        if requirement_group_count:
            strengths.append(f"✅ Partial requirement match: {requirement_group_match_count}/{requirement_group_count} required skill groups ({skill_match_pct:.0f}%)")
        else:
            strengths.append(f"✅ Partial skill match: {len(matched_skills)}/{len(core_jd_skills)} core JD skills ({skill_match_pct:.0f}%)")

    if exp_years >= req_exp and req_exp > 0:
        strengths.append(f"✅ Experience meets/exceeds requirement: {exp_years:.1f} yrs (need {req_exp:.0f}+)")
    elif exp_years > 0 and req_exp == 0:
        strengths.append(f"✅ Has {exp_years:.1f} years of experience")

    if edu_level >= 3:
        strengths.append(f"✅ Education: {_EDU_LABELS.get(edu_level, 'Unknown')}")

    if len(extra_skills) > 3:
        strengths.append(f"✅ Brings {len(extra_skills)} additional skills beyond JD requirements")

    if bert_sim >= 7.5:
        strengths.append(f"✅ High semantic relevance ({bert_sim:.1f}/10)")

    if mandatory_skills and not missing_mandatory_skills:
        strengths.append(f"✅ Covers all mandatory skills identified from the JD ({len(mandatory_skills)})")

    if requirement_score >= 0.85:
        strengths.append("✅ Meets the core requirement profile well")

    if ranking_confidence >= 0.75:
        strengths.append(f"✅ High ranking confidence ({ranking_confidence * 100:.0f}%)")

    if role_fit_score >= 0.95 and features.get("role_level") == "junior":
        strengths.append("✅ Experience level aligns well with a junior role")

    # Gaps
    gaps = []
    if missing_skills:
        missing_cs = missing_skills & CS_SKILLS
        missing_data = missing_skills & DATA_SKILLS
        missing_soft = missing_skills & SOFT_SKILLS

        if missing_mandatory_skills:
            gaps.append(f"⚠️ Missing mandatory skills: {', '.join(sorted(missing_mandatory_skills)[:5])}")
        if missing_cs:
            gaps.append(f"⚠️ Missing technical skills: {', '.join(sorted(missing_cs)[:5])}")
        if missing_data:
            gaps.append(f"⚠️ Missing data skills: {', '.join(sorted(missing_data)[:5])}")
        if missing_soft:
            gaps.append(f"⚠️ Missing soft skills: {', '.join(sorted(missing_soft)[:3])}")

    if req_exp > 0 and exp_years < req_exp:
        gap = req_exp - exp_years
        gaps.append(f"⚠️ Experience gap: needs {gap:.1f} more years (has {exp_years:.1f}, needs {req_exp:.0f}+)")

    if edu_level < 3:
        gaps.append(f"⚠️ Education: {_EDU_LABELS.get(edu_level, 'Not detected')} (may be underqualified)")

    if bert_sim < 5.0:
        gaps.append(f"⚠️ Low semantic match ({bert_sim:.1f}/10) — resume may not align with job domain")

    if text_quality < 0.45:
        gaps.append("⚠️ Low extraction confidence — this PDF may be image-based, noisy, or partially unreadable")

    if role_fit_score < 0.7 and features.get("role_level") == "junior":
        gaps.append("⚠️ Experience level may be less aligned with a junior role")

    # Recommendations
    recommendations = []
    if missing_skills:
        top_missing = sorted(missing_skills)[:5]
        recommendations.append(f"📚 Focus on learning: {', '.join(top_missing)}")

    if req_exp > 0 and exp_years < req_exp:
        recommendations.append("💼 Consider gaining more relevant work experience or internships")

    if bert_sim < 6.0:
        recommendations.append("📝 Tailor resume language to better match this job description")

    if missing_mandatory_skills:
        recommendations.append("📌 Highlight mandatory skills explicitly if you have them, or close those gaps first")

    if text_quality < 0.45:
        recommendations.append("📄 Re-upload a text-based PDF resume with clearer section headings and readable text")

    # Suggested roles
    suggested_roles = suggest_job_roles(all_skills)

    if requirement_score >= 0.85:
        eligibility_label = "Meets core requirements"
    elif requirement_score >= 0.60:
        eligibility_label = "Partially meets core requirements"
    else:
        eligibility_label = "Low core requirement fit"

    return {
        "verdict": verdict,
        "verdict_msg": verdict_msg,
        "strengths": strengths,
        "gaps": gaps,
        "recommendations": recommendations,
        "skill_match_pct": round(skill_match_pct, 1),
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(missing_skills),
        "mandatory_skills": sorted(mandatory_skills),
        "missing_mandatory_skills": sorted(missing_mandatory_skills),
        "extra_skills": sorted(extra_skills),
        "suggested_roles": suggested_roles,
        "eligibility_label": eligibility_label,
        "ranking_confidence": round(ranking_confidence * 100, 1),
        "requirement_fit_pct": round(requirement_score * 100, 1),
    }


# ─────────────────────────────────────────────
# MAIN RANKING FUNCTION
# ─────────────────────────────────────────────

def rank_resumes(
    jd_text: str,
    resume_pdf_paths: List[str],
    micro_dict: Optional[Dict] = None,
    college_tier: str = "Unknown",
    progress_callback=None,
) -> List[Dict]:
    """
    Rank resumes against a JD.

    Args:
        jd_text: Job description text
        resume_pdf_paths: List of PDF file paths
        micro_dict: Optional weighted skill criteria
        college_tier: College tier label
        progress_callback: Optional callback(current, total, message) for progress updates

    Returns:
        List of result dicts sorted by predicted_score descending
    """
    print("Initializing Pipeline...")
    pipe = InferencePipeline()
    results = []
    total = len(resume_pdf_paths)

    for i, path in enumerate(resume_pdf_paths):
        filename = os.path.basename(path)
        print(f"Processing [{i+1}/{total}] {filename}...")

        if progress_callback:
            progress_callback(i, total, f"Processing {filename}...")

        # Extract text
        resume_text = pdf_to_text(path)

        if not resume_text or len(resume_text.strip()) < 50:
            print(f"  ⚠️ Skipping {filename}: too little text extracted")
            results.append({
                "rank": 0,
                "index": i,
                "file_path": path,
                "predicted_score": 0.0,
                "features": {"experience_years": 0, "bert_sim_score": 0},
                "feedback": {
                    "verdict": "🔴 Unreadable",
                    "verdict_msg": "Could not extract enough text from this PDF.",
                    "strengths": [],
                    "gaps": ["⚠️ PDF text extraction failed or returned too little content"],
                    "recommendations": ["📄 Ensure the PDF is not image-based or corrupted"],
                    "skill_match_pct": 0,
                    "matched_skills": [],
                    "missing_skills": [],
                    "extra_skills": [],
                    "suggested_roles": [],
                },
                "skills_found": [],
                "cs_skills": [],
                "data_skills": [],
                "soft_skills": [],
            })
            continue

        # Build features
        features, all_skills, cs_skills, data_skills, soft_skills = build_structured_features(
            resume_text, jd_text, micro_dict, college_tier
        )

        # Get embeddings (already computed in build_structured_features)
        resume_emb = features.pop("_resume_emb")
        jd_emb = features.pop("_jd_emb")

        # Predict
        pred = pipe.predict_single(resume_emb, jd_emb, features)

        # Generate feedback
        feedback = generate_feedback(features, pred, resume_text, jd_text)

        # Clean up internal keys
        for k in ["_all_skills", "_cs_skills", "_data_skills", "_soft_skills", "_jd_skills"]:
            features.pop(k, None)

        results.append({
            "rank": 0,  # will be set after sorting
            "index": i,
            "file_path": path,
            "predicted_score": float(pred),
            "features": features,
            "feedback": feedback,
            "skills_found": sorted(all_skills),
            "cs_skills": sorted(cs_skills),
            "data_skills": sorted(data_skills),
            "soft_skills": sorted(soft_skills),
        })

    if progress_callback:
        progress_callback(total, total, "Ranking complete!")

    # Sort by score descending
    ranked = sorted(results, key=lambda x: x["predicted_score"], reverse=True)
    for rnk, r in enumerate(ranked, start=1):
        r["rank"] = rnk

    return ranked


def rank_resumes(
    jd_text: str,
    resume_pdf_paths: List[str],
    micro_dict: Optional[Dict] = None,
    college_tier: str = "Unknown",
    progress_callback=None,
) -> List[Dict]:
    """
    Rank resumes against a JD with confidence-aware re-ranking.

    Returns:
        List of result dicts sorted by predicted_score descending
    """
    print("Initializing Pipeline...")
    pipe = InferencePipeline()
    results = []
    total = len(resume_pdf_paths)
    jd_quality = assess_text_quality(jd_text)

    for i, path in enumerate(resume_pdf_paths):
        filename = os.path.basename(path)
        print(f"Processing [{i+1}/{total}] {filename}...")

        if progress_callback:
            progress_callback(i, total, f"Processing {filename}...")

        resume_text = pdf_to_text(path)
        text_quality = assess_text_quality(resume_text)

        if not resume_text or len(resume_text.strip()) < 50 or text_quality["extraction_confidence"] < 0.12:
            print(f"  Warning: Skipping {filename}: too little text extracted")
            results.append({
                "rank": 0,
                "index": i,
                "file_path": path,
                "predicted_score": 0.0,
                "raw_model_score": 0.0,
                "features": {
                    "experience_years": 0,
                    "bert_sim_score": 0,
                    "text_quality_score": text_quality["extraction_confidence"],
                    "ranking_confidence": 0.0,
                    "requirement_score": 0.0,
                    "jd_quality_score": jd_quality["extraction_confidence"],
                },
                "feedback": {
                    "verdict": "Unreadable",
                    "verdict_msg": "Could not extract enough text from this PDF.",
                    "strengths": [],
                    "gaps": ["PDF text extraction failed or returned too little content"],
                    "recommendations": ["Ensure the PDF is text-based, readable, and not corrupted"],
                    "skill_match_pct": 0,
                    "matched_skills": [],
                    "missing_skills": [],
                    "mandatory_skills": [],
                    "missing_mandatory_skills": [],
                    "extra_skills": [],
                    "suggested_roles": [],
                    "eligibility_label": "Unable to assess",
                    "ranking_confidence": 0.0,
                    "requirement_fit_pct": 0.0,
                },
                "skills_found": [],
                "cs_skills": [],
                "data_skills": [],
                "soft_skills": [],
            })
            continue

        features, all_skills, cs_skills, data_skills, soft_skills = build_structured_features(
            resume_text, jd_text, micro_dict, college_tier
        )

        resume_emb = features.pop("_resume_emb")
        jd_emb = features.pop("_jd_emb")

        raw_pred = pipe.predict_single(resume_emb, jd_emb, features)
        score_info = rerank_prediction(raw_pred, features)
        features.update(score_info)
        features["jd_quality_score"] = jd_quality["extraction_confidence"]

        feedback = generate_feedback(features, score_info["final_score"], resume_text, jd_text)

        for key in ["_all_skills", "_cs_skills", "_data_skills", "_soft_skills", "_jd_skills", "_core_jd_skills", "_mandatory_skills", "_requirement_groups", "_requirement_group_specs", "_skill_evidence_scores"]:
            features.pop(key, None)

        results.append({
            "rank": 0,
            "index": i,
            "file_path": path,
            "predicted_score": float(score_info["final_score"]),
            "raw_model_score": float(raw_pred),
            "features": features,
            "feedback": feedback,
            "skills_found": sorted(all_skills),
            "cs_skills": sorted(cs_skills),
            "data_skills": sorted(data_skills),
            "soft_skills": sorted(soft_skills),
        })

    if progress_callback:
        progress_callback(total, total, "Ranking complete!")

    ranked = sorted(results, key=lambda x: x["predicted_score"], reverse=True)
    for rnk, result in enumerate(ranked, start=1):
        result["rank"] = rnk

    return ranked


if __name__ == "__main__":
    print("SkillScout v2.0 ranking_feedback_nlp module loaded.")
