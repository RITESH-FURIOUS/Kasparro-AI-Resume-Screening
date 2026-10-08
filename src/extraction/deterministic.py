from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.loader import LoadedResume


# ============================================================
# Regex patterns
# ============================================================

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"""
    (?:
        (?:\+?91[\s\-]?)?
        [6-9]\d{9}
    )
    """,
    re.VERBOSE,
)

GITHUB_PATTERN = re.compile(
    r"""
    (?:
        https?://
    )?
    (?:
        www\.
    )?
    github\.com/
    [A-Za-z0-9][A-Za-z0-9-]*
    """,
    re.IGNORECASE | re.VERBOSE,
)

LINKEDIN_PATTERN = re.compile(
    r"""
    (?:
        https?://
    )?
    (?:
        www\.
    )?
    linkedin\.com/in/
    [A-Za-z0-9][A-Za-z0-9\-_%]*
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ============================================================
# Technology vocabulary
# ============================================================

# This is intentionally broader than the assignment's eligibility
# vocabulary. These are signals that can later help the LLM and
# deterministic scorer understand the candidate.
#
# Detection here does NOT mean the candidate automatically gets
# points for the skill.

TECHNOLOGY_PATTERNS: dict[str, list[str]] = {
    # -------------------------
    # Programming languages
    # -------------------------
    "Python": [
        r"\bpython\b",
    ],
    "Java": [
        r"\bjava\b",
    ],
    "C++": [
        r"\bc\+\+\b",
    ],
    "C": [
        r"\bc\b",
    ],
    "JavaScript": [
        r"\bjavascript\b",
        r"\bjs\b",
    ],
    "TypeScript": [
        r"\btypescript\b",
    ],
    "Go": [
        r"\bgolang\b",
        r"\bgo\b",
    ],
    "Rust": [
        r"\brust\b",
    ],

    # -------------------------
    # Backend
    # -------------------------
    "Flask": [
        r"\bflask\b",
    ],
    "FastAPI": [
        r"\bfastapi\b",
        r"\bfast\s*api\b",
    ],
    "Django": [
        r"\bdjango\b",
    ],
    "Node.js": [
        r"\bnode\.?js\b",
        r"\bnodejs\b",
    ],
    "Express.js": [
        r"\bexpress\.?js\b",
        r"\bexpressjs\b",
    ],
    "REST API": [
        r"\brest(?:ful)?\s*apis?\b",
        r"\brest\s*api\b",
    ],
    "GraphQL": [
        r"\bgraphql\b",
    ],

    # -------------------------
    # Databases
    # -------------------------
    "SQL": [
        r"\bsql\b",
    ],
    "MySQL": [
        r"\bmysql\b",
    ],
    "PostgreSQL": [
        r"\bpostgres(?:ql)?\b",
    ],
    "MongoDB": [
        r"\bmongodb\b",
    ],
    "Oracle": [
        r"\boracle\b",
    ],
    "Redis": [
        r"\bredis\b",
    ],

    # -------------------------
    # AI / ML
    # -------------------------
    "Machine Learning": [
        r"\bmachine\s+learning\b",
        r"\bml\b",
    ],
    "Deep Learning": [
        r"\bdeep\s+learning\b",
    ],
    "TensorFlow": [
        r"\btensorflow\b",
    ],
    "PyTorch": [
        r"\bpytorch\b",
    ],
    "scikit-learn": [
        r"\bscikit[-\s]?learn\b",
        r"\bsklearn\b",
    ],

    # -------------------------
    # LLM / Agentic AI
    # -------------------------
    "LLM": [
        r"\bllm(?:s)?\b",
        r"\blarge\s+language\s+models?\b",
    ],
    "Generative AI": [
        r"\bgenerative\s+ai\b",
        r"\bgenai\b",
    ],
    "LangChain": [
        r"\blangchain\b",
    ],
    "LangGraph": [
        r"\blanggraph\b",
    ],
    "LlamaIndex": [
        r"\bllamaindex\b",
        r"\bllama\s*index\b",
    ],
    "RAG": [
        r"\brag\b",
        r"\bretrieval[-\s]+augmented\s+generation\b",
    ],
    "Embeddings": [
        r"\bembeddings?\b",
    ],
    "Vector Search": [
        r"\bvector\s+(?:search|database|store|storage)\b",
        r"\bvector\s+db\b",
    ],
    "Vector Database": [
        r"\bvector\s+databases?\b",
        r"\bvector\s+db\b",
    ],
    "AI Agents": [
        r"\bai\s+agents?\b",
        r"\bagentic\s+ai\b",
        r"\bagentic\s+systems?\b",
        r"\bai\s+agentic\b",
    ],
    "Multi-Agent": [
        r"\bmulti[-\s]?agent\b",
        r"\bmulti[-\s]?agents\b",
    ],
    "Tool Calling": [
        r"\btool[-\s]?calling\b",
        r"\bfunction[-\s]?calling\b",
        r"\btool\s+use\b",
    ],
    "Prompt Engineering": [
        r"\bprompt\s+engineering\b",
        r"\bprompt\s+engineer(?:ing)?\b",
    ],
    "Google ADK": [
        r"\bgoogle\s+adk\b",
        r"\badk\b",
    ],
    "CrewAI": [
        r"\bcrewai\b",
    ],

    # -------------------------
    # Cloud / deployment
    # -------------------------
    "AWS": [
        r"\baws\b",
        r"\bamazon\s+web\s+services\b",
    ],
    "GCP": [
        r"\bgcp\b",
        r"\bgoogle\s+cloud\b",
    ],
    "Azure": [
        r"\bazure\b",
        r"\bmicrosoft\s+azure\b",
    ],
    "OCI": [
        r"\boci\b",
        r"\boracle\s+cloud\b",
        r"\boracle\s+cloud\s+infrastructure\b",
    ],
    "Docker": [
        r"\bdocker\b",
    ],
    "Kubernetes": [
        r"\bkubernetes\b",
        r"\bk8s\b",
    ],
    "CI/CD": [
        r"\bci\s*/\s*cd\b",
        r"\bcontinuous\s+integration\b",
        r"\bcontinuous\s+deployment\b",
    ],

    # -------------------------
    # Frontend
    # -------------------------
    "React": [
        r"\breact(?:\.js|js)?\b",
    ],
    "Next.js": [
        r"\bnext\.?js\b",
        r"\bnextjs\b",
    ],
    "HTML": [
        r"\bhtml5?\b",
    ],
    "CSS": [
        r"\bcss3?\b",
    ],

    # -------------------------
    # Testing / engineering
    # -------------------------
    "Pytest": [
        r"\bpytest\b",
    ],
    "Unit Testing": [
        r"\bunit\s+tests?\b",
        r"\bunit\s+testing\b",
    ],
    "Integration Testing": [
        r"\bintegration\s+tests?\b",
        r"\bintegration\s+testing\b",
    ],
    "Git": [
        r"\bgit\b",
    ],
    "GitHub": [
        r"\bgithub\b",
    ],
}


# ============================================================
# AI-specific signals
# ============================================================

AI_SIGNAL_PATTERNS: dict[str, list[str]] = {
    # --------------------------------------------------------
    # Strong AI / LLM / agentic signals
    # --------------------------------------------------------

    "llm": [
        r"\bllm(?:s)?\b",
        r"\blarge\s+language\s+models?\b",
    ],

    "tool_calling": [
        r"\btool[-\s]?calling\b",
        r"\bfunction[-\s]?calling\b",
        r"\btool\s+use\b",
        r"\btool[-\s]?using\s+agents?\b",
    ],

    "multi_agent": [
        r"\bmulti[-\s]?agent\b",
        r"\bmulti[-\s]?agents\b",
        r"\bmulti[-\s]?agent\s+system\b",
        r"\bmulti[-\s]?agent\s+workflow\b",
    ],

    "retrieval": [
        r"\bretrieval\b",
        r"\bretriever\b",
        r"\bretrieval[-\s]+augmented\b",
    ],

    "rag": [
        r"\brag\b",
        r"\bretrieval[-\s]+augmented\s+generation\b",
    ],

    "embeddings": [
        r"\bembeddings?\b",
        r"\bembedding\s+model\b",
    ],

    "vector_search": [
        r"\bvector\s+search\b",
        r"\bvector\s+database\b",
        r"\bvector\s+store\b",
        r"\bvector\s+db\b",
    ],

    "ai_agents": [
        r"\bai\s+agents?\b",
        r"\bagentic\s+ai\b",
        r"\bagentic\s+systems?\b",
        r"\bagentic\s+workflow\b",
    ],

    "prompt_engineering": [
        r"\bprompt\s+engineering\b",
        r"\bprompt\s+engineer(?:ing)?\b",
    ],

    "langchain": [
        r"\blangchain\b",
    ],

    "langgraph": [
        r"\blanggraph\b",
    ],

    "llamaindex": [
        r"\bllamaindex\b",
        r"\bllama\s*index\b",
    ],

    "crewai": [
        r"\bcrewai\b",
    ],

    "google_adk": [
        r"\bgoogle\s+adk\b",
    ],

    # --------------------------------------------------------
    # Supporting implementation signals
    #
    # These should NEVER independently make someone AI-eligible.
    # They become useful when combined with strong AI evidence.
    # --------------------------------------------------------

    "state_management": [
        r"\bconversation\s+state\b",
        r"\bstate\s+management\b",
        r"\bagent\s+state\b",
        r"\bworkflow\s+state\b",
    ],

    "external_api": [
        r"\bexternal\s+apis?\b",
        r"\bthird[-\s]?party\s+apis?\b",
        r"\bapi\s+integration\b",
    ],

    "persistence": [
        r"\bdatabase\b",
        r"\bpersistence\b",
        r"\bpersisted\b",
        r"\bstored\s+agent\b",
    ],

    "evaluation": [
        r"\bevaluation\b",
        r"\bevaluated\b",
        r"\bbenchmark(?:ed|ing)?\b",
        r"\bab\s+test(?:ing)?\b",
    ],

    "validation": [
        r"\bstructured\s+output\b",
        r"\bpydantic\b",
        r"\bschema\s+validation\b",
    ],
}


# ============================================================
# Data structures
# ============================================================


@dataclass
class DeterministicEvidence:
    """
    Signals extracted without using an LLM.

    This is not the final candidate profile.
    It is the evidence available to downstream extraction/scoring.
    """

    normalized_text: str

    emails: list[str] = field(default_factory=list)

    phones: list[str] = field(default_factory=list)

    github_urls: list[str] = field(default_factory=list)

    linkedin_urls: list[str] = field(default_factory=list)

    technologies: dict[str, list[str]] = field(
        default_factory=dict
    )

    ai_signals: dict[str, list[str]] = field(
        default_factory=dict
    )

    first_lines: list[str] = field(default_factory=list)


# ============================================================
# Text normalization
# ============================================================


def normalize_resume_text(text: str) -> str:
    """
    Normalize common PDF extraction artifacts while preserving
    enough structure for later section/evidence extraction.

    We deliberately avoid aggressive NLP cleanup.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", "")

    # Normalize non-breaking spaces.
    text = text.replace("\u00a0", " ")

    # Common PDF extraction issue:
    #
    #   "inReact.js"
    #
    # The original PDF may have placed two text boxes adjacent to
    # each other. We do NOT blindly insert spaces between every
    # lowercase-uppercase boundary because that can damage normal
    # words.
    #
    # Instead, technology matching later also handles many variants.

    # Normalize bullets.
    text = text.replace("•", "• ")

    # Normalize repeated spaces, but preserve newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces immediately before punctuation.
    text = re.sub(r"\s+([,.;:])", r"\1", text)

    # Normalize excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# Regex helpers
# ============================================================


def unique_preserve_order(values: list[str]) -> list[str]:
    """
    Remove duplicates while preserving their original order.
    """

    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        key = value.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result


def extract_emails(text: str) -> list[str]:
    matches = EMAIL_PATTERN.findall(text)

    return unique_preserve_order(
        [match.strip(".,;:()[]<>") for match in matches]
    )


def extract_phones(text: str) -> list[str]:
    matches = PHONE_PATTERN.findall(text)

    return unique_preserve_order(
        [match.strip(".,;:()[]<>") for match in matches]
    )


def normalize_url(url: str) -> str:
    """
    Convert detected URLs into a consistent representation.
    """

    url = url.strip(".,;:()[]<>")

    if not url.lower().startswith("http"):
        return f"https://{url}"

    return url


def extract_github_urls(text: str) -> list[str]:
    matches = GITHUB_PATTERN.findall(text)

    return unique_preserve_order(
        [normalize_url(match) for match in matches]
    )


def extract_linkedin_urls(text: str) -> list[str]:
    matches = LINKEDIN_PATTERN.findall(text)

    return unique_preserve_order(
        [normalize_url(match) for match in matches]
    )


# ============================================================
# Technology detection
# ============================================================


def find_evidence_snippets(
    text: str,
    pattern: str,
    context_chars: int = 140,
) -> list[str]:
    """
    Return small surrounding snippets for each regex match.

    These snippets are useful evidence for the LLM and for
    debugging why a deterministic signal was detected.
    """

    try:
        matches = list(
            re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
        )
    except re.error:
        return []

    snippets: list[str] = []

    for match in matches:
        start = max(
            0,
            match.start() - context_chars,
        )

        end = min(
            len(text),
            match.end() + context_chars,
        )

        snippet = text[start:end].strip()

        # Collapse internal whitespace for compact evidence.
        snippet = re.sub(
            r"\s+",
            " ",
            snippet,
        )

        snippets.append(snippet)

    return unique_preserve_order(snippets)


def detect_technologies(
    text: str,
) -> dict[str, list[str]]:
    """
    Detect technology mentions and retain surrounding evidence.

    IMPORTANT:
    A detected technology is only a signal.

    Example:
        "Interested in learning Python"

    should not automatically receive the same weight as:

        "Built Flask REST APIs using Python"
    """

    detected: dict[str, list[str]] = {}

    for technology, patterns in TECHNOLOGY_PATTERNS.items():
        snippets: list[str] = []

        for pattern in patterns:
            snippets.extend(
                find_evidence_snippets(
                    text,
                    pattern,
                )
            )

        snippets = unique_preserve_order(snippets)

        if snippets:
            detected[technology] = snippets

    return detected


# ============================================================
# AI signal detection
# ============================================================


def detect_ai_signals(
    text: str,
) -> dict[str, list[str]]:
    """
    Detect meaningful AI/agentic implementation signals.

    These signals are deliberately more specific than merely
    checking for the word 'AI'.
    """

    detected: dict[str, list[str]] = {}

    for signal, patterns in AI_SIGNAL_PATTERNS.items():
        snippets: list[str] = []

        for pattern in patterns:
            snippets.extend(
                find_evidence_snippets(
                    text,
                    pattern,
                )
            )

        snippets = unique_preserve_order(snippets)

        if snippets:
            detected[signal] = snippets

    return detected


# ============================================================
# Candidate name detection
# ============================================================


def looks_like_contact_line(line: str) -> bool:
    """
    Identify lines that are clearly contact information.
    """

    lowered = line.lower()

    contact_markers = [
        "@",
        "phone:",
        "mobile:",
        "linkedin",
        "github",
        "http://",
        "https://",
        "+91",
    ]

    return any(
        marker in lowered
        for marker in contact_markers
    )


def looks_like_section_heading(line: str) -> bool:
    """
    Avoid choosing obvious section headings as the candidate name.
    """

    normalized = re.sub(
        r"[^a-zA-Z ]",
        "",
        line,
    ).strip().lower()

    headings = {
        "summary",
        "profile",
        "objective",
        "skills",
        "technical skills",
        "education",
        "experience",
        "work experience",
        "professional experience",
        "projects",
        "project",
        "certifications",
        "achievements",
        "internships",
        "internship",
        "contact",
        "about",
        "references",
    }

    return normalized in headings


def extract_candidate_name(
    text: str,
    filename: str,
) -> str:
    """
    Conservative candidate-name extraction.

    PDF page markers such as:
        --- PAGE 1 ---

    are ignored.

    We inspect the first meaningful lines rather than assuming
    that the name must be at one exact position.
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    for line in lines[:20]:

        # Ignore our own PDF page markers.
        if re.fullmatch(
            r"-{2,}\s*PAGE\s+\d+\s*-{2,}",
            line,
            flags=re.IGNORECASE,
        ):
            continue

        if looks_like_contact_line(line):
            continue

        if looks_like_section_heading(line):
            continue

        words = line.split()

        # Names are normally short.
        if not 1 <= len(words) <= 5:
            continue

        # Names should not contain digits.
        if any(char.isdigit() for char in line):
            continue

        lowered = line.lower()

        # Avoid selecting job titles / resume labels.
        if any(
            marker in lowered
            for marker in [
                "developer",
                "engineer",
                "student",
                "resume",
                "curriculum vitae",
                "software",
                "profile",
                "intern",
                "consultant",
                "analyst",
            ]
        ):
            continue

        alphabetic = sum(
            char.isalpha()
            for char in line
        )

        if alphabetic < max(
            2,
            len(line) * 0.5,
        ):
            continue

        return line

    # Filename fallback.
    fallback = Path(filename).stem

    fallback = re.sub(
        r"[_\-]+",
        " ",
        fallback,
    )

    fallback = re.sub(
        r"\b(?:resume|cv)\b",
        "",
        fallback,
        flags=re.IGNORECASE,
    )

    fallback = re.sub(
        r"\s+",
        " ",
        fallback,
    ).strip()

    return fallback or "Unknown"


# ============================================================
# Main deterministic extraction
# ============================================================


def extract_deterministic_evidence(
    resume: LoadedResume,
) -> DeterministicEvidence:
    """
    Extract reliable, provider-independent signals from a resume.
    """

    normalized_text = normalize_resume_text(
        resume.text
    )

    emails = extract_emails(
        normalized_text
    )

    phones = extract_phones(
        normalized_text
    )

    github_urls = extract_github_urls(
        normalized_text
    )

    linkedin_urls = extract_linkedin_urls(
        normalized_text
    )

    technologies = detect_technologies(
        normalized_text
    )

    ai_signals = detect_ai_signals(
        normalized_text
    )

    first_lines = [
        line.strip()
        for line in normalized_text.splitlines()
        if line.strip()
    ][:20]

    return DeterministicEvidence(
        normalized_text=normalized_text,
        emails=emails,
        phones=phones,
        github_urls=github_urls,
        linkedin_urls=linkedin_urls,
        technologies=technologies,
        ai_signals=ai_signals,
        first_lines=first_lines,
    )