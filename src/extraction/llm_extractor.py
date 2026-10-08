import json
import re
from typing import Any

from pydantic import ValidationError

from config.settings import settings
from src.extraction.prompts import SYSTEM_PROMPT, build_extraction_prompt
from src.llm.base import LLMError
from src.llm.factory import create_llm_provider
from src.schemas import (
    CandidateEvidence,
    EvidenceItem,
    EvidenceStrength,
    ProjectEvidence,
)


class ExtractionError(Exception):
    """Raised when structured resume extraction fails completely."""


def _clean_json_response(response: str) -> str:
    """
    Remove common Markdown code fences from an LLM response.

    The model is instructed to return JSON only, but this makes the parser
    resilient if the model wraps the JSON in ```json ... ``` fences.
    """

    text = response.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    return text.strip()


def _parse_candidate_json(response: str) -> CandidateEvidence:
    """Validate the LLM response against the Pydantic candidate schema."""

    cleaned = _clean_json_response(response)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"LLM returned invalid JSON: {exc}"
        ) from exc

    try:
        return CandidateEvidence.model_validate(payload)
    except ValidationError as exc:
        raise ExtractionError(
            f"LLM response failed schema validation: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

_TECHNOLOGY_PATTERNS = [
    ("Python", r"\bpython\b"),
    ("FastAPI", r"\bfastapi\b"),
    ("Flask", r"\bflask\b"),
    ("Django", r"\bdjango\b"),
    ("REST API", r"\brest(?:ful)?\s*api\b"),
    ("SQL", r"\bsql\b"),
    ("PostgreSQL", r"\bpostgres(?:ql)?\b"),
    ("MySQL", r"\bmysql\b"),
    ("MongoDB", r"\bmongodb\b"),
    ("Redis", r"\bredis\b"),
    ("Java", r"\bjava\b"),
    ("JavaScript", r"\bjavascript\b"),
    ("TypeScript", r"\btypescript\b"),
    ("React", r"\breact(?:\.js|js)?\b"),
    ("Node.js", r"\bnode(?:\.js)?\b"),
    ("AWS", r"\baws\b"),
    ("GCP", r"\bgcp\b"),
    ("Azure", r"\bazure\b"),
    ("OCI", r"\boci\b"),
    ("Docker", r"\bdocker\b"),
    ("Kubernetes", r"\bkubernetes\b"),
    ("TensorFlow", r"\btensorflow\b"),
    ("PyTorch", r"\bpytorch\b"),
    ("scikit-learn", r"\bscikit[- ]learn\b"),
    ("Machine Learning", r"\bmachine learning\b"),
    ("Deep Learning", r"\bdeep learning\b"),
    ("LLM", r"\bllm\b"),
    ("Generative AI", r"\bgenerative ai\b"),
    ("LangChain", r"\blangchain\b"),
    ("LangGraph", r"\blanggraph\b"),
    ("LlamaIndex", r"\bllamaindex\b"),
    ("RAG", r"\brag\b"),
    ("Embeddings", r"\bembeddings?\b"),
    ("Vector Search", r"\bvector search\b"),
    ("Vector Database", r"\bvector database\b"),
    ("AI Agents", r"\bai agents?\b"),
    ("Multi-Agent", r"\bmulti[- ]agent\b"),
    ("Tool Calling", r"\btool calling\b"),
    ("CrewAI", r"\bcrewai\b"),
    ("Prompt Engineering", r"\bprompt engineering\b"),
    ("CI/CD", r"\bci/cd\b"),
    ("GitHub", r"\bgithub\b"),
]


_AI_SIGNAL_PATTERNS = {
    "llm": [
        r"\bllm\b",
        r"\blarge language model",
        r"\bgenerative ai\b",
        r"\bgemini\b",
        r"\bopenai\b",
        r"\bchatgpt\b",
        r"\bclaude\b",
    ],
    "rag": [
        r"\brag\b",
        r"\bretrieval[- ]augmented generation\b",
    ],
    "retrieval": [
        r"\bretrieval\b",
        r"\bretriever\b",
    ],
    "embeddings": [
        r"\bembeddings?\b",
    ],
    "vector_search": [
        r"\bvector search\b",
        r"\bvector database\b",
        r"\bvector db\b",
    ],
    "tool_calling": [
        r"\btool calling\b",
        r"\bfunction calling\b",
        r"\btools? integration\b",
    ],
    "multi_agent": [
        r"\bmulti[- ]agent\b",
        r"\bmultiagent\b",
        r"\bcrewai\b",
        r"\bagentic\b",
    ],
    "evaluation": [
        r"\bevaluation\b",
        r"\bevaluat(?:ed|ing|ion)\b",
        r"\bab testing\b",
        r"\ba/b testing\b",
    ],
    "state_management": [
        r"\bstate management\b",
        r"\bmemory\b",
        r"\bagent memory\b",
    ],
    "persistence": [
        r"\bpersistence\b",
        r"\bdatabase\b",
        r"\bsql\b",
        r"\bmongodb\b",
        r"\bpostgres(?:ql)?\b",
        r"\bmysql\b",
    ],
    "validation": [
        r"\bvalidation\b",
        r"\bvalidated\b",
        r"\bfault tolerance\b",
        r"\berror handling\b",
    ],
    "external_api": [
        r"\bexternal api\b",
        r"\brest api\b",
        r"\bapi integration\b",
        r"\bapi integrations\b",
    ],
}


def _as_text(value: Any) -> str:
    """Convert deterministic evidence or raw text into searchable text."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    normalized_text = getattr(value, "normalized_text", None)

    if normalized_text:
        return str(normalized_text)

    return str(value)


def _extract_email(text: str) -> str | None:
    match = re.search(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(0) if match else None


def _extract_phone(text: str) -> str | None:
    match = re.search(
        r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",
        text,
    )
    return match.group(0) if match else None


def _extract_github(text: str) -> str | None:
    match = re.search(
        r"https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/?",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(0).rstrip("/") if match else None


def _extract_linkedin(text: str) -> str | None:
    match = re.search(
        r"https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9_.-]+/?",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(0).rstrip("/") if match else None


def _extract_name(text: str) -> str:
    """
    Conservative name extraction.

    If the deterministic extractor already produced name candidates,
    those are preferred. Otherwise use the first plausible non-heading line.
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    for line in lines[:15]:
        cleaned = re.sub(r"^---\s*PAGE\s*\d+\s*---$", "", line, flags=re.I).strip()

        if not cleaned:
            continue

        if "@" in cleaned:
            continue

        if re.search(r"https?://|www\.", cleaned, flags=re.I):
            continue

        if re.search(r"\b(resume|curriculum vitae|cv)\b", cleaned, flags=re.I):
            continue

        if len(cleaned) > 60:
            continue

        words = cleaned.split()

        if 2 <= len(words) <= 5 and all(
            re.match(r"^[A-Za-z][A-Za-z.'-]*$", word)
            for word in words
        ):
            return cleaned

    return "Unknown"


def _find_technologies(text: str) -> list[str]:
    technologies = []

    for technology, pattern in _TECHNOLOGY_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            technologies.append(technology)

    return technologies


def _find_ai_signals(text: str) -> list[str]:
    signals = []

    for signal, patterns in _AI_SIGNAL_PATTERNS.items():
        if any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in patterns
        ):
            signals.append(signal)

    return signals


def _evidence(
    claim: str,
    evidence: str,
    strength: EvidenceStrength = EvidenceStrength.MODERATE,
) -> EvidenceItem:
    return EvidenceItem(
        claim=claim,
        evidence=evidence[:500],
        source_section=None,
        strength=strength,
    )


def _build_deterministic_fallback(
    resume_text: str,
    deterministic_signals: Any,
    failure_reason: str,
) -> CandidateEvidence:
    """
    Build a conservative CandidateEvidence object without an LLM.

    This fallback is intentionally evidence-driven. It does not invent
    projects or experience. It only converts observable resume signals into
    structured evidence so the batch can continue when an LLM provider is
    unavailable.
    """

    # Prefer the normalized deterministic text when the caller supplied
    # the DeterministicEvidence object. Otherwise use the resume text.
    text = _as_text(deterministic_signals) or resume_text
    text = text[: settings.max_resume_chars]

    # If the object contains structured deterministic fields, use them too.
    deterministic_obj = (
        deterministic_signals
        if not isinstance(deterministic_signals, str)
        else None
    )

    technologies = list(
        getattr(deterministic_obj, "technologies", None) or []
    )

    if not technologies:
        technologies = _find_technologies(text)

    # Keep order while removing duplicates.
    technologies = list(dict.fromkeys(technologies))

    ai_signals = list(
        getattr(deterministic_obj, "ai_signals", None) or []
    )

    if not ai_signals:
        ai_signals = _find_ai_signals(text)

    ai_signals = list(dict.fromkeys(ai_signals))

    name_candidates = list(
        getattr(deterministic_obj, "name_candidates", None) or []
    )

    name = (
        str(name_candidates[0]).strip()
        if name_candidates
        else _extract_name(text)
    )

    email_candidates = list(
        getattr(deterministic_obj, "emails", None) or []
    )

    github_candidates = list(
        getattr(deterministic_obj, "github_urls", None) or []
    )

    linkedin_candidates = list(
        getattr(deterministic_obj, "linkedin_urls", None) or []
    )

    email = email_candidates[0] if email_candidates else _extract_email(text)
    github_url = (
        github_candidates[0]
        if github_candidates
        else _extract_github(text)
    )
    linkedin_url = (
        linkedin_candidates[0]
        if linkedin_candidates
        else _extract_linkedin(text)
    )

    phone_candidates = list(
        getattr(deterministic_obj, "phones", None) or []
    )

    phone = phone_candidates[0] if phone_candidates else _extract_phone(text)

    # Genuine Python evidence requires Python itself to appear in observable
    # resume content, not merely a generic backend/API signal.
    has_python = bool(
        re.search(r"\bpython\b", text, flags=re.IGNORECASE)
        or any(tech.lower() == "python" for tech in technologies)
    )

    python_evidence: list[EvidenceItem] = []

    if has_python:
        python_evidence.append(
            _evidence(
                "Python implementation",
                "Python is explicitly listed or used in the resume.",
                EvidenceStrength.STRONG,
            )
        )

    # Only genuine AI/LLM/agentic signals satisfy the AI eligibility gate.
    # Generic signals such as REST APIs or validation are deliberately not
    # sufficient on their own.
    meaningful_ai_signals = {
        "llm",
        "rag",
        "retrieval",
        "embeddings",
        "vector_search",
        "tool_calling",
        "multi_agent",
    }

    matched_ai_signals = [
        signal
        for signal in ai_signals
        if signal in meaningful_ai_signals
    ]

    # Also allow strong explicit AI technologies detected directly.
    strong_ai_technologies = {
        "LLM",
        "Generative AI",
        "LangChain",
        "LangGraph",
        "LlamaIndex",
        "RAG",
        "Embeddings",
        "Vector Search",
        "Vector Database",
        "AI Agents",
        "Multi-Agent",
        "Tool Calling",
        "CrewAI",
    }

    matched_ai_technologies = [
        technology
        for technology in technologies
        if technology in strong_ai_technologies
    ]

    ai_evidence: list[EvidenceItem] = []

    for signal in matched_ai_signals:
        ai_evidence.append(
            _evidence(
                f"AI implementation signal: {signal}",
                f"The resume contains evidence of {signal.replace('_', ' ')}.",
                EvidenceStrength.MODERATE,
            )
        )

    for technology in matched_ai_technologies:
        ai_evidence.append(
            _evidence(
                f"AI technology: {technology}",
                f"The resume explicitly references {technology}.",
                EvidenceStrength.STRONG,
            )
        )

    # Deduplicate AI evidence.
    seen_claims: set[str] = set()
    ai_evidence = [
        item
        for item in ai_evidence
        if not (item.claim in seen_claims or seen_claims.add(item.claim))
    ]

    has_meaningful_ai = bool(ai_evidence)

    backend_evidence: list[EvidenceItem] = []

    backend_terms = {
        "FastAPI",
        "Flask",
        "Django",
        "REST API",
        "SQL",
        "PostgreSQL",
        "MySQL",
        "MongoDB",
        "Redis",
    }

    for technology in technologies:
        if technology in backend_terms:
            backend_evidence.append(
                _evidence(
                    f"Backend technology: {technology}",
                    f"The resume explicitly references {technology}.",
                )
            )

    cloud_terms = {"AWS", "GCP", "Azure", "OCI", "Docker", "Kubernetes", "CI/CD"}

    cloud_evidence = [
        _evidence(
            f"Cloud/deployment technology: {technology}",
            f"The resume explicitly references {technology}.",
        )
        for technology in technologies
        if technology in cloud_terms
    ]

    frontend_terms = {"React", "JavaScript", "TypeScript", "Node.js"}

    frontend_evidence = [
        _evidence(
            f"Frontend/full-stack technology: {technology}",
            f"The resume explicitly references {technology}.",
        )
        for technology in technologies
        if technology in frontend_terms
    ]

    github_evidence: list[EvidenceItem] = []

    if github_url:
        github_evidence.append(
            _evidence(
                "GitHub profile",
                github_url,
                EvidenceStrength.STRONG,
            )
        )

    project_evidence: list[EvidenceItem] = []

    for signal in ai_signals:
        project_evidence.append(
            _evidence(
                f"Project implementation signal: {signal}",
                f"The resume contains the implementation signal '{signal.replace('_', ' ')}'.",
            )
        )

    project_evidence.extend(
        _evidence(
            f"Technology used: {technology}",
            f"The resume references {technology}.",
        )
        for technology in technologies
    )

    # Build one conservative project representation. This does not claim
    # details that the fallback cannot establish.
    project = ProjectEvidence(
        name="Resume evidence",
        description="Project-level evidence extracted without LLM assistance.",
        technologies=technologies,
        evidence=project_evidence,
        python_used=has_python,
        ai_or_llm_used=has_meaningful_ai,
        llm="llm" in ai_signals or "LLM" in matched_ai_technologies,
        rag="rag" in ai_signals or "RAG" in matched_ai_technologies,
        retrieval="retrieval" in ai_signals,
        embeddings=(
            "embeddings" in ai_signals
            or "Embeddings" in matched_ai_technologies
        ),
        vector_search=(
            "vector_search" in ai_signals
            or "Vector Search" in matched_ai_technologies
            or "Vector Database" in matched_ai_technologies
        ),
        tool_calling=(
            "tool_calling" in ai_signals
            or "Tool Calling" in matched_ai_technologies
        ),
        multi_agent=(
            "multi_agent" in ai_signals
            or "Multi-Agent" in matched_ai_technologies
            or "AI Agents" in matched_ai_technologies
            or "CrewAI" in matched_ai_technologies
        ),
        evaluation="evaluation" in ai_signals,
        state_management="state_management" in ai_signals,
        persistence="persistence" in ai_signals,
        validation="validation" in ai_signals,
        external_api="external_api" in ai_signals,
    )

    concerns = [
        "LLM extraction was unavailable; this candidate was processed using deterministic fallback extraction.",
        f"LLM failure reason: {failure_reason[:300]}",
    ]

    strengths = []

    if has_python:
        strengths.append("Explicit Python evidence detected.")

    if has_meaningful_ai:
        strengths.append(
            "Meaningful AI/LLM/agentic implementation signals detected."
        )

    if backend_evidence:
        strengths.append("Backend/API technology evidence detected.")

    if cloud_evidence:
        strengths.append("Cloud/deployment evidence detected.")

    summary_parts = []

    if technologies:
        summary_parts.append(
            f"Detected technologies: {', '.join(technologies[:15])}."
        )

    if matched_ai_signals:
        summary_parts.append(
            f"AI signals: {', '.join(matched_ai_signals)}."
        )

    if not summary_parts:
        summary_parts.append(
            "Limited structured evidence could be extracted without LLM assistance."
        )

    return CandidateEvidence(
        name=name or "Unknown",
        email=email,
        phone=phone,
        github_url=github_url,
        linkedin_url=linkedin_url,
        skills=technologies,
        python_evidence=python_evidence,
        ai_evidence=ai_evidence,
        backend_evidence=backend_evidence,
        cloud_evidence=cloud_evidence,
        frontend_evidence=frontend_evidence,
        github_evidence=github_evidence,
        projects=[project],
        summary=" ".join(summary_parts),
        concerns=concerns,
        strengths=strengths,
    )


def extract_candidate_evidence(
    resume_text: str,
    deterministic_signals: str = "",
) -> CandidateEvidence:
    """
    Extract structured evidence from one resume.

    Normal path:
        Resume -> Gemini/LLM -> validated CandidateEvidence

    Reliability path:
        Resume -> LLM failure -> deterministic CandidateEvidence

    The fallback is important for batch processing because a provider outage,
    quota exhaustion, timeout, or malformed LLM response must not make the
    entire screening run unusable.
    """

    if not resume_text.strip():
        raise ExtractionError("Resume text is empty.")

    truncated_text = resume_text[: settings.max_resume_chars]

    try:
        provider = create_llm_provider()

        prompt = build_extraction_prompt(
            resume_text=truncated_text,
            deterministic_signals=deterministic_signals,
        )

        response = provider.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
        )

        return _parse_candidate_json(response)

    except (LLMError, ExtractionError) as exc:
        # IMPORTANT:
        # Do not retry here. Provider-specific retries already happen inside
        # the provider adapter. A 429 quota exhaustion will not be fixed by
        # repeatedly calling the same provider.
        #
        # Instead, preserve the batch and fall back to deterministic evidence.
        return _build_deterministic_fallback(
            resume_text=truncated_text,
            deterministic_signals=deterministic_signals,
            failure_reason=str(exc),
        )

    except Exception as exc:
        # Last-resort protection for unexpected provider/SDK errors.
        # The fallback remains conservative and evidence-based.
        return _build_deterministic_fallback(
            resume_text=truncated_text,
            deterministic_signals=deterministic_signals,
            failure_reason=f"Unexpected extraction error: {exc}",
        )