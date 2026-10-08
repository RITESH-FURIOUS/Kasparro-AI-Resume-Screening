from src.schemas import CandidateEvidence, ScoreBreakdown

from src.scoring.rules import (
    AI_MAX,
    PYTHON_BACKEND_MAX,
    CLOUD_FULLSTACK_MAX,
    GITHUB_MAX,
    ENGINEERING_MAX,
    SHALLOW_AI_PENALTY,
)


def _clamp(value: float, maximum: float) -> float:
    """Keep a score between 0 and its category maximum."""
    return max(0.0, min(float(value), float(maximum)))


def score_ai_depth(candidate: CandidateEvidence) -> float:
    """
    Score AI / Agentic / RAG depth.

    The system rewards actual implementation depth rather than
    simply mentioning AI or an LLM in a skills section.
    """

    score = 0.0
    project_signal_counts = []

    for project in candidate.projects:
        if not project.ai_or_llm_used:
            continue

        project_score = 0.0

        if project.llm:
            project_score += 5

        if project.rag:
            project_score += 6

        if project.retrieval:
            project_score += 4

        if project.embeddings:
            project_score += 4

        if project.vector_search:
            project_score += 4

        if project.tool_calling:
            project_score += 5

        if project.multi_agent:
            project_score += 5

        if project.evaluation:
            project_score += 3

        if project.state_management:
            project_score += 2

        if project.validation:
            project_score += 2

        if project.persistence:
            project_score += 2

        if project.external_api:
            project_score += 1

        project_signal_counts.append(
            project.ai_depth_signal_count
        )

        score += project_score

    score = _clamp(score, AI_MAX)

    # A project that only wraps a basic LLM/API gets penalized.
    if project_signal_counts and max(project_signal_counts) <= 1:
        score = max(
            0.0,
            score - SHALLOW_AI_PENALTY,
        )

    return score


def score_python_backend(candidate: CandidateEvidence) -> float:
    """
    Score Python and backend engineering depth.

    Actual project implementation evidence is weighted more heavily
    than generic skill mentions.
    """

    score = 0.0

    for project in candidate.projects:
        if project.python_used:
            score += 5

        if project.external_api:
            score += 2

        if project.persistence:
            score += 2

        if project.validation:
            score += 2

        if project.business_logic:
            score += 2

    # Internship and work experience can provide additional
    # implementation evidence.
    experience_text = " ".join(
        str(item)
        for item in (
            candidate.internships
            + candidate.experience
        )
    ).lower()

    if "python" in experience_text:
        score += 4

    backend_terms = (
        "rest",
        "api",
        "flask",
        "fastapi",
        "django",
        "backend",
        "sql",
        "database",
        "postgresql",
        "mysql",
    )

    matched_backend_terms = sum(
        1
        for term in backend_terms
        if term in experience_text
    )

    score += min(
        matched_backend_terms,
        4,
    )

    return _clamp(
        score,
        PYTHON_BACKEND_MAX,
    )


def score_cloud_fullstack(
    candidate: CandidateEvidence,
) -> float:
    """
    Score cloud, deployment and full-stack evidence.
    """

    score = 0.0

    cloud_keywords = {
        "aws",
        "gcp",
        "google cloud",
        "azure",
        "oci",
        "oracle cloud",
        "docker",
        "kubernetes",
        "ci/cd",
        "cicd",
    }

    fullstack_keywords = {
        "react",
        "reactjs",
        "javascript",
        "typescript",
        "html",
        "css",
        "frontend",
        "full stack",
        "fullstack",
    }

    for skill in candidate.skills:
        skill_name = skill.name.lower().strip()

        if any(
            keyword in skill_name
            for keyword in cloud_keywords
        ):
            score += 2

        if any(
            keyword in skill_name
            for keyword in fullstack_keywords
        ):
            score += 1

    for project in candidate.projects:
        if project.external_api:
            score += 1

        if project.persistence:
            score += 1

    return _clamp(
        score,
        CLOUD_FULLSTACK_MAX,
    )


def score_engineering_depth(
    candidate: CandidateEvidence,
) -> float:
    """
    Score engineering maturity signals.
    """

    score = 0.0

    for project in candidate.projects:
        if project.validation:
            score += 1

        if project.persistence:
            score += 1

        if project.evaluation:
            score += 1

        if project.business_logic:
            score += 1

        if project.external_api:
            score += 1

    return _clamp(
        score,
        ENGINEERING_MAX,
    )


def score_github(github_score: float) -> float:
    """Cap GitHub enrichment at the allowed maximum."""

    return _clamp(
        github_score,
        GITHUB_MAX,
    )


def calculate_score(
    candidate: CandidateEvidence,
    github_score: float = 0.0,
) -> ScoreBreakdown:
    """
    Calculate the deterministic 100-point score.

    Maximum:
        AI / Agentic / RAG       40
        Python / Backend         30
        Cloud / Full Stack       15
        GitHub Activity          10
        Engineering Depth         5
        --------------------------------
        Total                   100
    """

    ai_score = score_ai_depth(candidate)

    python_backend_score = score_python_backend(
        candidate
    )

    cloud_fullstack_score = score_cloud_fullstack(
        candidate
    )

    github_activity_score = score_github(
        github_score
    )

    engineering_depth_score = score_engineering_depth(
        candidate
    )

    return ScoreBreakdown(
        ai_agentic_rag=ai_score,
        python_backend=python_backend_score,
        cloud_full_stack=cloud_fullstack_score,
        github_activity=github_activity_score,
        engineering_depth=engineering_depth_score,
    )