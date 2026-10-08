from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ============================================================
# ENUMS
# ============================================================

class EvidenceStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class EligibilityStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    REJECTED = "REJECTED"


class ProcessingStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


# ============================================================
# EVIDENCE
# ============================================================

class EvidenceItem(BaseModel):
    """
    A concrete claim extracted from a resume together with
    the supporting resume text/evidence.
    """

    model_config = ConfigDict(extra="forbid")

    claim: str
    evidence: str
    source_section: str | None = None
    strength: EvidenceStrength = EvidenceStrength.MODERATE


# ============================================================
# PROJECT EVIDENCE
# ============================================================

class ProjectEvidence(BaseModel):
    """
    Structured representation of one candidate project.

    Boolean signals are intentionally explicit because the LLM
    should identify evidence, while deterministic Python logic
    makes the final scoring decision.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = ""

    description: str = ""

    technologies: list[str] = Field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Core eligibility signals
    # --------------------------------------------------------

    python_used: bool = False
    ai_or_llm_used: bool = False

    # --------------------------------------------------------
    # AI / Agentic / RAG depth signals
    # --------------------------------------------------------

    llm: bool = False
    tool_calling: bool = False
    multi_agent: bool = False
    retrieval: bool = False
    embeddings: bool = False
    vector_search: bool = False
    rag: bool = False

    state_management: bool = False
    external_api: bool = False
    persistence: bool = False
    evaluation: bool = False
    validation: bool = False
    business_logic: bool = False

    # --------------------------------------------------------
    # Supporting evidence
    # --------------------------------------------------------

    evidence: list[EvidenceItem] = Field(
        default_factory=list
    )

    @property
    def ai_depth_signal_count(self) -> int:
        """
        Number of meaningful AI/engineering depth signals
        present in this project.
        """

        signals = [
            self.llm,
            self.tool_calling,
            self.multi_agent,
            self.retrieval,
            self.embeddings,
            self.vector_search,
            self.rag,
            self.state_management,
            self.external_api,
            self.persistence,
            self.evaluation,
            self.validation,
            self.business_logic,
        ]

        return sum(bool(signal) for signal in signals)


# ============================================================
# CANDIDATE EVIDENCE
# ============================================================

class CandidateEvidence(BaseModel):
    """
    Structured evidence extracted from a complete resume.

    The LLM populates this structure. Eligibility and ranking
    are subsequently determined by deterministic Python rules.
    """

    model_config = ConfigDict(extra="forbid")

    # --------------------------------------------------------
    # Identity / contact
    # --------------------------------------------------------

    name: str = "Unknown"

    email: str | None = None

    phone: str | None = None

    location: str | None = None

    github_url: str | None = None

    linkedin_url: str | None = None

    # --------------------------------------------------------
    # Skills
    # --------------------------------------------------------

    skills: list[str] = Field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Category-specific evidence
    # --------------------------------------------------------

    python_evidence: list[EvidenceItem] = Field(
        default_factory=list
    )

    ai_evidence: list[EvidenceItem] = Field(
        default_factory=list
    )

    backend_evidence: list[EvidenceItem] = Field(
        default_factory=list
    )

    cloud_evidence: list[EvidenceItem] = Field(
        default_factory=list
    )

    frontend_evidence: list[EvidenceItem] = Field(
        default_factory=list
    )

    github_evidence: list[EvidenceItem] = Field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Projects / experience
    # --------------------------------------------------------

    projects: list[ProjectEvidence] = Field(
        default_factory=list
    )

    internships: list[str] = Field(
        default_factory=list
    )

    experience: list[str] = Field(
        default_factory=list
    )

    certifications: list[str] = Field(
        default_factory=list
    )

    education: list[str] = Field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------

    summary: str = ""

    concerns: list[str] = Field(
        default_factory=list
    )

    strengths: list[str] = Field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Deterministic eligibility helpers
    # --------------------------------------------------------

    @property
    def has_python_evidence(self) -> bool:
        """
        Python eligibility requires actual evidence rather than
        Python appearing only as a keyword in a skills section.
        """

        if self.python_evidence:
            return True

        for project in self.projects:
            if project.python_used:
                return True

        # Experience/internship evidence can also establish
        # genuine Python usage.
        python_terms = (
            "python",
            "flask",
            "django",
            "fastapi",
            "pandas",
            "numpy",
            "tensorflow",
            "pytorch",
        )

        experience_text = " ".join(
            self.experience + self.internships
        ).lower()

        return any(
            term in experience_text
            for term in python_terms
        )

    @property
    def has_ai_evidence(self) -> bool:
        """
        AI eligibility requires meaningful implementation evidence,
        not merely an AI keyword in a skills list.
        """

        if self.ai_evidence:
            return True

        return any(
            project.ai_or_llm_used
            and project.ai_depth_signal_count >= 1
            for project in self.projects
        )

    @property
    def has_meaningful_ai_project(self) -> bool:
        """
        Determines whether at least one project contains
        meaningful AI/LLM/agentic implementation evidence.
        """

        return any(
            project.ai_or_llm_used
            and project.ai_depth_signal_count >= 2
            for project in self.projects
        )


# ============================================================
# ELIGIBILITY
# ============================================================

class EligibilityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: EligibilityStatus

    python_gate: bool = False

    ai_gate: bool = False

    reason: str = ""

    matched_requirements: list[str] = Field(
        default_factory=list
    )

    missing_requirements: list[str] = Field(
        default_factory=list
    )


# ============================================================
# PROJECT QUALITY
# ============================================================

class ProjectQuality(BaseModel):
    """
    AI project depth classification.

    This is deliberately separate from the final score.
    """

    model_config = ConfigDict(extra="forbid")

    level: str = "UNKNOWN"

    depth_score: float = Field(
        default=0.0,
        ge=0,
        le=10,
    )

    shallow_ai_penalty: float = Field(
        default=0.0,
        ge=0,
    )

    reasoning: str = ""

    signals: list[str] = Field(
        default_factory=list
    )


# ============================================================
# SCORE BREAKDOWN
# ============================================================

class ScoreBreakdown(BaseModel):
    """
    Deterministic 100-point scoring model.
    """

    model_config = ConfigDict(extra="forbid")

    ai_agentic_rag: float = Field(
        default=0.0,
        ge=0,
        le=40,
    )

    python_backend: float = Field(
        default=0.0,
        ge=0,
        le=30,
    )

    cloud_full_stack: float = Field(
        default=0.0,
        ge=0,
        le=15,
    )

    github_activity: float = Field(
        default=0.0,
        ge=0,
        le=10,
    )

    engineering_depth: float = Field(
        default=0.0,
        ge=0,
        le=5,
    )

    @property
    def total(self) -> float:
        return round(
            self.ai_agentic_rag
            + self.python_backend
            + self.cloud_full_stack
            + self.github_activity
            + self.engineering_depth,
            2,
        )


# ============================================================
# GITHUB
# ============================================================

class GitHubSummary(BaseModel):
    """
    GitHub information is enrichment only.

    GitHub availability must never determine basic eligibility.
    """

    model_config = ConfigDict(extra="forbid")

    available: bool = False

    username: str | None = None

    profile_url: str | None = None

    public_repositories: int = 0

    repositories: list[dict[str, Any]] = Field(
        default_factory=list
    )

    recent_activity_score: float = Field(
        default=0.0,
        ge=0,
        le=5,
    )

    relevant_repositories_score: float = Field(
        default=0.0,
        ge=0,
        le=5,
    )

    score: float = Field(
        default=0.0,
        ge=0,
        le=10,
    )

    summary: str = ""

    error: str | None = None

    concerns: list[str] = Field(
        default_factory=list
    )


# ============================================================
# DECISION TRACE
# ============================================================

class DecisionTrace(BaseModel):
    """
    Makes the final screening decision explainable.
    """

    model_config = ConfigDict(extra="forbid")

    python_gate: bool = False

    ai_gate: bool = False

    project_depth: str = "UNKNOWN"

    github_enrichment: str = "not_run"

    ranking_status: str = ""


# ============================================================
# RANKED CANDIDATE
# ============================================================

class RankedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(
        ge=1
    )

    name: str = "Unknown"

    email: str | None = None

    eligible: bool = True

    total_score: float = Field(
        ge=0,
        le=100,
    )

    score_breakdown: ScoreBreakdown

    matched_skills: list[str] = Field(
        default_factory=list
    )

    project_summary: list[str] = Field(
        default_factory=list
    )

    github_summary: GitHubSummary = Field(
        default_factory=GitHubSummary
    )

    strengths: list[str] = Field(
        default_factory=list
    )

    concerns: list[str] = Field(
        default_factory=list
    )

    evidence: dict[str, Any] = Field(
        default_factory=dict
    )

    decision_trace: DecisionTrace = Field(
        default_factory=DecisionTrace
    )


# ============================================================
# REJECTED CANDIDATE
# ============================================================

class RejectedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "Unknown"

    email: str | None = None

    eligible: bool = False

    rejection_reason: str = ""

    missing_requirements: list[str] = Field(
        default_factory=list
    )

    matched_requirements: list[str] = Field(
        default_factory=list
    )

    evidence: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# PROCESSING FAILURE
# ============================================================

class ProcessingFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str

    status: ProcessingStatus = ProcessingStatus.FAILED

    stage: str

    error_type: str

    message: str


# ============================================================
# RUN SUMMARY
# ============================================================

class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_resumes: int = Field(
        ge=0
    )

    processed: int = Field(
        ge=0
    )

    eligible: int = Field(
        ge=0
    )

    rejected: int = Field(
        ge=0
    )

    failed: int = Field(
        ge=0
    )

    @model_validator(mode="after")
    def validate_counts(self) -> "RunSummary":
        """
        The counters should always add up to the total number
        of discovered resumes.
        """

        if (
            self.processed
            + self.failed
            != self.total_resumes
        ):
            raise ValueError(
                "Run summary counts are inconsistent: "
                "processed + failed must equal total_resumes."
            )

        if (
            self.eligible
            + self.rejected
            != self.processed
        ):
            raise ValueError(
                "Run summary counts are inconsistent: "
                "eligible + rejected must equal processed."
            )

        return self


# ============================================================
# FINAL SCREENING RESULTS
# ============================================================

class ScreeningResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_summary: RunSummary

    ranked_candidates: list[RankedCandidate] = Field(
        default_factory=list
    )

    rejected_candidates: list[RejectedCandidate] = Field(
        default_factory=list
    )

    processing_failures: list[ProcessingFailure] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_ranks(self) -> "ScreeningResults":
        """
        Eligible candidates must have sequential ranks.
        """

        expected_rank = 1

        for candidate in self.ranked_candidates:
            if candidate.rank != expected_rank:
                raise ValueError(
                    "Ranked candidates must have sequential ranks."
                )

            expected_rank += 1

        return self