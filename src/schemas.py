from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================
# Enums
# ============================================================


class ProcessingStatus(str, Enum):
    PROCESSED = "PROCESSED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class EligibilityStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    REJECTED = "REJECTED"


class EvidenceStrength(str, Enum):
    NONE = "NONE"
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


# ============================================================
# Evidence
# ============================================================


class EvidenceItem(BaseModel):
    """
    A concrete piece of resume evidence supporting a claim.

    The system should prefer actual project/work evidence over
    isolated keyword mentions.
    """

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(
        ...,
        description="The capability or fact being supported.",
    )

    evidence: str = Field(
        ...,
        description="Exact or closely paraphrased evidence from the resume.",
    )

    source_section: str | None = Field(
        default=None,
        description="Section where the evidence was found, if identifiable.",
    )

    strength: EvidenceStrength = EvidenceStrength.MODERATE


class SkillEvidence(BaseModel):
    """
    Evidence for a particular skill/category.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    present: bool = False
    evidence: list[EvidenceItem] = Field(default_factory=list)


# ============================================================
# Project Evidence
# ============================================================


class ProjectEvidence(BaseModel):
    """
    Structured representation of a project.

    The goal is not simply to detect words like 'AI'.
    We want to understand what the project actually did.
    """

    model_config = ConfigDict(extra="forbid")

    name: str

    description: str = ""

    technologies: list[str] = Field(default_factory=list)

    python_used: bool = False

    ai_or_llm_used: bool = False

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

    evidence: list[EvidenceItem] = Field(default_factory=list)

    @property
    def ai_depth_signal_count(self) -> int:
        """
        Number of meaningful AI/agentic implementation signals.

        This is intentionally not the final score.
        It is simply an intermediate quality signal.
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

        return sum(signals)


# ============================================================
# Candidate Evidence
# ============================================================


class CandidateEvidence(BaseModel):
    """
    Main evidence ledger for one candidate.

    The LLM can populate this structure, but eligibility and
    scoring decisions will ultimately be made by deterministic
    Python logic.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "Unknown"
    email: str | None = None

    phone: str | None = None

    location: str | None = None

    github_url: str | None = None

    linkedin_url: str | None = None

    skills: list[str] = Field(default_factory=list)

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

    projects: list[ProjectEvidence] = Field(
        default_factory=list
    )

    internships: list[str] = Field(default_factory=list)

    experience: list[str] = Field(default_factory=list)

    certifications: list[str] = Field(default_factory=list)

    education: list[str] = Field(default_factory=list)

    summary: str = ""

    concerns: list[str] = Field(default_factory=list)

    strengths: list[str] = Field(default_factory=list)

    @property
    def has_python_evidence(self) -> bool:
        return bool(self.python_evidence)

    @property
    def has_ai_evidence(self) -> bool:
        return bool(self.ai_evidence)

    @property
    def has_meaningful_ai_project(self) -> bool:
        """
        Returns True when at least one project contains
        meaningful AI/LLM/agentic implementation evidence.
        """

        for project in self.projects:
            if not project.ai_or_llm_used:
                continue

            if project.ai_depth_signal_count >= 2:
                return True

        return False


# ============================================================
# Eligibility
# ============================================================


class EligibilityDecision(BaseModel):
    """
    Deterministic eligibility decision.

    Python code will generate this after examining the
    extracted evidence.
    """

    model_config = ConfigDict(extra="forbid")

    status: EligibilityStatus

    python_gate: bool

    ai_gate: bool

    reason: str

    matched_requirements: list[str] = Field(
        default_factory=list
    )

    missing_requirements: list[str] = Field(
        default_factory=list
    )


# ============================================================
# Project Quality
# ============================================================


class ProjectQuality(BaseModel):
    """
    Quality assessment of the candidate's AI/agentic work.

    This is an intermediate representation, not the final
    100-point candidate score.
    """

    model_config = ConfigDict(extra="forbid")

    level: str = "UNKNOWN"

    depth_score: float = 0.0

    shallow_ai_penalty: float = 0.0

    reasoning: str = ""

    signals: list[str] = Field(default_factory=list)


# ============================================================
# Score Breakdown
# ============================================================


class ScoreBreakdown(BaseModel):
    """
    Final deterministic scoring breakdown.

    Maximum = 100.
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
# GitHub Enrichment
# ============================================================


class GitHubSummary(BaseModel):
    """
    GitHub information is enrichment only.

    GitHub availability must never determine basic eligibility.
    """

    model_config = ConfigDict(extra="forbid")

    available: bool = False

    username: str | None = None

    public_repositories: int = 0

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

    summary: str = ""

    concerns: list[str] = Field(default_factory=list)


# ============================================================
# Decision Trace
# ============================================================


class DecisionTrace(BaseModel):
    """
    Human-readable trace of why the candidate reached a
    particular final state.
    """

    model_config = ConfigDict(extra="forbid")

    python_gate: str

    ai_gate: str

    project_depth: str

    github_enrichment: str

    ranking_status: str


# ============================================================
# Ranked Candidate
# ============================================================


class RankedCandidate(BaseModel):
    """
    Final candidate representation appearing in results.json.
    """

    model_config = ConfigDict(extra="forbid")

    rank: int

    name: str

    email: str | None = None

    eligible: bool

    total_score: float

    score_breakdown: ScoreBreakdown

    matched_skills: list[str] = Field(
        default_factory=list
    )

    project_summary: list[str] = Field(
        default_factory=list
    )

    github_summary: GitHubSummary

    strengths: list[str] = Field(
        default_factory=list
    )

    concerns: list[str] = Field(
        default_factory=list
    )

    evidence: dict[str, Any] = Field(
        default_factory=dict
    )

    decision_trace: DecisionTrace


# ============================================================
# Rejected Candidate
# ============================================================


class RejectedCandidate(BaseModel):
    """
    Candidate who was processed successfully but failed
    the mandatory Python + AI/LLM eligibility gate.
    """

    model_config = ConfigDict(extra="forbid")

    name: str

    email: str | None = None

    eligible: bool = False

    rejection_reason: str

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
# Processing Failure
# ============================================================


class ProcessingFailure(BaseModel):
    """
    A resume that could not be processed.

    This is different from rejection.

    REJECTED = processed successfully but not eligible.

    FAILED = system could not reliably process the resume.
    """

    model_config = ConfigDict(extra="forbid")

    filename: str

    status: ProcessingStatus = ProcessingStatus.FAILED

    stage: str

    error_type: str

    message: str


# ============================================================
# Batch Summary
# ============================================================


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_resumes: int = 0

    processed: int = 0

    eligible: int = 0

    rejected: int = 0

    failed: int = 0


# ============================================================
# Final Results
# ============================================================


class ScreeningResults(BaseModel):
    """
    Top-level object written to results.json.
    """

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

    @field_validator("ranked_candidates")
    @classmethod
    def validate_rank_order(
        cls,
        candidates: list[RankedCandidate],
    ) -> list[RankedCandidate]:
        """
        Ensure ranked candidates are actually ordered by rank.
        """

        expected_rank = 1

        for candidate in candidates:
            if candidate.rank != expected_rank:
                raise ValueError(
                    "Ranked candidates must have sequential ranks "
                    "starting from 1."
                )

            expected_rank += 1

        return candidates