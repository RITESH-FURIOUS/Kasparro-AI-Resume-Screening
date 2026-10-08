from src.eligibility.filter import evaluate_eligibility
from src.schemas import (
    CandidateEvidence,
    EvidenceItem,
    EvidenceStrength,
    ProjectEvidence,
)


def make_evidence(
    claim: str,
    evidence: str,
) -> EvidenceItem:
    return EvidenceItem(
        claim=claim,
        evidence=evidence,
        source_section="Projects",
        strength=EvidenceStrength.STRONG,
    )


def make_candidate(
    *,
    python: bool,
    ai: bool,
    meaningful_ai: bool = False,
) -> CandidateEvidence:

    projects = []

    python_evidence = []
    ai_evidence = []

    if python:
        python_evidence.append(
            make_evidence(
                claim="Python implementation",
                evidence="Implemented the project using Python.",
            )
        )

    if ai:
        ai_evidence.append(
            make_evidence(
                claim="AI/LLM implementation",
                evidence=(
                    "Implemented an AI/LLM system with "
                    "actual model integration."
                ),
            )
        )

    if meaningful_ai:
        ai_evidence.append(
            make_evidence(
                claim="RAG implementation",
                evidence="Implemented retrieval and RAG.",
            )
        )

        ai_evidence.append(
            make_evidence(
                claim="Tool calling",
                evidence=(
                    "Integrated external tools with the AI system."
                ),
            )
        )

    if python or ai:
        projects.append(
            ProjectEvidence(
                name="Test Project",
                description="Test project",
                technologies=["Python"] if python else [],
                evidence=[],
                python_used=python,
                ai_or_llm_used=ai,
                llm=ai,
                rag=meaningful_ai,
                retrieval=meaningful_ai,
                tool_calling=meaningful_ai,
            )
        )

    return CandidateEvidence(
        name="Test Candidate",
        python_evidence=python_evidence,
        ai_evidence=ai_evidence,
        projects=projects,
    )


def test_python_and_ai_is_eligible():
    candidate = make_candidate(
        python=True,
        ai=True,
        meaningful_ai=True,
    )

    decision = evaluate_eligibility(candidate)

    assert decision.python_gate is True
    assert decision.ai_gate is True
    assert decision.status.value == "ELIGIBLE"


def test_python_without_ai_is_rejected():
    candidate = make_candidate(
        python=True,
        ai=False,
    )

    decision = evaluate_eligibility(candidate)

    assert decision.python_gate is True
    assert decision.ai_gate is False
    assert decision.status.value == "REJECTED"


def test_ai_without_python_is_rejected():
    candidate = make_candidate(
        python=False,
        ai=True,
        meaningful_ai=True,
    )

    decision = evaluate_eligibility(candidate)

    assert decision.python_gate is False
    assert decision.ai_gate is True
    assert decision.status.value == "REJECTED"


def test_neither_python_nor_ai_is_rejected():
    candidate = make_candidate(
        python=False,
        ai=False,
    )

    decision = evaluate_eligibility(candidate)

    assert decision.python_gate is False
    assert decision.ai_gate is False
    assert decision.status.value == "REJECTED"