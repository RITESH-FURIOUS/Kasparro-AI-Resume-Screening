from src.schemas import (
    CandidateEvidence,
    EligibilityDecision,
    EligibilityStatus,
)


def evaluate_eligibility(
    candidate: CandidateEvidence,
) -> EligibilityDecision:
    """
    Apply the hard eligibility gate.

    A candidate must have:
    1. Genuine Python evidence.
    2. Meaningful AI / LLM / agentic evidence.

    The LLM extracts evidence, but this deterministic Python
    function makes the final eligibility decision.
    """

    python_gate = candidate.has_python_evidence
    ai_gate = candidate.has_ai_evidence

    if python_gate and ai_gate:
        return EligibilityDecision(
            status=EligibilityStatus.ELIGIBLE,
            python_gate=True,
            ai_gate=True,
            reason=(
                "Eligible: genuine Python evidence and meaningful "
                "AI/LLM/agentic evidence were found."
            ),
        )

    reasons = []

    if not python_gate:
        reasons.append(
            "No genuine Python evidence found in projects, "
            "internships, or work experience."
        )

    if not ai_gate:
        reasons.append(
            "No meaningful AI/LLM/agentic implementation evidence found."
        )

    return EligibilityDecision(
        status=EligibilityStatus.REJECTED,
        python_gate=python_gate,
        ai_gate=ai_gate,
        reason=" ".join(reasons),
    )