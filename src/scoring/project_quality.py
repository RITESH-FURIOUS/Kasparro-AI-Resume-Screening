from src.schemas import CandidateEvidence, ProjectQuality


def evaluate_project_quality(
    candidate: CandidateEvidence,
) -> ProjectQuality:
    """
    Evaluate the depth of the candidate's AI projects.

    This does not assign the final 100-point score.
    It provides deterministic signals used by the scoring layer.
    """

    ai_projects = [
        project
        for project in candidate.projects
        if project.ai_or_llm_used
    ]

    if not ai_projects:
        return ProjectQuality(
            depth="none",
            score=0,
            reasons=["No AI/LLM project evidence found."],
        )

    depth_signals = 0
    reasons: list[str] = []

    for project in ai_projects:
        signals = project.ai_depth_signal_count

        depth_signals += signals

        if project.rag:
            reasons.append(f"{project.name}: RAG implementation")

        if project.tool_calling:
            reasons.append(f"{project.name}: tool calling")

        if project.multi_agent:
            reasons.append(f"{project.name}: multi-agent workflow")

        if project.embeddings:
            reasons.append(f"{project.name}: embeddings")

        if project.vector_search:
            reasons.append(f"{project.name}: vector search")

        if project.retrieval:
            reasons.append(f"{project.name}: retrieval")

        if project.evaluation:
            reasons.append(f"{project.name}: evaluation")

        if project.persistence:
            reasons.append(f"{project.name}: persistence")

        if project.validation:
            reasons.append(f"{project.name}: validation")

    if depth_signals >= 6:
        depth = "deep"
    elif depth_signals >= 3:
        depth = "moderate"
    else:
        depth = "shallow"

    return ProjectQuality(
        depth=depth,
        score=depth_signals,
        reasons=reasons,
    )