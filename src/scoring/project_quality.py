from __future__ import annotations

from src.schemas import CandidateEvidence, ProjectQuality


def _normalise_signal(value: str) -> str:
    """Normalise a project-depth signal into the scoring vocabulary."""

    value = value.strip().lower()

    aliases = {
        "llm": "llm",
        "large language model": "llm",
        "rag": "rag",
        "retrieval": "retrieval",
        "embeddings": "embeddings",
        "embedding": "embeddings",
        "vector search": "vector_search",
        "vector database": "vector_search",
        "vector db": "vector_search",
        "tool calling": "tool_calling",
        "tool-calling": "tool_calling",
        "function calling": "tool_calling",
        "multi-agent": "multi_agent",
        "multi agent": "multi_agent",
        "ai agents": "multi_agent",
        "agentic": "multi_agent",
        "evaluation": "evaluation",
        "eval": "evaluation",
        "state management": "state_management",
        "persistence": "persistence",
        "validation": "validation",
        "external api": "external_api",
        "external apis": "external_api",
        "api integration": "external_api",
    }

    return aliases.get(value, value.replace(" ", "_"))


def evaluate_project_quality(
    candidate: CandidateEvidence,
) -> ProjectQuality:
    """
    Evaluate the depth of AI/LLM/agentic project implementation.

    This is deliberately deterministic. The LLM extracts evidence;
    Python decides how much engineering depth that evidence represents.

    Levels:
        SHALLOW  -> 0-2 meaningful signals
        MODERATE -> 3-5 meaningful signals
        DEEP     -> 6+ meaningful signals
    """

    ai_projects = [
        project
        for project in candidate.projects
        if getattr(project, "ai_or_llm_used", False)
    ]

    if not ai_projects:
        return ProjectQuality(
            level="UNKNOWN",
            depth_score=0.0,
            shallow_ai_penalty=0.0,
            reasoning="No AI/LLM/agentic project evidence was identified.",
            signals=[],
        )

    # We evaluate the strongest AI project rather than adding unrelated
    # projects together. This prevents a candidate from appearing deep
    # merely because several shallow projects each contain one signal.
    best_signals: set[str] = set()

    for project in ai_projects:
        signals: set[str] = set()

        # Explicit boolean fields from ProjectEvidence.
        boolean_fields = {
            "llm": "llm",
            "rag": "rag",
            "retrieval": "retrieval",
            "embeddings": "embeddings",
            "vector_search": "vector_search",
            "tool_calling": "tool_calling",
            "multi_agent": "multi_agent",
            "evaluation": "evaluation",
            "state_management": "state_management",
            "persistence": "persistence",
            "validation": "validation",
            "external_api": "external_api",
        }

        for field_name, signal_name in boolean_fields.items():
            if bool(getattr(project, field_name, False)):
                signals.add(signal_name)

        # Some schema versions may expose explicit signal names/counts.
        explicit_signals = getattr(project, "ai_depth_signals", None)

        if explicit_signals:
            for signal in explicit_signals:
                if signal:
                    signals.add(_normalise_signal(str(signal)))

        # If the project exposes a precomputed count, use it only as a
        # conservative supplement. Never invent signal names.
        declared_count = getattr(project, "ai_depth_signal_count", None)

        if declared_count:
            try:
                declared_count = int(declared_count)
            except (TypeError, ValueError):
                declared_count = 0
        else:
            declared_count = 0

        # Evidence text can provide additional explicit implementation
        # signals when the project booleans are incomplete.
        for item in getattr(project, "evidence", []) or []:
            claim = str(getattr(item, "claim", "") or "")
            evidence = str(getattr(item, "evidence", "") or "")

            combined = f"{claim} {evidence}".lower()

            evidence_patterns = {
                "rag": ("rag", "retrieval augmented"),
                "retrieval": ("retrieval",),
                "embeddings": ("embedding",),
                "vector_search": (
                    "vector search",
                    "vector database",
                    "vector db",
                ),
                "tool_calling": (
                    "tool calling",
                    "tool-calling",
                    "function calling",
                    "tools",
                ),
                "multi_agent": (
                    "multi-agent",
                    "multi agent",
                    "multiple agents",
                    "agentic workflow",
                ),
                "evaluation": (
                    "evaluation pipeline",
                    "evaluated",
                    "evaluation",
                ),
                "state_management": (
                    "state management",
                    "agent state",
                ),
                "persistence": (
                    "database",
                    "persistence",
                    "stored",
                    "storage",
                ),
                "validation": (
                    "validation",
                    "validated",
                ),
                "external_api": (
                    "external api",
                    "external apis",
                    "third-party api",
                    "rest api",
                ),
            }

            for signal_name, patterns in evidence_patterns.items():
                if any(pattern in combined for pattern in patterns):
                    signals.add(signal_name)

        # A declared count may indicate additional structured evidence that
        # is not represented as individual booleans. Preserve it without
        # inventing fake signal names.
        if declared_count > len(signals):
            # Add only generic structured-depth markers. These contribute
            # to the depth decision but are not exposed as named signals.
            missing = declared_count - len(signals)
            for index in range(missing):
                signals.add(f"structured_signal_{index + 1}")

        if len(signals) > len(best_signals):
            best_signals = signals

    signal_count = len(best_signals)

    if signal_count >= 6:
        level = "DEEP"
    elif signal_count >= 3:
        level = "MODERATE"
    else:
        level = "SHALLOW"

    depth_score = float(min(signal_count, 10))

    shallow_penalty = 5.0 if level == "SHALLOW" else 0.0

    visible_signals = sorted(
        signal
        for signal in best_signals
        if not signal.startswith("structured_signal_")
    )

    if level == "DEEP":
        reasoning = (
            f"Strong AI/agentic implementation depth with "
            f"{signal_count} distinct engineering signals."
        )
    elif level == "MODERATE":
        reasoning = (
            f"Moderate AI implementation depth with "
            f"{signal_count} distinct engineering signals."
        )
    else:
        reasoning = (
            f"Limited AI implementation depth with only "
            f"{signal_count} distinct engineering signals; "
            f"shallow-AI penalty applied."
        )

    return ProjectQuality(
        level=level,
        depth_score=depth_score,
        shallow_ai_penalty=shallow_penalty,
        reasoning=reasoning,
        signals=visible_signals,
    )