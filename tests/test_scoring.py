from src.schemas import CandidateEvidence, ProjectEvidence
from src.scoring.score import calculate_score


def test_empty_candidate_scores_zero():
    candidate = CandidateEvidence(
        name="Empty Candidate",
        projects=[],
    )

    score = calculate_score(candidate)

    assert score.ai_agentic_rag == 0
    assert score.python_backend == 0
    assert score.cloud_full_stack == 0
    assert score.github_activity == 0
    assert score.engineering_depth == 0
    assert score.total == 0


def test_deep_ai_project_scores_higher():
    shallow = CandidateEvidence(
        name="Shallow",
        projects=[
            ProjectEvidence(
                name="Simple AI App",
                description="LLM API application",
                technologies=["Python"],
                evidence=[],
                python_used=True,
                ai_or_llm_used=True,
                llm=True,
            )
        ],
    )

    deep = CandidateEvidence(
        name="Deep",
        projects=[
            ProjectEvidence(
                name="Agentic RAG System",
                description="Agentic retrieval application",
                technologies=["Python"],
                evidence=[],
                python_used=True,
                ai_or_llm_used=True,
                llm=True,
                rag=True,
                retrieval=True,
                embeddings=True,
                vector_search=True,
                tool_calling=True,
                multi_agent=True,
                state_management=True,
                persistence=True,
                evaluation=True,
                validation=True,
                external_api=True,
            )
        ],
    )

    shallow_score = calculate_score(shallow)
    deep_score = calculate_score(deep)

    assert (
        deep_score.ai_agentic_rag
        > shallow_score.ai_agentic_rag
    )

    assert deep_score.total > shallow_score.total


def test_github_score_is_capped_at_ten():
    candidate = CandidateEvidence(
        name="GitHub Candidate",
        projects=[],
    )

    score = calculate_score(
        candidate,
        github_score=50,
    )

    assert score.github_activity == 10
    assert score.total <= 100