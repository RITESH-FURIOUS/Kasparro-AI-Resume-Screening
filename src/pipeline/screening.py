from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eligibility.filter import evaluate_eligibility
from src.extraction.deterministic import extract_deterministic_evidence
from src.extraction.llm_extractor import extract_candidate_evidence
from src.github.client import GitHubClient
from src.ingestion.loader import discover_resume_files, load_resume
from src.schemas import (
    CandidateEvidence,
    DecisionTrace,
    GitHubSummary,
    ProcessingFailure,
    RankedCandidate,
    RejectedCandidate,
    RunSummary,
    ScreeningResults,
)
from src.scoring.project_quality import evaluate_project_quality
from src.scoring.score import calculate_score


class ResumeScreeningPipeline:

    def __init__(self, enable_github: bool = True):
        self.enable_github = enable_github
        self.github_client = (
            GitHubClient() if enable_github else None
        )

    def _build_deterministic_context(
        self,
        deterministic: Any,
    ) -> str:

        parts: list[str] = []

        normalized_text = getattr(
            deterministic,
            "normalized_text",
            "",
        )

        if normalized_text:
            parts.append(normalized_text)

        technologies = getattr(
            deterministic,
            "technologies",
            {},
        )

        if technologies:
            parts.append(
                "Detected technologies: "
                + ", ".join(
                    str(key)
                    for key in technologies.keys()
                )
            )

        ai_signals = getattr(
            deterministic,
            "ai_signals",
            [],
        )

        if ai_signals:
            parts.append(
                "Detected AI signals: "
                + ", ".join(
                    str(signal)
                    for signal in ai_signals
                )
            )

        return "\n".join(parts)

    def _matched_skills(
        self,
        candidate: CandidateEvidence,
    ) -> list[str]:

        return sorted(
            {
                str(skill).strip()
                for skill in candidate.skills
                if str(skill).strip()
            },
            key=str.lower,
        )

    def _project_summaries(
        self,
        candidate: CandidateEvidence,
    ) -> list[str]:

        summaries: list[str] = []

        for project in candidate.projects:

            description = project.description.strip()

            if description:
                summaries.append(
                    f"{project.name}: {description}"
                )
            else:
                summaries.append(project.name)

        return summaries

    def _build_evidence(
        self,
        candidate: CandidateEvidence,
    ) -> dict[str, Any]:

        return {
            "python": [
                item.model_dump(mode="json")
                for item in candidate.python_evidence
            ],
            "ai": [
                item.model_dump(mode="json")
                for item in candidate.ai_evidence
            ],
            "backend": [
                item.model_dump(mode="json")
                for item in candidate.backend_evidence
            ],
            "cloud": [
                item.model_dump(mode="json")
                for item in candidate.cloud_evidence
            ],
            "frontend": [
                item.model_dump(mode="json")
                for item in candidate.frontend_evidence
            ],
            "github": [
                item.model_dump(mode="json")
                for item in candidate.github_evidence
            ],
        }

    def _decision_trace(
        self,
        candidate: CandidateEvidence,
        eligible: bool,
        project_quality: Any | None = None,
        github_summary: GitHubSummary | None = None,
    ) -> DecisionTrace:

        python_gate = (
            "PASS: genuine Python evidence found."
            if candidate.has_python_evidence
            else "FAIL: no genuine Python evidence found."
        )

        ai_gate = (
            "PASS: meaningful AI/LLM/agentic evidence found."
            if candidate.has_ai_evidence
            else "FAIL: no meaningful AI/LLM/agentic evidence found."
        )

        if project_quality is None:

            project_depth = (
                "Not evaluated because the candidate was rejected."
            )

        else:

            depth = getattr(
                project_quality,
                "depth",
                "UNKNOWN",
            )

            score = getattr(
                project_quality,
                "score",
                0,
            )

            project_depth = (
                f"AI project depth: {depth}; "
                f"depth signal score: {score}."
            )

        if github_summary is None:

            github_enrichment = (
                "GitHub enrichment unavailable or disabled."
            )

        elif github_summary.error:

            github_enrichment = (
                "GitHub enrichment attempted but unavailable: "
                f"{github_summary.error}"
            )

        else:

            github_enrichment = (
                "GitHub enrichment completed for "
                f"{github_summary.username or 'unknown user'}."
            )

        ranking_status = (
            "Candidate is eligible and included in ranking."
            if eligible
            else "Candidate is rejected and excluded from ranking."
        )

        return DecisionTrace(
            python_gate=python_gate,
            ai_gate=ai_gate,
            project_depth=project_depth,
            github_enrichment=github_enrichment,
            ranking_status=ranking_status,
        )

    def _default_github_summary(
        self,
    ) -> GitHubSummary:

        return GitHubSummary(
            username=None,
            profile_url=None,
            public_repositories=0,
            recent_activity_score=0.0,
            relevant_repository_score=0.0,
            score=0.0,
            repositories=[],
            error=None,
        )

    def _convert_github_summary(
        self,
        summary: Any,
    ) -> GitHubSummary:

        if isinstance(
            summary,
            GitHubSummary,
        ):
            return summary

        repositories = []

        for repo in (
            getattr(
                summary,
                "repositories",
                [],
            )
            or []
        ):

            if hasattr(
                repo,
                "__dict__",
            ):

                repositories.append(
                    dict(repo.__dict__)
                )

            elif isinstance(
                repo,
                dict,
            ):

                repositories.append(repo)

            else:

                repositories.append(
                    str(repo)
                )

        return GitHubSummary(
            username=getattr(
                summary,
                "username",
                None,
            ),
            profile_url=getattr(
                summary,
                "profile_url",
                None,
            ),
            public_repositories=int(
                getattr(
                    summary,
                    "public_repositories",
                    0,
                )
                or 0
            ),
            recent_activity_score=float(
                getattr(
                    summary,
                    "recent_activity_score",
                    0.0,
                )
                or 0.0
            ),
            relevant_repository_score=float(
                getattr(
                    summary,
                    "relevant_repository_score",
                    0.0,
                )
                or 0.0
            ),
            score=float(
                getattr(
                    summary,
                    "score",
                    0.0,
                )
                or 0.0
            ),
            repositories=repositories,
            error=getattr(
                summary,
                "error",
                None,
            ),
        )

    def _process_one(
        self,
        resume_path: Path,
    ) -> RankedCandidate | RejectedCandidate:

        loaded_resume = load_resume(
            resume_path
        )

        deterministic = extract_deterministic_evidence(
            loaded_resume
        )

        # The LLM extractor already receives the resume
        # and deterministic evidence. It does not accept
        # a separate "context" keyword argument.
        candidate = extract_candidate_evidence(
            loaded_resume.text,
            deterministic,
        )

        eligibility = evaluate_eligibility(
            candidate
        )

        evidence = self._build_evidence(
            candidate
        )

        evidence["python_gate"] = {
            "status": (
                "PASS"
                if candidate.has_python_evidence
                else "FAIL"
            ),
            "reason": (
                "Genuine Python evidence found."
                if candidate.has_python_evidence
                else (
                    "No genuine Python evidence found "
                    "in projects, internships, or work experience."
                )
            ),
        }

        evidence["ai_gate"] = {
            "status": (
                "PASS"
                if candidate.has_ai_evidence
                else "FAIL"
            ),
            "reason": (
                "Meaningful AI/LLM/agentic evidence found."
                if candidate.has_ai_evidence
                else (
                    "No meaningful AI/LLM/agentic "
                    "implementation evidence found."
                )
            ),
        }

        if eligibility.status.value != "ELIGIBLE":

            return RejectedCandidate(
                name=candidate.name,
                email=candidate.email,
                eligible=False,
                rejection_reason=eligibility.reason,
                missing_requirements=(
                    eligibility.missing_requirements
                ),
                matched_requirements=(
                    eligibility.matched_requirements
                ),
                evidence=evidence,
            )

        project_quality = evaluate_project_quality(
            candidate
        )

        github_summary = (
            self._default_github_summary()
        )

        if (
            self.enable_github
            and self.github_client is not None
            and candidate.github_url
        ):

            try:

                raw_github_summary = (
                    self.github_client.summarize(
                        candidate.github_url
                    )
                )

                github_summary = (
                    self._convert_github_summary(
                        raw_github_summary
                    )
                )

            except Exception as exc:

                github_summary = GitHubSummary(
                    username=None,
                    profile_url=candidate.github_url,
                    public_repositories=0,
                    recent_activity_score=0.0,
                    relevant_repository_score=0.0,
                    score=0.0,
                    repositories=[],
                    error=(
                        f"GitHub enrichment failed: {exc}"
                    ),
                )

        score_breakdown = calculate_score(
            candidate=candidate,
            project_quality=project_quality,
            github_score=github_summary.score,
        )

        total_score = round(
            score_breakdown.ai_agentic_rag
            + score_breakdown.python_backend
            + score_breakdown.cloud_full_stack
            + score_breakdown.github_activity
            + score_breakdown.engineering_depth,
            2,
        )

        decision_trace = self._decision_trace(
            candidate=candidate,
            eligible=True,
            project_quality=project_quality,
            github_summary=github_summary,
        )

        return RankedCandidate(
            rank=0,
            name=candidate.name,
            email=candidate.email,
            eligible=True,
            total_score=total_score,
            score_breakdown=score_breakdown,
            matched_skills=self._matched_skills(
                candidate
            ),
            project_summary=self._project_summaries(
                candidate
            ),
            github_summary=github_summary,
            strengths=candidate.strengths,
            concerns=candidate.concerns,
            evidence=evidence,
            decision_trace=decision_trace,
        )

    def process_resume(
        self,
        resume_path: Path,
    ) -> (
        RankedCandidate
        | RejectedCandidate
        | ProcessingFailure
    ):

        filename = resume_path.name

        try:

            return self._process_one(
                resume_path
            )

        except Exception as exc:

            return ProcessingFailure(
                filename=filename,
                stage="processing",
                message=str(exc),
                error_type=type(exc).__name__,
            )

    def run(
        self,
        input_dir: Path,
    ) -> ScreeningResults:

        input_dir = Path(
            input_dir
        )

        resume_files = discover_resume_files(
            input_dir
        )

        ranked_candidates: list[
            RankedCandidate
        ] = []

        rejected_candidates: list[
            RejectedCandidate
        ] = []

        processing_failures: list[
            ProcessingFailure
        ] = []

        for resume_path in resume_files:

            result = self.process_resume(
                resume_path
            )

            if isinstance(
                result,
                RankedCandidate,
            ):

                ranked_candidates.append(
                    result
                )

            elif isinstance(
                result,
                RejectedCandidate,
            ):

                rejected_candidates.append(
                    result
                )

            else:

                processing_failures.append(
                    result
                )

        ranked_candidates.sort(
            key=lambda candidate: candidate.total_score,
            reverse=True,
        )

        for index, candidate in enumerate(
            ranked_candidates,
            start=1,
        ):

            candidate.rank = index

        summary = RunSummary(
            total_resumes=len(
                resume_files
            ),
            processed=(
                len(ranked_candidates)
                + len(rejected_candidates)
            ),
            eligible=len(
                ranked_candidates
            ),
            rejected=len(
                rejected_candidates
            ),
            failed=len(
                processing_failures
            ),
        )

        return ScreeningResults(
            run_summary=summary,
            ranked_candidates=ranked_candidates,
            rejected_candidates=rejected_candidates,
            processing_failures=processing_failures,
        )