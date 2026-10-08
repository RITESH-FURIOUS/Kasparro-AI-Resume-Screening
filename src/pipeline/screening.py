from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eligibility.filter import evaluate_eligibility
from src.extraction.deterministic import extract_deterministic_evidence
from src.extraction.llm_extractor import extract_candidate_evidence
from src.github.client import GitHubClient
from src.ingestion.loader import load_all_resumes
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
    """
    End-to-end batch-safe resume screening pipeline.

    Core principle:

        LLM understands the resume.
        Python makes the final decision.
    """

    def __init__(self, enable_github: bool = True):
        self.enable_github = enable_github
        self.github_client = (
            GitHubClient()
            if enable_github
            else None
        )

    # ------------------------------------------------------------------
    # GitHub
    # ------------------------------------------------------------------

    def _empty_github_summary(
        self,
        reason: str | None = None,
    ) -> GitHubSummary:
        return GitHubSummary(
            available=False,
            username=None,
            public_repositories=0,
            recent_activity_score=0.0,
            relevant_repositories_score=0.0,
            summary=reason or "GitHub enrichment unavailable.",
            concerns=[reason] if reason else [],
        )

    def _convert_github_result(
        self,
        github_result: Any,
    ) -> GitHubSummary:

        if github_result is None:
            return self._empty_github_summary(
                "GitHub enrichment returned no result."
            )

        username = getattr(
            github_result,
            "username",
            None,
        )

        repositories = getattr(
            github_result,
            "repositories",
            None,
        ) or []

        try:
            public_repository_count = len(repositories)
        except TypeError:
            public_repository_count = 0

        available = bool(
            getattr(
                github_result,
                "available",
                False,
            )
        )

        if not available and username:
            available = True

        recent_activity_score = float(
            getattr(
                github_result,
                "recent_activity_score",
                0.0,
            )
            or 0.0
        )

        relevant_repositories_score = float(
            getattr(
                github_result,
                "relevant_repository_score",
                0.0,
            )
            or 0.0
        )

        recent_activity_score = max(
            0.0,
            min(5.0, recent_activity_score),
        )

        relevant_repositories_score = max(
            0.0,
            min(5.0, relevant_repositories_score),
        )

        summary = (
            getattr(
                github_result,
                "summary",
                "",
            )
            or ""
        )

        error = getattr(
            github_result,
            "error",
            None,
        )

        concerns: list[str] = []

        if error:
            concerns.append(str(error))

        if not summary:
            if available:
                summary = (
                    "GitHub profile found with "
                    f"{public_repository_count} public repositories."
                )
            else:
                summary = (
                    "GitHub profile unavailable or no public "
                    "repository evidence found."
                )

        return GitHubSummary(
            available=available,
            username=username,
            public_repositories=public_repository_count,
            recent_activity_score=recent_activity_score,
            relevant_repositories_score=relevant_repositories_score,
            summary=summary,
            concerns=concerns,
        )

    def _enrich_github(
        self,
        candidate: CandidateEvidence,
    ) -> GitHubSummary:

        if not self.enable_github:
            return self._empty_github_summary(
                "GitHub enrichment is disabled."
            )

        if self.github_client is None:
            return self._empty_github_summary(
                "GitHub client is unavailable."
            )

        if not candidate.github_url:
            return self._empty_github_summary(
                "No GitHub profile URL was found in the resume."
            )

        try:
            github_result = self.github_client.enrich(
                candidate.github_url
            )

            return self._convert_github_result(
                github_result
            )

        except Exception as exc:
            return self._empty_github_summary(
                f"GitHub enrichment failed: "
                f"{type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matched_skills(
        candidate: CandidateEvidence,
    ) -> list[str]:

        seen: set[str] = set()
        result: list[str] = []

        for skill in candidate.skills:
            if not skill:
                continue

            cleaned = skill.strip()

            if not cleaned:
                continue

            key = cleaned.lower()

            if key in seen:
                continue

            seen.add(key)
            result.append(cleaned)

        return result

    @staticmethod
    def _project_summaries(
        candidate: CandidateEvidence,
    ) -> list[str]:

        summaries: list[str] = []

        for project in candidate.projects:

            name = getattr(
                project,
                "name",
                "Unnamed project",
            )

            description = getattr(
                project,
                "description",
                "",
            )

            technologies = getattr(
                project,
                "technologies",
                [],
            )

            parts: list[str] = []

            if description:
                parts.append(
                    description.strip()
                )

            if technologies:
                tech_text = ", ".join(
                    str(item)
                    for item in technologies
                    if item
                )

                if tech_text:
                    parts.append(
                        f"Technologies: {tech_text}"
                    )

            if parts:
                summaries.append(
                    f"{name}: " + " | ".join(parts)
                )
            else:
                summaries.append(str(name))

        return summaries

    @staticmethod
    def _candidate_evidence_dict(
        candidate: CandidateEvidence,
    ) -> dict[str, Any]:

        return candidate.model_dump(
            mode="json"
        )

    # ------------------------------------------------------------------
    # Candidate processing
    # ------------------------------------------------------------------

    def _process_candidate(
        self,
        loaded_resume: Any,
    ) -> tuple[
        RankedCandidate | RejectedCandidate | None,
        ProcessingFailure | None,
    ]:

        filename = Path(
            str(
                getattr(
                    loaded_resume,
                    "filename",
                    "unknown",
                )
            )
        ).name

        resume_text = str(
            getattr(
                loaded_resume,
                "text",
                "",
            )
        )

        try:

            if not resume_text.strip():
                return (
                    None,
                    ProcessingFailure(
                        filename=filename,
                        stage="ingestion",
                        message="Resume produced no readable text.",
                        error_type="EmptyResumeText",
                    ),
                )

            # ----------------------------------------------------------
            # 1. Deterministic extraction
            #
            # IMPORTANT:
            # extract_deterministic_evidence() expects LoadedResume,
            # NOT a string.
            # ----------------------------------------------------------

            deterministic = extract_deterministic_evidence(
                loaded_resume
            )

            # ----------------------------------------------------------
            # 2. Structured LLM/fallback extraction
            #
            # LLM extractor expects raw resume text plus the
            # deterministic evidence object.
            # ----------------------------------------------------------

            candidate = extract_candidate_evidence(
                resume_text,
                deterministic,
            )

            # ----------------------------------------------------------
            # 3. Hard eligibility gate
            # ----------------------------------------------------------

            eligibility = evaluate_eligibility(
                candidate
            )

            # ----------------------------------------------------------
            # 4. Reject immediately if requirements fail
            # ----------------------------------------------------------

            if (
                eligibility.status.value.upper()
                != "ELIGIBLE"
            ):

                missing_requirements: list[str] = []

                if not eligibility.python_gate:
                    missing_requirements.append(
                        "Python evidence"
                    )

                if not eligibility.ai_gate:
                    missing_requirements.append(
                        "AI/LLM/agentic evidence"
                    )

                matched_requirements: list[str] = []

                if eligibility.python_gate:
                    matched_requirements.append(
                        "Python evidence"
                    )

                if eligibility.ai_gate:
                    matched_requirements.append(
                        "AI/LLM/agentic evidence"
                    )

                rejected = RejectedCandidate(
                    name=candidate.name,
                    email=candidate.email,
                    eligible=False,
                    rejection_reason=eligibility.reason,
                    missing_requirements=missing_requirements,
                    matched_requirements=matched_requirements,
                    evidence=self._candidate_evidence_dict(
                        candidate
                    ),
                )

                return rejected, None

            # ----------------------------------------------------------
            # 5. Project quality
            # ----------------------------------------------------------

            project_quality = evaluate_project_quality(
                candidate
            )

            # ----------------------------------------------------------
            # 6. GitHub enrichment
            # ----------------------------------------------------------

            github_summary = self._enrich_github(
                candidate
            )

            github_score = (
                github_summary.recent_activity_score
                + github_summary.relevant_repositories_score
            )

            github_score = max(
                0.0,
                min(10.0, github_score),
            )

            # ----------------------------------------------------------
            # 7. Deterministic scoring
            # ----------------------------------------------------------

            score = calculate_score(
                candidate,
                github_score=github_score,
            )

            score_values = score.model_dump(
                mode="python"
            )

            total_score = round(
                sum(
                    float(value)
                    for value in score_values.values()
                ),
                2,
            )

            # ----------------------------------------------------------
            # 8. Decision trace
            # ----------------------------------------------------------

            decision_trace = DecisionTrace(
                python_gate=(
                    "PASS — genuine Python evidence found."
                    if eligibility.python_gate
                    else "FAIL — no genuine Python evidence found."
                ),
                ai_gate=(
                    "PASS — meaningful AI/LLM/agentic evidence found."
                    if eligibility.ai_gate
                    else "FAIL — no meaningful AI/LLM/agentic evidence found."
                ),
                project_depth=(
                    f"{project_quality.level} — "
                    f"depth score "
                    f"{project_quality.depth_score:.1f}; "
                    f"{project_quality.reasoning}"
                ),
                github_enrichment=github_summary.summary,
                ranking_status=(
                    f"Eligible candidate scored "
                    f"{total_score:.1f}/100."
                ),
            )

            # ----------------------------------------------------------
            # 9. Final candidate
            # ----------------------------------------------------------

            ranked = RankedCandidate(
                rank=0,
                name=candidate.name,
                email=candidate.email,
                eligible=True,
                total_score=total_score,
                score_breakdown=score,
                matched_skills=self._matched_skills(
                    candidate
                ),
                project_summary=self._project_summaries(
                    candidate
                ),
                github_summary=github_summary,
                strengths=list(candidate.strengths),
                concerns=list(candidate.concerns),
                evidence=self._candidate_evidence_dict(
                    candidate
                ),
                decision_trace=decision_trace,
            )

            return ranked, None

        except Exception as exc:

            failure = ProcessingFailure(
                filename=filename,
                stage="processing",
                message=str(exc),
                error_type=type(exc).__name__,
            )

            return None, failure

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def run(
        self,
        input_dir: str | Path,
    ) -> ScreeningResults:

        input_path = Path(input_dir)

        # Loader returns:
        #
        #   loaded resumes
        #   ingestion failures
        #
        loaded_resumes, ingestion_failures = load_all_resumes(
            input_path
        )

        ranked_candidates: list[RankedCandidate] = []
        rejected_candidates: list[RejectedCandidate] = []
        processing_failures: list[ProcessingFailure] = []

        # Preserve ingestion failures.
        for failure in ingestion_failures:

            processing_failures.append(
                ProcessingFailure(
                    filename=failure.filename,
                    stage=failure.stage,
                    message=failure.message,
                    error_type=failure.error_type,
                )
            )

        processed = 0

        for loaded_resume in loaded_resumes:

            result, failure = self._process_candidate(
                loaded_resume
            )

            if failure is not None:

                processing_failures.append(
                    failure
                )

                continue

            processed += 1

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

        # --------------------------------------------------------------
        # Final ranking
        # --------------------------------------------------------------

        ranked_candidates.sort(
            key=lambda candidate: candidate.total_score,
            reverse=True,
        )

        for index, candidate in enumerate(
            ranked_candidates,
            start=1,
        ):
            candidate.rank = index

        total_resumes = (
            len(loaded_resumes)
            + len(ingestion_failures)
        )

        run_summary = RunSummary(
            total_resumes=total_resumes,
            processed=processed,
            eligible=len(ranked_candidates),
            rejected=len(rejected_candidates),
            failed=len(processing_failures),
        )

        return ScreeningResults(
            run_summary=run_summary,
            ranked_candidates=ranked_candidates,
            rejected_candidates=rejected_candidates,
            processing_failures=processing_failures,
        )