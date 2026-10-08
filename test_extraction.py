from pathlib import Path

from src.extraction.deterministic import extract_deterministic_evidence
from src.extraction.llm_extractor import extract_candidate_evidence
from src.ingestion.loader import load_resume


def build_deterministic_context(deterministic) -> str:
    """
    Build compact deterministic hints for the LLM.
    """

    lines = []

    useful_fields = {
        "ai_signals",
        "name_candidates",
        "emails",
        "phones",
        "github_urls",
        "linkedin_urls",
    }

    for field_name in useful_fields:
        if not hasattr(deterministic, field_name):
            continue

        value = getattr(
            deterministic,
            field_name,
        )

        if not value:
            continue

        if isinstance(value, (list, tuple, set)):
            lines.append(
                f"{field_name}: "
                + ", ".join(
                    str(item)
                    for item in value
                )
            )
        else:
            lines.append(
                f"{field_name}: {value}"
            )

    return "\n".join(lines)


def main() -> None:
    print("=" * 60)
    print("LLM RESUME EXTRACTION TEST")
    print("=" * 60)

    resume_files = sorted(
        Path("resumes").glob("*.pdf")
    )

    if not resume_files:
        raise FileNotFoundError(
            "No PDF resumes found in the resumes directory."
        )

    resume_path = resume_files[0]

    print(f"Testing resume: {resume_path.name}")

    # ---------------------------------------------------------
    # 1. Load resume
    # ---------------------------------------------------------

    loaded = load_resume(resume_path)

    print(f"Resume loaded: {len(loaded.text)} characters")

    # ---------------------------------------------------------
    # 2. Deterministic extraction
    # ---------------------------------------------------------

    deterministic = extract_deterministic_evidence(
        loaded
    )

    deterministic_context = build_deterministic_context(
        deterministic
    )

    print("\nDeterministic signals:")
    print(
        deterministic_context
        if deterministic_context
        else "  None"
    )

    # ---------------------------------------------------------
    # 3. LLM structured extraction
    # ---------------------------------------------------------

    print("\nSending resume to Gemini...")

    candidate = extract_candidate_evidence(
        resume_text=loaded.text,
        deterministic_signals=deterministic_context,
    )

    # ---------------------------------------------------------
    # 4. Display extracted candidate
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("EXTRACTED CANDIDATE")
    print("-" * 60)

    print(f"Name: {candidate.name}")
    print(f"Email: {candidate.email}")
    print(f"Phone: {candidate.phone}")
    print(f"Location: {candidate.location}")
    print(f"GitHub: {candidate.github_url}")
    print(f"LinkedIn: {candidate.linkedin_url}")

    # ---------------------------------------------------------
    # Skills
    # ---------------------------------------------------------

    print("\nSkills:")

    if candidate.skills:
        for skill in candidate.skills:
            print(
                f"  - {skill.name} "
                f"[{skill.strength.value}]"
            )
            print(f"    Evidence: {skill.evidence}")
    else:
        print("  None")

    # ---------------------------------------------------------
    # Evidence
    # ---------------------------------------------------------

    print("\nEvidence:")

    if candidate.evidence:
        for item in candidate.evidence:
            print(
                f"  - [{item.category}] "
                f"{item.claim} "
                f"[{item.strength.value}]"
            )
            print(f"    Source: {item.source}")
            print(f"    Evidence: {item.evidence}")
    else:
        print("  None")

    # ---------------------------------------------------------
    # Projects
    # ---------------------------------------------------------

    print("\nProjects:")

    if candidate.projects:
        for project in candidate.projects:
            print(f"  - {project.name}")
            print(f"    Description: {project.description}")
            print(
                "    Technologies: "
                + ", ".join(project.technologies)
            )

            print(f"    Python: {project.python_used}")
            print(f"    AI/LLM: {project.ai_or_llm_used}")
            print(f"    LLM: {project.llm}")
            print(f"    Tool calling: {project.tool_calling}")
            print(f"    Multi-agent: {project.multi_agent}")
            print(f"    Retrieval: {project.retrieval}")
            print(f"    Embeddings: {project.embeddings}")
            print(f"    Vector search: {project.vector_search}")
            print(f"    RAG: {project.rag}")
            print(
                f"    State management: "
                f"{project.state_management}"
            )
            print(
                f"    External API: "
                f"{project.external_api}"
            )
            print(
                f"    Persistence: "
                f"{project.persistence}"
            )
            print(
                f"    Evaluation: "
                f"{project.evaluation}"
            )
            print(
                f"    Validation: "
                f"{project.validation}"
            )
            print(
                f"    Business logic: "
                f"{project.business_logic}"
            )

    else:
        print("  None")

    # ---------------------------------------------------------
    # Internships
    # ---------------------------------------------------------

    print("\nInternships:")

    if candidate.internships:
        for internship in candidate.internships:
            print(f"  - {internship}")
    else:
        print("  None")

    # ---------------------------------------------------------
    # Experience
    # ---------------------------------------------------------

    print("\nExperience:")

    if candidate.experience:
        for experience in candidate.experience:
            print(f"  - {experience}")
    else:
        print("  None")

    # ---------------------------------------------------------
    # Certifications
    # ---------------------------------------------------------

    print("\nCertifications:")

    if candidate.certifications:
        for certification in candidate.certifications:
            print(f"  - {certification}")
    else:
        print("  None")

    # ---------------------------------------------------------
    # Education
    # ---------------------------------------------------------

    print("\nEducation:")

    if candidate.education:
        for education in candidate.education:
            print(f"  - {education}")
    else:
        print("  None")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\nSummary:")
    print(candidate.summary or "None")

    # ---------------------------------------------------------
    # Strengths
    # ---------------------------------------------------------

    print("\nStrengths:")

    if candidate.strengths:
        for strength in candidate.strengths:
            print(f"  - {strength}")
    else:
        print("  None")

    # ---------------------------------------------------------
    # Concerns
    # ---------------------------------------------------------

    print("\nConcerns:")

    if candidate.concerns:
        for concern in candidate.concerns:
            print(f"  - {concern}")
    else:
        print("  None")

    # ---------------------------------------------------------
    # Success
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXTRACTION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()