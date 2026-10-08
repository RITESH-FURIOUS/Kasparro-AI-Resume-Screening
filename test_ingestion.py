from pathlib import Path

from src.extraction.deterministic import (
    extract_candidate_name,
    extract_deterministic_evidence,
)
from src.ingestion.loader import load_all_resumes


input_dir = Path("resumes")

resumes, failures = load_all_resumes(input_dir)

print("=" * 70)
print("DETERMINISTIC EXTRACTION TEST")
print("=" * 70)

print(f"Successfully loaded: {len(resumes)}")
print(f"Failed to load:      {len(failures)}")
print()


for resume in resumes:
    evidence = extract_deterministic_evidence(resume)

    name = extract_candidate_name(
        evidence.normalized_text,
        resume.filename,
    )

    print("-" * 70)
    print(resume.filename)

    print(f"Name guess: {name}")

    print(f"Email: {evidence.emails}")

    print(f"GitHub: {evidence.github_urls}")

    print(f"LinkedIn: {evidence.linkedin_urls}")

    print(
        "Technologies:",
        ", ".join(evidence.technologies.keys())
    )

    print(
        "AI signals:",
        ", ".join(evidence.ai_signals.keys())
    )


print()
print("=" * 70)
print("INGESTION FAILURES")
print("=" * 70)

for failure in failures:
    print(
        f"{failure.filename}: "
        f"{failure.error_type} - "
        f"{failure.message}"
    )