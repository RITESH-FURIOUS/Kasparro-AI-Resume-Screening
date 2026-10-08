from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.settings import load_settings
from src.pipeline.screening import ResumeScreeningPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Resume Screening & Ranking System"
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Directory containing resumes",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Path for the JSON output file",
    )

    return parser.parse_args()


def main() -> None:
    settings = load_settings()
    args = parse_args()

    input_dir = Path(
        args.input or settings.input_dir
    )

    output_file = Path(
        args.output or settings.output_file
    )

    print("=" * 70)
    print("AI RESUME SCREENING & RANKING SYSTEM")
    print("=" * 70)

    print(
        f"Input directory : {input_dir}"
    )

    print(
        f"Output file     : {output_file}"
    )

    print(
        f"LLM provider    : {settings.llm_provider}"
    )

    print(
        f"LLM model       : {settings.llm_model}"
    )

    print()
    print("Starting batch screening...")
    print()

    if not input_dir.exists():
        raise SystemExit(
            f"Input directory does not exist: {input_dir}"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline = ResumeScreeningPipeline(
        enable_github=settings.enable_github
    )

    results = pipeline.run(
        input_dir=input_dir
    )

    output_data = results.model_dump(
        mode="json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    summary = results.run_summary

    print("=" * 70)
    print("SCREENING COMPLETE")
    print("=" * 70)

    print(
        f"Total resumes : {summary.total_resumes}"
    )

    print(
        f"Processed     : {summary.processed}"
    )

    print(
        f"Eligible      : {summary.eligible}"
    )

    print(
        f"Rejected      : {summary.rejected}"
    )

    print(
        f"Failed        : {summary.failed}"
    )

    print()
    print(
        f"Results saved : {output_file}"
    )

    if results.ranked_candidates:
        print()
        print("TOP ELIGIBLE CANDIDATES")
        print("-" * 70)

        for candidate in results.ranked_candidates[:10]:
            print(
                f"#{candidate.rank:<3} "
                f"{candidate.name:<30} "
                f"{candidate.total_score:>6.2f}/100"
            )

    if results.rejected_candidates:
        print()
        print("REJECTED CANDIDATES")
        print("-" * 70)

        for candidate in results.rejected_candidates:
            print(
                f"{candidate.name}: "
                f"{candidate.rejection_reason}"
            )

    if results.processing_failures:
        print()
        print("PROCESSING FAILURES")
        print("-" * 70)

        for failure in results.processing_failures:
            print(
                f"{failure.filename}: "
                f"[{failure.stage}] "
                f"{failure.error_type} - "
                f"{failure.message}"
            )


if __name__ == "__main__":
    main()