import json
import re

from pydantic import ValidationError

from config.settings import settings
from src.extraction.prompts import (
    SYSTEM_PROMPT,
    build_extraction_prompt,
)
from src.llm.base import LLMError
from src.llm.factory import create_llm_provider
from src.schemas import CandidateEvidence


class ExtractionError(Exception):
    """Raised when structured resume extraction fails."""


def _clean_json_response(response: str) -> str:
    """
    Remove common Markdown code fences from an LLM response.

    The model is instructed to return JSON only, but this makes the parser
    resilient if the model wraps the JSON in ```json ... ``` fences.
    """

    text = response.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    return text.strip()


def _parse_candidate_json(response: str) -> CandidateEvidence:
    """Validate the LLM response against the Pydantic candidate schema."""

    cleaned = _clean_json_response(response)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"LLM returned invalid JSON: {exc}"
        ) from exc

    try:
        return CandidateEvidence.model_validate(payload)
    except ValidationError as exc:
        raise ExtractionError(
            f"LLM response failed schema validation: {exc}"
        ) from exc


def extract_candidate_evidence(
    resume_text: str,
    deterministic_signals: str = "",
) -> CandidateEvidence:
    """
    Extract structured evidence from one resume.

    LLM failures are surfaced to the caller so the batch pipeline can record
    the individual failure without stopping the entire batch.
    """

    if not resume_text.strip():
        raise ExtractionError("Resume text is empty.")

    truncated_text = resume_text[: settings.max_resume_chars]

    provider = create_llm_provider()

    prompt = build_extraction_prompt(
        resume_text=truncated_text,
        deterministic_signals=deterministic_signals,
    )

    try:
        response = provider.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
        )
    except LLMError as exc:
        raise ExtractionError(
            f"LLM extraction failed: {exc}"
        ) from exc

    return _parse_candidate_json(response)