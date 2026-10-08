from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Load .env from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """
    Central application configuration.

    Environment variables are used for secrets and deployment-
    specific configuration so no API key is hard-coded.
    """

    llm_provider: str
    llm_model: str

    gemini_api_key: str | None
    openai_api_key: str | None
    anthropic_api_key: str | None

    github_token: str | None

    input_dir: Path
    output_file: Path

    max_resume_chars: int

    enable_github: bool


def _get_optional_env(name: str) -> str | None:
    """
    Return an environment variable if it has a non-empty value.
    """

    value = os.getenv(name)

    if value is None:
        return None

    value = value.strip()

    return value or None


def _get_bool_env(
    name: str,
    default: bool,
) -> bool:
    """
    Parse a boolean environment variable safely.
    """

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _get_int_env(
    name: str,
    default: int,
) -> int:
    """
    Parse an integer environment variable.
    """

    value = os.getenv(name)

    if value is None:
        return default

    try:
        parsed = int(value)

    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer, got: {value!r}"
        ) from exc

    if parsed <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
        )

    return parsed


def load_settings() -> Settings:
    """
    Load and validate application configuration.
    """

    provider = (
        os.getenv(
            "LLM_PROVIDER",
            "gemini",
        )
        .strip()
        .lower()
    )

    supported_providers = {
        "gemini",
        "openai",
        "anthropic",
    }

    if provider not in supported_providers:
        raise ValueError(
            "Unsupported LLM_PROVIDER={!r}. "
            "Supported providers: {}".format(
                provider,
                ", ".join(sorted(supported_providers)),
            )
        )

    model = os.getenv(
        "LLM_MODEL",
        "",
    ).strip()

    if not model:
        raise ValueError(
            "LLM_MODEL is not configured in .env"
        )

    input_dir = Path(
        os.getenv(
            "INPUT_DIR",
            "./resumes",
        )
    )

    output_file = Path(
        os.getenv(
            "OUTPUT_FILE",
            "./output/results.json",
        )
    )

    return Settings(
        llm_provider=provider,
        llm_model=model,

        gemini_api_key=_get_optional_env(
            "GEMINI_API_KEY"
        ),

        openai_api_key=_get_optional_env(
            "OPENAI_API_KEY"
        ),

        anthropic_api_key=_get_optional_env(
            "ANTHROPIC_API_KEY"
        ),

        github_token=_get_optional_env(
            "GITHUB_TOKEN"
        ),

        input_dir=input_dir,
        output_file=output_file,

        max_resume_chars=_get_int_env(
            "MAX_RESUME_CHARS",
            50000,
        ),

        enable_github=_get_bool_env(
            "ENABLE_GITHUB",
            True,
        ),
    )


settings = load_settings()