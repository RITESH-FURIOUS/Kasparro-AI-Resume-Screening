import time

from google import genai
from google.genai import types

from src.llm.base import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMRequestError,
    LLMResponseError,
)
from config.settings import settings


class GeminiProvider(BaseLLMProvider):
    """Gemini implementation of the LLM provider interface."""

    MAX_RETRIES = 3
    RETRY_DELAYS = (2, 5, 10)

    @property
    def provider_name(self) -> str:
        return "gemini"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model or settings.llm_model)

        if not settings.gemini_api_key:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(
            api_key=settings.gemini_api_key
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a text response from Gemini with retry handling."""

        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1,
                    ),
                )

                text = getattr(response, "text", None)

                if not text or not text.strip():
                    raise LLMResponseError(
                        "Gemini returned an empty response."
                    )

                return text.strip()

            except LLMResponseError:
                raise

            except Exception as exc:
                last_error = exc

                error_text = str(exc).lower()

                temporary_error = (
                    "503" in error_text
                    or "unavailable" in error_text
                    or "429" in error_text
                    or "resource exhausted" in error_text
                    or "too many requests" in error_text
                )

                if not temporary_error:
                    raise LLMRequestError(
                        f"Gemini request failed: {exc}"
                    ) from exc

                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(
                        self.RETRY_DELAYS[attempt]
                    )

        raise LLMRequestError(
            f"Gemini request failed after "
            f"{self.MAX_RETRIES} attempts: {last_error}"
        ) from last_error