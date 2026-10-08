from openai import OpenAI

from src.llm.base import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMRequestError,
    LLMResponseError,
)
from config.settings import settings


class OpenAIProvider(BaseLLMProvider):
    """OpenAI implementation of the LLM provider interface."""

    @property
    def provider_name(self) -> str:
        return "openai"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model or settings.llm_model)

        if not settings.openai_api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=settings.openai_api_key
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a text response from OpenAI."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.1,
            )
        except Exception as exc:
            raise LLMRequestError(
                f"OpenAI request failed: {exc}"
            ) from exc

        if not response.choices:
            raise LLMResponseError(
                "OpenAI returned no choices."
            )

        text = response.choices[0].message.content

        if not text or not text.strip():
            raise LLMResponseError(
                "OpenAI returned an empty response."
            )

        return text.strip()