from anthropic import Anthropic

from src.llm.base import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMRequestError,
    LLMResponseError,
)
from config.settings import settings


class AnthropicProvider(BaseLLMProvider):
    """Anthropic implementation of the LLM provider interface."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model or settings.llm_model)

        if not settings.anthropic_api_key:
            raise LLMConfigurationError(
                "ANTHROPIC_API_KEY is not configured."
            )

        self.client = Anthropic(
            api_key=settings.anthropic_api_key
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a text response from Anthropic."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )
        except Exception as exc:
            raise LLMRequestError(
                f"Anthropic request failed: {exc}"
            ) from exc

        if not response.content:
            raise LLMResponseError(
                "Anthropic returned an empty response."
            )

        text_parts = []

        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)

        text = "\n".join(text_parts).strip()

        if not text:
            raise LLMResponseError(
                "Anthropic returned no text content."
            )

        return text