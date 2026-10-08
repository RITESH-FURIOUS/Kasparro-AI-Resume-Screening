from src.llm.base import BaseLLMProvider, LLMConfigurationError
from src.llm.gemini import GeminiProvider
from src.llm.openai import OpenAIProvider
from src.llm.anthropic import AnthropicProvider
from config.settings import settings


def create_llm_provider(
    provider: str | None = None,
    model: str | None = None,
) -> BaseLLMProvider:
    """
    Create the configured LLM provider.

    The rest of the application should use this factory
    instead of importing provider-specific implementations.
    """

    provider_name = (
        provider or settings.llm_provider
    ).strip().lower()

    if provider_name == "gemini":
        return GeminiProvider(model=model)

    if provider_name == "openai":
        return OpenAIProvider(model=model)

    if provider_name == "anthropic":
        return AnthropicProvider(model=model)

    raise LLMConfigurationError(
        f"Unsupported LLM provider: {provider_name}. "
        "Supported providers: gemini, openai, anthropic."
    )