from __future__ import annotations

from abc import ABC, abstractmethod


class LLMError(Exception):
    """
    Base exception for all LLM-related failures.
    """


class LLMConfigurationError(LLMError):
    """
    Raised when a provider is incorrectly configured.
    """


class LLMRequestError(LLMError):
    """
    Raised when an LLM request fails.
    """


class LLMResponseError(LLMError):
    """
    Raised when the provider responds but the response
    cannot be interpreted as expected.
    """


class BaseLLMProvider(ABC):
    """
    Common interface for every supported LLM provider.

    The rest of the application should depend on this interface,
    not on Gemini/OpenAI/Anthropic-specific SDKs.
    """

    def __init__(
        self,
        model: str,
    ) -> None:
        self.model = model

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Human-readable provider name.
        """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Send a prompt to the provider and return text.

        Provider-specific SDK handling belongs inside each
        concrete provider implementation.
        """

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(model={self.model!r})"
        )