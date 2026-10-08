from src.llm.gemini import GeminiProvider


def main() -> None:
    print("=" * 60)
    print("GEMINI PROVIDER TEST")
    print("=" * 60)

    provider = GeminiProvider()

    print(f"Provider: {provider.provider_name}")
    print(f"Model: {provider.model}")
    print("Sending test request...")

    response = provider.generate(
        system_prompt=(
            "You are a helpful assistant. "
            "Answer briefly and clearly."
        ),
        user_prompt=(
            "Reply with exactly one sentence explaining "
            "what a Python function is."
        ),
    )

    print("\nGemini response:")
    print(response)

    print("\nGemini provider test passed.")


if __name__ == "__main__":
    main()