from src.llm.factory import create_llm_provider


def main() -> None:
    print("=" * 60)
    print("LLM FACTORY TEST")
    print("=" * 60)

    provider = create_llm_provider()

    print(f"Provider: {provider.provider_name}")
    print(f"Model: {provider.model}")

    response = provider.generate(
        system_prompt="Answer briefly.",
        user_prompt="What does REST stand for?",
    )

    print("\nResponse:")
    print(response)

    print("\nLLM factory test passed.")


if __name__ == "__main__":
    main()