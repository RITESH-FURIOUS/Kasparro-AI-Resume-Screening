from config.settings import settings


print("=" * 60)
print("CONFIGURATION TEST")
print("=" * 60)

print("Provider:", settings.llm_provider)
print("Model:", settings.llm_model)
print("Input directory:", settings.input_dir)
print("Output file:", settings.output_file)
print("Max resume chars:", settings.max_resume_chars)
print("GitHub enabled:", settings.enable_github)

print()
print("Gemini key configured:",
      bool(settings.gemini_api_key))

print("OpenAI key configured:",
      bool(settings.openai_api_key))

print("Anthropic key configured:",
      bool(settings.anthropic_api_key))

print()
print("Configuration loaded successfully.")