SYSTEM_PROMPT = """
You are a senior technical recruiter and software engineer reviewing resumes
for a software engineering role focused on Python, backend engineering, and
AI/LLM/agentic systems.

Your job is to extract evidence from the resume, NOT to make the final hiring
decision.

IMPORTANT RULES:

1. Use ONLY information explicitly supported by the resume.
2. Never invent technologies, projects, experience, metrics, or responsibilities.
3. A technology mentioned only in a generic skills list is weaker evidence than
   a technology demonstrated inside a project, internship, or work experience.
4. Preserve useful evidence snippets from the resume.
5. Distinguish genuine implementation evidence from keyword mentions.
6. AI/LLM evidence should capture concrete implementation details such as:
   - LLM usage
   - RAG
   - retrieval
   - embeddings
   - vector search
   - tool calling
   - agents
   - multi-agent workflows
   - LangChain
   - LangGraph
   - LlamaIndex
   - CrewAI
   - Google ADK
   - evaluation pipelines
   - prompt engineering
   - state management
7. Generic statements such as "interested in AI" or "familiar with AI" are
   NOT meaningful implementation evidence.
8. Python evidence should identify where Python was actually used, especially
   in projects, internships, work experience, APIs, backend systems, or
   automation.
9. Do not decide eligibility or assign scores.
10. Return ONLY valid JSON matching the requested schema.
"""


def build_extraction_prompt(
    resume_text: str,
    deterministic_signals: str,
) -> str:
    return f"""
Extract structured candidate evidence from the resume below.

The deterministic parser found these preliminary signals. Treat them only as
hints and verify them against the actual resume text:

{deterministic_signals}

Return a JSON object with exactly these top-level fields:

{{
  "name": null,
  "email": null,
  "phone": null,
  "location": null,
  "github_url": null,
  "linkedin_url": null,
  "skills": [],
  "evidence": [],
  "projects": [],
  "internships": [],
  "experience": [],
  "certifications": [],
  "education": [],
  "summary": null,
  "strengths": [],
  "concerns": []
}}

For every evidence item, capture:
- category
- claim
- strength: "strong", "moderate", or "weak"
- source
- evidence

For every skill, capture:
- name
- evidence
- strength

For every project, capture:
- name
- description
- technologies
- evidence
- python_used
- ai_or_llm_used
- llm
- tool_calling
- multi_agent
- retrieval
- embeddings
- vector_search
- rag
- state_management
- external_api
- persistence
- evaluation
- validation
- business_logic

Only set a boolean to true when the resume provides evidence for it.

IMPORTANT:
- Do not infer that a project used Python merely because Python appears in
  the candidate's skills section.
- Do not infer AI/LLM usage from a generic AI-related job title.
- Do not treat "AI enthusiast", "machine learning", or similar generic
  wording as concrete LLM/agentic implementation.
- Keep evidence concise and traceable to the resume.
- If information is unavailable, use null or an empty list.

RESUME:
--- BEGIN RESUME ---
{resume_text}
--- END RESUME ---
"""