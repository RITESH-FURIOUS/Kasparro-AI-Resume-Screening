# 🤖 AI Resume Screening & Ranking System

> **Kasparro SDE Intern Assignment — AI Resume Screening, Eligibility Filtering & Ranking**

A production-minded Python system that processes a batch of resumes, extracts structured candidate evidence, applies hard eligibility rules, evaluates AI/LLM project depth, enriches candidates with GitHub activity, and produces a transparent ranked JSON report.

The core design principle is:

> **LLM understands the resume. Python makes the decision.**

---

## 🚀 Quick Start — Install & Run

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Kasparro-AI-Resume-Screening
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy:

```text
.env.example
```

to:

```text
.env
```

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Then add your LLM API key and configuration:

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.7-flash
GEMINI_API_KEY=your_api_key_here

GITHUB_TOKEN=
INPUT_DIR=./resumes
OUTPUT_FILE=./output/results.json
MAX_RESUME_CHARS=50000
ENABLE_GITHUB=true
```

> ⚠️ Never commit `.env` or API keys. `.env` is already included in `.gitignore`.

### 5. Add resumes

Place candidate resumes inside:

```text
resumes/
```

Supported formats:

- PDF — required
- DOCX — supported
- TXT — supported

Example:

```text
resumes/
├── candidate_01.pdf
├── candidate_02.pdf
├── candidate_03.docx
└── candidate_04.txt
```

### 6. Run the screening pipeline

```bash
python main.py
```

Or explicitly provide input/output paths:

```bash
python main.py --input ./resumes --output ./output/results.json
```

### 7. Run tests

```bash
pytest -q
```

Expected result for the included test suite:

```text
7 passed
```

---

# 📌 What This Project Does

The system is designed for a recruiter who has to screen a large batch of resumes for an engineering role requiring:

1. Genuine Python experience
2. Meaningful AI / LLM / agentic experience
3. Strong AI project depth
4. Backend engineering ability
5. Cloud / deployment / full-stack exposure
6. GitHub activity
7. Evidence-backed and explainable ranking

Instead of simply searching for keywords, the system tries to determine **what the candidate actually built**.

---

# 🎯 Assignment Requirements

| Requirement | Implementation |
|---|---|
| Process ~50 resumes | Batch-oriented pipeline |
| PDF support | `pypdf` parser |
| DOCX support | `python-docx` |
| TXT support | Native text parser |
| Variable resume layouts | Evidence extraction rather than fixed positions |
| Multi-page resumes | Page-aware PDF extraction |
| Malformed resume isolation | Per-file failure handling |
| Name/email extraction | Deterministic + structured extraction |
| Skills extraction | Technology/evidence extraction |
| Project extraction | Structured `ProjectEvidence` |
| GitHub URL extraction | Deterministic URL detection |
| Python eligibility | Hard gate |
| AI/LLM eligibility | Hard gate |
| AI depth scoring | Evidence-based project signals |
| 100-point ranking | Deterministic scoring engine |
| GitHub enrichment | Optional API client |
| LLM support | Provider abstraction |
| LLM failure handling | Batch-safe processing |
| Tests | Eligibility + scoring tests |
| JSON output | Structured final report |

---

# 🧠 Architecture

```text
                         ┌──────────────────────┐
                         │     Resume Folder    │
                         │   PDF / DOCX / TXT   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Ingestion       │
                         │ Parse + Normalize    │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌──────────────────┐             ┌──────────────────┐
          │ Deterministic    │             │       LLM        │
          │ Extraction       │             │ Structured       │
          │ URLs / Email /   │             │ Evidence         │
          │ Tech Signals     │             │ Extraction       │
          └────────┬─────────┘             └────────┬─────────┘
                   └───────────────┬────────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │   Evidence Ledger    │
                         │ Python / AI / Backend │
                         │ Cloud / GitHub / etc. │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Hard Eligibility     │
                         │ Python + AI required │
                         └──────────┬───────────┘
                                    │
                           ┌────────┴────────┐
                           │                 │
                        Reject             Eligible
                           │                 │
                           │                 ▼
                           │       ┌──────────────────────┐
                           │       │ Project Quality      │
                           │       │ AI / RAG / Agents    │
                           │       │ Backend / Cloud      │
                           │       └──────────┬───────────┘
                           │                  │
                           │                  ▼
                           │       ┌──────────────────────┐
                           │       │ GitHub Enrichment    │
                           │       └──────────┬───────────┘
                           │                  │
                           │                  ▼
                           │       ┌──────────────────────┐
                           │       │ Deterministic Score   │
                           │       │        /100            │
                           │       └──────────┬───────────┘
                           │                  │
                           └──────────┬───────┘
                                      ▼
                           ┌──────────────────────┐
                           │ Ranked JSON Report   │
                           └──────────────────────┘
```

---

# 🔑 Core Design Principle

## LLM understands. Code decides.

The LLM is useful for understanding unstructured resume language:

- project descriptions
- engineering responsibilities
- AI implementation details
- tool usage
- RAG architecture
- agent workflows
- backend implementation
- deployment evidence

However, the LLM does **not** directly decide the final score.

The system converts the extracted information into structured evidence and applies deterministic Python rules for:

- eligibility
- project depth
- category scores
- penalties
- final ranking

This makes the output easier to test, debug, and explain.

---

# ✅ Hard Eligibility Filter

A candidate must have **both**:

### 1. Genuine Python evidence

Examples:

```text
Built a Flask REST API in Python
Implemented a Python data processing pipeline
Developed backend services using Python
Used Python extensively in an internship
```

### 2. Meaningful AI / LLM / Agentic evidence

Examples include:

- LangChain
- LangGraph
- CrewAI
- Google ADK
- LlamaIndex
- RAG
- embeddings
- vector search
- retrieval systems
- tool-calling agents
- multi-agent workflows
- evaluation pipelines
- meaningful custom agent implementations

A resume that only says:

```text
Skills: Python, AI, Machine Learning
```

should not automatically pass.

The system looks for implementation evidence.

---

# 🧩 Evidence-First Extraction

The internal candidate representation separates evidence by category:

```text
python_evidence
ai_evidence
backend_evidence
cloud_evidence
frontend_evidence
github_evidence
projects
internships
experience
certifications
education
```

Each evidence item contains:

```text
claim
evidence
source_section
strength
```

This allows the final decision to answer:

> **Why did this candidate receive this score?**

rather than simply returning a number.

---

# 🧠 AI / Agentic Project Depth

Not all AI projects are equally valuable.

A thin project such as:

```text
Prompt → LLM API → Response
```

is different from a system containing:

```text
LLM
+ retrieval
+ embeddings
+ vector search
+ tool calling
+ multi-agent coordination
+ persistence
+ validation
+ evaluation
```

The scoring system therefore rewards implementation depth.

### Strong signals

- RAG
- retrieval
- embeddings
- vector search
- tool calling
- multi-agent workflows
- evaluation
- state management
- persistence
- validation
- external API/tool integration

### Shallow AI penalty

Projects that appear to be thin LLM/API wrappers can receive a deduction within the AI category.

The goal is to reward **engineering depth**, not AI buzzwords.

---

# 🏆 Scoring Model — 100 Points

Eligible candidates are scored across five categories.

| Category | Maximum |
|---|---:|
| AI / Agentic / RAG Depth | **40** |
| Python & Backend Engineering | **30** |
| Cloud / Deployment / Full Stack | **15** |
| GitHub Activity | **10** |
| Engineering Depth | **5** |
| **Total** | **100** |

---

## 1. AI / Agentic / RAG Depth — 40

The system evaluates project-level implementation signals such as:

- LLM usage
- RAG
- retrieval
- embeddings
- vector search
- tool calling
- multi-agent architecture
- evaluation
- state management
- validation
- persistence
- external API integration

---

## 2. Python & Backend Engineering — 30

The score rewards actual engineering evidence.

Examples:

- Python implementation
- REST APIs
- Flask/backend development
- persistence
- validation
- business logic
- external integrations
- backend responsibilities in projects or internships

Keyword-only skills are not treated the same as project evidence.

---

## 3. Cloud / Deployment / Full Stack — 15

Relevant evidence can include:

- AWS
- GCP
- Azure
- OCI
- CI/CD
- containers
- deployment
- frontend + backend integration
- end-to-end application development

---

## 4. GitHub Activity — 10

GitHub is treated as an **enrichment signal**, not an eligibility requirement.

The system can use:

- recent repository activity
- relevant repositories
- repository maintenance
- public project evidence

Missing, private, invalid, or inaccessible GitHub profiles should not automatically reject a candidate.

---

## 5. Engineering Depth — 5

Small additional signals reward:

- validation
- persistence
- evaluation
- business logic
- external integrations
- fault-tolerant engineering

---

# 🐙 GitHub Enrichment

GitHub enrichment is intentionally separated from resume eligibility.

This prevents a candidate from being rejected simply because:

- they have no GitHub URL
- their repository is private
- the API is temporarily unavailable
- GitHub rate limits are reached

The system attempts enrichment when possible and gracefully continues when it cannot.

---

# 🛡️ Batch Reliability

A major requirement is that one bad resume should not stop the entire batch.

Conceptually:

```python
for resume in resumes:
    try:
        process_resume(resume)
    except Exception:
        record_failure(resume)
        continue
```

This means:

```text
49 valid resumes + 1 malformed resume
```

should still produce results for the 49 valid candidates.

Processing failures are recorded separately in the final output.

---

# 🤖 LLM Provider Architecture

LLM providers are isolated behind a small interface.

Current provider structure:

```text
src/llm/
├── base.py
├── factory.py
├── gemini.py
├── openai.py
└── anthropic.py
```

The rest of the application does not need to know provider-specific SDK details.

This makes it possible to switch providers without rewriting:

- extraction
- eligibility
- scoring
- pipeline logic

---

# 📦 Project Structure

```text
Kasparro-AI-Resume-Screening/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py
│
├── config/
│   └── settings.py
│
├── src/
│   ├── schemas.py
│   │
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── pdf_parser.py
│   │   ├── docx_parser.py
│   │   └── text_parser.py
│   │
│   ├── extraction/
│   │   ├── deterministic.py
│   │   ├── llm_extractor.py
│   │   └── prompts.py
│   │
│   ├── llm/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── gemini.py
│   │   ├── openai.py
│   │   └── anthropic.py
│   │
│   ├── eligibility/
│   │   └── filter.py
│   │
│   ├── scoring/
│   │   ├── project_quality.py
│   │   ├── score.py
│   │   └── rules.py
│   │
│   ├── github/
│   │   └── client.py
│   │
│   └── pipeline/
│       └── screening.py
│
├── tests/
│   ├── test_eligibility.py
│   └── test_scoring.py
│
├── resumes/
├── output/
└── docs/
    └── DESIGN.md
```

---

# 🛠️ Technology Stack

### Language

- Python 3

### Resume processing

- `pypdf`
- `python-docx`

### Data validation

- Pydantic

### LLM

- Google Gemini
- OpenAI adapter
- Anthropic adapter

### HTTP / APIs

- HTTPX

### Configuration

- python-dotenv

### Testing

- pytest

### External enrichment

- GitHub REST API

---

# ⚙️ Configuration

`.env.example`:

```env
LLM_PROVIDER=gemini
LLM_MODEL=
GEMINI_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GITHUB_TOKEN=
INPUT_DIR=./resumes
OUTPUT_FILE=./output/results.json
MAX_RESUME_CHARS=50000
ENABLE_GITHUB=true
```

### Recommended Gemini configuration

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.7-flash
GEMINI_API_KEY=your_key
```

A GitHub token is optional.

---

# 📥 Input

Place resumes in:

```text
resumes/
```

The loader recursively discovers supported files.

Example:

```text
resumes/
├── backend/
│   ├── candidate_01.pdf
│   └── candidate_02.pdf
│
├── ai/
│   ├── candidate_03.pdf
│   └── candidate_04.docx
│
└── other/
    └── candidate_05.txt
```

---

# ▶️ Usage

## Default

```bash
python main.py
```

Uses:

```text
INPUT_DIR=./resumes
OUTPUT_FILE=./output/results.json
```

## Custom input/output

```bash
python main.py --input ./resumes --output ./output/results.json
```

## Tests

```bash
pytest -q
```

---

# 📤 Output

The system writes:

```text
output/results.json
```

The output contains:

```text
run_summary
ranked_candidates
rejected_candidates
processing_failures
```

---

# 📊 Ranked Candidate Output

An eligible candidate includes information such as:

```json
{
  "rank": 1,
  "name": "Candidate Name",
  "eligible": true,
  "total_score": 87.0,
  "score_breakdown": {
    "ai_agentic_rag": 35,
    "python_backend": 27,
    "cloud_full_stack": 11,
    "github_activity": 9,
    "engineering_depth": 5
  },
  "matched_skills": [],
  "project_summary": [],
  "github_summary": {},
  "strengths": [],
  "concerns": [],
  "decision_trace": {
    "python_gate": "",
    "ai_gate": "",
    "project_depth": "",
    "github_enrichment": "",
    "ranking_status": ""
  }
}
```

The exact output depends on the resumes processed.

---

# 🔍 Example Decision Logic

### Candidate A

```text
Python:
✓ Flask backend project

AI:
✓ CrewAI
✓ Gemini
✓ Tool calling
✓ Multi-agent workflow
✓ External APIs

Result:
ELIGIBLE
```

### Candidate B

```text
Python:
✗ No meaningful implementation evidence

AI:
✓ Chatbot project

Result:
REJECTED
Reason:
No genuine Python evidence found.
```

### Candidate C

```text
Python:
✓ Python backend

AI:
✓ "Built an AI chatbot"

Implementation depth:
✗ No retrieval
✗ No tools
✗ No agent workflow
✗ No meaningful architecture evidence

Result:
Potentially eligible, but receives lower AI depth score.
```

---

# 🧪 Testing Strategy

The test suite focuses on the decision layer rather than making tests dependent on external APIs.

Current coverage includes:

### Eligibility

- Python + AI → eligible
- Python only → rejected
- AI only → rejected
- neither → rejected

### Scoring

- Empty candidate → zero score
- Deep AI project → higher AI score
- Shallow AI project → lower AI score
- GitHub score → capped at 10

Run:

```bash
pytest -q
```

---

# 🧱 Engineering Tradeoffs

## Why deterministic scoring?

Because final ranking should be reproducible.

If an LLM directly returned:

```text
Score: 87/100
```

the result could change between calls.

Instead:

```text
Resume
  ↓
LLM extracts evidence
  ↓
Structured Pydantic object
  ↓
Python scoring rules
  ↓
Deterministic score
```

This gives a clearer audit trail.

---

## Why use both deterministic extraction and an LLM?

Regex and deterministic extraction are excellent for:

- emails
- URLs
- GitHub links
- obvious technology names

LLMs are better at understanding:

```text
"Designed a multi-agent travel planning system where specialized agents
used external APIs to retrieve live information."
```

The hybrid approach combines both strengths.

---

# 👨‍💻 Engineering Focus

This project reflects the type of engineering work I want to build toward:

- backend systems
- REST APIs
- structured data
- AI/LLM applications
- agentic workflows
- external tool integration
- validation
- fault isolation
- debugging
- production-oriented architecture

One important design consideration is that AI systems can produce **plausible but incorrect outputs**.

Therefore, the system emphasizes:

```text
Evidence
→ Validation
→ Deterministic Rules
→ Traceable Decision
```

rather than treating an LLM response as unquestionable truth.

---

# 📈 Future Improvements

If more development time were available, the next improvements would be:

### 1. Better resume normalization

Handle:

- tables
- columns
- unusual PDF layouts
- scanned resumes using OCR

### 2. Async batch processing

Process multiple resumes concurrently with bounded concurrency.

### 3. Caching

Cache:

- parsed resumes
- LLM extraction results
- GitHub API responses

This reduces cost and improves repeatability.

### 4. FastAPI

Expose the screening engine as an API:

```text
POST /screen
GET /results
```

### 5. Evaluation dataset

Create a manually labeled benchmark containing:

```text
eligible
rejected
shallow AI
deep AI
strong backend
weak backend
```

Then measure precision/recall of the eligibility layer and ranking quality.

### 6. Better observability

Add structured logs for:

- processing time
- extraction failures
- LLM failures
- GitHub failures
- per-stage latency

---

# 🔐 Security

Never commit secrets.

Before pushing:

```bash
git status
```

Verify that:

```text
.env
```

is not listed.

You can also check:

**Windows PowerShell**

```powershell
git check-ignore .env
```

If `.env` was accidentally tracked previously:

```bash
git rm --cached .env
```

Then commit the removal.

---

# 🚨 Troubleshooting

## `ModuleNotFoundError`

Make sure the virtual environment is active:

```powershell
.venv\Scripts\Activate.ps1
```

Then:

```bash
pip install -r requirements.txt
```

---

## API key error

Check `.env`:

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.7-flash
GEMINI_API_KEY=your_key
```

Restart the terminal/process after changing environment variables.

---

## No resumes found

Check:

```text
resumes/
```

and make sure files have one of the supported extensions:

```text
.pdf
.docx
.txt
```

---

## GitHub enrichment fails

This should not make the candidate ineligible.

You can disable it:

```env
ENABLE_GITHUB=false
```

---

# 📝 Design Summary

The complete decision pipeline is:

```text
Resume
  ↓
File validation
  ↓
Text extraction
  ↓
Deterministic signals
  ↓
LLM structured extraction
  ↓
Evidence ledger
  ↓
Python eligibility gate
  ↓
AI project depth analysis
  ↓
Deterministic 100-point scoring
  ↓
Optional GitHub enrichment
  ↓
Decision trace
  ↓
Ranked JSON output
```

The important distinction is:

> **The LLM extracts and interprets evidence; deterministic Python rules control eligibility and ranking.**

This keeps the system explainable, testable, and easier to extend.

---

# 📌 Current Scope

### Implemented

- PDF ingestion
- DOCX ingestion
- TXT ingestion
- Batch processing
- Deterministic extraction
- Structured Pydantic schemas
- LLM provider abstraction
- Gemini integration
- OpenAI adapter
- Anthropic adapter
- Hard eligibility filtering
- AI project-depth scoring
- 100-point ranking
- GitHub enrichment
- Per-resume failure isolation
- JSON result generation
- Unit tests

### Optional extensions

- FastAPI
- OCR
- Async concurrency
- Caching
- Evaluation benchmark
- Web dashboard
- Database persistence

---

# 👤 Author

**Ritesh Kumar Muduli**

B.Tech — Computer Science & Engineering

Focus:

**Backend Engineering • AI/LLM Systems • Agentic AI • REST APIs**

---

# ⭐ Final Note

This project is intentionally designed as an **engineering system**, not just an LLM prompt.

The key idea is:

```text
Unstructured Resume
        ↓
Structured Evidence
        ↓
Explicit Rules
        ↓
Explainable Decision
        ↓
Ranked Candidates
```

That separation makes the system easier to reason about, test, debug, and evolve into a production recruitment workflow.
