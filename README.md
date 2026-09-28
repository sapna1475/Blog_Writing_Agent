# 📝 Blog Writer Agent

An autonomous, multi-stage blog-writing agent built on **LangGraph**, served over **FastAPI**, with a **Streamlit** frontend and a **human-in-the-loop plan approval** step. Runs on a free, open-source LLM (Qwen2.5-7B via Hugging Face Inference) — no OpenAI/Anthropic key required.

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async%20jobs%20%2B%20SSE-009688)
![LangGraph](https://img.shields.io/badge/LangGraph-stateful%20agent-6f42c1)
![Streamlit](https://img.shields.io/badge/Streamlit-frontend-ff4b4b)

---

## What it does

Give it a topic, and the agent will:

1. Decide whether the topic needs live web research.
2. Research it (if needed) via Tavily.
3. Draft a section-by-section outline (a **Plan**).
4. **Pause and hand the plan to you for approval or editing** before writing a single word.
5. Write, self-critique, and (if needed) revise every section — in parallel.
6. Merge everything into one document, decide whether it needs diagrams, generate them with Gemini, and save the final Markdown post (with images) to disk.

Every step is checkpointed to Postgres, so a run can be paused (at the approval step) and resumed later, even across server restarts.

---

## Architecture

```mermaid
flowchart TD
    START([start]) --> router[router]
    router -->|needs_research = True| research[research]
    router -->|needs_research = False| orchestrator[orchestrator]
    research --> orchestrator
    orchestrator --> plan_review{{"plan_review — human approval (interrupt)"}}
    plan_review -->|Send() fan-out, one per section| worker

    subgraph worker["worker — one instance per section, run in parallel"]
        direction TB
        draft[draft] --> critic[critic]
        critic -->|failed & attempt < 2| revise[revise]
        revise --> critic
        critic -->|passed, or attempt cap hit| finalize[finalize]
    end

    worker -->|fan-in, all sections done| reducer

    subgraph reducer["reducer"]
        direction TB
        merge_content[merge_content] --> decide_images[decide_images] --> generate_and_place_images[generate_and_place_images]
    end

    reducer --> END([end])
```

`worker` and `reducer` are separate compiled subgraphs. `worker` runs once per outline section — LangGraph's `Send()` fans out one parallel instance per section after `plan_review` resumes, and `worker_node` (a thin async wrapper around `worker_subgraph`) returns only `{"sections": [...]}` back to the parent graph, so the parallel branches never collide on shared state keys like `topic` or `plan`.

To regenerate this diagram yourself straight from the compiled graph:

```python
from backend import build_graph
from IPython.display import Image, display

graph = build_graph(checkpointer=None)
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
```

### Rendered directly from the code

<p align="center">
  <img src="assets/main_graph.png" alt="Main graph, top-level view" width="260">
</p>

<p align="center"><em>Main graph — default (non-<code>xray</code>) view. This collapses <code>worker</code> and <code>reducer</code> into single boxes and doesn't expand the <code>plan_review</code> interrupt step; use <code>xray=True</code> (snippet above) or the Mermaid diagram further up for the full picture including human approval.</em></p>

<p align="center">
  <img src="assets/reducer_subgraph.png" alt="Reducer subgraph" width="260">
</p>

<p align="center"><em>The <code>reducer</code> subgraph on its own — get it directly with:</em></p>

```python
from backend import reducer_subgraph
from IPython.display import Image, display

display(Image(reducer_subgraph.get_graph().draw_mermaid_png()))
```

*(Swap in `worker_subgraph` the same way to render the draft → critic → revise → finalize loop on its own.)*

---

## Project structure

```
.
├── backend.py            # FastAPI + LangGraph agent (the whole pipeline)
├── frontend.py            # Streamlit UI — talks to backend.py over HTTP
├── docker-compose.yml     # Postgres, for LangGraph's checkpointer
├── requirements.txt
├── .env                   # your secrets/config (not committed)
└── blog_output/           # generated markdown + images (created at runtime)
    └── images/
```

---

## How it works, node by node

| Node | What it does |
|---|---|
| `router` | Classifies the topic as `closed_book` / `hybrid` / `open_book` and decides `needs_research`. Uses structured LLM output (see below). |
| `research` | Runs Tavily searches for the router's queries, has the LLM synthesize results into `EvidenceItem`s, dedupes by URL, and (for `open_book` mode) filters by recency. |
| `orchestrator` | Turns the topic + evidence into a `Plan`: 5–9 sections, each with a goal, bullets, target word count, and citation/code/research flags. |
| `plan_review` | Calls LangGraph's `interrupt()` — the graph **pauses here** until a human approves the plan (as-is, or edited). This is what makes the run resumable across requests/restarts. |
| `worker` *(fanned out, one per section)* | `draft` → `critic` → `revise` (max one revision pass) → `finalize`. The critic is a deterministic, rule-based check (word count, bullet coverage, citation validity) — no extra LLM call. `finalize` always strips any citation not in the approved evidence list, regardless of what the critic caught. |
| `reducer` | Merges all sections into one document, asks the LLM whether diagrams would help (`decide_images`), generates any requested images with Gemini, and writes the final `.md` + images to `blog_output/`. |

### Structured output on an open-source model

Small open-source models served via `HuggingFaceEndpoint` don't support OpenAI-style function calling, so `llm.with_structured_output(...)` isn't available. Instead, `invoke_structured()`:

1. Appends a `PydanticOutputParser`'s format instructions to the system prompt.
2. Strips markdown fences / stray prose from the response.
3. Parses it into the target Pydantic model.
4. On failure, shows the model its own broken output + the parser error and asks for a corrected JSON-only reply (up to `STRUCTURED_REPAIR_ATTEMPTS` times) before giving up.

### Human-in-the-loop plan approval

`plan_review_node` calls `interrupt({"plan": plan})`. The FastAPI layer surfaces this as job status `waiting_approval` with the pending plan attached; `POST /jobs/{id}/approve` resumes the graph via `Command(resume=...)`. The resume payload is always non-empty (`{"approved": True, ...}`) — LangGraph treats a falsy resume as "nothing to resume with" and silently re-pauses otherwise.

---

## API (backend.py)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/generate` | Start a new job. Body: `{"topic": str, "as_of": "YYYY-MM-DD"}`. Returns `{"job_id": ...}`. |
| `GET` | `/jobs/{job_id}` | Current status, pending plan (if `waiting_approval`), final markdown (if `completed`), or error. |
| `POST` | `/jobs/{job_id}/approve` | Resume a paused job. Body: `{}` to approve as-is, or `{"plan": {...}}` to approve with edits. |
| `GET` | `/jobs/{job_id}/stream` | Server-Sent Events stream of graph progress for the job's current run segment. |
| `GET` | `/health` | Liveness check. |

---

## Setup

### 1. Clone & install

```bash
git clone <your-repo-url>
cd Blog_Writing_Agent
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure `.env`

```env
# LLM (free Hugging Face inference)
HF_REPO_ID=Qwen/Qwen2.5-7B-Instruct
HF_PROVIDER=featherless-ai
HF_MAX_NEW_TOKENS=1536
HF_TEMPERATURE=0.7
HUGGINGFACEHUB_API_TOKEN=hf_...

# Research
TAVILY_API_KEY=tvly-...

# Image generation
GOOGLE_API_KEY=...

# Persistence (matches docker-compose.yml)
DATABASE_URL=postgresql://bloguser:blogpass@localhost:5432/blogagent

# Observability (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=blog-writer-agent

# Output
BLOG_OUTPUT_DIR=./blog_output
```

| Variable | Default | Notes |
|---|---|---|
| `HF_REPO_ID` | `Qwen/Qwen2.5-7B-Instruct` | Any chat model served by your `HF_PROVIDER`. |
| `HF_PROVIDER` | `featherless-ai` | Hugging Face Inference provider. |
| `HF_MAX_NEW_TOKENS` | `1536` | Needs to be generous — the `Plan` schema alone can run long. |
| `HF_TEMPERATURE` | `0.7` | |
| `HUGGINGFACEHUB_API_TOKEN` | — | Required. |
| `TAVILY_API_KEY` | — | Optional — without it, `research` logs a warning and returns no evidence. |
| `GOOGLE_API_KEY` | — | Optional — without it, image generation fails gracefully per-image (`[IMAGE GENERATION FAILED]` block), everything else still completes. |
| `DATABASE_URL` | `postgresql://bloguser:blogpass@localhost:5432/blogagent` | Must match `docker-compose.yml`. |
| `LANGCHAIN_TRACING_V2` | `false` | Set `true` to enable LangSmith tracing. |
| `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` | — / `blog-writer-agent` | Only used if tracing is enabled. |
| `BLOG_OUTPUT_DIR` | `./blog_output` | Where final markdown + `images/` are written. Must match what you point the frontend's **Output dir** field at. |
| `STRUCTURED_REPAIR_ATTEMPTS` | `2` | How many times to ask the model to fix malformed structured JSON before failing. |

### 3. Start Postgres

```bash
docker compose up -d
```

### 4. Run the backend

```bash
uvicorn backend:app --reload
```

### 5. Run the frontend

```bash
streamlit run frontend.py
```

In the sidebar, set **Backend URL** (e.g. `http://localhost:8000`) and **Output dir** to match `BLOG_OUTPUT_DIR`.

---

## Usage

1. Enter a topic and an as-of date, click **Generate Blog**.
2. Watch live progress (router → research → orchestrator).
3. When the plan is ready, review it — edit any section's title, goal, bullets, word target, or flags — then **Approve as-is** or **Approve with my edits**.
4. Sections draft, self-critique, and revise in parallel; the final post (with any generated diagrams) appears in the **Markdown Preview** tab, with download buttons for the raw `.md` and a zipped bundle.
5. Past posts are listed in the sidebar and can be reloaded for viewing at any time.

---

## Known limitations

- **In-memory job registry.** `JOBS: Dict[str, JobRecord]` lives in process memory — jobs don't survive a backend restart (though the *graph's own* state does, via the Postgres checkpointer). Swap in Redis for production.
- **No evidence/image-spec exposure via the API.** `/jobs/{id}` only returns the pending plan and final markdown — not the evidence list or per-image detail — so the frontend can't show a full Evidence tab. Extend the endpoint if you need that.
- **`requires_research` on a `Task` is informational only.** It's passed into the worker's prompt as context but doesn't currently trigger a section-scoped research call — all research happens once, globally, before planning.
- **Free-tier HF inference can be slow**, especially on cold starts, and small models sometimes need several structured-output repair attempts — a run can take a few minutes.
- **The image-decision LLM call tends to under-request images.** A deterministic fallback inserts one default diagram for `system_design` / `tutorial` / `comparison` posts if the model returns zero.
- **Single revision pass, hard-capped.** `route_after_critic` allows exactly one revise → critic loop per section, to bound cost/latency — a section can still finish "failed" if the second attempt doesn't pass.
- **No auth, no rate limiting.** The FastAPI app is wide open — fine for local/demo use, not for a public deployment.
- **No automated tests.** The graph is validated interactively rather than via a test suite.

---

## Future scope

- **Durable job store.** Replace the in-memory `JOBS` dict with Redis (or a Postgres table) so job status survives backend restarts and can be shared across multiple worker processes.
- **Richer API surface.** Expose evidence and per-image specs on `/jobs/{id}` so the frontend can show full Evidence/Images-plan tabs instead of only the final markdown.
- **Section-scoped research.** Make `requires_research=True` on an individual `Task` actually trigger a targeted Tavily search for that section, instead of relying solely on the single global research pass.
- **Pluggable LLM backend.** Add a config switch to use `with_structured_output()` directly when running against a model that supports function calling (OpenAI, Anthropic, etc.), falling back to the `invoke_structured()` parser workaround only for open-source models.
- **Section-level regeneration.** Let a user regenerate or edit a single section/image after the run completes, without re-running the whole graph.
- **Image editing / retry controls.** Surface a "regenerate this image" action in the frontend, and expose the failure reason inline instead of only in the markdown as `[IMAGE GENERATION FAILED]`.
- **Auth + per-user job isolation**, for anything beyond local/demo use.
- **Concurrency controls.** Queue or rate-limit concurrent jobs against the free HF inference tier to avoid throttling/slow-downs under load.
- **Export integrations.** Publish directly to a CMS (e.g. WordPress, Ghost, Notion) or export to PDF/DOCX in addition to Markdown.
- **Automated tests.** Unit tests for `critique_section`/`verify_and_strip_citations`, and an integration test that runs the graph end-to-end against a mocked LLM.

---

## Tech stack

- **Orchestration:** LangGraph (`StateGraph`, `Send` fan-out, `interrupt()`, `AsyncPostgresSaver`)
- **LLM:** Qwen2.5-7B-Instruct via `langchain-huggingface` (any HF-hosted chat model works)
- **Research:** Tavily (`langchain-community`)
- **Images:** Google Gemini (`google-genai`)
- **API:** FastAPI, async jobs, Server-Sent Events
- **Persistence:** Postgres (via Docker Compose)
- **Frontend:** Streamlit
- **Validation/resilience:** Pydantic, `pydantic-settings`, Tenacity (retry/backoff)
- **Observability:** LangSmith 

---

## License

Add your license here (e.g. MIT).
