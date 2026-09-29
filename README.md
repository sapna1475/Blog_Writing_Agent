# 📝 Blog Writer Agent

An autonomous, multi-stage blog-writing agent built on **LangGraph**, served over **FastAPI**, with a **Streamlit** frontend and a **human-in-the-loop plan approval** step. Runs on a free, open-source LLM (Qwen2.5-7B via Hugging Face Inference) — no OpenAI/Anthropic key required. Containerized with **Docker** and deployed on **AWS (ECR + ECS Fargate + Application Load Balancer)**.

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async%20jobs%20%2B%20SSE-009688)
![LangGraph](https://img.shields.io/badge/LangGraph-stateful%20agent-6f42c1)
![Streamlit](https://img.shields.io/badge/Streamlit-frontend-ff4b4b)
![Docker](https://img.shields.io/badge/Docker-containerized-2496ED)
![AWS](https://img.shields.io/badge/AWS-ECS%20Fargate-FF9900)

**Live backend health check:** <http://blog-writing-agent-lb-428148850.us-east-1.elb.amazonaws.com/health>
*(Hosted on AWS. The demo may be taken down at any time to avoid running costs.)*

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [How it works, node by node](#how-it-works-node-by-node)
- [API](#api-backendpy)
- [Local setup](#local-setup)
- [Usage](#usage)
- [Docker](#docker)
- [Deployment on AWS](#deployment-on-aws)
- [Known limitations](#known-limitations)
- [Future scope](#future-scope)
- [Tech stack](#tech-stack)

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

### Agent graph

```mermaid
flowchart TD
    START([Start]) --> router
    router -->|needs_research| research
    router -->|closed_book| orchestrator
    research --> orchestrator
    orchestrator --> plan_review{{"plan_review<br/>(human approval · interrupt)"}}
    plan_review -->|"Send() × N sections"| worker
    worker --> reducer
    reducer --> END([End])

    subgraph worker_subgraph [worker: one instance per section]
        direction LR
        draft --> critic
        critic -->|fails, first pass| revise
        revise --> critic
        critic -->|passes or 2nd pass| finalize
    end
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

<p align="center"><em>Main graph — default (non-<code>xray</code>) view. This collapses <code>worker</code> and <code>reducer</code> into single boxes and doesn't expand the <code>plan_review</code> interrupt step; use <code>xray=True</code> (snippet above) or the Mermaid diagram above for the full picture including human approval.</em></p>

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

### Cloud architecture (AWS)

```mermaid
flowchart LR
    User([User / Streamlit UI]) -->|HTTP :80| ALB[Application Load Balancer<br/>blog-writing-agent-lb]
    ALB -->|forward to target group<br/>health check: /health| ECS

    subgraph AWS [AWS · us-east-1]
        ALB
        subgraph ECS [ECS Fargate · blog-writing-agent-cluster]
            Task[Task: FastAPI + LangGraph<br/>container port 8000]
        end
        ECR[(Amazon ECR<br/>blog-writing-agent:latest)] -.->|image pull| Task
        Task -.->|logs| CW[CloudWatch Logs]
    end

    Task -->|inference| HF[Hugging Face Inference]
    Task -->|search| Tavily[Tavily]
    Task -->|images| Gemini[Google Gemini]
    Task -->|checkpoints| PG[(Postgres)]
```

---

### Agent graph

```mermaid
flowchart TD
    START([Start]) --> router
    router -->|needs_research| research
    router -->|closed_book| orchestrator
    research --> orchestrator
    orchestrator --> plan_review{{"plan_review<br/>(human approval · interrupt)"}}
    plan_review -->|"Send() × N sections"| worker
    worker --> reducer
    reducer --> END([End])

    subgraph worker_subgraph [worker: one instance per section]
        direction LR
        draft --> critic
        critic -->|fails, first pass| revise
        revise --> critic
        critic -->|passes or 2nd pass| finalize
    end
```

`worker` and `reducer` are separate compiled subgraphs. `worker` runs once per outline section — LangGraph's `Send()` fans out one parallel instance per section after `plan_review` resumes, and `worker_node` (a thin async wrapper around `worker_subgraph`) returns only `{"sections": [...]}` back to the parent graph, so the parallel branches never collide on shared state keys like `topic` or `plan`.


## Project structure

```
.
├── backend.py             # FastAPI + LangGraph agent (the whole pipeline)
├── frontend.py            # Streamlit UI — talks to backend.py over HTTP
├── Dockerfile             # Container image for the backend
├── .dockerignore
├── docker-compose.yml     # Postgres, for LangGraph's checkpointer (local dev)
├── requirements.txt
├── .env                   # your secrets/config (not committed)
├── assets/                # README images (graph renders + AWS screenshots)
│   └── aws/
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
| `GET` | `/health` | Liveness check (also used as the ALB target-group health check). |

Quick check against the deployed service:

```bash
curl http://blog-writing-agent-lb-428148850.us-east-1.elb.amazonaws.com/health
```

---

## Local setup

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
| `DATABASE_URL` | `postgresql://bloguser:blogpass@localhost:5432/blogagent` | Must match `docker-compose.yml` locally; point at a managed Postgres in the cloud. |
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

In the sidebar, set **Backend URL** (e.g. `http://localhost:8000`, or the ALB URL below to use the deployed backend) and **Output dir** to match `BLOG_OUTPUT_DIR`.

---

## Usage

1. Enter a topic and an as-of date, click **Generate Blog**.
2. Watch live progress (router → research → orchestrator).
3. When the plan is ready, review it — edit any section's title, goal, bullets, word target, or flags — then **Approve as-is** or **Approve with my edits**.
4. Sections draft, self-critique, and revise in parallel; the final post (with any generated diagrams) appears in the **Markdown Preview** tab, with download buttons for the raw `.md` and a zipped bundle.
5. Past posts are listed in the sidebar and can be reloaded for viewing at any time.

---

## Docker

Build and run the backend container locally:

```bash
# Build the image
docker build -t blog-writing-agent .

# Run it (secrets and config come from .env; never bake them into the image)
docker run -d -p 8000:8000 --env-file .env --name blog-writing-agent blog-writing-agent

# Verify
curl http://localhost:8000/health
```

Useful commands:

```bash
docker ps                          # is it running?
docker logs blog-writing-agent     # startup errors, request logs
docker rm -f blog-writing-agent    # remove before re-running with the same name
```

> `.env` is listed in `.dockerignore` so secrets never end up inside the image layers.
> If `DATABASE_URL` uses `localhost`, remember that inside a container `localhost` is the container itself — use `host.docker.internal` (Docker Desktop) or a Compose network service name instead.

---

## Deployment on AWS

The backend is packaged as a Docker image, stored in **Amazon ECR**, run on **Amazon ECS with Fargate** (no servers to manage), and exposed to the internet through an **Application Load Balancer**.

**Live health check:** <http://blog-writing-agent-lb-428148850.us-east-1.elb.amazonaws.com/health>

| Component | Resource |
|---|---|
| Region | `us-east-1` (N. Virginia) |
| Container registry | ECR repository `blog-writing-agent` |
| Compute | ECS cluster `blog-writing-agent-cluster` (Fargate) |
| Task definition | `blog-writing-agent-task` |
| Load balancer | Application Load Balancer `blog-writing-agent-lb` (internet-facing, IPv4) |
| Health check | `GET /health` on the target group |

### Step 1 — Build the image

```powershell
docker build -t blog-writing-agent .
```

> If you build on an ARM machine (e.g. Apple Silicon), add `--platform linux/amd64` so the image matches Fargate's default x86_64 architecture.

### Step 2 — Create the ECR repository

```powershell
aws ecr create-repository --repository-name blog-writing-agent --region us-east-1
```

### Step 3 — Authenticate Docker to ECR

ECR is a private registry, so Docker needs a temporary token (valid for 12 hours) before it can push.

PowerShell (AWS Tools for PowerShell):

```powershell
(Get-ECRLoginCommand).Password | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
```

Or with the AWS CLI:

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
```

You should see `Login Succeeded`.

### Step 4 — Tag and push

```powershell
docker tag blog-writing-agent:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/blog-writing-agent:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/blog-writing-agent:latest
```

Once the push completes, the image appears in the repository:

![ECR repository showing the pushed blog-writing-agent image tagged latest](assets/aws/ecr-repository.png)

### Step 5 — Create the ECS cluster, task definition, and service

1. **Cluster:** create an ECS cluster (`blog-writing-agent-cluster`) using **AWS Fargate**.
2. **Task definition** (`blog-writing-agent-task`):
   - Launch type: Fargate, Linux/X86_64
   - Container image: the ECR URI from step 4
   - Container port: `8000` (TCP)
   - CPU / memory: size for your workload (the agent is I/O-bound on external APIs, so a small task is fine)
   - Environment variables: `HF_REPO_ID`, `HF_PROVIDER`, `BLOG_OUTPUT_DIR`, etc.
   - **Secrets** (`HUGGINGFACEHUB_API_TOKEN`, `TAVILY_API_KEY`, `GOOGLE_API_KEY`, `DATABASE_URL`, `LANGCHAIN_API_KEY`): inject from **AWS Secrets Manager** or **SSM Parameter Store** using the task definition's `secrets` field rather than plain-text environment variables.
   - Logging: `awslogs` driver → CloudWatch Logs
3. **Service:** create a service from the task definition with a desired count of `1`, attached to the load balancer below.

### Step 6 — Put an Application Load Balancer in front

1. Create an **internet-facing Application Load Balancer** (`blog-writing-agent-lb`) across the VPC's public subnets, with a listener on port `80`.
2. Create a **target group** (target type `IP` for Fargate) on port `8000` with health check path `/health`.
3. Security groups:
   - **ALB SG:** allow inbound `80` from `0.0.0.0/0`.
   - **Task SG:** allow inbound `8000` **only from the ALB's security group**.

![Application Load Balancer details showing Active status and DNS name](assets/aws/alb-details.png)

### Step 7 — Verify

The ECS service reports `Active` with the desired task running and a successful deployment, and the load balancer's target health is passing:

![ECS service overview showing Active status, 1 running task, and successful deployment](assets/aws/ecs-service.png)

```bash
curl http://blog-writing-agent-lb-428148850.us-east-1.elb.amazonaws.com/health
```

### Updating the deployment

After changing code, rebuild, push, and roll the service:

```powershell
docker build -t blog-writing-agent .
docker tag blog-writing-agent:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/blog-writing-agent:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/blog-writing-agent:latest

aws ecs update-service --cluster blog-writing-agent-cluster --service <SERVICE_NAME> --force-new-deployment --region us-east-1
```

### Cost and cleanup

Fargate tasks and ALBs bill by the hour even when idle. To stop charges, delete the ECS service, the load balancer, the target group, and (optionally) the ECR images and CloudWatch log group.

---

## Known limitations

- **In-memory job registry.** `JOBS: Dict[str, JobRecord]` lives in process memory — jobs don't survive a backend restart (though the *graph's own* state does, via the Postgres checkpointer). Swap in Redis for production.
- **No evidence/image-spec exposure via the API.** `/jobs/{id}` only returns the pending plan and final markdown — not the evidence list or per-image detail — so the frontend can't show a full Evidence tab. Extend the endpoint if you need that.
- **`requires_research` on a `Task` is informational only.** It's passed into the worker's prompt as context but doesn't currently trigger a section-scoped research call — all research happens once, globally, before planning.
- **Free-tier HF inference can be slow**, especially on cold starts, and small models sometimes need several structured-output repair attempts — a run can take a few minutes.
- **The image-decision LLM call tends to under-request images.** A deterministic fallback inserts one default diagram for `system_design` / `tutorial` / `comparison` posts if the model returns zero.
- **Single revision pass, hard-capped.** `route_after_critic` allows exactly one revise → critic loop per section, to bound cost/latency — a section can still finish "failed" if the second attempt doesn't pass.
- **No auth, no rate limiting.** The FastAPI app is wide open, and the AWS deployment is reachable over plain HTTP from the public internet. Anyone with the URL can trigger jobs that consume your Hugging Face, Tavily, and Gemini quotas. Fine for a demo; add authentication, HTTPS (ACM certificate + port 443 listener), and rate limiting before real use.
- **Generated files live on the container's ephemeral disk.** On Fargate, `blog_output/` is lost when a task restarts. Mount EFS or write to S3 for durable output.
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
- **Production AWS hardening.** HTTPS via ACM, private subnets with NAT, EFS/S3 for output, RDS for Postgres, autoscaling, and Infrastructure as Code (Terraform/CDK).
- **CI/CD.** GitHub Actions workflow to build, push to ECR, and force a new ECS deployment on every merge to `main`.
- **Export integrations.** Publish directly to a CMS (e.g. WordPress, Ghost, Notion) or export to PDF/DOCX in addition to Markdown.
- **Automated tests.** Unit tests for `critique_section`/`verify_and_strip_citations`, and an integration test that runs the graph end-to-end against a mocked LLM.

---

## Tech stack

- **Orchestration:** LangGraph (`StateGraph`, `Send` fan-out, `interrupt()`, `AsyncPostgresSaver`)
- **LLM:** Qwen2.5-7B-Instruct via `langchain-huggingface` (any HF-hosted chat model works)
- **Research:** Tavily (`langchain-community`)
- **Images:** Google Gemini (`google-genai`)
- **API:** FastAPI, async jobs, Server-Sent Events
- **Persistence:** Postgres (Docker Compose locally)
- **Frontend:** Streamlit
- **Containerization:** Docker
- **Cloud:** AWS — ECR, ECS Fargate, Application Load Balancer, CloudWatch Logs
- **Validation/resilience:** Pydantic, `pydantic-settings`, Tenacity (retry/backoff)
- **Observability:** LangSmith

---

## License

Add your license here (e.g. MIT).
