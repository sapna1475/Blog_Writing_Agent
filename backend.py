"""
backend.py — Blog Writer Agent


  1. Settings & setup (env vars, logging, LangSmith tracing)
  2. Resilience helper (retry/backoff for LLM calls)
  3. LLM client (free Hugging Face model)
  4. Schemas (Task, Plan, Evidence, Critique, Images...)
  5. Citation-grounding verification helper
  6. Graph nodes: router -> research -> orchestrator -> plan_review (human
     approval) -> worker subgraph (draft -> critic -> revise -> finalize,
     fanned out per section) -> reducer subgraph (merge -> images)
  7. Graph assembly with a Postgres checkpointer (persistence + interrupts)
  8. FastAPI app: async jobs + SSE streaming + plan-approval endpoint

Run:
    docker compose up -d          # starts Postgres
    uvicorn backend:app --reload  # starts the API
"""
from __future__ import annotations

import asyncio
import json
import logging
import operator
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("blog_agent")


# ============================================================
# 1) Settings
# ============================================================
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (free Hugging Face inference)
    hf_repo_id: str = "Qwen/Qwen2.5-7B-Instruct"
    hf_provider: str = "featherless-ai"
    hf_max_new_tokens: int = 1536
    hf_temperature: float = 0.7
    huggingfacehub_api_token: str | None = None

    # Third-party APIs
    tavily_api_key: str | None = None
    google_api_key: str | None = None

    # Persistence (Postgres, run via docker-compose)
    database_url: str = "postgresql://bloguser:blogpass@localhost:5432/blogagent"

    # Observability
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "blog-writer-agent"

    # Output
    output_dir: Path = Path(os.getenv("BLOG_OUTPUT_DIR", "./blog_output")).resolve()
    images_subdir: str = "images"

    # Retry / backoff
    llm_max_retries: int = 4
    llm_retry_min_wait_s: float = 1.0
    llm_retry_max_wait_s: float = 20.0

    # Structured-output repair (open-source models don't support function
    # calling, so we parse free-text JSON and give the model a chance to
    # fix its own malformed output before giving up).
    structured_repair_attempts: int = 2


settings = Settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)
(settings.output_dir / settings.images_subdir).mkdir(parents=True, exist_ok=True)

# LangSmith reads these exact env var names — set them explicitly so tracing
# works regardless of how the person configured their .env file.
if settings.langchain_tracing_v2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    logger.info("LangSmith tracing enabled for project=%r", settings.langchain_project)


# ============================================================
# 2) Resilience — retry/backoff wrapper for every LLM call
# ============================================================
def _log_retry(retry_state) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "LLM call failed (attempt %s/%s): %r — retrying...",
        retry_state.attempt_number,
        settings.llm_max_retries,
        exc,
    )


llm_retry = retry(
    reraise=True,
    stop=stop_after_attempt(settings.llm_max_retries),
    wait=wait_exponential_jitter(initial=settings.llm_retry_min_wait_s, max=settings.llm_retry_max_wait_s),
    retry=retry_if_exception_type(Exception),
    before_sleep=_log_retry,
)


@llm_retry
def invoke_with_retry(runnable: Any, messages: list) -> Any:
    """Call `.invoke(messages)` with exponential-backoff retries + logging."""
    return runnable.invoke(messages)


# ============================================================
# 2b) Structured output WITHOUT function calling
# ============================================================
# `llm.with_structured_output(Model)` defaults to method="function_calling",
# which raises NotImplementedError on open-source models served via
# HuggingFaceEndpoint/ChatHuggingFace (they don't expose an OpenAI-style
# tools API). Instead we:
#   1) ask the model for JSON via a PydanticOutputParser's format
#      instructions appended to the system prompt,
#   2) strip markdown code fences / stray prose the model may add,
#   3) parse it into the Pydantic model,
#   4) if parsing fails, show the model its own broken output + the parser
#      error and ask it to return corrected JSON only (repeated up to
#      settings.structured_repair_attempts times) before giving up.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _clean_json_candidate(text: str) -> str:
    """Best-effort extraction of a bare JSON object from raw model text."""
    text = text.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()
    # No fences — if there's leading/trailing prose, grab the outermost {...}
    obj_match = _JSON_OBJECT_RE.search(text)
    if obj_match:
        return obj_match.group(0).strip()
    return text


def invoke_structured(model_cls: type[BaseModel], system_prompt: str, human_content: str) -> BaseModel:
    """
    Structured-output replacement for `llm.with_structured_output(model_cls)`
    that works with any chat model — including open-source models with no
    function-calling support.
    """
    parser = PydanticOutputParser(pydantic_object=model_cls)
    format_instructions = parser.get_format_instructions()
    system = (
        f"{system_prompt}\n\n"
        f"{format_instructions}\n\n"
        "Return ONLY the JSON object. No markdown code fences, no explanation, "
        "no text before or after the JSON."
    )
    messages: list = [SystemMessage(content=system), HumanMessage(content=human_content)]

    last_error: Optional[Exception] = None
    last_raw: str = ""
    max_attempts = settings.structured_repair_attempts + 1

    for attempt in range(1, max_attempts + 1):
        response = invoke_with_retry(llm, messages)
        raw = response.content if hasattr(response, "content") else str(response)
        last_raw = raw
        candidate = _clean_json_candidate(raw)
        try:
            return parser.parse(candidate)
        except Exception as e:  # noqa: BLE001 — any parse/validation failure
            last_error = e
            logger.warning(
                "Structured parse failed for %s (attempt %s/%s): %s",
                model_cls.__name__, attempt, max_attempts, e,
            )
            if attempt == max_attempts:
                break
            # Give the model its own bad output + the error, ask for a fix.
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=human_content),
                AIMessage(content=raw),
                HumanMessage(
                    content=(
                        "That response could not be parsed as valid JSON matching the "
                        f"required schema. Parser error:\n{e}\n\n"
                        "Return ONLY the corrected JSON object — no markdown fences, "
                        "no explanation, nothing else."
                    )
                ),
            ]

    raise ValueError(
        f"Could not get valid structured output for {model_cls.__name__} after "
        f"{max_attempts} attempt(s). Last error: {last_error}\nLast raw output:\n{last_raw}"
    )


# ============================================================
# 3) LLM client
# ============================================================
def build_llm() -> ChatHuggingFace:
    endpoint = HuggingFaceEndpoint(
        repo_id=settings.hf_repo_id,
        provider=settings.hf_provider,
        max_new_tokens=settings.hf_max_new_tokens,
        temperature=settings.hf_temperature,
        huggingfacehub_api_token=settings.huggingfacehub_api_token,
    )
    return ChatHuggingFace(llm=endpoint)


llm = build_llm()


# ============================================================
# 4) Schemas
# ============================================================
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120-550).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str
    filename: str
    alt: str
    caption: str
    prompt: str
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


class CritiqueResult(BaseModel):
    """Rule-based scorecard for one drafted section — no extra LLM call needed."""

    passed: bool
    word_count: int
    target_words: int
    bullet_coverage: float
    missing_bullets: List[str] = Field(default_factory=list)
    invalid_citations: List[str] = Field(default_factory=list)
    feedback: str


# ============================================================
# 5) Citation-grounding verification
# ============================================================
# Matches markdown links like "[Source](https://example.com/page)".
_CITATION_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


def verify_and_strip_citations(section_md: str, evidence_urls: set[str]) -> tuple[str, List[str]]:
    """
    Regex out every markdown link in `section_md`. Any URL that is NOT one of
    the URLs we actually gave the model as evidence is a hallucinated
    citation: drop the hyperlink (keep the link text) and report it.

    Returns (cleaned_markdown, list_of_invalid_urls).
    """
    invalid: List[str] = []

    def _check(match: re.Match) -> str:
        text, url = match.group(1), match.group(2)
        if url in evidence_urls:
            return match.group(0)
        invalid.append(url)
        return text  # keep the readable text, drop the unverifiable link

    cleaned = _CITATION_RE.sub(_check, section_md)
    return cleaned, invalid


def critique_section(section_md: str, task: Task, evidence_urls: set[str]) -> CritiqueResult:
    """Deterministic, cheap check: word count, bullet coverage, citation validity."""
    word_count = len(section_md.split())
    lower_bound = int(task.target_words * 0.85)
    upper_bound = int(task.target_words * 1.15)
    word_count_ok = lower_bound <= word_count <= upper_bound

    missing_bullets = [b for b in task.bullets if not _bullet_is_covered(b, section_md)]
    bullet_coverage = round(1.0 - (len(missing_bullets) / len(task.bullets)), 2) if task.bullets else 1.0

    _, invalid_citations = verify_and_strip_citations(section_md, evidence_urls)

    passed = word_count_ok and bullet_coverage >= 0.7 and not invalid_citations

    notes = []
    if not word_count_ok:
        notes.append(f"Word count is {word_count}; target range is {lower_bound}-{upper_bound}.")
    if missing_bullets:
        notes.append("These points are not clearly covered: " + "; ".join(missing_bullets))
    if invalid_citations:
        notes.append(
            "These cited URLs are NOT in the approved evidence list — remove or replace them: "
            + ", ".join(invalid_citations)
        )

    return CritiqueResult(
        passed=passed,
        word_count=word_count,
        target_words=task.target_words,
        bullet_coverage=bullet_coverage,
        missing_bullets=missing_bullets,
        invalid_citations=invalid_citations,
        feedback=" ".join(notes) if notes else "Looks good.",
    )


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your",
    "their", "which", "these", "those", "about", "where", "when", "have",
    "will", "been", "being", "such", "also",
}


def _bullet_is_covered(bullet: str, section_md: str) -> bool:
    """Heuristic: at least one distinctive keyword from the bullet appears in the section."""
    keywords = {w for w in re.findall(r"[a-zA-Z]{5,}", bullet.lower()) if w not in _STOPWORDS}
    if not keywords:
        return True  # nothing distinctive to check for
    section_lower = section_md.lower()
    return any(kw in section_lower for kw in keywords)


# ============================================================
# 6) Graph state
# ============================================================
class State(TypedDict):
    """
    NOTE: `plan` and `evidence` are plain dicts, not `Plan`/`EvidenceItem`
    instances. LangGraph checkpoints state to Postgres between steps (that's
    what makes interrupt()/resume work), and a custom Pydantic class isn't
    known to the checkpoint serializer — round-tripping it silently comes
    back as a dict instead of the original type. Storing dicts everywhere
    and reconstructing `Plan(**state["plan"])` locally inside a node avoids
    that trap entirely.
    """

    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[dict]
    plan: Optional[dict]
    as_of: str
    recency_days: int
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str


class WorkerState(TypedDict):
    """State for the per-section worker subgraph (draft -> critic -> revise -> finalize)."""

    task: dict
    topic: str
    mode: str
    as_of: str
    recency_days: int
    plan: dict
    evidence: List[dict]
    draft: str
    attempt: int
    critique: dict
    sections: Annotated[List[tuple[int, str]], operator.add]  # merges into the parent State


# ============================================================
# 7a) Router
# ============================================================
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.
Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true, output 3-10 high-signal, scoped queries.
"""


def router_node(state: State) -> dict:
    decision: RouterDecision = invoke_structured(
        RouterDecision,
        ROUTER_SYSTEM,
        f"Topic: {state['topic']}\nAs-of date: {state['as_of']}",
    )
    recency_days = {"open_book": 7, "hybrid": 45}.get(decision.mode, 3650)
    logger.info("Router: mode=%s needs_research=%s", decision.mode, decision.needs_research)
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# ============================================================
# 7b) Research (Tavily)
# ============================================================
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not settings.tavily_api_key:
        logger.warning("TAVILY_API_KEY not set — skipping search for: %r", query)
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore

        results = TavilySearchResults(max_results=max_results).invoke({"query": query})
        return [
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
            for r in results or []
        ]
    except Exception:
        logger.exception("Tavily search failed for query: %r", query)
        return []


def _iso_to_date(s: Optional[str]):
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


RESEARCH_SYSTEM = """You are a research synthesizer.
Given raw web search results, produce EvidenceItem objects.
Only include items with a non-empty url. Normalize published_at to ISO
YYYY-MM-DD if reliably inferable, else null (do NOT guess). Deduplicate by URL.
"""


def research_node(state: State) -> dict:
    raw: List[dict] = []
    for q in (state.get("queries") or [])[:10]:
        raw.extend(_tavily_search(q, max_results=6))

    if not raw:
        logger.warning("No search results — evidence will be empty.")
        return {"evidence": []}

    pack: EvidencePack = invoke_structured(
        EvidencePack,
        RESEARCH_SYSTEM,
        f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n\nRaw results:\n{raw}",
    )

    evidence = list({e.url: e for e in pack.evidence if e.url}.values())
    if state.get("mode") == "open_book":
        cutoff = date.fromisoformat(state["as_of"]) - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    logger.info("Research: %d evidence item(s) after dedup/filtering.", len(evidence))
    return {"evidence": [e.model_dump() for e in evidence]}


# ============================================================
# 7c) Orchestrator (Plan)
# ============================================================
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post: 5-9 tasks,
each with a goal, 3-6 bullets, and a target word count.

- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks
  requires_research=True and requires_citations=True.
- open_book: set blog_kind="news_roundup"; no tutorial content; if evidence
  is weak, say so rather than inventing events.
"""


def orchestrator_node(state: State) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])  # already list[dict] — see State's note above

    plan: Plan = invoke_structured(
        Plan,
        ORCH_SYSTEM,
        (
            f"Topic: {state['topic']}\nMode: {mode}\n"
            f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n\n"
            f"Evidence:\n{evidence[:16]}"
        ),
    )
    if mode == "open_book":
        plan.blog_kind = "news_roundup"

    logger.info("Plan created: %r with %d tasks", plan.blog_title, len(plan.tasks))
    return {"plan": plan.model_dump()}


# ============================================================
# 7d) Plan review — human-in-the-loop approval (interrupt)
# ============================================================
def plan_review_node(state: State) -> dict:
    """
    Pauses the graph and hands the plan to a human. Resuming with
    `Command(resume={"approved": True, "plan": {...}})` swaps in an edited
    plan; resuming with `Command(resume={"approved": True})` approves the
    plan as-is. Requires a checkpointer (see build_graph) — without one,
    interrupt() has nothing to persist to.

    NOTE: the resume value must never be empty/falsy (e.g. `{}` or `None`) —
    LangGraph treats a falsy resume as "nothing to resume with" and re-pauses
    at the same interrupt instead of continuing, which looks like the
    approval silently did nothing. Always send a payload with at least one
    truthy key, which is why we always include `"approved": True`.
    """
    plan = state["plan"]  # already a dict — see State's note
    assert plan is not None
    decision = interrupt({"plan": plan})
    if isinstance(decision, dict) and decision.get("plan"):
        logger.info("Plan approved WITH edits.")
        return {"plan": decision["plan"]}
    logger.info("Plan approved as-is.")
    return {}


# ============================================================
# 7e) Fanout
# ============================================================
def fanout(state: State):
    plan = state["plan"]
    assert plan is not None
    return [
        Send(
            "worker",
            {
                "task": task,
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": plan,
                "evidence": state.get("evidence", []),
            },
        )
        for task in plan["tasks"]
    ]


# ============================================================
# 7f) Worker subgraph: draft -> critic -> (revise ->) critic -> finalize
# ============================================================
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

- Cover ALL bullets in order. Target words +/-15%.
- Output only section markdown starting with "## <Section Title>".
- If blog_kind=="news_roundup", do NOT drift into tutorials — focus on events + implications.
- If mode=="open_book": only make claims supported by the given Evidence URLs,
  and cite each with a Markdown link ([Source](URL)). If unsupported, write
  "Not found in provided sources."
- If requires_citations==true, cite Evidence URLs for external claims.
- If requires_code==true, include at least one minimal snippet.
"""


def _worker_messages(task: Task, plan: Plan, payload: dict, evidence: List[EvidenceItem], feedback: Optional[str]) -> list:
    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}" for e in evidence[:20])
    feedback_block = (
        f"\n\nA previous draft had issues — rewrite the FULL section fixing them:\n{feedback}\n" if feedback else ""
    )
    return [
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(
            content=(
                f"Blog title: {plan.blog_title}\nAudience: {plan.audience}\nTone: {plan.tone}\n"
                f"Blog kind: {plan.blog_kind}\nConstraints: {plan.constraints}\n"
                f"Topic: {payload['topic']}\nMode: {payload.get('mode')}\n"
                f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                f"Section title: {task.title}\nGoal: {task.goal}\nTarget words: {task.target_words}\n"
                f"Tags: {task.tags}\nrequires_research: {task.requires_research}\n"
                f"requires_citations: {task.requires_citations}\nrequires_code: {task.requires_code}\n"
                f"Bullets:{bullets_text}\n\nEvidence (ONLY cite these URLs):\n{evidence_text}\n{feedback_block}"
            )
        ),
    ]


def draft_node(state: WorkerState) -> dict:
    task, plan = Task(**state["task"]), Plan(**state["plan"])
    evidence = [EvidenceItem(**e) for e in state.get("evidence", [])]
    logger.info("Drafting section task=%s title=%r", task.id, task.title)
    response = invoke_with_retry(llm, _worker_messages(task, plan, state, evidence, feedback=None))
    return {"draft": response.content.strip(), "attempt": 1}


def critic_node(state: WorkerState) -> dict:
    task = Task(**state["task"])
    evidence_urls = {e["url"] for e in state.get("evidence", [])}
    result = critique_section(state["draft"], task, evidence_urls)
    logger.info(
        "Critic task=%s attempt=%s passed=%s coverage=%.2f invalid_citations=%d",
        task.id, state.get("attempt"), result.passed, result.bullet_coverage, len(result.invalid_citations),
    )
    return {"critique": result.model_dump()}


def route_after_critic(state: WorkerState) -> str:
    critique = state.get("critique") or {}
    # Only one revision pass, ever — avoids infinite loops and runaway cost.
    if not critique.get("passed") and state.get("attempt", 1) < 2:
        return "revise"
    return "finalize"


def revise_node(state: WorkerState) -> dict:
    task, plan = Task(**state["task"]), Plan(**state["plan"])
    evidence = [EvidenceItem(**e) for e in state.get("evidence", [])]
    feedback = (state.get("critique") or {}).get("feedback")
    logger.info("Revising section task=%s based on critic feedback.", task.id)
    response = invoke_with_retry(llm, _worker_messages(task, plan, state, evidence, feedback=feedback))
    return {"draft": response.content.strip(), "attempt": state.get("attempt", 1) + 1}


def finalize_node(state: WorkerState) -> dict:
    """Last line of defense: strip any citation the critic may have missed, no matter what."""
    task = Task(**state["task"])
    evidence_urls = {e["url"] for e in state.get("evidence", [])}
    cleaned, invalid = verify_and_strip_citations(state["draft"], evidence_urls)
    if invalid:
        logger.warning("Task=%s: stripped %d unsupported citation(s): %s", task.id, len(invalid), invalid)
    return {"sections": [(task.id, cleaned)]}


worker_graph = StateGraph(WorkerState)
worker_graph.add_node("draft", draft_node)
worker_graph.add_node("critic", critic_node)
worker_graph.add_node("revise", revise_node)
worker_graph.add_node("finalize", finalize_node)
worker_graph.add_edge(START, "draft")
worker_graph.add_edge("draft", "critic")
worker_graph.add_conditional_edges("critic", route_after_critic, {"revise": "revise", "finalize": "finalize"})
worker_graph.add_edge("revise", "critic")
worker_graph.add_edge("finalize", END)
worker_subgraph = worker_graph.compile()


async def worker_node(payload: dict) -> dict:
    """
    Thin wrapper around worker_subgraph, used as the actual graph node
    instead of adding worker_subgraph directly.

    fanout() dispatches one Send("worker", {...}) per task, so several
    worker branches run concurrently. worker_subgraph's WorkerState shares
    several key names with the parent State (topic, mode, as_of,
    recency_days, plan, evidence). If worker_subgraph were added directly
    as a node, LangGraph would treat those as shared channels and try to
    write each parallel branch's copy of them back into the parent state —
    and since none of those keys are Annotated with a reducer (only
    `sections` is, via operator.add), concurrent writes to them raise
    InvalidUpdateError ("Can receive only one value per step"), even when
    every branch's value is identical.

    Calling worker_subgraph as a plain function and returning only
    {"sections": ...} sidesteps this entirely: each parallel branch then
    only ever writes to the one channel designed to accept concurrent
    writes.
    """
    result = await worker_subgraph.ainvoke(payload)
    return {"sections": result["sections"]}


# ============================================================
# 7g) Reducer subgraph: merge -> decide images -> generate images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]  # dict — see State's note
    assert plan is not None
    ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    return {"merged_md": f"# {plan['blog_title']}\n\n" + "\n\n".join(ordered).strip() + "\n"}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor. Decide if images/diagrams
are needed for THIS blog. Max 3, each must materially improve understanding.
Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]]. If none are
needed, md_with_placeholders must equal the input and images=[].

For system_design, tutorial, and comparison posts that describe an
architecture, workflow, or step-by-step process, you should almost always
include at least ONE diagram — readers benefit far more from a labeled
flow/architecture diagram than from an all-text explanation. Reserve
images=[] for purely conceptual or opinion pieces with nothing visual to show.
"""

# Small open-source models tend to play it safe under uncertainty, and an
# empty images list trivially satisfies GlobalImagePlan's schema while a
# correctly-filled ImageSpec (7 fields, 2 of them constrained enums) does
# not — so they often return images=[] even for clearly diagram-friendly
# topics. For blog kinds that almost always benefit from a diagram, force
# one in deterministically if the model didn't ask for any, rather than
# relying entirely on the model's judgment.
_DIAGRAM_FRIENDLY_KINDS = {"system_design", "tutorial", "comparison"}


def _default_image_spec(topic: str) -> ImageSpec:
    return ImageSpec(
        placeholder="[[IMAGE_1]]",
        filename="overview_diagram.png",
        alt=f"Diagram illustrating {topic}",
        caption=f"A high-level overview of {topic}.",
        prompt=(
            f"A clean, modern technical diagram illustrating {topic}. "
            "Flat design, labeled boxes and arrows, minimal color palette, "
            "white background, no photorealism, suitable for a blog post."
        ),
        size="1024x1024",
        quality="medium",
    )


def decide_images(state: State) -> dict:
    planner_input_plan = state["plan"]  # dict — see State's note
    assert planner_input_plan is not None
    image_plan: GlobalImagePlan = invoke_structured(
        GlobalImagePlan,
        DECIDE_IMAGES_SYSTEM,
        f"Blog kind: {planner_input_plan['blog_kind']}\nTopic: {state['topic']}\n\n{state['merged_md']}",
    )

    md = image_plan.md_with_placeholders
    images = list(image_plan.images)

    if not images and planner_input_plan.get("blog_kind") in _DIAGRAM_FRIENDLY_KINDS:
        logger.info(
            "Model requested 0 images for a %s post — inserting a default diagram.",
            planner_input_plan["blog_kind"],
        )
        spec = _default_image_spec(state["topic"])
        images = [spec]
        md = f"{spec.placeholder}\n\n{md}"  # place it right under the title

    logger.info("Image plan: %d image(s) requested.", len(images))
    return {"md_with_placeholders": md, "image_specs": [i.model_dump() for i in images]}


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    from google import genai
    from google.genai import types

    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    client = genai.Client(api_key=settings.google_api_key)
    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH")],
        ),
    )
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        parts = resp.candidates[0].content.parts
    for part in parts or []:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data
    raise RuntimeError("No inline image bytes found in response.")


def _safe_slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9 _-]+", "", title.strip().lower())
    return re.sub(r"\s+", "_", s).strip("_") or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]  # dict — see State's note
    assert plan is not None
    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []
    images_dir = settings.output_dir / settings.images_subdir
    images_dir.mkdir(parents=True, exist_ok=True)

    for spec in image_specs:
        out_path = images_dir / spec["filename"]
        if not out_path.exists():
            try:
                out_path.write_bytes(_gemini_generate_image_bytes(spec["prompt"]))
                logger.info("Generated image %s", out_path)
            except Exception as e:
                logger.exception("Image generation failed for %s", spec["filename"])
                md = md.replace(
                    spec["placeholder"],
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption', '')}\n> **Error:** {e}\n",
                )
                continue
        md = md.replace(spec["placeholder"], f"![{spec['alt']}]({settings.images_subdir}/{spec['filename']})\n*{spec['caption']}*")

    out_path = settings.output_dir / f"{_safe_slug(plan['blog_title'])}.md"
    out_path.write_text(md, encoding="utf-8")
    logger.info("Wrote final markdown to %s", out_path)
    return {"final": md}


reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()


# ============================================================
# 8) Assemble the main graph
# ============================================================
def build_graph(checkpointer) -> Any:
    g = StateGraph(State)
    g.add_node("router", router_node)
    g.add_node("research", research_node)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("plan_review", plan_review_node)
    g.add_node("worker", worker_node)
    g.add_node("reducer", reducer_subgraph)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
    g.add_edge("research", "orchestrator")
    g.add_edge("orchestrator", "plan_review")
    g.add_conditional_edges("plan_review", fanout, ["worker"])
    g.add_edge("worker", "reducer")
    g.add_edge("reducer", END)

    # checkpointer=... is what makes interrupt()/Command(resume=...) and
    # cross-request persistence possible: without it, graph state lives only
    # in memory for the duration of a single .astream_events() call.
    return g.compile(checkpointer=checkpointer)


# ============================================================
# 9) FastAPI app — async jobs + SSE streaming + plan approval
# ============================================================
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerateRequest(BaseModel):
    topic: str
    as_of: Optional[str] = None


class GenerateResponse(BaseModel):
    job_id: str


class ApprovePlanRequest(BaseModel):
    plan: Optional[dict] = None  # omit/None to approve as-is; pass an edited plan to override it


class JobRecord:
    """In-memory job record — fine for a demo/single process. For production,
    swap this for Redis so jobs survive restarts and multiple workers can share state."""

    def __init__(self, job_id: str, topic: str):
        self.job_id = job_id
        self.topic = topic
        self.status = JobStatus.PENDING
        self.events: List[Dict[str, Any]] = []
        self.result: Optional[str] = None
        self.error: Optional[str] = None
        self.pending_plan: Optional[dict] = None
        self.queue: "asyncio.Queue[Optional[Dict[str, Any]]]" = asyncio.Queue()


JOBS: Dict[str, JobRecord] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer:
        await checkpointer.setup()  # creates checkpoint tables if they don't exist yet
        app.state.graph = build_graph(checkpointer)
        logger.info("Graph compiled with Postgres checkpointer (%s).", settings.database_url)
        yield


app = FastAPI(title="Blog Writer Agent", version="0.2.0", lifespan=lifespan)


async def _stream_graph(job: JobRecord, graph_input: Any, config: dict) -> None:
    """Drives one segment of the graph (initial run OR a resume-after-approval
    run) and fans events out to /jobs/{id}/stream, then checks whether the
    graph finished or paused on a human-approval interrupt."""
    job.status = JobStatus.RUNNING
    graph = app.state.graph
    # astream_events fires one on_chat_model_stream event PER TOKEN generated.
    # Forwarding those to the SSE stream floods slow frontends (and
    # Streamlit in particular) with hundreds/thousands of lines that make a
    # perfectly healthy run look "stuck". We only care about node/model
    # start & end boundaries for progress reporting, so drop token events.
    _NOISY_EVENTS = {"on_chat_model_stream", "on_llm_stream"}
    try:
        async for event in graph.astream_events(graph_input, config=config, version="v2"):
            if event.get("event") in _NOISY_EVENTS:
                continue
            simplified: Dict[str, Any] = {"event": event.get("event"), "name": event.get("name")}
            if event.get("event") == "on_chain_end":
                output = (event.get("data", {}) or {}).get("output")
                if isinstance(output, dict):
                    if "sections" in output:
                        simplified["sections_added"] = len(output["sections"])
                    if output.get("plan") is not None:
                        simplified["plan_ready"] = True
                    if "final" in output:
                        simplified["final_ready"] = True
                        job.result = output["final"]
            job.events.append(simplified)
            await job.queue.put(simplified)

        snapshot = await graph.aget_state(config)
        if snapshot.next:
            # Graph paused at plan_review's interrupt() call — surface the plan.
            interrupt_payload = next(
                (t.interrupts[0].value for t in snapshot.tasks if t.interrupts), None
            )
            job.pending_plan = (interrupt_payload or {}).get("plan")
            job.status = JobStatus.WAITING_APPROVAL
            await job.queue.put({"event": "on_plan_ready_for_approval"})
        else:
            job.status = JobStatus.COMPLETED

    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed", job.job_id)
        job.status = JobStatus.FAILED
        job.error = str(exc)
        await job.queue.put({"event": "on_error", "error": str(exc)})
    finally:
        await job.queue.put(None)  # sentinel: this streaming segment is done


@app.post("/generate", response_model=GenerateResponse, status_code=202)
async def generate(req: GenerateRequest) -> GenerateResponse:
    job_id = str(uuid.uuid4())
    job = JOBS[job_id] = JobRecord(job_id, req.topic)
    initial_state = {"topic": req.topic, "as_of": req.as_of or date.today().isoformat(), "sections": []}
    config = {"configurable": {"thread_id": job_id}}  # thread_id ties this run to its checkpoint
    asyncio.create_task(_stream_graph(job, initial_state, config))
    logger.info("Queued job %s for topic=%r", job_id, req.topic)
    return GenerateResponse(job_id=job_id)


@app.post("/jobs/{job_id}/approve")
async def approve_plan(job_id: str, req: ApprovePlanRequest) -> Dict[str, str]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.WAITING_APPROVAL:
        raise HTTPException(status_code=400, detail=f"Job is not waiting for approval (status={job.status}).")

    config = {"configurable": {"thread_id": job_id}}
    # Always a non-empty/truthy payload — see the note in plan_review_node
    # about why `Command(resume={})` silently fails to resume.
    resume_value: Dict[str, Any] = {"approved": True}
    if req.plan:
        resume_value["plan"] = req.plan
    job.queue = asyncio.Queue()  # fresh queue so /stream clients see only this segment's events
    asyncio.create_task(_stream_graph(job, Command(resume=resume_value), config))
    logger.info("Resuming job %s after plan approval (edited=%s).", job_id, bool(req.plan))
    return {"status": "resumed"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.job_id,
        "topic": job.topic,
        "status": job.status,
        "num_events": len(job.events),
        "pending_plan": job.pending_plan if job.status == JobStatus.WAITING_APPROVAL else None,
        "result": job.result,
        "error": job.error,
    }


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str) -> StreamingResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        for evt in list(job.events):
            yield f"data: {json.dumps(evt)}\n\n"
        while True:
            evt = await job.queue.get()
            if evt is None:
                yield f"data: {json.dumps({'event': 'on_stream_end'})}\n\n"
                break
            yield f"data: {json.dumps(evt)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}