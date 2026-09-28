from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# ============================================================
# This frontend talks to backend.py over HTTP.
#
# backend.py is a FastAPI service (run separately with
# `uvicorn backend:app --reload`), not a Python module we import. The graph
# itself now runs as an async background job with Postgres-backed
# persistence, so the flow here is:
#
#   1. POST /generate                 -> {job_id}
#   2. GET  /jobs/{id}/stream (SSE)   -> progress events for ONE segment
#      of the run (either "start -> plan_review interrupt" or
#      "resume -> finish"), ending with an "on_stream_end" sentinel.
#   3. GET  /jobs/{id}                -> current status/pending_plan/result
#   4. If status == "waiting_approval": show the plan, let the user edit
#      it, then POST /jobs/{id}/approve with the (possibly edited) plan.
#      Go back to step 2 to stream the rest of the run.
#   5. If status == "completed": job["result"] has the final markdown.
#
# The API does not expose evidence or per-image detail, only the plan
# (while pending approval) and the final markdown — so this frontend
# doesn't have an Evidence tab. Images are generated to disk by the
# backend and referenced in the markdown as relative paths
# ("images/xxx.png"), so we read them straight from BLOG_OUTPUT_DIR on
# the same machine/filesystem as the backend.
# ============================================================

BLOG_KIND_OPTIONS = ["explainer", "tutorial", "news_roundup", "comparison", "system_design"]


# -----------------------------
# HTTP helpers
# -----------------------------
def api_generate(base_url: str, topic: str, as_of: str) -> str:
    resp = requests.post(f"{base_url}/generate", json={"topic": topic, "as_of": as_of}, timeout=30)
    resp.raise_for_status()
    return resp.json()["job_id"]


def api_get_job(base_url: str, job_id: str) -> Dict[str, Any]:
    resp = requests.get(f"{base_url}/jobs/{job_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_approve(base_url: str, job_id: str, plan: Optional[dict]) -> None:
    body = {"plan": plan} if plan else {}
    resp = requests.post(f"{base_url}/jobs/{job_id}/approve", json=body, timeout=30)
    resp.raise_for_status()


def api_health(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=3)
        return resp.ok
    except Exception:
        return False


def consume_stream(base_url: str, job_id: str) -> List[Dict[str, Any]]:
    """
    Blocks until the CURRENT streaming segment ends (the backend always
    closes with an "on_stream_end" sentinel — see backend.py's
    event_generator), then returns every event seen.
    """
    events: List[Dict[str, Any]] = []
    try:
        with requests.get(f"{base_url}/jobs/{job_id}/stream", stream=True, timeout=900) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                events.append(evt)
                if evt.get("event") == "on_stream_end":
                    break
    except Exception as e:  # noqa: BLE001
        events.append({"event": "on_client_stream_error", "error": str(e)})
    return events


def describe_event(evt: Dict[str, Any]) -> str:
    ev, name = evt.get("event") or "", evt.get("name") or ""
    line = f"`{ev}`" + (f" — **{name}**" if name else "")
    if evt.get("plan_ready"):
        line += "  🧩 plan ready"
    if "sections_added" in evt:
        line += f"  📄 +{evt['sections_added']} section(s)"
    if evt.get("final_ready"):
        line += "  ✅ final markdown ready"
    if ev == "on_plan_ready_for_approval":
        line += "  ⏸️ waiting on your approval"
    if ev == "on_error":
        line += f"  ❌ {evt.get('error')}"
    if ev == "on_client_stream_error":
        line += f"  ⚠️ connection issue: {evt.get('error')}"
    return line


# -----------------------------
# Local filesystem helpers (backend writes markdown/images here)
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def referenced_images(md: str, output_dir: Path) -> List[Tuple[str, Path]]:
    """Every local (non-http) image path the markdown references, resolved against output_dir."""
    found = []
    for m in _MD_IMG_RE.finditer(md):
        src = (m.group("src") or "").strip()
        if src.startswith("http://") or src.startswith("https://"):
            continue
        found.append((m.group("alt") or "", (output_dir / src.lstrip("./")).resolve()))
    return found


def render_markdown_with_local_images(md: str, output_dir: Path) -> None:
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last: m.start()]
        if before:
            parts.append(("md", before))
        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()
    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]
        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)
        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = (output_dir / src.lstrip("./")).resolve()
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")
        i += 1


def bundle_zip(md_text: str, md_filename: str, image_paths: List[Path]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        for p in image_paths:
            if p.exists():
                z.write(p, arcname=f"images/{p.name}")
    return buf.getvalue()


def images_only_zip(image_paths: List[Path]) -> Optional[bytes]:
    existing = [p for p in image_paths if p.exists()]
    if not existing:
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in existing:
            z.write(p, arcname=p.name)
    return buf.getvalue()


def list_past_blogs(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    files = [p for p in output_dir.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


# -----------------------------
# Plan review form (used for the human-in-the-loop approval step)
# -----------------------------
def render_plan_editor(plan: dict, key_prefix: str) -> Tuple[dict, bool, bool]:
    """Renders an editable plan; returns (edited_plan_dict, approve_as_is_clicked, approve_edited_clicked)."""
    with st.form(key=f"{key_prefix}_form"):
        blog_title = st.text_input("Blog title", value=plan.get("blog_title", ""))
        c1, c2 = st.columns(2)
        audience = c1.text_input("Audience", value=plan.get("audience", ""))
        tone = c2.text_input("Tone", value=plan.get("tone", ""))
        kind_val = plan.get("blog_kind", "explainer")
        blog_kind = st.selectbox(
            "Blog kind", BLOG_KIND_OPTIONS,
            index=BLOG_KIND_OPTIONS.index(kind_val) if kind_val in BLOG_KIND_OPTIONS else 0,
        )
        constraints_text = st.text_area(
            "Constraints (one per line)", value="\n".join(plan.get("constraints", []) or []), height=80,
        )

        st.markdown("**Sections**")
        edited_tasks = []
        for t in plan.get("tasks", []) or []:
            tid = t.get("id")
            with st.expander(f"Section {tid}: {t.get('title', '')}", expanded=False):
                title = st.text_input("Title", value=t.get("title", ""), key=f"{key_prefix}_title_{tid}")
                goal = st.text_area("Goal", value=t.get("goal", ""), key=f"{key_prefix}_goal_{tid}", height=60)
                bullets_text = st.text_area(
                    "Bullets (one per line)", value="\n".join(t.get("bullets", []) or []),
                    key=f"{key_prefix}_bullets_{tid}", height=100,
                )
                wc1, wc2 = st.columns(2)
                target_words = wc1.number_input(
                    "Target words", min_value=50, max_value=2000, step=10,
                    value=int(t.get("target_words", 200)), key=f"{key_prefix}_words_{tid}",
                )
                tags_text = wc2.text_input(
                    "Tags (comma-separated)", value=", ".join(t.get("tags", []) or []), key=f"{key_prefix}_tags_{tid}",
                )
                fc1, fc2, fc3 = st.columns(3)
                requires_research = fc1.checkbox("Needs research", value=bool(t.get("requires_research")), key=f"{key_prefix}_rr_{tid}")
                requires_citations = fc2.checkbox("Needs citations", value=bool(t.get("requires_citations")), key=f"{key_prefix}_rc_{tid}")
                requires_code = fc3.checkbox("Needs code", value=bool(t.get("requires_code")), key=f"{key_prefix}_code_{tid}")

                edited_tasks.append({
                    "id": tid,
                    "title": title,
                    "goal": goal,
                    "bullets": [b.strip() for b in bullets_text.splitlines() if b.strip()],
                    "target_words": int(target_words),
                    "tags": [x.strip() for x in tags_text.split(",") if x.strip()],
                    "requires_research": requires_research,
                    "requires_citations": requires_citations,
                    "requires_code": requires_code,
                })

        bcol1, bcol2 = st.columns(2)
        approve_as_is = bcol1.form_submit_button("✅ Approve as-is")
        approve_edited = bcol2.form_submit_button("✏️ Approve with my edits")

    edited_plan = {
        "blog_title": blog_title,
        "audience": audience,
        "tone": tone,
        "blog_kind": blog_kind,
        "constraints": [c.strip() for c in constraints_text.splitlines() if c.strip()],
        "tasks": edited_tasks,
    }
    return edited_plan, approve_as_is, approve_edited


# ============================================================
# Streamlit app
# ============================================================
st.set_page_config(page_title="Blog Writer Agent", layout="wide")
st.title("📝 Blog Writer Agent")

for key, default in {
    "base_url": "http://localhost:8000",
    "output_dir": "./blog_output",
    "job_id": None,
    "status": None,           # None | running | waiting_approval | completed | failed
    "pending_plan": None,
    "approved_plan": None,
    "final_md": None,
    "error": None,
    "event_log": [],
    "last_job": None,
    "last_topic": "",
    "last_as_of": None,
}.items():
    st.session_state.setdefault(key, default)

# -----------------------------
# Sidebar: connection + generate + past blogs
# -----------------------------
with st.sidebar:
    st.header("⚙️ Connection")
    base_url = st.text_input("Backend URL", value=st.session_state["base_url"]).rstrip("/")
    st.session_state["base_url"] = base_url
    output_dir_str = st.text_input(
        "Output dir",
        value=st.session_state["output_dir"],
        help="Must match the backend's BLOG_OUTPUT_DIR — markdown and images are read straight from this folder.",
    )
    st.session_state["output_dir"] = output_dir_str
    output_dir = Path(output_dir_str)

    if api_health(base_url):
        st.success("Backend reachable", icon="✅")
    else:
        st.error("Backend not reachable at this URL", icon="⚠️")

    st.divider()
    st.header("🆕 Generate blog")
    topic = st.text_area("Topic", height=100, key="topic_input")
    as_of = st.date_input("As-of date", value=date.today())
    gen_disabled = st.session_state["status"] == "running"

    if st.button("🚀 Generate Blog", type="primary", disabled=gen_disabled):
        if not topic.strip():
            st.warning("Please enter a topic.")
            st.stop()
        for key in ("pending_plan", "approved_plan", "final_md", "error"):
            st.session_state[key] = None
        st.session_state["event_log"] = []
        st.session_state["last_topic"] = topic.strip()
        st.session_state["last_as_of"] = as_of.isoformat()
        try:
            job_id = api_generate(base_url, topic.strip(), as_of.isoformat())
        except Exception as e:
            st.error(f"Could not reach backend at {base_url}: {e}")
            st.stop()
        st.session_state["job_id"] = job_id
        st.session_state["status"] = "running"
        st.rerun()

    st.divider()
    st.header("📂 Past blogs")
    past_files = list_past_blogs(output_dir)
    if not past_files:
        st.caption(f"No *.md files found in `{output_dir}`.")
    else:
        options, file_by_label = [], {}
        for p in past_files[:50]:
            try:
                title = extract_title_from_md(p.read_text(encoding="utf-8", errors="replace"), p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p
        selected_label = st.radio("Select a blog to load", options=options, index=0, label_visibility="collapsed")
        if st.button("📂 Load selected blog"):
            selected = file_by_label[selected_label]
            st.session_state["final_md"] = selected.read_text(encoding="utf-8", errors="replace")
            st.session_state["job_id"] = None
            st.session_state["status"] = "completed"
            st.session_state["approved_plan"] = None
            st.rerun()

# -----------------------------
# Drive the job: stream the current segment, then refresh status
# -----------------------------
job_id = st.session_state["job_id"]

if job_id and st.session_state["status"] == "running":
    with st.status("Running graph…", expanded=True) as status_box:
        events = consume_stream(base_url, job_id)
        for evt in events:
            status_box.write(describe_event(evt))
        st.session_state["event_log"].extend(events)

    try:
        job = api_get_job(base_url, job_id)
    except Exception as e:
        st.error(f"Could not fetch job status: {e}")
        st.stop()

    st.session_state["last_job"] = job
    st.session_state["status"] = job["status"]
    st.session_state["pending_plan"] = job.get("pending_plan")
    st.session_state["error"] = job.get("error")
    if job.get("result"):
        st.session_state["final_md"] = job["result"]
    st.rerun()

# -----------------------------
# Human-in-the-loop plan approval
# -----------------------------
if job_id and st.session_state["status"] == "waiting_approval" and st.session_state["pending_plan"]:
    st.subheader("🧩 Review the plan before writing begins")
    st.caption("Edit any section below, or approve it as generated.")
    edited_plan, approve_as_is, approve_edited = render_plan_editor(st.session_state["pending_plan"], key_prefix="review")

    if approve_as_is or approve_edited:
        plan_to_send = edited_plan if approve_edited else None
        try:
            api_approve(base_url, job_id, plan_to_send)
        except Exception as e:
            st.error(f"Could not approve the plan: {e}")
            st.stop()
        st.session_state["approved_plan"] = edited_plan if approve_edited else st.session_state["pending_plan"]
        st.session_state["pending_plan"] = None
        st.session_state["status"] = "running"
        st.rerun()

# -----------------------------
# Failed state
# -----------------------------
if job_id and st.session_state["status"] == "failed":
    st.error(f"Job failed: {st.session_state.get('error') or 'unknown error'}")
    if st.button("🔁 Try again with the same topic"):
        try:
            new_job_id = api_generate(base_url, st.session_state["last_topic"], st.session_state["last_as_of"])
        except Exception as e:
            st.error(f"Could not reach backend at {base_url}: {e}")
            st.stop()
        st.session_state["job_id"] = new_job_id
        st.session_state["status"] = "running"
        for key in ("pending_plan", "approved_plan", "final_md", "error"):
            st.session_state[key] = None
        st.session_state["event_log"] = []
        st.rerun()

# -----------------------------
# Result tabs (works for a completed job OR a loaded past blog)
# -----------------------------
final_md = st.session_state.get("final_md")
if final_md:
    tab_plan, tab_preview, tab_images, tab_logs = st.tabs(
        ["🧩 Plan", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs"]
    )

    with tab_plan:
        plan_dict = st.session_state.get("approved_plan")
        if not plan_dict:
            st.info("No plan available for this blog (e.g. it was loaded from a saved file).")
        else:
            st.write("**Title:**", plan_dict.get("blog_title"))
            c1, c2, c3 = st.columns(3)
            c1.write("**Audience:** " + str(plan_dict.get("audience")))
            c2.write("**Tone:** " + str(plan_dict.get("tone")))
            c3.write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))
            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame([
                    {
                        "id": t.get("id"),
                        "title": t.get("title"),
                        "target_words": t.get("target_words"),
                        "requires_research": t.get("requires_research"),
                        "requires_citations": t.get("requires_citations"),
                        "requires_code": t.get("requires_code"),
                        "tags": ", ".join(t.get("tags") or []),
                    }
                    for t in tasks
                ]).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)
                with st.expander("Full task details"):
                    st.json(tasks)

    with tab_preview:
        render_markdown_with_local_images(final_md, output_dir)
        blog_title = extract_title_from_md(final_md, (st.session_state.get("approved_plan") or {}).get("blog_title", "blog"))
        md_filename = f"{safe_slug(blog_title)}.md"
        image_paths = [p for _, p in referenced_images(final_md, output_dir)]

        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download Markdown", data=final_md.encode("utf-8"), file_name=md_filename, mime="text/markdown")
        bundle = bundle_zip(final_md, md_filename, image_paths)
        d2.download_button(
            "📦 Download Bundle (MD + images)", data=bundle,
            file_name=f"{safe_slug(blog_title)}_bundle.zip", mime="application/zip",
        )

    with tab_images:
        image_paths = [p for _, p in referenced_images(final_md, output_dir)]
        if not image_paths:
            st.info("No images referenced in this blog's markdown.")
        else:
            missing = [p for p in image_paths if not p.exists()]
            if missing:
                st.warning(f"{len(missing)} referenced image(s) not found on disk (generation may have failed for these).")
            for p in image_paths:
                if p.exists():
                    st.image(str(p), caption=p.name, use_container_width=True)
            zdata = images_only_zip(image_paths)
            if zdata:
                st.download_button("⬇️ Download Images (zip)", data=zdata, file_name="images.zip", mime="application/zip")

    with tab_logs:
        st.subheader("Job")
        if st.session_state.get("last_job"):
            st.json(st.session_state["last_job"])
        st.subheader("Event stream")
        log_text = "\n".join(describe_event(e).replace("**", "").replace("`", "") for e in st.session_state["event_log"])
        st.text_area("Events", value=log_text or "(no events captured)", height=300)

elif not job_id:
    st.info("Enter a topic in the sidebar and click **Generate Blog**, or load a past blog.")