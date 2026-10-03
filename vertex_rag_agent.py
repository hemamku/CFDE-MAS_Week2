"""Vertex AI / Google ADK RAG agent for GCP deployment.

This module builds a Google ADK agent that uses Gemini 2.5 Flash on Vertex AI
and retrieves facts from a local document directory before answering.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

CONFIG_PATH = Path(__file__).with_name("vertex_rag_agent.yaml")
DEFAULT_DOCS_DIR = Path(__file__).with_name("docs")


def resolve_gcloud_command() -> list[str] | None:
    """Find the gcloud executable across common Windows install locations."""
    candidates = []

    gcloud_path = shutil.which("gcloud")
    if gcloud_path:
        candidates.append(gcloud_path)

    if os.name == "nt":
        candidates.extend(
            [
                str(Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Cloud SDK" / "google-cloud-sdk" / "bin" / "gcloud.cmd"),
                str(Path("C:/Program Files/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd")),
                str(Path("C:/Users/HEMA/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd")),
            ]
        )

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return [str(candidate)]

    return None


def detect_active_gcloud_project() -> str:
    """Look up the active project from the local gcloud configuration."""
    command = resolve_gcloud_command()
    if not command:
        return ""

    try:
        result = subprocess.run(
            command + ["config", "get-value", "project"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            project = result.stdout.strip()
            if project:
                return project
    except Exception:
        pass

    return ""


def ensure_vertex_ai_env(project: str | None = None, location: str | None = None) -> tuple[str, str]:
    """Ensure the Google ADK app uses Vertex AI rather than the Gemini API key flow."""
    project = project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    location = location or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"

    if not project:
        project = detect_active_gcloud_project()

    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    os.environ["GOOGLE_CLOUD_LOCATION"] = location

    if not project:
        raise RuntimeError(
            "Missing GOOGLE_CLOUD_PROJECT. Set it before running the Vertex AI agent, e.g.:\n"
            "$env:GOOGLE_CLOUD_PROJECT = 'your-project-id'\n"
            "$env:GOOGLE_CLOUD_LOCATION = 'us-central1'\n"
            "$env:GOOGLE_GENAI_USE_VERTEXAI = 'true'\n"
            "gcloud auth application-default login\n"
            "gcloud config set project YOUR_PROJECT_ID\n"
            "or pass --project your-project-id --location us-central1"
        )

    os.environ["GOOGLE_CLOUD_PROJECT"] = project
    return project, location


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_text_from_file(file_path: str | os.PathLike[str]) -> str:
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("Install pypdf to read PDF files: pip install pypdf")
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [normalized[i : i + chunk_size] for i in range(0, len(normalized), chunk_size)]


def retrieve_documents(query: str, docs_dir: str | None = None, top_k: int = 5) -> str:
    """Return a ranked list of relevant document chunks as JSON.

    This acts as the retrieval tool used by the ADK agent.
    """
    docs_root = Path(docs_dir or os.getenv("VERTEX_RAG_DOCS_DIR", str(DEFAULT_DOCS_DIR)))
    if not docs_root.exists():
        return json.dumps({"results": [], "message": f"No docs directory found at {docs_root}"}, ensure_ascii=False)

    query_terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    scored: list[tuple[float, dict[str, str]]] = []

    for file_path in sorted(docs_root.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        try:
            text = extract_text_from_file(file_path)
        except Exception:
            continue

        for index, chunk in enumerate(chunk_text(text)):
            chunk_text_norm = chunk.lower()
            overlap = len(set(re.findall(r"[a-zA-Z0-9]+", chunk_text_norm)) & query_terms)
            score = overlap + sum(2 for term in query_terms if term in chunk_text_norm)
            if score > 0:
                scored.append(
                    (
                        score,
                        {
                            "source": str(file_path),
                            "chunk_index": str(index),
                            "content": chunk,
                        },
                    )
                )

    ranked = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]
    results = [item for _, item in ranked]
    return json.dumps({"results": results}, ensure_ascii=False)


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def build_agent(project: str | None = None, location: str | None = None) -> Agent:
    ensure_vertex_ai_env(project=project, location=location)
    config = load_config()
    return Agent(
        name=config["name"],
        model=os.getenv("GOOGLE_ADK_MODEL", config["model"]),
        description=config["description"],
        instruction=config["instruction"],
        tools=[retrieve_documents],
    )


async def ask_rag(question: str, docs_dir: str | None = None, project: str | None = None, location: str | None = None) -> str:
    """Execute the ADK agent and return the final model answer."""
    ensure_vertex_ai_env(project=project, location=location)
    config = load_config()
    agent = build_agent(project=project, location=location)
    runner = Runner(
        app_name=config["app_name"],
        agent=agent,
        session_service=InMemorySessionService(),
    )

    session_id = "rag-session"
    user_id = "rag-user"
    await runner.session_service.create_session(
        app_name=config["app_name"],
        user_id=user_id,
        session_id=session_id,
    )

    resolved_docs_dir = docs_dir or os.getenv("VERTEX_RAG_DOCS_DIR", str(DEFAULT_DOCS_DIR))
    retrieval_context = retrieve_documents(question, resolved_docs_dir)
    prompt = (
        f"Use the following retrieved context to answer the user's question. "
        f"If the answer is not supported by the context, say so clearly.\n\n"
        f"Retrieved context:\n{retrieval_context}\n\n"
        f"User question: {question}"
    )

    final_text = ""
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text or final_text

    return final_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex AI Gemini RAG agent")
    parser.add_argument("--question", default="Summarize the policy documents in the knowledge base.", help="Question to ask the agent")
    parser.add_argument("--docs-dir", default=os.getenv("VERTEX_RAG_DOCS_DIR", str(DEFAULT_DOCS_DIR)), help="Directory containing knowledge-base docs")
    parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT_ID"), help="Google Cloud project ID for Vertex AI")
    parser.add_argument("--location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"), help="Vertex AI region, for example us-central1")
    args = parser.parse_args()

    ensure_vertex_ai_env(project=args.project, location=args.location)
    print(asyncio.run(ask_rag(args.question, args.docs_dir, project=args.project, location=args.location)))


if __name__ == "__main__":
    main()
